// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use anyhow::{anyhow, Context, Result};
use cubic_world::{
    Chunk, ChunkLocalPos, ChunkPos, FaceDef, WorldGenerator, CHUNK_SIZE, CHUNK_VOLUME,
};
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use wasmtime::{Engine, Linker, Memory, Module, Store, TypedFunc};

// ---------------------------------------------------------------------------
// WIT-mangled names
// ---------------------------------------------------------------------------
//
// We call into the guest's core wasm module directly (no Component Model on
// the host side), but the guest still uses wit-bindgen, which mangles names
// for interfaces declared in the WIT world. Verified against an actual
// compiled guest (`wasm-tools print` on a wit-bindgen-generated module for
// this exact wit/game.wit): imports are named
// "<namespace>:<package>/<interface>@<version>" (function name unmangled,
// since core wasm imports already have a two-part module/name key), and
// *exports* are named "<namespace>:<package>/<interface>@<version>#<function>"
// (core wasm exports have only a flat name, so wit-bindgen folds the
// interface path in). These must stay in sync with the `package
// cubic:game@0.1.0` declaration in wit/game.wit.
const IMPORT_BLOCK_REGISTRY_MODULE: &str = "cubic:game/block-registry@0.1.0";
const EXPORT_ON_LOAD: &str = "cubic:game/world-gen@0.1.0#on-load";
const EXPORT_GENERATE: &str = "cubic:game/world-gen@0.1.0#generate";
const EXPORT_IS_DEFINITELY_AIR: &str = "cubic:game/world-gen@0.1.0#is-definitely-air";
const IMPORT_DATA_MODULE: &str = "cubic:game/data@0.1.0";
const IMPORT_DATA_LIST_DIR: &str = "cubic:game/data@0.1.0";
const IMPORT_PHYSICS_MODULE: &str = "cubic:game/physics@0.1.0";
const IMPORT_INPUT_MODULE: &str = "cubic:game/input@0.1.0";
const IMPORT_CAMERA_MODULE: &str = "cubic:game/camera@0.1.0";
const EXPORT_ON_TICK: &str = "cubic:game/tick@0.1.0#on-tick";

// ---------------------------------------------------------------------------
// Memory layout
// ---------------------------------------------------------------------------

/// CHUNK_VOLUME * 4 bytes per chunk output buffer.
const CHUNK_BUFFER_BYTES: usize = CHUNK_VOLUME * 4;

/// How many bytes to reserve at the start of WASM memory for the guest's
/// own stack and heap before our output buffers begin.
const GUEST_RESERVED_BYTES: usize = 2 * 1024 * 1024; // 2MB

pub struct WasmMemoryLayout {
    worker_count: usize,
    /// Byte offset in WASM linear memory where per-worker output buffers start.
    buffer_base: usize,
}

impl WasmMemoryLayout {
    pub fn new(worker_count: usize) -> Self {
        Self {
            worker_count,
            buffer_base: GUEST_RESERVED_BYTES,
        }
    }

    /// Total bytes needed for output buffers.
    pub fn buffer_bytes(&self) -> usize {
        self.worker_count * CHUNK_BUFFER_BYTES
    }

    /// Minimum total WASM memory in bytes.
    pub fn min_memory_bytes(&self) -> usize {
        GUEST_RESERVED_BYTES + self.buffer_bytes()
    }

    /// Byte offset in WASM linear memory for worker `id`'s output buffer.
    pub fn worker_buffer_offset(&self, worker_id: usize) -> u32 {
        (self.buffer_base + worker_id * CHUNK_BUFFER_BYTES) as u32
    }
}

// ---------------------------------------------------------------------------
// Host state (passed into Store)
// ---------------------------------------------------------------------------
pub struct DataStore {
    /// Data directories in load order. Later entries win for the same path.
    dirs: Vec<std::path::PathBuf>,
}

impl DataStore {
    pub fn new(game_dir: &std::path::Path) -> Self {
        Self {
            dirs: vec![game_dir.join("data")],
        }
    }

    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        // Iterate in reverse — last dir wins
        for dir in self.dirs.iter().rev() {
            let full = dir.join(path);
            if full.exists() {
                return std::fs::read(&full).ok();
            }
        }
        None
    }

    pub fn add_mod_dir(&mut self, mod_dir: &std::path::Path) {
        self.dirs.push(mod_dir.join("data"));
    }

    pub fn list_dir(&self, path: &str) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        // Iterate in reverse so later dirs (mods) override earlier (base game)
        // For listing we want all files, deduped by filename — later wins
        for dir in self.dirs.iter().rev() {
            let full = dir.join(path);
            if let Ok(entries) = std::fs::read_dir(&full) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if seen.insert(name.clone()) {
                        result.push(name);
                    }
                }
            }
        }
        result.sort(); // deterministic load order
        result
    }
}

struct HostState {
    block_registry: Arc<Mutex<cubic_world::BlockRegistry>>,
    data_store: Arc<DataStore>,
}

// ---------------------------------------------------------------------------
// Tick-scoped chunk query (thread-local)
// ---------------------------------------------------------------------------
//
// Physics host functions (is-solid, sweep-aabb) need read access to live
// chunk data during on_tick, but func_wrap closures must be 'static while
// the borrow they need is scoped to a single tick call. A thread-local raw
// pointer bridges that gap without locking or copying chunk data.
//
// Safety: CHUNK_QUERY_PTR holds a raw pointer to a ChunkQueryView that
// borrows AsyncWorldStream::inner.chunks. This is valid because:
// 1. set_tick_query is called before any physics host functions
// 2. clear_tick_query is called before the streaming update mutates chunks
// 3. All of this happens on the main thread sequentially
// If this ever moves to worker threads, replace with Arc<RwLock<>> instead.
thread_local! {
    static CHUNK_QUERY_PTR: Cell<Option<*const dyn cubic_world::ChunkQuery>> =
        const { Cell::new(None) };
}

/// Set the chunk query pointer for the current tick. Must be cleared after
/// with `clear_tick_query` before the borrow it points to ends.
pub fn set_tick_query(q: &dyn cubic_world::ChunkQuery) {
    // Safety: erasing the borrow's lifetime to 'static here is sound only
    // because every caller clears it (via clear_tick_query) before the
    // borrow it points to actually ends — see the module-level safety note
    // above. Both sides of the transmute are fat pointers of identical
    // layout (data ptr + vtable ptr); only the lifetime bound differs.
    let ptr: *const dyn cubic_world::ChunkQuery = unsafe { std::mem::transmute(q) };
    CHUNK_QUERY_PTR.with(|c| c.set(Some(ptr)));
}

pub fn clear_tick_query() {
    CHUNK_QUERY_PTR.with(|c| c.set(None));
}

fn with_chunk_query<F, R>(f: F) -> R
where
    F: FnOnce(Option<&dyn cubic_world::ChunkQuery>) -> R,
{
    CHUNK_QUERY_PTR.with(|c| match c.get() {
        None => f(None),
        // Safety: see the module-level safety note on CHUNK_QUERY_PTR.
        Some(ptr) => f(Some(unsafe { &*ptr })),
    })
}

// ---------------------------------------------------------------------------
// Tick-scoped input/camera (thread-local)
// ---------------------------------------------------------------------------
//
// Same pattern as CHUNK_QUERY_PTR above, but these carry plain Copy data
// rather than a borrow, so no unsafe erasure is needed: set_tick_input is
// called before on_tick, the guest reads it via get-input during the tick,
// and any set-camera call during that same tick stashes its result here for
// cubic-app to pick up right after via take_camera_update.
thread_local! {
    static TICK_INPUT: Cell<InputSnapshot> = Cell::new(InputSnapshot::default());
    static CAMERA_UPDATE: Cell<Option<CameraUpdate>> = const { Cell::new(None) };
}

#[derive(Clone, Copy, Default)]
pub struct InputSnapshot {
    pub move_forward: bool,
    pub move_back: bool,
    pub move_left: bool,
    pub move_right: bool,
    pub jump: bool,
    pub sneak: bool,
    pub look_dx: f32,
    pub look_dy: f32,
}

#[derive(Clone, Copy)]
pub struct CameraUpdate {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// Set the input snapshot for the current tick. cubic-app calls this once
/// per frame, immediately before calling `WasmWorldGenerator::tick`.
pub fn set_tick_input(input: InputSnapshot) {
    TICK_INPUT.with(|c| c.set(input));
}

/// Take (and clear) the camera update the guest set via `set-camera` during
/// the most recent tick, if any.
pub fn take_camera_update() -> Option<CameraUpdate> {
    CAMERA_UPDATE.with(|c| c.take())
}

// ---------------------------------------------------------------------------
// Per-worker WASM instance
// ---------------------------------------------------------------------------

struct WasmInstance {
    store: Store<HostState>,
    memory: Memory,
    fn_generate: TypedFunc<(u32, i32, i32, i32, u32), u32>,
    fn_is_definitely_air: TypedFunc<(u32, i32, i32, i32), u32>,
    // None until cubic-game implements the `tick` interface's `on-tick`
    // export — see the `.ok()` lookup below. tick() is then a no-op.
    fn_on_tick: Option<TypedFunc<f32, ()>>,
    generator_handle: u32,
}

impl WasmInstance {
    fn new(
        engine: &Engine,
        module: &Module,
        block_registry: Arc<Mutex<cubic_world::BlockRegistry>>,
        data_store: Arc<DataStore>,
        memory_bytes: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut store = Store::new(
            engine,
            HostState {
                block_registry: Arc::clone(&block_registry),
                data_store,
            },
        );

        let mut linker = Linker::new(engine);

        // Host function: block_registry::register_block
        linker.func_wrap(
            IMPORT_BLOCK_REGISTRY_MODULE,
            "register-block",
            |mut caller: wasmtime::Caller<'_, HostState>, ptr: i32, len: i32| -> i32 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");
                let data = mem.data(&caller);
                let name_bytes = &data[ptr as usize..(ptr + len) as usize];
                let name = std::str::from_utf8(name_bytes).unwrap_or("unknown");
                let mut reg = caller.data().block_registry.lock().unwrap();
                let id = reg.register(name);
                id.0 as i32
            },
        )?;

        linker.func_wrap(
            IMPORT_BLOCK_REGISTRY_MODULE,
            "register-block-with-faces",
            |mut caller: wasmtime::Caller<'_, HostState>,
             name_ptr: i32,
             name_len: i32,
             top_ptr: i32,
             top_len: i32,
             bot_ptr: i32,
             bot_len: i32,
             front_ptr: i32,
             front_len: i32,
             back_ptr: i32,
             back_len: i32,
             left_ptr: i32,
             left_len: i32,
             right_ptr: i32,
             right_len: i32|
             -> i32 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");

                let read_str = |data: &[u8], ptr: i32, len: i32| -> String {
                    std::str::from_utf8(&data[ptr as usize..(ptr + len) as usize])
                        .unwrap_or("")
                        .to_owned()
                };

                let (name, faces) = {
                    let data = mem.data(&caller);
                    let name = read_str(data, name_ptr, name_len);
                    let faces = FaceDef {
                        top: read_str(data, top_ptr, top_len),
                        bottom: read_str(data, bot_ptr, bot_len),
                        front: read_str(data, front_ptr, front_len),
                        back: read_str(data, back_ptr, back_len),
                        left: read_str(data, left_ptr, left_len),
                        right: read_str(data, right_ptr, right_len),
                    };
                    (name, faces)
                };

                let mut reg = caller.data().block_registry.lock().unwrap();
                let id = reg.register_with_faces(&name, faces);
                id.0 as i32
            },
        )?;

        linker.func_wrap(
            IMPORT_DATA_MODULE,
            "read-file",
            |mut caller: wasmtime::Caller<'_, HostState>,
             path_ptr: i32,
             path_len: i32,
             out_ptr: i32,
             max_len: i32|
             -> i32 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");

                // Read path string from guest memory
                let path = {
                    let data = mem.data(&caller);
                    let bytes = &data[path_ptr as usize..(path_ptr + path_len) as usize];
                    std::str::from_utf8(bytes).unwrap_or("").to_owned()
                };

                let store = Arc::clone(&caller.data().data_store);
                match store.read_file(&path) {
                    None => 0,
                    Some(contents) => {
                        let write_len = contents.len().min(max_len as usize);
                        let data = mem.data_mut(&mut caller);
                        data[out_ptr as usize..out_ptr as usize + write_len]
                            .copy_from_slice(&contents[..write_len]);
                        write_len as i32
                    }
                }
            },
        )?;

        linker.func_wrap(
            IMPORT_DATA_LIST_DIR,
            "list-dir",
            |mut caller: wasmtime::Caller<'_, HostState>,
             path_ptr: i32,
             path_len: i32,
             out_ptr: i32,
             max_len: i32|
             -> i32 {
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");

                let path = {
                    let data = mem.data(&caller);
                    let bytes = &data[path_ptr as usize..(path_ptr + path_len) as usize];
                    std::str::from_utf8(bytes).unwrap_or("").to_owned()
                };

                let store = Arc::clone(&caller.data().data_store);
                let listing = store.list_dir(&path);
                let result = listing.join("\n");
                let bytes = result.as_bytes();
                let write_len = bytes.len().min(max_len as usize);
                let data = mem.data_mut(&mut caller);
                data[out_ptr as usize..out_ptr as usize + write_len]
                    .copy_from_slice(&bytes[..write_len]);
                write_len as i32
            },
        )?;

        linker.func_wrap(
            IMPORT_PHYSICS_MODULE,
            "is-solid",
            |_caller: wasmtime::Caller<'_, HostState>, x: f32, y: f32, z: f32| -> i32 {
                with_chunk_query(|q| q.map(|q| q.is_solid(x, y, z) as i32).unwrap_or(0))
            },
        )?;

        linker.func_wrap(
            IMPORT_PHYSICS_MODULE,
            "sweep-aabb",
            |mut caller: wasmtime::Caller<'_, HostState>,
             ox: f32,
             oy: f32,
             oz: f32,
             dx: f32,
             dy: f32,
             dz: f32,
             hw: f32,
             height: f32,
             hd: f32,
             out_ptr: i32|
             -> i32 {
                let result = with_chunk_query(|q| match q {
                    Some(q) => cubic_world::sweep_aabb(
                        q,
                        cubic_math::Vec3::new(ox, oy, oz),
                        cubic_math::Vec3::new(dx, dy, dz),
                        hw,
                        height,
                        hd,
                    ),
                    None => cubic_world::SweepResult {
                        pos: cubic_math::Vec3::new(ox, oy, oz),
                        hit_x: false,
                        hit_y: false,
                        hit_z: false,
                    },
                });

                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");
                let data = mem.data_mut(&mut caller);
                let base = out_ptr as usize;
                data[base..base + 4].copy_from_slice(&result.pos.x.to_le_bytes());
                data[base + 4..base + 8].copy_from_slice(&result.pos.y.to_le_bytes());
                data[base + 8..base + 12].copy_from_slice(&result.pos.z.to_le_bytes());
                data[base + 12..base + 16].copy_from_slice(&(result.hit_x as i32).to_le_bytes());
                data[base + 16..base + 20].copy_from_slice(&(result.hit_y as i32).to_le_bytes());
                data[base + 20..base + 24].copy_from_slice(&(result.hit_z as i32).to_le_bytes());
                24i32
            },
        )?;

        linker.func_wrap(
            IMPORT_INPUT_MODULE,
            "get-input",
            |mut caller: wasmtime::Caller<'_, HostState>, out_ptr: i32| -> i32 {
                let snap = TICK_INPUT.with(|c| c.get());
                let mem = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .expect("guest has no memory export");
                let data = mem.data_mut(&mut caller);
                let base = out_ptr as usize;
                data[base..base + 4].copy_from_slice(&(snap.move_forward as i32).to_le_bytes());
                data[base + 4..base + 8].copy_from_slice(&(snap.move_back as i32).to_le_bytes());
                data[base + 8..base + 12].copy_from_slice(&(snap.move_left as i32).to_le_bytes());
                data[base + 12..base + 16]
                    .copy_from_slice(&(snap.move_right as i32).to_le_bytes());
                data[base + 16..base + 20].copy_from_slice(&(snap.jump as i32).to_le_bytes());
                data[base + 20..base + 24].copy_from_slice(&(snap.sneak as i32).to_le_bytes());
                data[base + 24..base + 28].copy_from_slice(&snap.look_dx.to_le_bytes());
                data[base + 28..base + 32].copy_from_slice(&snap.look_dy.to_le_bytes());
                32i32
            },
        )?;

        linker.func_wrap(
            IMPORT_CAMERA_MODULE,
            "set-camera",
            |_caller: wasmtime::Caller<'_, HostState>,
             x: f32,
             y: f32,
             z: f32,
             yaw: f32,
             pitch: f32| {
                CAMERA_UPDATE.with(|c| c.set(Some(CameraUpdate { x, y, z, yaw, pitch })));
            },
        )?;

        let instance = linker.instantiate(&mut store, module)?;

        // Grow memory to required size if needed
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("guest has no 'memory' export"))?;
        let current_bytes = memory.data_size(&store);
        if current_bytes < memory_bytes {
            let pages_needed = (memory_bytes - current_bytes).div_ceil(65536);
            memory.grow(&mut store, pages_needed as u64)?;
        }

        // Get exported functions. Names are the wit-bindgen-mangled forms —
        // see the EXPORT_* constants above.
        let fn_on_load = instance
            .get_typed_func::<u64, u32>(&mut store, EXPORT_ON_LOAD)
            .map_err(anyhow::Error::from)
            .context("guest missing 'on-load' export")?;
        let fn_generate = instance
            .get_typed_func::<(u32, i32, i32, i32, u32), u32>(&mut store, EXPORT_GENERATE)
            .map_err(anyhow::Error::from)
            .context("guest missing 'generate' export")?;
        let fn_is_definitely_air = instance
            .get_typed_func::<(u32, i32, i32, i32), u32>(&mut store, EXPORT_IS_DEFINITELY_AIR)
            .map_err(anyhow::Error::from)
            .context("guest missing 'is-definitely-air' export")?;
        // Optional: cubic-game doesn't implement `tick` yet (next card), so
        // this lookup is allowed to fail — tick() below just no-ops until
        // it does.
        let fn_on_tick = instance
            .get_typed_func::<f32, ()>(&mut store, EXPORT_ON_TICK)
            .ok();

        // Call on_load — guest registers blocks and returns generator handle
        let generator_handle = fn_on_load.call(&mut store, seed)?;
        tracing::info!("WASM guest on_load complete, generator_handle={generator_handle}");

        Ok(Self {
            store,
            memory,
            fn_generate,
            fn_is_definitely_air,
            fn_on_tick,
            generator_handle,
        })
    }

    fn generate(&mut self, pos: ChunkPos, out_ptr: u32) -> Result<u32> {
        let handle = self.generator_handle;
        let n = self
            .fn_generate
            .call(&mut self.store, (handle, pos.x, pos.y, pos.z, out_ptr))?;
        Ok(n)
    }

    fn is_definitely_air(&mut self, pos: ChunkPos) -> Result<bool> {
        let handle = self.generator_handle;
        let result = self
            .fn_is_definitely_air
            .call(&mut self.store, (handle, pos.x, pos.y, pos.z))?;
        Ok(result != 0)
    }

    fn tick(&mut self, dt: f32) -> Result<()> {
        if let Some(f) = &self.fn_on_tick {
            f.call(&mut self.store, dt)?;
        }
        Ok(())
    }

    fn read_chunk(&self, out_ptr: u32) -> Chunk {
        let data = self.memory.data(&self.store);
        let offset = out_ptr as usize;
        let mut chunk = Chunk::new();
        for i in 0..CHUNK_VOLUME {
            let id_bytes = &data[offset + i * 4..offset + i * 4 + 4];
            let id = u32::from_le_bytes(id_bytes.try_into().unwrap());
            if id != 0 {
                let x = (i % CHUNK_SIZE) as u8;
                let y = (i / CHUNK_SIZE / CHUNK_SIZE) as u8;
                let z = ((i / CHUNK_SIZE) % CHUNK_SIZE) as u8;
                chunk.set(ChunkLocalPos::new(x, y, z), cubic_world::BlockTypeId(id));
            }
        }
        chunk
    }
}

// ---------------------------------------------------------------------------
// WasmPlugin — owns Module and per-worker instances
// ---------------------------------------------------------------------------

/// Shared across threads via Arc. Owns the compiled Module and registry.
/// Per-worker instances are stored in thread-locals inside WasmWorldGenerator.
pub struct WasmPlugin {
    engine: Engine,
    module: Module,
    layout: WasmMemoryLayout,
    memory_bytes: usize,
    block_registry: Arc<Mutex<cubic_world::BlockRegistry>>,
    data_store: Arc<DataStore>,
    seed: u64,
}

impl WasmPlugin {
    pub fn load(path: &str, worker_count: usize, memory_mb: usize, seed: u64) -> Result<Self> {
        let game_dir = std::path::Path::new(path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        let engine = Engine::default();
        let bytes =
            std::fs::read(path).with_context(|| format!("failed to read game plugin: {path}"))?;
        let module = Module::new(&engine, &bytes)
            .map_err(anyhow::Error::from)
            .context("failed to compile game plugin")?;

        let layout = WasmMemoryLayout::new(worker_count);
        let memory_bytes = (memory_mb * 1024 * 1024).max(layout.min_memory_bytes());

        tracing::info!(
            "WASM plugin loaded: {path}, workers={worker_count}, memory={}MB",
            memory_bytes / 1024 / 1024
        );

        Ok(Self {
            engine,
            module,
            layout,
            memory_bytes,
            block_registry: Arc::new(Mutex::new(cubic_world::BlockRegistry::new())),
            data_store: Arc::new(DataStore::new(&game_dir)),
            seed,
        })
    }

    fn make_instance(&self) -> Result<WasmInstance> {
        WasmInstance::new(
            &self.engine,
            &self.module,
            Arc::clone(&self.block_registry),
            Arc::clone(&self.data_store),
            self.memory_bytes,
            self.seed,
        )
    }

    /// Shared handle to the block registry the guest populates via
    /// `on_load` (see `register-block`/`register-block-with-faces`).
    pub fn block_registry(&self) -> Arc<Mutex<cubic_world::BlockRegistry>> {
        Arc::clone(&self.block_registry)
    }

    /// Eagerly create a WASM instance on the calling thread, running the
    /// guest's `on_load` synchronously so `block_registry` is populated
    /// immediately — otherwise it only happens lazily on a worker thread's
    /// first `generate()`/`is_definitely_air()` call, which is too late for
    /// callers (like texture loading) that need it right after `load()`.
    /// Worker threads still lazily create their own instances later; this
    /// only guarantees one exists up front.
    pub fn warm_up(&self) {
        WASM_INSTANCE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                *opt = Some(
                    self.make_instance()
                        .expect("failed to warm up WASM instance"),
                );
            }
        });
    }
}

// ---------------------------------------------------------------------------
// WasmWorldGenerator — implements WorldGenerator
// ---------------------------------------------------------------------------

// Thread-local storage for per-worker WASM instances.
// Initialized lazily on first generate() call per thread.
thread_local! {
    static WASM_INSTANCE: std::cell::RefCell<Option<WasmInstance>> =
        const { std::cell::RefCell::new(None) };
    static WORKER_ID: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// Called by each streaming worker thread at startup to assign its WASM
/// shared memory buffer slot. Must be called before any generate() calls
/// on that thread.
pub fn set_worker_id(id: usize) {
    WORKER_ID.set(id);
}

pub struct WasmWorldGenerator {
    plugin: Arc<WasmPlugin>,
}

impl WasmWorldGenerator {
    pub fn new(plugin: Arc<WasmPlugin>) -> Self {
        Self { plugin }
    }

    /// Run the guest's `on-tick` export (a no-op if it doesn't implement
    /// `tick` yet — see `WasmInstance::fn_on_tick`) using the main thread's
    /// WASM_INSTANCE. Worker threads have their own instances, but those are
    /// only ever used for `generate`/`is_definitely_air` — they never tick.
    pub fn tick(&self, dt: f32) {
        WASM_INSTANCE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                let plugin = Arc::clone(&self.plugin);
                *opt = Some(
                    plugin
                        .make_instance()
                        .expect("failed to create WASM instance"),
                );
            }
            if let Some(instance) = opt.as_mut() {
                let _ = instance.tick(dt);
            }
        });
    }
}

impl WorldGenerator for WasmWorldGenerator {
    fn generate(&self, pos: ChunkPos, _seed: u64) -> Chunk {
        let worker_id = WORKER_ID.get();
        let out_ptr = if worker_id == usize::MAX {
            // Main thread fallback — use slot 0
            self.plugin.layout.worker_buffer_offset(0)
        } else {
            self.plugin
                .layout
                .worker_buffer_offset(worker_id % self.plugin.layout.worker_count)
        };

        let plugin = Arc::clone(&self.plugin);

        // NB: don't lock block_registry here. The lazy `make_instance()` call
        // below runs the guest's on-load export, which calls back into the
        // host's register-block import — that import also locks
        // block_registry, so holding the lock across instance creation would
        // self-deadlock on every worker thread's first generate() call.
        WASM_INSTANCE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                *opt = Some(
                    plugin
                        .make_instance()
                        .expect("failed to create WASM instance"),
                );
            }
            let instance = opt.as_mut().unwrap();
            instance
                .generate(pos, out_ptr)
                .expect("WASM generate failed");
            instance.read_chunk(out_ptr)
        })
    }

    fn is_definitely_air(&self, pos: ChunkPos) -> bool {
        WASM_INSTANCE.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                let plugin = Arc::clone(&self.plugin);
                *opt = Some(
                    plugin
                        .make_instance()
                        .expect("failed to create WASM instance"),
                );
            }
            opt.as_mut()
                .unwrap()
                .is_definitely_air(pos)
                .unwrap_or(false)
        })
    }
}
