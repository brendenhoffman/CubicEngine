// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::{Cell, RefCell};

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

/// Input state snapshotted by cubic-app before each on_tick call.
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
    // toggle_third_person deliberately absent: it used to be a continuous
    // boolean the guest edge-detected itself, which meant it fired on any
    // press no matter what trigger kind was configured for it (the exact
    // bug this InputEvent-based toggle_third_person, spectate, and fly all
    // now avoid — see InputTracker::update on the host). All fully discrete
    // controls go through get-events instead of get-input.
    /// cfg.player.* from cubic.toml (layered through game_overrides.toml /
    /// profile.toml), host-resolved and sent every tick — same pattern as
    /// look_dx/look_dy already being sensitivity-scaled — so the guest
    /// doesn't need its own copy of engine config, and Settings-tab edits
    /// take effect immediately without a reload. Distinct from
    /// cfg.camera.move_speed, which only drives the free-fly debug camera
    /// used when no game is loaded.
    pub walk_speed: f32,
    pub fly_speed: f32,
    pub jump_velocity: f32,
    pub gravity: f32,
    // A generic movement-speed multiplier, not tied to any specific engine
    // feature — cubic-game's "sprint" (double-tap-forward) is entirely
    // guest-side and just multiplies horiz_speed by this while its own
    // `sprinting` flag is set; the host has no notion of sprinting at all,
    // it only carries the configured number through, same as walk_speed.
    pub sprint_multiplier: f32,
}

thread_local! {
    static TICK_INPUT: Cell<InputSnapshot> = const { Cell::new(InputSnapshot {
        move_forward: false,
        move_back: false,
        move_left: false,
        move_right: false,
        jump: false,
        sneak: false,
        look_dx: 0.0,
        look_dy: 0.0,
        walk_speed: 0.0,
        fly_speed: 0.0,
        jump_velocity: 0.0,
        gravity: 0.0,
        sprint_multiplier: 1.0,
    }) };
}

/// Set the input snapshot for the current tick. cubic-app calls this once
/// per frame immediately before calling `WasmWorldGenerator::tick`.
pub fn set_tick_input(input: InputSnapshot) {
    TICK_INPUT.with(|c| c.set(input));
}

/// Read the current tick's input snapshot. Called by the `get-input` host function.
pub fn get_tick_input() -> InputSnapshot {
    TICK_INPUT.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Camera position and orientation set by the game during on_tick via `set-camera`.
#[derive(Clone, Copy)]
pub struct CameraUpdate {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
}

thread_local! {
    static CAMERA_UPDATE: Cell<Option<CameraUpdate>> = const { Cell::new(None) };
}

/// Store a camera update from the guest's `set-camera` call.
pub fn set_camera_update(update: CameraUpdate) {
    CAMERA_UPDATE.with(|c| c.set(Some(update)));
}

/// Take (and clear) the camera update the guest set during the most recent tick.
pub fn take_camera_update() -> Option<CameraUpdate> {
    CAMERA_UPDATE.with(|c| c.take())
}

// ---------------------------------------------------------------------------
// Player feet position
// ---------------------------------------------------------------------------

/// Player feet (pos, not eye) position set by the game during on_tick via
/// `set-player-feet` — tracked separately from the camera since third-person
/// orbit moves the camera away from the player's actual position.
#[derive(Clone, Copy, Default)]
pub struct PlayerFeet {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

thread_local! {
    static PLAYER_FEET: Cell<PlayerFeet> =
        const { Cell::new(PlayerFeet { x: 0.0, y: 0.0, z: 0.0 }) };
}

/// Store the player feet position from the guest's `set-player-feet` call.
pub fn set_player_feet(feet: PlayerFeet) {
    PLAYER_FEET.with(|c| c.set(feet));
}

/// Read the most recently set player feet position (zeroed until the first
/// `set-player-feet` call, e.g. before a game is loaded).
pub fn get_player_feet() -> PlayerFeet {
    PLAYER_FEET.with(|c| c.get())
}

// ---------------------------------------------------------------------------
// Draw queue
// ---------------------------------------------------------------------------

/// A draw request queued by the game via `draw-mesh` during on_tick.
/// Flushed by cubic-app after on_tick returns, before render().
#[derive(Clone)]
pub struct DrawRequest {
    pub mesh_handle: u32,
    pub tex_index: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
}

thread_local! {
    static DRAW_QUEUE: RefCell<Vec<DrawRequest>> = const { RefCell::new(Vec::new()) };
}

/// Push a draw request from the guest's `draw-mesh` call.
pub fn push_draw_request(req: DrawRequest) {
    DRAW_QUEUE.with(|q| q.borrow_mut().push(req));
}

/// Take (and clear) all queued draw requests. Called by cubic-app after on_tick.
pub fn take_draw_queue() -> Vec<DrawRequest> {
    DRAW_QUEUE.with(|q| std::mem::take(&mut *q.borrow_mut()))
}

// ---------------------------------------------------------------------------
// Block edit queue
// ---------------------------------------------------------------------------

/// A block edit (break/place) requested by the guest via `request-set-block`
/// during on_tick. Queued rather than applied immediately for the same
/// reason as CHUNK_QUERY_PTR's safety note explains: the chunk data this
/// tick's is-solid/get-block calls are reading from can't be mutated until
/// that borrow ends, which is after on_tick returns — cubic-app drains this
/// queue then, applying each edit via AsyncWorldStream::set_block_at.
#[derive(Clone, Copy)]
pub struct BlockEditRequest {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub block_id: u32,
}

thread_local! {
    static BLOCK_EDITS: RefCell<Vec<BlockEditRequest>> = const { RefCell::new(Vec::new()) };
}

/// Push a block edit request from the guest's `request-set-block` call.
pub fn push_block_edit(req: BlockEditRequest) {
    BLOCK_EDITS.with(|q| q.borrow_mut().push(req));
}

/// Take (and clear) all queued block edits. Called by cubic-app after
/// on_tick returns and clear_tick_query() has run.
pub fn take_block_edits() -> Vec<BlockEditRequest> {
    BLOCK_EDITS.with(|q| std::mem::take(&mut *q.borrow_mut()))
}

// ---------------------------------------------------------------------------
// Input events
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct InputEvent {
    /// Action name matching ControlsCfg field e.g. "jump", "forward"
    pub name: String,
    /// 0=Pressed, 1=Released, 2=DoubleTap
    pub kind: u32,
    pub payload: [f32; 3],
}

thread_local! {
    static INPUT_EVENTS: RefCell<Vec<InputEvent>> =
        const { RefCell::new(Vec::new()) };
}

pub fn push_input_event(event: InputEvent) {
    INPUT_EVENTS.with(|q| q.borrow_mut().push(event));
}

pub fn take_input_events() -> Vec<InputEvent> {
    INPUT_EVENTS.with(|q| std::mem::take(&mut *q.borrow_mut()))
}

// ---------------------------------------------------------------------------
// Asset loading callbacks
// ---------------------------------------------------------------------------
//
// Set once at load_world() time — before warm_up() runs on_load — so the
// guest can call load-mesh/load-texture synchronously during on_load.
// The closures capture raw pointers to the backend and App fields that are
// valid for the lifetime of the app; they are never called after the backend
// is dropped.

type LoadFn = RefCell<Option<Box<dyn Fn(&str) -> u32>>>;

thread_local! {
    static LOAD_MESH_FN: LoadFn = const { RefCell::new(None) };
    static LOAD_TEX_FN: LoadFn = const { RefCell::new(None) };
}

/// Register the callbacks used by `load-mesh` and `load-texture` host functions.
/// Must be called before warm_up() so on_load can call them.
pub fn set_load_fns(
    mesh_fn: impl Fn(&str) -> u32 + 'static,
    tex_fn: impl Fn(&str) -> u32 + 'static,
) {
    LOAD_MESH_FN.with(|c| *c.borrow_mut() = Some(Box::new(mesh_fn)));
    LOAD_TEX_FN.with(|c| *c.borrow_mut() = Some(Box::new(tex_fn)));
}

/// Invoke the registered load-mesh callback. Returns 0 on failure or if not set.
pub fn call_load_mesh(path: &str) -> u32 {
    LOAD_MESH_FN.with(|f| f.borrow().as_ref().map(|f| f(path)).unwrap_or(0))
}

/// Invoke the registered load-texture callback. Returns 0 on failure or if not set.
pub fn call_load_tex(path: &str) -> u32 {
    LOAD_TEX_FN.with(|f| f.borrow().as_ref().map(|f| f(path)).unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Chunk query (physics)
// ---------------------------------------------------------------------------
//
// Physics host functions (is-solid, sweep-aabb) need read access to live
// chunk data during on_tick, but func_wrap closures must be 'static while
// the borrow they need is scoped to a single tick call. A thread-local raw
// pointer bridges that gap without locking or copying chunk data.
//
// Safety invariant: CHUNK_QUERY_PTR holds a raw pointer to a ChunkQueryView
// that borrows AsyncWorldStream::inner.chunks. This is valid because:
//   1. set_tick_query is called before any physics host functions
//   2. clear_tick_query is called before the streaming update mutates chunks
//   3. All of this happens on the main thread sequentially
// If this ever moves to worker threads, replace with Arc<RwLock<>> instead.

thread_local! {
    static CHUNK_QUERY_PTR: Cell<Option<*const dyn cubic_world::ChunkQuery>> =
        const { Cell::new(None) };
}

/// Set the chunk query pointer for the current tick.
/// Must be paired with clear_tick_query before the borrowed data is mutated.
pub fn set_tick_query(q: &dyn cubic_world::ChunkQuery) {
    // Safety: erasing the borrow's lifetime to 'static here is sound only
    // because every caller clears it (via clear_tick_query) before the
    // borrow it points to actually ends — see the module-level safety note
    // above. Both sides of the transmute are fat pointers of identical
    // layout (data ptr + vtable ptr); only the lifetime bound differs.
    let ptr: *const dyn cubic_world::ChunkQuery =
        unsafe { std::mem::transmute(q as *const dyn cubic_world::ChunkQuery) };
    CHUNK_QUERY_PTR.with(|c| c.set(Some(ptr)));
}

/// Clear the chunk query pointer. Must be called before the streaming update
/// runs (which mutates inner.chunks).
pub fn clear_tick_query() {
    CHUNK_QUERY_PTR.with(|c| c.set(None));
}

/// Call `f` with the current tick's chunk query, or `None` if not set.
pub fn with_chunk_query<F, R>(f: F) -> R
where
    F: FnOnce(Option<&dyn cubic_world::ChunkQuery>) -> R,
{
    CHUNK_QUERY_PTR.with(|c| match c.get() {
        None => f(None),
        // Safety: see the module-level safety note on CHUNK_QUERY_PTR.
        Some(ptr) => f(Some(unsafe { &*ptr })),
    })
}
