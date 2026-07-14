// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::physics::{world_to_chunk_local, ChunkQuery};
use crate::region::{apply_diff, diff_from_chunks, RegionCache};
use crate::{
    mesh_chunk, BlockFaceTextures, BlockTypeId, Chunk, ChunkPos, StreamDelta, WorldGenerator,
    WorldStream, CHUNK_SIZE,
};
use cubic_render::Vertex;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// Internal channel types
// ---------------------------------------------------------------------------

struct WorkItem {
    pos: ChunkPos,
    seed: u64,
    generator: Arc<dyn WorldGenerator>,
    face_textures: Arc<BlockFaceTextures>,
    region_cache: Option<Arc<Mutex<RegionCache>>>,
}

struct WorkResult {
    pos: ChunkPos,
    chunk: Option<Chunk>, // None = air, Some = has geometry
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

// ---------------------------------------------------------------------------
// AsyncWorldStream
// ---------------------------------------------------------------------------

pub struct ChunkQueryView<'a> {
    chunks: &'a HashMap<ChunkPos, Chunk>,
}

impl ChunkQuery for ChunkQueryView<'_> {
    fn get_block_at(&self, wx: f64, wy: f64, wz: f64) -> BlockTypeId {
        self.chunks.get_block_at(wx, wy, wz)
    }
}

/// Wraps `WorldStream` with a background worker pool so chunk generation
/// never blocks the main thread. Call `update` each frame exactly like
/// `WorldStream::update`; completed chunks trickle in via the result channel
/// and appear in `StreamDelta::loaded` once ready.
pub struct AsyncWorldStream {
    inner: WorldStream,
    in_flight: HashSet<ChunkPos>,
    discard: HashSet<ChunkPos>,
    work_tx: Sender<WorkItem>,
    result_rx: Receiver<WorkResult>,
    pub ready_meshes: Vec<(ChunkPos, Vec<Vertex>, Vec<u32>)>,
    pub remesh_queue: Vec<ChunkPos>,
    // Tracks which neighbors were present when this chunk was last remeshed.
    // Key: chunk position. Value: bitmask of which of the 6 neighbors were present
    // (bit 0 = -X, 1 = +X, 2 = -Y, 3 = +Y, 4 = -Z, 5 = +Z).
    remeshed_with: HashMap<ChunkPos, u8>,
    // Positions whose last mesh attempt produced no geometry — either pure air
    // or fully buried solid. Not inserted into inner.chunks. Solid entries will
    // need revisiting when block removal exists (dirty chunk system).
    known_empty: HashSet<ChunkPos>,
    _workers: Vec<JoinHandle<()>>,
    dirty: HashSet<ChunkPos>,
    region_cache: Option<Arc<Mutex<RegionCache>>>,
    generator: Option<Arc<dyn WorldGenerator>>,
    seed: u64,
    diff_threshold: usize,
}

impl AsyncWorldStream {
    pub fn new(
        radius_xz: i32,
        radius_y: i32,
        on_worker_start: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    ) -> Self {
        let worker_count = thread::available_parallelism()
            .map_or(4, |n| n.get())
            .saturating_sub(1)
            .max(1);

        let (work_tx, work_rx) = mpsc::channel::<WorkItem>();
        let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

        // Wrap the receiver so multiple workers can pull from it
        let work_rx = Arc::new(Mutex::new(work_rx));

        let workers = (0..worker_count)
            .enumerate()
            .map(|(i, _)| {
                let work_rx = Arc::clone(&work_rx);
                let result_tx = result_tx.clone();
                let cb = on_worker_start.clone(); // clone the Option<Arc<...>>
                thread::spawn(move || {
                    if let Some(ref f) = cb {
                        f(i);
                    }
                    loop {
                        let item = {
                            let rx = work_rx.lock().unwrap();
                            rx.recv()
                        };
                        match item {
                            Ok(work) => {
                                let mut chunk = work.generator.generate(work.pos, work.seed);

                                if let Some(cache) = &work.region_cache {
                                    if let Ok(mut cache) = cache.lock() {
                                        match cache.read_chunk(work.pos.x, work.pos.y, work.pos.z) {
                                            Ok(Some(diff)) => apply_diff(&mut chunk, &diff),
                                            Ok(None) => {}
                                            Err(e) => tracing::warn!(
                                                "failed to read diff for {:?}: {e:#}",
                                                work.pos
                                            ),
                                        }
                                    }
                                }

                                let (vertices, indices) =
                                    mesh_chunk(&chunk, [None; 6], &work.face_textures);
                                if vertices.is_empty() {
                                    // No geometry — pure air or fully buried solid.
                                    // Neighbors don't need to know since this chunk
                                    // contributes no faces. A future "dirty chunk"
                                    // system will handle the fully-buried case when
                                    // block removal is added.
                                    let _ = result_tx.send(WorkResult {
                                        pos: work.pos,
                                        chunk: None,
                                        vertices: Vec::new(),
                                        indices: Vec::new(),
                                    });
                                } else {
                                    let _ = result_tx.send(WorkResult {
                                        pos: work.pos,
                                        chunk: Some(chunk),
                                        vertices,
                                        indices,
                                    });
                                }
                            }
                            // Channel closed — main thread dropped the sender, time to exit
                            Err(_) => break,
                        }
                    }
                })
            })
            .collect();

        Self {
            inner: WorldStream::new(radius_xz, radius_y),
            in_flight: HashSet::new(),
            discard: HashSet::new(),
            work_tx,
            result_rx,
            ready_meshes: Vec::new(),
            remesh_queue: Vec::new(),
            remeshed_with: HashMap::new(),
            known_empty: HashSet::new(),
            _workers: workers,
            dirty: HashSet::new(),
            region_cache: None,
            generator: None,
            seed: 0,
            diff_threshold: 512,
        }
    }

    pub fn set_persistence(
        &mut self,
        region_cache: Arc<Mutex<RegionCache>>,
        generator: Arc<dyn WorldGenerator>,
        seed: u64,
        diff_threshold: usize,
    ) {
        self.region_cache = Some(region_cache);
        self.generator = Some(generator);
        self.seed = seed;
        self.diff_threshold = diff_threshold;
    }

    /// Same contract as `WorldStream::update` but generation is async.
    /// Loaded chunks in the returned delta are only the ones that completed
    /// this frame — newly dispatched ones will appear in future frames.
    pub fn update(
        &mut self,
        center: ChunkPos,
        generator: &Arc<dyn WorldGenerator>,
        seed: u64,
        face_textures: &Arc<BlockFaceTextures>,
    ) -> StreamDelta {
        let rxz = self.inner.radius_xz;
        let ry = self.inner.radius_y;
        let mut to_unload = Vec::new();
        let mut loaded = Vec::new();

        // --- Drain completed results ---
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.pos);
            if self.discard.remove(&result.pos) {
                continue;
            }
            let chunk = match result.chunk {
                Some(chunk) => chunk,
                None => {
                    self.known_empty.insert(result.pos);
                    continue;
                }
            };
            self.inner.chunks.insert(result.pos, chunk);
            if !result.vertices.is_empty() {
                self.ready_meshes
                    .push((result.pos, result.vertices, result.indices));
            }
            // Queue self and all loaded neighbors for boundary remesh — but only
            // if their neighbor set actually changed since they were last
            // remeshed, so a chunk isn't re-remeshed for a neighbor it already
            // accounted for.
            if needs_remesh(&self.remeshed_with, &self.inner.chunks, result.pos) {
                self.remesh_queue.push(result.pos);
            }
            for neighbor_pos in six_neighbors(result.pos) {
                if self.inner.chunks.contains_key(&neighbor_pos)
                    && !self.known_empty.contains(&neighbor_pos)
                    && needs_remesh(&self.remeshed_with, &self.inner.chunks, neighbor_pos)
                {
                    self.remesh_queue.push(neighbor_pos);
                }
            }
            loaded.push(result.pos);
        }

        // --- Compute desired set and find what needs loading ---
        for x in (center.x - rxz)..=(center.x + rxz) {
            for y in (center.y - ry)..=(center.y + ry) {
                for z in (center.z - rxz)..=(center.z + rxz) {
                    let pos = ChunkPos { x, y, z };
                    if self.inner.chunks.contains_key(&pos)
                        || self.in_flight.contains(&pos)
                        || self.known_empty.contains(&pos)
                    {
                        continue;
                    }
                    if generator.is_definitely_air(pos, seed) && !self.has_saved_diff(pos) {
                        // The generator's heuristic is coarse (e.g. "this
                        // chunk's Y range is entirely above the highest
                        // possible terrain height") and knows nothing about
                        // player edits — see set_block_at's doc comment,
                        // which can materialize exactly this kind of chunk
                        // on demand when a diff *does* exist. Record it as
                        // known_empty so every later frame hits the cheap
                        // in-memory check above instead of repeating this
                        // (rare, one-time-per-position) disk lookup.
                        self.known_empty.insert(pos);
                        continue;
                    }
                    self.in_flight.insert(pos);
                    // best-effort send — if the channel is full we'll retry next frame
                    let _ = self.work_tx.send(WorkItem {
                        pos,
                        seed,
                        generator: Arc::clone(generator),
                        face_textures: Arc::clone(face_textures),
                        region_cache: self.region_cache.clone(),
                    });
                }
            }
        }

        // --- Find what needs unloading ---
        for pos in self.inner.chunks.keys().copied() {
            if (pos.x - center.x).abs() > rxz
                || (pos.y - center.y).abs() > ry
                || (pos.z - center.z).abs() > rxz
            {
                to_unload.push(pos);
            }
        }
        // Cancel in-flight chunks that are now out of range
        let to_discard: Vec<ChunkPos> = self
            .in_flight
            .iter()
            .copied()
            .filter(|pos| {
                (pos.x - center.x).abs() > rxz
                    || (pos.y - center.y).abs() > ry
                    || (pos.z - center.z).abs() > rxz
            })
            .collect();
        for pos in to_discard {
            self.in_flight.remove(&pos);
            self.discard.insert(pos);
        }

        // Apply unloads
        for pos in &to_unload {
            if self.dirty.contains(pos) {
                if let (Some(chunk), Some(cache), Some(gen)) = (
                    self.inner.chunks.get(pos),
                    &self.region_cache,
                    &self.generator,
                ) {
                    let baseline = gen.generate(*pos, self.seed);
                    match diff_from_chunks(&baseline, chunk, self.diff_threshold) {
                        Some(diff) => {
                            if let Ok(mut cache) = cache.lock() {
                                if let Err(e) = cache.write_chunk(pos.x, pos.y, pos.z, &diff) {
                                    tracing::error!("failed to save chunk {pos:?}: {e:#}");
                                }
                            }
                        }
                        None => {
                            // Chunk reverted to virgin -- remove any previously saved diff
                            if let Ok(mut cache) = cache.lock() {
                                if let Err(e) = cache.remove_chunk(pos.x, pos.y, pos.z) {
                                    tracing::warn!("failed to tombstone chunk {pos:?}: {e:#}");
                                }
                            }
                        }
                    }
                }
                self.dirty.remove(pos);
            }

            self.inner.chunks.remove(pos);
            self.remeshed_with.remove(pos);
            // Clear this position's bit in each neighbor's recorded mask so
            // they get re-queued if a new chunk later fills this slot.
            for (i, neighbor_pos) in six_neighbors(*pos).iter().enumerate() {
                if let Some(mask) = self.remeshed_with.get_mut(neighbor_pos) {
                    *mask &= !(1 << (i ^ 1));
                }
            }
        }

        // known_empty positions never enter `inner.chunks`, so they can't be
        // found via the loop above — drop the ones that fell out of range
        // directly so the set doesn't grow unbounded as the camera moves.
        self.known_empty.retain(|pos| {
            (pos.x - center.x).abs() <= rxz
                && (pos.y - center.y).abs() <= ry
                && (pos.z - center.z).abs() <= rxz
        });

        StreamDelta {
            loaded,
            unloaded: to_unload,
        }
    }

    /// Cheap existence-only region-cache lookup (see
    /// `RegionFile::has_chunk`), used to override
    /// `WorldGenerator::is_definitely_air`'s coarse heuristic when a player
    /// has actually edited a chunk it would otherwise skip. `false` (not
    /// `true`) on any I/O error or missing region cache — worst case that
    /// just means trusting the heuristic, same as before this existed.
    fn has_saved_diff(&self, pos: ChunkPos) -> bool {
        let Some(cache) = &self.region_cache else {
            return false;
        };
        let Ok(mut cache) = cache.lock() else {
            return false;
        };
        cache.has_chunk(pos.x, pos.y, pos.z).unwrap_or(false)
    }

    /// Delegate to inner stream for neighbor lookups
    pub fn neighbors(&self, pos: ChunkPos) -> [Option<&Chunk>; 6] {
        self.inner.neighbors(pos)
    }

    /// Direct chunk access for meshing
    pub fn chunks(&self) -> &std::collections::HashMap<ChunkPos, Chunk> {
        &self.inner.chunks
    }

    /// Record which neighbors were present the last time `pos` was remeshed,
    /// so future arrivals that don't change its neighbor set won't re-queue it.
    pub fn mark_remeshed(&mut self, pos: ChunkPos) {
        let mask = neighbor_mask(&self.inner.chunks, pos);
        self.remeshed_with.insert(pos, mask);
    }

    pub fn query_view(&self) -> ChunkQueryView<'_> {
        ChunkQueryView {
            chunks: &self.inner.chunks,
        }
    }

    /// Edit a single voxel in the currently loaded world (break/place),
    /// queuing this chunk — and any face-adjacent neighbor whose mesh culls
    /// faces against this voxel — for remesh via the same `remesh_queue`
    /// the boundary-stitching path already drains every frame (see
    /// RedrawRequested in cubic-app), so callers don't need any separate
    /// "apply this edit's mesh" step.
    ///
    /// If the containing chunk isn't materialized, this generates it on
    /// demand (via `self.generator`, reapplying any saved diff, exactly
    /// like the async worker's normal generate path) before applying the
    /// edit. Three different cases land here and all need this:
    /// - `known_empty`: streamed once, turned out to have zero visible
    ///   faces (pure air or fully buried solid), and so was discarded
    ///   rather than stored (see `known_empty`'s doc comment).
    /// - Skipped by `WorldGenerator::is_definitely_air` in `update()`'s
    ///   request loop — that check is a coarse, cheap-to-compute bound
    ///   (e.g. "this chunk's Y range is entirely above the highest
    ///   possible terrain height"), so a chunk it flags is never even
    ///   requested, let alone generated. Building straight up past that
    ///   bound is exactly how a player reaches one of these.
    /// - Simply never requested yet, e.g. the streaming loop hasn't reached
    ///   it this frame.
    ///
    /// All three are safe to materialize synchronously here: a block edit
    /// is only ever raycast-bounded to within reach of the player, so it's
    /// always near the currently active area, and this pays the same
    /// per-chunk generation cost the async worker would pay anyway, just
    /// off the worker thread. The one case that must stay a no-op is
    /// `in_flight` — a worker is already generating this exact chunk, and
    /// generating a second copy here would race its result, silently
    /// overwriting this edit when the worker's result lands.
    pub fn set_block_at(&mut self, wx: f64, wy: f64, wz: f64, id: BlockTypeId) -> bool {
        let (cp, lp) = world_to_chunk_local(wx, wy, wz);
        if !self.inner.chunks.contains_key(&cp) {
            if self.in_flight.contains(&cp) {
                return false;
            }
            let Some(generator) = &self.generator else {
                return false;
            };
            let mut chunk = generator.generate(cp, self.seed);
            if let Some(cache) = &self.region_cache {
                if let Ok(mut cache) = cache.lock() {
                    if let Ok(Some(diff)) = cache.read_chunk(cp.x, cp.y, cp.z) {
                        apply_diff(&mut chunk, &diff);
                    }
                }
            }
            self.known_empty.remove(&cp);
            self.inner.chunks.insert(cp, chunk);
        }
        let chunk = self
            .inner
            .chunks
            .get_mut(&cp)
            .expect("just materialized or already present above");
        chunk.set(lp, id);
        self.dirty.insert(cp);
        self.remesh_queue.push(cp);
        self.remesh_queue.extend(boundary_neighbors(cp, lp));
        true
    }

    /// Save all dirty loaded chunks to disk without unloading them.
    /// Called by the autosave timer and on clean quit.
    pub fn flush_dirty(&mut self) {
        let (Some(cache), Some(gen)) = (&self.region_cache, &self.generator) else {
            return;
        };
        let dirty_positions: Vec<ChunkPos> = self.dirty.iter().copied().collect();
        for pos in dirty_positions {
            let Some(chunk) = self.inner.chunks.get(&pos) else {
                self.dirty.remove(&pos);
                continue;
            };
            let baseline = gen.generate(pos, self.seed);
            match diff_from_chunks(&baseline, chunk, self.diff_threshold) {
                Some(diff) => {
                    if let Ok(mut cache) = cache.lock() {
                        if let Err(e) = cache.write_chunk(pos.x, pos.y, pos.z, &diff) {
                            tracing::error!("autosave failed for {pos:?}: {e:#}");
                        }
                    }
                }
                None => {
                    if let Ok(mut cache) = cache.lock() {
                        let _ = cache.remove_chunk(pos.x, pos.y, pos.z);
                    }
                    self.dirty.remove(&pos);
                }
            }
        }
    }
}

/// Chunk positions adjacent to `cp` that a mesh-culling boundary edit at
/// `lp` also affects — i.e. `lp` sits on one of the 6 chunk faces, so the
/// neighbor's mesh (which culls its own faces against this chunk's
/// occupancy) needs to be recomputed too. Pure and separately testable from
/// `set_block_at`, which just needs live chunk data to exercise otherwise.
fn boundary_neighbors(cp: ChunkPos, lp: crate::ChunkLocalPos) -> Vec<ChunkPos> {
    let max = (CHUNK_SIZE - 1) as u8;
    let mut out = Vec::new();
    if lp.x == 0 {
        out.push(ChunkPos { x: cp.x - 1, ..cp });
    }
    if lp.x == max {
        out.push(ChunkPos { x: cp.x + 1, ..cp });
    }
    if lp.y == 0 {
        out.push(ChunkPos { y: cp.y - 1, ..cp });
    }
    if lp.y == max {
        out.push(ChunkPos { y: cp.y + 1, ..cp });
    }
    if lp.z == 0 {
        out.push(ChunkPos { z: cp.z - 1, ..cp });
    }
    if lp.z == max {
        out.push(ChunkPos { z: cp.z + 1, ..cp });
    }
    out
}

fn neighbor_mask(chunks: &std::collections::HashMap<ChunkPos, Chunk>, pos: ChunkPos) -> u8 {
    let mut mask = 0u8;
    for (i, neighbor_pos) in six_neighbors(pos).iter().enumerate() {
        if chunks.contains_key(neighbor_pos) {
            mask |= 1 << i;
        }
    }
    mask
}

fn needs_remesh(
    remeshed_with: &HashMap<ChunkPos, u8>,
    chunks: &std::collections::HashMap<ChunkPos, Chunk>,
    pos: ChunkPos,
) -> bool {
    let current_neighbor_mask = neighbor_mask(chunks, pos);
    let last_mask = remeshed_with.get(&pos).copied().unwrap_or(0);
    current_neighbor_mask != last_mask
}

fn six_neighbors(pos: ChunkPos) -> [ChunkPos; 6] {
    [
        ChunkPos {
            x: pos.x - 1,
            ..pos
        },
        ChunkPos {
            x: pos.x + 1,
            ..pos
        },
        ChunkPos {
            y: pos.y - 1,
            ..pos
        },
        ChunkPos {
            y: pos.y + 1,
            ..pos
        },
        ChunkPos {
            z: pos.z - 1,
            ..pos
        },
        ChunkPos {
            z: pos.z + 1,
            ..pos
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChunkLocalPos;

    #[test]
    fn boundary_neighbors_empty_for_interior_voxel() {
        let cp = ChunkPos { x: 0, y: 0, z: 0 };
        let lp = ChunkLocalPos::new(5, 5, 5);
        assert!(boundary_neighbors(cp, lp).is_empty());
    }

    #[test]
    fn boundary_neighbors_one_face() {
        let cp = ChunkPos { x: 2, y: 0, z: 0 };
        let lp = ChunkLocalPos::new(0, 5, 5); // on the -X face only
        let out = boundary_neighbors(cp, lp);
        assert_eq!(out, vec![ChunkPos { x: 1, y: 0, z: 0 }]);
    }

    #[test]
    fn boundary_neighbors_corner_hits_three_faces() {
        let cp = ChunkPos { x: 0, y: 0, z: 0 };
        let max = (CHUNK_SIZE - 1) as u8;
        let lp = ChunkLocalPos::new(0, max, 0); // -X, +Y, -Z corner
        let out = boundary_neighbors(cp, lp);
        assert_eq!(out.len(), 3);
        assert!(out.contains(&ChunkPos { x: -1, y: 0, z: 0 }));
        assert!(out.contains(&ChunkPos { x: 0, y: 1, z: 0 }));
        assert!(out.contains(&ChunkPos { x: 0, y: 0, z: -1 }));
    }

    /// Always generates an all-air chunk and always reports itself as
    /// "definitely air" — stands in for both the `known_empty` case (a
    /// chunk streamed once, found empty, and discarded) and the
    /// never-generated-at-all case (a chunk `update()`'s request loop
    /// skipped outright because `is_definitely_air` said not to bother —
    /// see `set_block_at`'s doc comment).
    struct AirGenerator;
    impl WorldGenerator for AirGenerator {
        fn generate(&self, _pos: ChunkPos, _seed: u64) -> Chunk {
            Chunk::new()
        }
        fn is_definitely_air(&self, _pos: ChunkPos, _seed: u64) -> bool {
            true
        }
    }

    #[test]
    fn set_block_at_materializes_known_empty_chunk() {
        // Building upward: the surface chunk is loaded, but the chunk
        // directly above it meshed to zero faces on first generation (pure
        // air) and was therefore never stored — only recorded in
        // known_empty. Placing a block into it (e.g. against the surface
        // chunk's top face) must succeed, not silently no-op.
        let mut stream = AsyncWorldStream::new(1, 1, None);
        stream.generator = Some(Arc::new(AirGenerator) as Arc<dyn WorldGenerator>);
        let pos = ChunkPos { x: 0, y: 1, z: 0 };
        stream.known_empty.insert(pos);

        let stone = BlockTypeId(1);
        // World Y 16.0 is the first metre of chunk y=1 (chunk size = 32 *
        // 0.5 = 16m), local voxel (2,0,2) within it.
        let placed = stream.set_block_at(1.0, 16.0, 1.0, stone);

        assert!(
            placed,
            "placing into a known_empty chunk should materialize it, not no-op"
        );
        assert!(!stream.known_empty.contains(&pos));
        let chunk = stream
            .chunks()
            .get(&pos)
            .expect("chunk should be materialized");
        assert_eq!(chunk.get(ChunkLocalPos::new(2, 0, 2)), stone);
        assert!(stream.remesh_queue.contains(&pos));
    }

    #[test]
    fn set_block_at_materializes_chunk_never_generated_at_all() {
        // The chunk was never streamed AT ALL — not known_empty, not
        // in_flight, not in inner.chunks — exactly what happens when
        // is_definitely_air short-circuits it out of update()'s request
        // loop entirely (e.g. building straight up past the generator's
        // "highest possible terrain" bound). Must still succeed.
        let mut stream = AsyncWorldStream::new(1, 1, None);
        stream.generator = Some(Arc::new(AirGenerator) as Arc<dyn WorldGenerator>);
        let pos = ChunkPos { x: 0, y: 3, z: 0 };

        let stone = BlockTypeId(1);
        let placed = stream.set_block_at(1.0, 49.0, 1.0, stone); // chunk y=3 spans world Y [48, 64)

        assert!(
            placed,
            "placing into a never-generated chunk should materialize it, not no-op"
        );
        let chunk = stream
            .chunks()
            .get(&pos)
            .expect("chunk should be materialized");
        assert_eq!(chunk.get(ChunkLocalPos::new(2, 2, 2)), stone);
    }

    #[test]
    fn set_block_at_noop_while_chunk_in_flight() {
        // A worker is already generating this exact chunk. Materializing a
        // second copy here would race the worker's own result landing,
        // which would silently overwrite this edit — must stay a no-op.
        let mut stream = AsyncWorldStream::new(1, 1, None);
        stream.generator = Some(Arc::new(AirGenerator) as Arc<dyn WorldGenerator>);
        let pos = ChunkPos { x: 0, y: 0, z: 0 };
        stream.in_flight.insert(pos);

        let placed = stream.set_block_at(1.0, 1.0, 1.0, BlockTypeId(1));
        assert!(!placed);
        assert!(!stream.chunks().contains_key(&pos));
    }

    /// Fresh, unique scratch directory per test — cleaned up on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "cubic_world_stream_pool_test_{tag}_{}_{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn update_promotes_definitely_air_chunk_with_saved_diff() {
        // A chunk the generator's is_definitely_air heuristic would skip
        // outright (e.g. above the highest possible terrain, as with
        // building above the surface) but which has a saved diff from a
        // previous session (e.g. the player built there, flew away far
        // enough to unload it, and came back) must still get requested for
        // generation, not silently dropped into known_empty forever.
        let dir = TempDir::new("with_diff");
        let mut cache = RegionCache::new(dir.0.clone(), 4);
        let pos = ChunkPos { x: 0, y: 3, z: 0 };
        let diff = crate::region::ChunkDiff::Sparse {
            entries: vec![crate::region::SparseDiffEntry {
                local_pos: crate::region::encode_local_pos(2, 2, 2),
                block_id: 1,
            }],
            cpd: vec![],
        };
        cache.write_chunk(pos.x, pos.y, pos.z, &diff).unwrap();

        let mut stream = AsyncWorldStream::new(1, 1, None);
        let generator = Arc::new(AirGenerator) as Arc<dyn WorldGenerator>;
        stream.set_persistence(Arc::new(Mutex::new(cache)), Arc::clone(&generator), 0, 512);

        stream.update(pos, &generator, 0, &Arc::new(BlockFaceTextures::new()));

        assert!(
            stream.in_flight.contains(&pos),
            "chunk with a saved diff should be requested despite is_definitely_air"
        );
        assert!(!stream.known_empty.contains(&pos));
    }

    #[test]
    fn update_still_skips_definitely_air_chunk_with_no_diff() {
        // Same generator/heuristic, but nothing was ever saved for this
        // position — the ordinary case (vast majority of "definitely air"
        // chunks) must still take the cheap known_empty short-circuit path,
        // not pay a worker generation for genuinely empty space.
        let dir = TempDir::new("no_diff");
        let cache = RegionCache::new(dir.0.clone(), 4);

        let mut stream = AsyncWorldStream::new(1, 1, None);
        let generator = Arc::new(AirGenerator) as Arc<dyn WorldGenerator>;
        stream.set_persistence(Arc::new(Mutex::new(cache)), Arc::clone(&generator), 0, 512);

        let pos = ChunkPos { x: 0, y: 3, z: 0 };
        stream.update(pos, &generator, 0, &Arc::new(BlockFaceTextures::new()));

        assert!(stream.known_empty.contains(&pos));
        assert!(!stream.in_flight.contains(&pos));
    }
}
