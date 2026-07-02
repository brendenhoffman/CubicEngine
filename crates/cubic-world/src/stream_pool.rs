// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{mesh_chunk, Chunk, ChunkPos, StreamDelta, WorldGenerator, WorldStream};
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
    // Positions a worker determined produce no geometry (pure air or fully
    // buried). Never inserted into `inner.chunks`, so tracked separately to
    // avoid re-dispatching them to workers every frame.
    known_air: HashSet<ChunkPos>,
    _workers: Vec<JoinHandle<()>>,
}

impl AsyncWorldStream {
    pub fn new(radius_xz: i32, radius_y: i32) -> Self {
        let worker_count = thread::available_parallelism()
            .map_or(4, |n| n.get())
            .saturating_sub(1)
            .max(1);

        let (work_tx, work_rx) = mpsc::channel::<WorkItem>();
        let (result_tx, result_rx) = mpsc::channel::<WorkResult>();

        // Wrap the receiver so multiple workers can pull from it
        let work_rx = Arc::new(Mutex::new(work_rx));

        let workers = (0..worker_count)
            .map(|_| {
                let work_rx = Arc::clone(&work_rx);
                let result_tx = result_tx.clone();
                thread::spawn(move || {
                    loop {
                        let item = {
                            let rx = work_rx.lock().unwrap();
                            rx.recv()
                        };
                        match item {
                            Ok(work) => {
                                let chunk = work.generator.generate(work.pos, work.seed);
                                let (vertices, indices) = mesh_chunk(&chunk, [None; 6]);
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
            known_air: HashSet::new(),
            _workers: workers,
        }
    }

    /// Same contract as `WorldStream::update` but generation is async.
    /// Loaded chunks in the returned delta are only the ones that completed
    /// this frame — newly dispatched ones will appear in future frames.
    pub fn update(
        &mut self,
        center: ChunkPos,
        generator: &Arc<dyn WorldGenerator>,
        seed: u64,
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
                    self.known_air.insert(result.pos);
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
                    if !self.inner.chunks.contains_key(&pos)
                        && !self.in_flight.contains(&pos)
                        && !self.known_air.contains(&pos)
                    {
                        self.in_flight.insert(pos);
                        // best-effort send — if the channel is full we'll retry next frame
                        let _ = self.work_tx.send(WorkItem {
                            pos,
                            seed,
                            generator: Arc::clone(generator),
                        });
                    }
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

        // known_air positions never enter `inner.chunks`, so they can't be
        // found via the loop above — drop the ones that fell out of range
        // directly so the set doesn't grow unbounded as the camera moves.
        self.known_air.retain(|pos| {
            (pos.x - center.x).abs() <= rxz
                && (pos.y - center.y).abs() <= ry
                && (pos.z - center.z).abs() <= rxz
        });

        StreamDelta {
            loaded,
            unloaded: to_unload,
        }
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
