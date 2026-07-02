// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{Chunk, ChunkPos, StreamDelta, WorldGenerator, WorldStream};
use std::collections::HashSet;
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
    chunk: Chunk,
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
    // Kept alive so threads don't exit when idle
    _workers: Vec<JoinHandle<()>>,
}

impl AsyncWorldStream {
    pub fn new(radius: i32) -> Self {
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
                                // If send fails the main thread has exited — just stop
                                if result_tx
                                    .send(WorkResult {
                                        pos: work.pos,
                                        chunk,
                                    })
                                    .is_err()
                                {
                                    break;
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
            inner: WorldStream::new(radius),
            in_flight: HashSet::new(),
            discard: HashSet::new(),
            work_tx,
            result_rx,
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
        let r = self.inner.radius;
        let mut to_unload = Vec::new();
        let mut loaded = Vec::new();

        // --- Drain completed results ---
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.pos);
            if self.discard.remove(&result.pos) {
                // Was unloaded while in flight — throw it away
                continue;
            }
            self.inner.chunks.insert(result.pos, result.chunk);
            loaded.push(result.pos);
        }

        // --- Compute desired set and find what needs loading ---
        for x in (center.x - r)..=(center.x + r) {
            for y in (center.y - r)..=(center.y + r) {
                for z in (center.z - r)..=(center.z + r) {
                    let pos = ChunkPos { x, y, z };
                    if !self.inner.chunks.contains_key(&pos) && !self.in_flight.contains(&pos) {
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
            if (pos.x - center.x).abs() > r
                || (pos.y - center.y).abs() > r
                || (pos.z - center.z).abs() > r
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
                (pos.x - center.x).abs() > r
                    || (pos.y - center.y).abs() > r
                    || (pos.z - center.z).abs() > r
            })
            .collect();
        for pos in to_discard {
            self.in_flight.remove(&pos);
        }

        // Apply unloads
        for pos in &to_unload {
            self.inner.chunks.remove(pos);
        }

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
}
