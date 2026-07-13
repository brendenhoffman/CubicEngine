// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Region file I/O for world persistence.
//!
//! One `.cbr` file covers a 32×32 XZ column of chunks, unbounded Y extent.
//! Named `r.X.Z.cbr` where X and Z are `chunk_x.div_euclid(32)` and
//! `chunk_z.div_euclid(32)`.
//!
//! # File layout
//! ```text
//! [magic: 4 bytes]        "CBRG"
//! [version: u16]          currently 1
//! [reserved: u16]         zero
//! [xz_table: 1024 × 8]   (column_offset: u32, column_len: u32) per XZ slot;
//!                         both zero means no data for that column
//! [data region ...]       Y indices and chunk payloads, written in append order
//! ```
//!
//! # Per-column Y index
//! At `column_offset`, a sequence of 10-byte `YEntry` records:
//! `[chunk_y: i16, data_offset: u32, data_len: u32]`
//! terminated by a sentinel where `chunk_y == i16::MIN`.
//! Entries are kept sorted by `chunk_y` for binary search.
//! A tombstone has `data_len == 0` with a non-sentinel `chunk_y`;
//! compaction removes them.
//!
//! # Compaction
//! Writes a fresh file to `<path>.tmp`, then atomically renames it over the
//! original. Tombstoned entries and orphaned data blobs are dropped.

use crate::{BlockTypeId, Chunk, ChunkLocalPos, CHUNK_VOLUME};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"CBRG";
const VERSION: u16 = 1;

/// Byte offset where the XZ lookup table begins.
const XZ_TABLE_OFFSET: u64 = 8; // 4 magic + 2 version + 2 reserved
/// Number of XZ slots (32 × 32).
const XZ_SLOTS: usize = 1024;
/// Bytes per XZ slot: column_offset (u32) + column_len (u32).
const XZ_SLOT_SIZE: u64 = 8;
/// Byte offset where the data region begins.
const DATA_START: u64 = XZ_TABLE_OFFSET + XZ_SLOTS as u64 * XZ_SLOT_SIZE;

/// Bytes per Y index entry: chunk_y (i16) + data_offset (u32) + data_len (u32).
const Y_ENTRY_SIZE: usize = 10;
const Y_SENTINEL: i16 = i16::MIN;

const DIFF_TAG_SPARSE: u8 = 0;
const DIFF_TAG_FULL: u8 = 1;

/// Soft warning threshold: log if a single chunk's saved data exceeds this.
const CHUNK_SIZE_WARN_BYTES: usize = 1024 * 1024; // 1 MiB

// ---------------------------------------------------------------------------
// Public diff types
// ---------------------------------------------------------------------------

/// A single changed voxel in a sparse diff.
/// `local_pos` encodes `x | (z << 5) | (y << 10)` — 5 bits each.
#[derive(Debug, Clone)]
pub struct SparseDiffEntry {
    pub local_pos: u16,
    pub block_id: u32,
}

/// Per-block instance data (CPD). Placeholder — not yet populated.
#[derive(Debug, Clone)]
pub struct CpdEntry {
    pub local_pos: u16,
    pub data: Vec<u8>,
}

/// The persisted delta for one chunk.
#[derive(Debug, Clone)]
pub enum ChunkDiff {
    /// Fewer changed voxels than `threshold` — only deltas stored.
    Sparse {
        entries: Vec<SparseDiffEntry>,
        cpd: Vec<CpdEntry>,
    },
    /// At or above `threshold` — full chunk LZ4-compressed.
    Full {
        compressed: Vec<u8>,
        cpd: Vec<CpdEntry>,
    },
}

// ---------------------------------------------------------------------------
// Local-pos encode/decode
// ---------------------------------------------------------------------------

#[inline]
pub fn encode_local_pos(x: u8, y: u8, z: u8) -> u16 {
    x as u16 | ((z as u16) << 5) | ((y as u16) << 10)
}

#[inline]
pub fn decode_local_pos(pos: u16) -> (u8, u8, u8) {
    (
        (pos & 0x1F) as u8,
        ((pos >> 10) & 0x1F) as u8,
        ((pos >> 5) & 0x1F) as u8,
    )
    // returns (x, y, z)
}

// ---------------------------------------------------------------------------
// Diff computation and application
// ---------------------------------------------------------------------------

/// Compute the delta between a generator baseline and the current chunk state.
/// Returns `None` if the chunk is still virgin (no changes).
pub fn diff_from_chunks(original: &Chunk, modified: &Chunk, threshold: usize) -> Option<ChunkDiff> {
    let mut entries: Vec<SparseDiffEntry> = Vec::new();

    for y in 0u8..32 {
        for z in 0u8..32 {
            for x in 0u8..32 {
                let pos = ChunkLocalPos::new(x, y, z);
                let orig = original.get(pos);
                let modif = modified.get(pos);
                if orig != modif {
                    entries.push(SparseDiffEntry {
                        local_pos: encode_local_pos(x, y, z),
                        block_id: modif.0,
                    });
                }
            }
        }
    }

    if entries.is_empty() {
        return None; // still virgin
    }

    if entries.len() >= threshold {
        Some(ChunkDiff::Full {
            compressed: compress_chunk(modified),
            cpd: vec![],
        })
    } else {
        Some(ChunkDiff::Sparse {
            entries,
            cpd: vec![],
        })
    }
}

/// Apply a saved diff onto a chunk in place.
pub fn apply_diff(chunk: &mut Chunk, diff: &ChunkDiff) {
    match diff {
        ChunkDiff::Sparse { entries, .. } => {
            for e in entries {
                let (x, y, z) = decode_local_pos(e.local_pos);
                chunk.set(ChunkLocalPos::new(x, y, z), BlockTypeId(e.block_id));
            }
        }
        ChunkDiff::Full { compressed, .. } => match decompress_chunk(compressed) {
            Ok(c) => *chunk = c,
            Err(e) => {
                tracing::error!("failed to decompress full chunk diff: {e:#} — chunk unchanged")
            }
        },
    }
}

// ---------------------------------------------------------------------------
// LZ4 full-chunk serialization
// ---------------------------------------------------------------------------

/// `[palette_len: u32][palette: palette_len × u32][data: CHUNK_VOLUME × u16]` then LZ4.
fn compress_chunk(chunk: &Chunk) -> Vec<u8> {
    let palette_len = chunk.palette.len() as u32;
    let mut raw = Vec::with_capacity(4 + chunk.palette.len() * 4 + CHUNK_VOLUME * 2);
    raw.extend_from_slice(&palette_len.to_le_bytes());
    for id in &chunk.palette {
        raw.extend_from_slice(&id.0.to_le_bytes());
    }
    for &idx in chunk.data.iter() {
        raw.extend_from_slice(&idx.to_le_bytes());
    }
    lz4_flex::compress_prepend_size(&raw)
}

fn decompress_chunk(compressed: &[u8]) -> Result<Chunk> {
    let raw = lz4_flex::decompress_size_prepended(compressed).context("lz4 decompress failed")?;

    let min_len = 4;
    if raw.len() < min_len {
        bail!("chunk payload too short");
    }
    let palette_len = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let palette_end = 4 + palette_len * 4;
    let expected_len = palette_end + CHUNK_VOLUME * 2;
    if raw.len() < expected_len {
        bail!(
            "chunk payload truncated (got {}, need {expected_len})",
            raw.len()
        );
    }

    let palette = (0..palette_len)
        .map(|i| {
            let off = 4 + i * 4;
            BlockTypeId(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()))
        })
        .collect();

    let mut data = Box::new([0u16; CHUNK_VOLUME]);
    for i in 0..CHUNK_VOLUME {
        let off = palette_end + i * 2;
        data[i] = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
    }

    Ok(Chunk {
        palette,
        data,
        instance_data: HashMap::new(),
    })
}

// ---------------------------------------------------------------------------
// ChunkDiff binary serialization (payload written into region data region)
// ---------------------------------------------------------------------------

fn serialize_diff(diff: &ChunkDiff) -> Vec<u8> {
    let mut out = Vec::new();
    match diff {
        ChunkDiff::Sparse { entries, .. } => {
            out.push(DIFF_TAG_SPARSE);
            out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for e in entries {
                out.extend_from_slice(&e.local_pos.to_le_bytes());
                out.extend_from_slice(&e.block_id.to_le_bytes());
            }
        }
        ChunkDiff::Full { compressed, .. } => {
            out.push(DIFF_TAG_FULL);
            out.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
            out.extend_from_slice(compressed);
        }
    }
    // CPD count placeholder (not yet used)
    out.extend_from_slice(&0u32.to_le_bytes());
    out
}

fn deserialize_diff(data: &[u8]) -> Result<ChunkDiff> {
    if data.is_empty() {
        bail!("empty diff payload");
    }
    let tag = data[0];
    let mut cur = 1usize;

    let read_u32 = |buf: &[u8], off: usize| -> Result<u32> {
        buf.get(off..off + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .context("diff payload truncated reading u32")
    };

    match tag {
        DIFF_TAG_SPARSE => {
            let count = read_u32(data, cur)? as usize;
            cur += 4;
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                if cur + 6 > data.len() {
                    bail!("sparse diff truncated at entry");
                }
                let local_pos = u16::from_le_bytes(data[cur..cur + 2].try_into().unwrap());
                let block_id = u32::from_le_bytes(data[cur + 2..cur + 6].try_into().unwrap());
                cur += 6;
                entries.push(SparseDiffEntry {
                    local_pos,
                    block_id,
                });
            }
            Ok(ChunkDiff::Sparse {
                entries,
                cpd: vec![],
            })
        }
        DIFF_TAG_FULL => {
            let len = read_u32(data, cur)? as usize;
            cur += 4;
            if cur + len > data.len() {
                bail!(
                    "full diff truncated (need {len}, have {})",
                    data.len() - cur
                );
            }
            Ok(ChunkDiff::Full {
                compressed: data[cur..cur + len].to_vec(),
                cpd: vec![],
            })
        }
        t => bail!("unknown diff tag {t:#04x}"),
    }
}

// ---------------------------------------------------------------------------
// Region path helper
// ---------------------------------------------------------------------------

/// Returns the path to the `.cbr` file for the region containing `(chunk_x, chunk_z)`.
pub fn region_path(world_dir: &Path, chunk_x: i32, chunk_z: i32) -> PathBuf {
    let rx = chunk_x.div_euclid(32);
    let rz = chunk_z.div_euclid(32);
    world_dir.join("regions").join(format!("r.{rx}.{rz}.cbr"))
}

#[inline]
fn xz_slot(chunk_x: i32, chunk_z: i32) -> usize {
    chunk_x.rem_euclid(32) as usize + chunk_z.rem_euclid(32) as usize * 32
}

// ---------------------------------------------------------------------------
// Y index entry (in-memory representation)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct YEntry {
    chunk_y: i16,
    data_offset: u32,
    /// 0 = tombstone
    data_len: u32,
}

impl YEntry {
    fn is_live(self) -> bool {
        self.data_len > 0
    }
}

// ---------------------------------------------------------------------------
// RegionFile
// ---------------------------------------------------------------------------

pub struct RegionFile {
    path: PathBuf,
    file: std::fs::File,
}

impl RegionFile {
    /// Open an existing region file or create a new empty one.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("opening region file {}", path.display()))?;

        let mut rf = Self {
            path: path.to_owned(),
            file,
        };

        let file_len = rf.file.seek(SeekFrom::End(0))?;
        if file_len == 0 {
            rf.write_fresh_header()?;
        } else {
            rf.validate_header()?;
        }

        Ok(rf)
    }

    // --- Header ---

    fn write_fresh_header(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(MAGIC)?;
        self.file.write_all(&VERSION.to_le_bytes())?;
        self.file.write_all(&0u16.to_le_bytes())?; // reserved
        self.file
            .write_all(&vec![0u8; XZ_SLOTS * XZ_SLOT_SIZE as usize])?;
        self.file.flush()?;
        Ok(())
    }

    fn validate_header(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 4];
        self.file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("bad magic bytes in {}", self.path.display());
        }
        let mut ver_buf = [0u8; 2];
        self.file.read_exact(&mut ver_buf)?;
        let ver = u16::from_le_bytes(ver_buf);
        if ver != VERSION {
            bail!(
                "unsupported region file version {ver} in {}",
                self.path.display()
            );
        }
        Ok(())
    }

    // --- XZ table ---

    fn xz_entry_offset(slot: usize) -> u64 {
        XZ_TABLE_OFFSET + slot as u64 * XZ_SLOT_SIZE
    }

    fn read_xz_entry(&mut self, slot: usize) -> Result<(u32, u32)> {
        self.file
            .seek(SeekFrom::Start(Self::xz_entry_offset(slot)))?;
        let mut buf = [0u8; 8];
        self.file.read_exact(&mut buf)?;
        Ok((
            u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        ))
    }

    fn write_xz_entry(&mut self, slot: usize, col_offset: u32, col_len: u32) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(Self::xz_entry_offset(slot)))?;
        self.file.write_all(&col_offset.to_le_bytes())?;
        self.file.write_all(&col_len.to_le_bytes())?;
        Ok(())
    }

    // --- Y index ---

    fn read_y_index(&mut self, col_offset: u32, col_len: u32) -> Result<Vec<YEntry>> {
        if col_offset == 0 && col_len == 0 {
            return Ok(vec![]);
        }
        self.file.seek(SeekFrom::Start(col_offset as u64))?;
        let mut entries = Vec::new();
        let max_entries = col_len as usize / Y_ENTRY_SIZE;
        for _ in 0..=max_entries {
            let mut buf = [0u8; Y_ENTRY_SIZE];
            self.file.read_exact(&mut buf)?;
            let chunk_y = i16::from_le_bytes(buf[0..2].try_into().unwrap());
            if chunk_y == Y_SENTINEL {
                break;
            }
            let data_offset = u32::from_le_bytes(buf[2..6].try_into().unwrap());
            let data_len = u32::from_le_bytes(buf[6..10].try_into().unwrap());
            entries.push(YEntry {
                chunk_y,
                data_offset,
                data_len,
            });
        }
        Ok(entries)
    }

    /// Append a new Y index block for a column and update its XZ slot.
    /// Old Y index block is abandoned in place (reclaimed by compaction).
    fn write_y_index(&mut self, slot: usize, entries: &[YEntry]) -> Result<()> {
        let col_offset = self.file.seek(SeekFrom::End(0))? as u32;
        for e in entries {
            self.file.write_all(&e.chunk_y.to_le_bytes())?;
            self.file.write_all(&e.data_offset.to_le_bytes())?;
            self.file.write_all(&e.data_len.to_le_bytes())?;
        }
        // Sentinel
        self.file.write_all(&Y_SENTINEL.to_le_bytes())?;
        self.file.write_all(&0u32.to_le_bytes())?;
        self.file.write_all(&0u32.to_le_bytes())?;

        let col_end = self.file.stream_position()? as u32;
        let col_len = col_end - col_offset;
        self.write_xz_entry(slot, col_offset, col_len)?;
        Ok(())
    }

    // --- Public API ---

    /// Read the saved diff for a chunk. Returns `None` if the chunk is virgin.
    pub fn read_chunk(
        &mut self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
    ) -> Result<Option<ChunkDiff>> {
        let slot = xz_slot(chunk_x, chunk_z);
        let (col_offset, col_len) = self.read_xz_entry(slot)?;
        let y_index = self.read_y_index(col_offset, col_len)?;

        let Some(entry) = y_index.iter().find(|e| e.chunk_y == chunk_y as i16) else {
            return Ok(None);
        };
        if !entry.is_live() {
            return Ok(None); // tombstoned
        }

        self.file.seek(SeekFrom::Start(entry.data_offset as u64))?;
        let mut buf = vec![0u8; entry.data_len as usize];
        self.file.read_exact(&mut buf)?;
        Ok(Some(deserialize_diff(&buf)?))
    }

    /// Write a diff for a chunk. Appends payload data then rewrites the Y index.
    pub fn write_chunk(
        &mut self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        diff: &ChunkDiff,
    ) -> Result<()> {
        let slot = xz_slot(chunk_x, chunk_z);
        let (col_offset, col_len) = self.read_xz_entry(slot)?;
        let mut y_index = self.read_y_index(col_offset, col_len)?;

        // Append payload
        let data_offset = self.file.seek(SeekFrom::End(0))? as u32;
        let payload = serialize_diff(diff);
        let data_len = payload.len() as u32;

        if data_len as usize >= CHUNK_SIZE_WARN_BYTES {
            tracing::warn!(
                "chunk ({chunk_x},{chunk_y},{chunk_z}) payload is {} bytes — exceeds soft 1 MiB limit",
                data_len
            );
        }

        self.file.write_all(&payload)?;

        // Update or insert Y entry, keeping sorted order
        if let Some(e) = y_index.iter_mut().find(|e| e.chunk_y == chunk_y as i16) {
            e.data_offset = data_offset;
            e.data_len = data_len;
        } else {
            let insert_at = y_index.partition_point(|e| e.chunk_y < chunk_y as i16);
            y_index.insert(
                insert_at,
                YEntry {
                    chunk_y: chunk_y as i16,
                    data_offset,
                    data_len,
                },
            );
        }

        self.write_y_index(slot, &y_index)?;
        self.file.flush()?;
        Ok(())
    }

    /// Mark a chunk entry as a tombstone (chunk reverted to virgin).
    pub fn remove_chunk(&mut self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Result<()> {
        let slot = xz_slot(chunk_x, chunk_z);
        let (col_offset, col_len) = self.read_xz_entry(slot)?;
        let mut y_index = self.read_y_index(col_offset, col_len)?;

        let Some(e) = y_index.iter_mut().find(|e| e.chunk_y == chunk_y as i16) else {
            return Ok(()); // not saved, nothing to do
        };
        e.data_len = 0; // tombstone

        self.write_y_index(slot, &y_index)?;
        self.file.flush()?;
        Ok(())
    }

    /// Rewrite the file compactly, dropping tombstones and orphaned data.
    /// Atomically replaces the original via a tmp file + rename.
    pub fn compact(&mut self) -> Result<()> {
        // --- Pass 1: collect all live entries and their payloads from old file ---

        struct LiveColumn {
            slot: usize,
            entries: Vec<(i16, Vec<u8>)>, // (chunk_y, payload)
        }

        let mut columns: Vec<LiveColumn> = Vec::new();

        for slot in 0..XZ_SLOTS {
            let (col_offset, col_len) = self.read_xz_entry(slot)?;
            let y_index = self.read_y_index(col_offset, col_len)?;
            let live: Vec<_> = y_index.into_iter().filter(|e| e.is_live()).collect();
            if live.is_empty() {
                continue;
            }
            let mut entries = Vec::with_capacity(live.len());
            for e in live {
                self.file.seek(SeekFrom::Start(e.data_offset as u64))?;
                let mut payload = vec![0u8; e.data_len as usize];
                self.file.read_exact(&mut payload)?;
                entries.push((e.chunk_y, payload));
            }
            columns.push(LiveColumn { slot, entries });
        }

        // --- Pass 2: build new file in memory (header + XZ table + data) ---

        // Start with header + blank XZ table
        let mut new_file: Vec<u8> = Vec::with_capacity(DATA_START as usize + columns.len() * 1024);
        new_file.extend_from_slice(MAGIC);
        new_file.extend_from_slice(&VERSION.to_le_bytes());
        new_file.extend_from_slice(&0u16.to_le_bytes()); // reserved
        let xz_table_start = new_file.len();
        new_file.resize(xz_table_start + XZ_SLOTS * XZ_SLOT_SIZE as usize, 0);

        // For each live column: write Y index then data blobs, record XZ entry
        for col in &columns {
            let col_start = new_file.len() as u32;

            // Y index size: entries × Y_ENTRY_SIZE + sentinel
            let y_index_size = (col.entries.len() + 1) * Y_ENTRY_SIZE;
            // Data blobs start immediately after Y index
            let mut blob_offset = col_start as usize + y_index_size;

            // Write Y index entries
            for (chunk_y, payload) in &col.entries {
                new_file.extend_from_slice(&chunk_y.to_le_bytes());
                new_file.extend_from_slice(&(blob_offset as u32).to_le_bytes());
                new_file.extend_from_slice(&(payload.len() as u32).to_le_bytes());
                blob_offset += payload.len();
            }
            // Sentinel
            new_file.extend_from_slice(&Y_SENTINEL.to_le_bytes());
            new_file.extend_from_slice(&0u32.to_le_bytes());
            new_file.extend_from_slice(&0u32.to_le_bytes());

            // Write data blobs
            for (_, payload) in &col.entries {
                new_file.extend_from_slice(payload);
            }

            let col_len = new_file.len() as u32 - col_start;

            // Patch XZ table slot
            let slot_off = xz_table_start + col.slot * XZ_SLOT_SIZE as usize;
            new_file[slot_off..slot_off + 4].copy_from_slice(&col_start.to_le_bytes());
            new_file[slot_off + 4..slot_off + 8].copy_from_slice(&col_len.to_le_bytes());
        }

        // --- Pass 3: atomic replace via tmp file ---

        let tmp_path = self.path.with_extension("cbr.tmp");
        std::fs::write(&tmp_path, &new_file)
            .with_context(|| format!("writing compact tmp {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &self.path)
            .with_context(|| format!("renaming compact tmp over {}", self.path.display()))?;

        // Reopen the fresh file so self.file stays valid
        self.file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .with_context(|| format!("reopening after compact {}", self.path.display()))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RegionCache
// ---------------------------------------------------------------------------

/// LRU-capped cache of open RegionFiles. Shared via Arc<Mutex<>> between the
/// main thread (writes on unload) and worker threads (reads on load).
pub struct RegionCache {
    world_dir: std::path::PathBuf,
    open_files: std::collections::HashMap<(i32, i32), RegionFile>,
    /// Insertion-order key list for LRU eviction.
    lru_order: std::collections::VecDeque<(i32, i32)>,
    max_open: usize,
}

impl RegionCache {
    pub fn new(world_dir: std::path::PathBuf, max_open: usize) -> Self {
        Self {
            world_dir,
            open_files: std::collections::HashMap::new(),
            lru_order: std::collections::VecDeque::new(),
            max_open,
        }
    }

    fn region_key(chunk_x: i32, chunk_z: i32) -> (i32, i32) {
        (chunk_x.div_euclid(32), chunk_z.div_euclid(32))
    }

    fn get_or_open(&mut self, chunk_x: i32, chunk_z: i32) -> anyhow::Result<&mut RegionFile> {
        let key = Self::region_key(chunk_x, chunk_z);

        if !self.open_files.contains_key(&key) {
            // Evict LRU if at capacity
            if self.open_files.len() >= self.max_open {
                if let Some(evict_key) = self.lru_order.pop_front() {
                    self.open_files.remove(&evict_key);
                }
            }
            let path = region_path(&self.world_dir, chunk_x, chunk_z);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let rf = RegionFile::open(&path)?;
            self.open_files.insert(key, rf);
            self.lru_order.push_back(key);
        } else {
            // Bubble to back (most recently used)
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push_back(key);
        }

        Ok(self.open_files.get_mut(&key).unwrap())
    }

    pub fn read_chunk(
        &mut self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
    ) -> anyhow::Result<Option<ChunkDiff>> {
        self.get_or_open(chunk_x, chunk_z)?
            .read_chunk(chunk_x, chunk_y, chunk_z)
    }

    pub fn write_chunk(
        &mut self,
        chunk_x: i32,
        chunk_y: i32,
        chunk_z: i32,
        diff: &ChunkDiff,
    ) -> anyhow::Result<()> {
        self.get_or_open(chunk_x, chunk_z)?
            .write_chunk(chunk_x, chunk_y, chunk_z, diff)
    }

    pub fn remove_chunk(&mut self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> anyhow::Result<()> {
        self.get_or_open(chunk_x, chunk_z)?
            .remove_chunk(chunk_x, chunk_y, chunk_z)
    }
}
