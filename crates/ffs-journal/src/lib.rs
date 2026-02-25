#![forbid(unsafe_code)]
//! JBD2-compatible journal replay, native COW journal, and per-core WAL buffers.
//!
//! This crate provides three complementary journaling mechanisms:
//! 1. A JBD2 replay engine for compatibility-mode ext4 recovery.
//! 2. A native append-only COW journal for FrankenFS MVCC commits.
//! 3. Per-core WAL buffers for lock-free concurrent MVCC writes.

pub mod wal_buffer;

use asupersync::Cx;
use ffs_block::BlockDevice;
use ffs_error::{FfsError, Result};
use ffs_types::{BlockNumber, CommitSeq};
use std::collections::{BTreeMap, BTreeSet};

const JBD2_MAGIC: u32 = 0xC03B_3998;
const JBD2_BLOCKTYPE_DESCRIPTOR: u32 = 1;
const JBD2_BLOCKTYPE_COMMIT: u32 = 2;
const JBD2_BLOCKTYPE_REVOKE: u32 = 5;
const JBD2_HEADER_SIZE: usize = 12;
const JBD2_REVOKE_HEADER_SIZE: usize = 16; // journal header (12) + r_count (4)
const JBD2_TAG_SIZE: usize = 8;
const JBD2_TAG_FLAG_LAST: u32 = 0x0000_0008;

const COW_MAGIC: u32 = 0x4A53_4646; // "FFSJ" in little-endian payload.
const COW_VERSION: u16 = 1;
const COW_RECORD_WRITE: u16 = 1;
const COW_RECORD_COMMIT: u16 = 2;
const COW_HEADER_SIZE: usize = 32;

/// Journal region expressed in block coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalSegment {
    pub start: BlockNumber,
    pub blocks: u64,
}

impl JournalSegment {
    /// Resolve a segment-relative index to an absolute block number.
    #[must_use]
    pub fn resolve(self, index: u64) -> Option<BlockNumber> {
        if index >= self.blocks {
            return None;
        }
        self.start.0.checked_add(index).map(BlockNumber)
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.blocks == 0
    }
}

/// Journal region expressed in block coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalRegion {
    pub start: BlockNumber,
    pub blocks: u64,
}

impl JournalRegion {
    /// Resolve a region-relative index to an absolute block number.
    #[must_use]
    pub fn resolve(self, index: u64) -> Option<BlockNumber> {
        if index >= self.blocks {
            return None;
        }
        self.start.0.checked_add(index).map(BlockNumber)
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.blocks == 0
    }
}

/// Aggregate replay counters for JBD2 recovery.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReplayStats {
    pub scanned_blocks: u64,
    pub descriptor_blocks: u64,
    pub descriptor_tags: u64,
    pub commit_blocks: u64,
    pub revoke_blocks: u64,
    pub revoke_entries: u64,
    pub replayed_blocks: u64,
    pub skipped_revoked_blocks: u64,
    pub orphaned_commit_blocks: u64,
    pub incomplete_transactions: u64,
}

/// Replay result including committed transaction sequence numbers.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReplayOutcome {
    pub committed_sequences: Vec<u32>,
    pub stats: ReplayStats,
}

/// A single committed transaction: a list of (target block, payload) writes
/// paired with the set of revoked block numbers for that transaction.
pub type CommittedTxn = (Vec<(BlockNumber, Vec<u8>)>, BTreeSet<BlockNumber>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Jbd2Header {
    magic: u32,
    block_type: u32,
    sequence: u32,
}

impl Jbd2Header {
    #[must_use]
    fn parse(bytes: &[u8]) -> Option<Self> {
        let magic = read_be_u32(bytes, 0)?;
        let block_type = read_be_u32(bytes, 4)?;
        let sequence = read_be_u32(bytes, 8)?;
        Some(Self {
            magic,
            block_type,
            sequence,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DescriptorTag {
    target: BlockNumber,
    flags: u32,
}

impl DescriptorTag {
    #[must_use]
    fn is_last(self) -> bool {
        (self.flags & JBD2_TAG_FLAG_LAST) != 0
    }
}

#[derive(Debug, Default)]
struct PendingTxn {
    writes: Vec<(BlockNumber, Vec<u8>)>,
    revoked: BTreeSet<BlockNumber>,
}

/// Replay JBD2 descriptor/commit/revoke blocks from a journal region.
///
/// Behavior:
/// - Descriptor blocks stage writes to target blocks.
/// - Revoke blocks mark target blocks as non-replayable for the same sequence.
/// - Commit blocks apply staged writes (except revoked entries).
/// - Uncommitted staged transactions are ignored.
pub fn replay_jbd2(
    cx: &Cx,
    dev: &dyn BlockDevice,
    journal_region: JournalRegion,
) -> Result<ReplayOutcome> {
    replay_jbd2_inner(cx, dev, journal_region.blocks, |index| {
        resolve_region_block(journal_region, index)
    })
}

/// Replay JBD2 descriptor/commit/revoke blocks from non-contiguous segments.
///
/// Segment order defines the journal index order, so the first block of the
/// second segment immediately follows the last block of the first segment.
pub fn replay_jbd2_segments(
    cx: &Cx,
    dev: &dyn BlockDevice,
    journal_segments: &[JournalSegment],
) -> Result<ReplayOutcome> {
    if journal_segments.is_empty() {
        return Err(FfsError::Format(
            "journal segment list must contain at least one segment".to_owned(),
        ));
    }
    if journal_segments.iter().any(|segment| segment.is_empty()) {
        return Err(FfsError::Format(
            "journal segments must not contain zero-length entries".to_owned(),
        ));
    }

    let total_blocks = journal_segments.iter().try_fold(0_u64, |acc, segment| {
        acc.checked_add(segment.blocks)
            .ok_or_else(|| FfsError::Format("journal segment length overflow".to_owned()))
    })?;

    replay_jbd2_inner(cx, dev, total_blocks, |index| {
        resolve_segment_block(journal_segments, index, total_blocks)
    })
}

#[allow(clippy::too_many_lines)]
fn replay_jbd2_inner(
    cx: &Cx,
    dev: &dyn BlockDevice,
    total_blocks: u64,
    mut resolve_block: impl FnMut(u64) -> Result<BlockNumber>,
) -> Result<ReplayOutcome> {
    if total_blocks == 0 {
        return Err(FfsError::Format(
            "journal region must contain at least one block".to_owned(),
        ));
    }

    let mut stats = ReplayStats::default();
    let mut pending: BTreeMap<u32, PendingTxn> = BTreeMap::new();
    let mut committed_sequences = BTreeSet::new();

    let mut idx = 0_u64;
    while idx < total_blocks {
        let absolute = resolve_block(idx)?;
        let raw = dev.read_block(cx, absolute)?;
        stats.scanned_blocks = stats.scanned_blocks.saturating_add(1);

        let Some(header) = Jbd2Header::parse(raw.as_slice()) else {
            idx = idx.saturating_add(1);
            continue;
        };

        if header.magic != JBD2_MAGIC {
            idx = idx.saturating_add(1);
            continue;
        }

        match header.block_type {
            JBD2_BLOCKTYPE_DESCRIPTOR => {
                stats.descriptor_blocks = stats.descriptor_blocks.saturating_add(1);
                let tags = parse_descriptor_tags(raw.as_slice());
                stats.descriptor_tags = stats
                    .descriptor_tags
                    .saturating_add(u64::try_from(tags.len()).unwrap_or(u64::MAX));

                // Check if all data blocks fit within the journal region.
                // A truncated descriptor (e.g. from power loss) means this
                // transaction is incomplete — stop scanning.
                let tag_count = u64::try_from(tags.len()).unwrap_or(u64::MAX);
                let last_data_idx = idx.checked_add(tag_count);
                if last_data_idx.is_none_or(|ld| ld >= total_blocks) {
                    break;
                }

                let mut staged = Vec::with_capacity(tags.len());
                for (tag_idx, tag) in tags.iter().enumerate() {
                    let offset_from_descriptor = u64::try_from(tag_idx)
                        .map_err(|_| {
                            FfsError::Format("descriptor tag index does not fit in u64".to_owned())
                        })?
                        .saturating_add(1);
                    let data_index = idx.checked_add(offset_from_descriptor).ok_or_else(|| {
                        FfsError::Format("journal descriptor data index overflow".to_owned())
                    })?;
                    let data_block = resolve_block(data_index)?;
                    let data = dev.read_block(cx, data_block)?.as_slice().to_vec();
                    staged.push((tag.target, data));
                }

                let txn = pending.entry(header.sequence).or_default();
                txn.writes.extend(staged);

                idx = idx.saturating_add(1).saturating_add(
                    u64::try_from(tags.len())
                        .map_err(|_| FfsError::Format("descriptor length overflow".to_owned()))?,
                );
            }
            JBD2_BLOCKTYPE_REVOKE => {
                stats.revoke_blocks = stats.revoke_blocks.saturating_add(1);
                let revokes = parse_revoke_entries(raw.as_slice());
                stats.revoke_entries = stats
                    .revoke_entries
                    .saturating_add(u64::try_from(revokes.len()).unwrap_or(u64::MAX));
                let txn = pending.entry(header.sequence).or_default();
                txn.revoked.extend(revokes);
                idx = idx.saturating_add(1);
            }
            JBD2_BLOCKTYPE_COMMIT => {
                stats.commit_blocks = stats.commit_blocks.saturating_add(1);
                committed_sequences.insert(header.sequence);
                idx = idx.saturating_add(1);
            }
            _ => {
                idx = idx.saturating_add(1);
            }
        }
    }

    let mut max_revoke_seq: BTreeMap<BlockNumber, u32> = BTreeMap::new();
    for &seq in &committed_sequences {
        if let Some(txn) = pending.get(&seq) {
            for &revoked_block in &txn.revoked {
                max_revoke_seq.insert(revoked_block, seq);
            }
        }
    }

    let mut final_writes: BTreeMap<BlockNumber, Vec<u8>> = BTreeMap::new();

    for &seq in &committed_sequences {
        if let Some(txn) = pending.get(&seq) {
            for (target, payload) in &txn.writes {
                let revoked_seq = max_revoke_seq.get(target).copied().unwrap_or(0);
                if seq <= revoked_seq {
                    stats.skipped_revoked_blocks = stats.skipped_revoked_blocks.saturating_add(1);
                    continue;
                }

                final_writes.insert(*target, payload.clone());
                stats.replayed_blocks = stats.replayed_blocks.saturating_add(1);
            }
        } else {
            stats.orphaned_commit_blocks = stats.orphaned_commit_blocks.saturating_add(1);
        }
    }

    for (target, payload) in final_writes {
        dev.write_block(cx, target, &payload)?;
    }

    stats.incomplete_transactions =
        u64::try_from(pending.len().saturating_sub(committed_sequences.len())).unwrap_or(u64::MAX);

    Ok(ReplayOutcome {
        committed_sequences: committed_sequences.into_iter().collect(),
        stats,
    })
}

/// Statistics from a journal write-back application.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ApplyStats {
    /// Number of blocks written to the device.
    pub blocks_written: u64,
    /// Number of blocks verified after write-back.
    pub blocks_verified: u64,
    /// Number of verification mismatches (should be 0 for correct operation).
    pub verify_mismatches: u64,
}

/// Apply the results of a JBD2 replay to the device by re-playing committed
/// transactions.
///
/// This is a separate "apply" step useful when replay was performed in
/// simulation mode. For normal replay the blocks are already written during
/// [`replay_jbd2`].
///
/// Each committed transaction's writes are written to their target locations
/// on the device, skipping revoked blocks. After writing, each block is
/// re-read and compared to detect silent corruption.
///
/// # Errors
///
/// Returns an error on I/O failure or if any verification read-back does
/// not match the written data.
pub fn apply_replay(
    cx: &Cx,
    dev: &dyn BlockDevice,
    committed: &[CommittedTxn],
) -> Result<ApplyStats> {
    let mut stats = ApplyStats::default();

    for (writes, revoked) in committed {
        for (target, payload) in writes {
            if revoked.contains(target) {
                continue;
            }
            dev.write_block(cx, *target, payload)?;
            stats.blocks_written = stats.blocks_written.saturating_add(1);

            // Verify: re-read the block and compare.
            let readback = dev.read_block(cx, *target)?;
            stats.blocks_verified = stats.blocks_verified.saturating_add(1);
            if readback.as_slice() != payload.as_slice() {
                return Err(FfsError::Corruption {
                    block: target.0,
                    detail: format!("JBD2 write-back verification failed for block {}", target.0),
                });
            }
        }
    }

    Ok(stats)
}

/// Clear a JBD2 journal region by zeroing all blocks.
///
/// This marks the journal as clean so a subsequent mount does not attempt
/// replay. The operation is safe to interrupt: an incomplete clear simply
/// means the next mount will replay again (idempotent).
///
/// # Errors
///
/// Returns an I/O error if any block write fails.
pub fn clear_journal(cx: &Cx, dev: &dyn BlockDevice, region: JournalRegion) -> Result<()> {
    let zero = vec![
        0_u8;
        usize::try_from(dev.block_size())
            .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?
    ];

    for idx in 0..region.blocks {
        let block = resolve_region_block(region, idx)?;
        dev.write_block(cx, block, &zero)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// JBD2-compatible journal writer
// ---------------------------------------------------------------------------

/// Statistics from a JBD2 journal write operation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Jbd2WriteStats {
    pub descriptor_blocks: u64,
    pub data_blocks: u64,
    pub revoke_blocks: u64,
    pub commit_blocks: u64,
}

/// A pending JBD2 transaction being assembled before commit.
#[derive(Debug, Clone)]
pub struct Jbd2Transaction {
    sequence: u32,
    writes: Vec<(BlockNumber, Vec<u8>)>,
    revokes: Vec<BlockNumber>,
}

impl Jbd2Transaction {
    /// Add a block write to this transaction.
    ///
    /// `target` is the home (destination) block number; `payload` is the data.
    pub fn add_write(&mut self, target: BlockNumber, payload: Vec<u8>) {
        self.writes.push((target, payload));
    }

    /// Add a revoke entry — `target` will be skipped during replay even if a
    /// prior descriptor references it.
    pub fn add_revoke(&mut self, target: BlockNumber) {
        self.revokes.push(target);
    }

    /// The sequence number assigned to this transaction.
    #[must_use]
    pub fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Number of writes staged so far.
    #[must_use]
    pub fn write_count(&self) -> usize {
        self.writes.len()
    }

    /// Number of revoke entries staged so far.
    #[must_use]
    pub fn revoke_count(&self) -> usize {
        self.revokes.len()
    }
}

/// JBD2-compatible journal writer.
///
/// Produces descriptor + data + revoke + commit blocks in a journal region
/// that [`replay_jbd2`] can successfully replay. This is the write-side
/// counterpart needed for compatibility-mode ext4 writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Jbd2Writer {
    region: JournalRegion,
    /// Next free index within the journal region (region-relative).
    head: u64,
    /// Next sequence number for a new transaction.
    next_seq: u32,
}

impl Jbd2Writer {
    /// Create a writer for an empty journal region starting at `start_seq`.
    #[must_use]
    pub fn new(region: JournalRegion, start_seq: u32) -> Self {
        Self {
            region,
            head: 0,
            next_seq: start_seq,
        }
    }

    /// Open an existing journal region, scanning forward to discover the tail.
    ///
    /// Scans JBD2 header blocks to find the first free slot. The next sequence
    /// number is set to one past the highest sequence seen, or `start_seq` if
    /// the region is empty.
    pub fn open(
        cx: &Cx,
        dev: &dyn BlockDevice,
        region: JournalRegion,
        start_seq: u32,
    ) -> Result<Self> {
        let mut head = 0_u64;
        let mut max_seq = start_seq;

        while head < region.blocks {
            let block = resolve_region_block(region, head)?;
            let raw = dev.read_block(cx, block)?;

            let Some(header) = Jbd2Header::parse(raw.as_slice()) else {
                break;
            };
            if header.magic != JBD2_MAGIC {
                break;
            }

            if header.sequence >= max_seq {
                max_seq = header.sequence.saturating_add(1);
            }

            match header.block_type {
                JBD2_BLOCKTYPE_DESCRIPTOR => {
                    let tags = parse_descriptor_tags(raw.as_slice());
                    // Advance past descriptor + its data blocks.
                    head = head
                        .saturating_add(1)
                        .saturating_add(u64::try_from(tags.len()).unwrap_or(u64::MAX));
                }
                _ => {
                    head = head.saturating_add(1);
                }
            }
        }

        Ok(Self {
            region,
            head,
            next_seq: max_seq,
        })
    }

    /// Begin a new transaction, consuming the next sequence number.
    pub fn begin_transaction(&mut self) -> Jbd2Transaction {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        Jbd2Transaction {
            sequence: seq,
            writes: Vec::new(),
            revokes: Vec::new(),
        }
    }

    /// Current head position (region-relative block index).
    #[must_use]
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Next sequence number that will be assigned.
    #[must_use]
    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Number of free blocks remaining in the journal region.
    #[must_use]
    pub fn free_blocks(&self) -> u64 {
        self.region.blocks.saturating_sub(self.head)
    }

    /// Compute how many journal blocks a transaction with `writes` data blocks
    /// and `revokes` revoke entries will consume.
    #[must_use]
    pub fn blocks_needed(block_size: u32, writes: usize, revokes: usize) -> u64 {
        let bs = block_size as usize;
        let tags_per_desc = max_tags_per_descriptor(bs);
        let entries_per_revoke = max_revoke_entries(bs);

        let mut total = 0_u64;
        if writes > 0 {
            let desc_blocks = writes.div_ceil(tags_per_desc) as u64;
            total = total
                .saturating_add(desc_blocks)
                .saturating_add(writes as u64);
        }
        if revokes > 0 {
            total = total.saturating_add(revokes.div_ceil(entries_per_revoke) as u64);
        }
        // Commit block.
        total.saturating_add(1)
    }

    /// Commit a transaction to the journal region.
    ///
    /// Layout produced:
    /// 1. One or more descriptor blocks, each followed by its data blocks.
    /// 2. Zero or more revoke blocks.
    /// 3. One commit block.
    ///
    /// Returns `(sequence_number, write_stats)`.
    pub fn commit_transaction(
        &mut self,
        cx: &Cx,
        dev: &dyn BlockDevice,
        txn: &Jbd2Transaction,
    ) -> Result<(u32, Jbd2WriteStats)> {
        let bs = usize::try_from(dev.block_size())
            .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?;

        let needed = Self::blocks_needed(dev.block_size(), txn.writes.len(), txn.revokes.len());
        if needed > self.free_blocks() {
            return Err(FfsError::NoSpace);
        }

        let mut stats = Jbd2WriteStats::default();
        let seq = txn.sequence;

        // --- Phase 1: descriptor + data blocks ---
        let tags_per_desc = max_tags_per_descriptor(bs);
        let mut write_idx = 0;
        while write_idx < txn.writes.len() {
            let chunk_end = (write_idx + tags_per_desc).min(txn.writes.len());
            let chunk = &txn.writes[write_idx..chunk_end];

            let mut desc = vec![0_u8; bs];
            encode_jbd2_header(&mut desc, JBD2_BLOCKTYPE_DESCRIPTOR, seq);

            let mut off = JBD2_HEADER_SIZE;
            for (i, (target, _)) in chunk.iter().enumerate() {
                let target_u32 = u32::try_from(target.0).map_err(|_| {
                    FfsError::Format(format!("target block {} exceeds u32 range", target.0))
                })?;
                let is_last_in_desc = i == chunk.len() - 1;
                let flags = if is_last_in_desc {
                    JBD2_TAG_FLAG_LAST
                } else {
                    0
                };
                desc[off..off + 4].copy_from_slice(&target_u32.to_be_bytes());
                desc[off + 4..off + 8].copy_from_slice(&flags.to_be_bytes());
                off += JBD2_TAG_SIZE;
            }

            let desc_block = self.alloc_block()?;
            dev.write_block(cx, desc_block, &desc)?;
            stats.descriptor_blocks = stats.descriptor_blocks.saturating_add(1);

            for (_, payload) in chunk {
                let data_block = self.alloc_block()?;
                let mut padded = vec![0_u8; bs];
                let copy_len = payload.len().min(bs);
                padded[..copy_len].copy_from_slice(&payload[..copy_len]);
                dev.write_block(cx, data_block, &padded)?;
                stats.data_blocks = stats.data_blocks.saturating_add(1);
            }

            write_idx = chunk_end;
        }

        // --- Phase 2: revoke blocks ---
        if !txn.revokes.is_empty() {
            let entries_per_revoke = max_revoke_entries(bs);
            let mut revoke_idx = 0;

            while revoke_idx < txn.revokes.len() {
                let chunk_end = (revoke_idx + entries_per_revoke).min(txn.revokes.len());
                let chunk = &txn.revokes[revoke_idx..chunk_end];

                let mut revoke = vec![0_u8; bs];
                encode_jbd2_header(&mut revoke, JBD2_BLOCKTYPE_REVOKE, seq);

                let r_count = u32::try_from(JBD2_REVOKE_HEADER_SIZE + chunk.len() * 4)
                    .map_err(|_| FfsError::Format("revoke r_count overflow".to_owned()))?;
                revoke[12..16].copy_from_slice(&r_count.to_be_bytes());

                let mut off = JBD2_REVOKE_HEADER_SIZE;
                for target in chunk {
                    let target_u32 = u32::try_from(target.0).map_err(|_| {
                        FfsError::Format(format!("revoke target {} exceeds u32 range", target.0))
                    })?;
                    revoke[off..off + 4].copy_from_slice(&target_u32.to_be_bytes());
                    off += 4;
                }

                let rev_block = self.alloc_block()?;
                dev.write_block(cx, rev_block, &revoke)?;
                stats.revoke_blocks = stats.revoke_blocks.saturating_add(1);

                revoke_idx = chunk_end;
            }
        }

        // --- Phase 3: commit block ---
        let mut commit = vec![0_u8; bs];
        encode_jbd2_header(&mut commit, JBD2_BLOCKTYPE_COMMIT, seq);

        let commit_blk = self.alloc_block()?;
        dev.write_block(cx, commit_blk, &commit)?;
        stats.commit_blocks = stats.commit_blocks.saturating_add(1);

        tracing::trace!(
            target: "ffs::journal",
            seq,
            data_blocks = stats.data_blocks,
            revoke_blocks = stats.revoke_blocks,
            head = self.head,
            free = self.free_blocks(),
            "jbd2_writer_committed"
        );

        Ok((seq, stats))
    }

    /// Allocate the next journal block, advancing head.
    fn alloc_block(&mut self) -> Result<BlockNumber> {
        if self.head >= self.region.blocks {
            return Err(FfsError::NoSpace);
        }
        let block = resolve_region_block(self.region, self.head)?;
        self.head = self.head.saturating_add(1);
        Ok(block)
    }
}

fn encode_jbd2_header(buf: &mut [u8], block_type: u32, sequence: u32) {
    buf[0..4].copy_from_slice(&JBD2_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&block_type.to_be_bytes());
    buf[8..12].copy_from_slice(&sequence.to_be_bytes());
}

#[must_use]
fn max_tags_per_descriptor(block_size: usize) -> usize {
    (block_size.saturating_sub(JBD2_HEADER_SIZE)) / JBD2_TAG_SIZE
}

#[must_use]
fn max_revoke_entries(block_size: usize) -> usize {
    (block_size.saturating_sub(JBD2_REVOKE_HEADER_SIZE)) / 4
}

/// A single recovered native COW write operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CowWrite {
    pub block: BlockNumber,
    pub bytes: Vec<u8>,
}

/// A recovered committed native COW transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredCommit {
    pub commit_seq: CommitSeq,
    pub writes: Vec<CowWrite>,
}

/// Native COW append-only journal writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCowJournal {
    region: JournalRegion,
    next_slot: u64,
}

impl NativeCowJournal {
    /// Open a native COW journal and discover the first free slot.
    pub fn open(cx: &Cx, dev: &dyn BlockDevice, region: JournalRegion) -> Result<Self> {
        if region.is_empty() {
            return Err(FfsError::Format(
                "native COW journal region must be non-empty".to_owned(),
            ));
        }

        let mut next_slot = 0_u64;
        while next_slot < region.blocks {
            let block = resolve_region_block(region, next_slot)?;
            let raw = dev.read_block(cx, block)?;
            if is_all_zero(raw.as_slice()) {
                break;
            }
            if !has_cow_magic(raw.as_slice()) {
                break;
            }
            next_slot = next_slot.saturating_add(1);
        }

        Ok(Self { region, next_slot })
    }

    /// Append a write record for `commit_seq`.
    pub fn append_write(
        &mut self,
        cx: &Cx,
        dev: &dyn BlockDevice,
        commit_seq: CommitSeq,
        block: BlockNumber,
        payload: &[u8],
    ) -> Result<()> {
        let encoded = encode_cow_record(
            dev.block_size(),
            &CowRecord::Write {
                commit_seq,
                block,
                payload,
            },
        )?;
        self.write_next(cx, dev, &encoded)
    }

    /// Append a commit marker for `commit_seq`.
    pub fn append_commit(
        &mut self,
        cx: &Cx,
        dev: &dyn BlockDevice,
        commit_seq: CommitSeq,
    ) -> Result<()> {
        let encoded = encode_cow_record(dev.block_size(), &CowRecord::Commit { commit_seq })?;
        self.write_next(cx, dev, &encoded)
    }

    fn write_next(&mut self, cx: &Cx, dev: &dyn BlockDevice, encoded: &[u8]) -> Result<()> {
        if self.next_slot >= self.region.blocks {
            return Err(FfsError::NoSpace);
        }
        let block = resolve_region_block(self.region, self.next_slot)?;
        dev.write_block(cx, block, encoded)?;
        self.next_slot = self.next_slot.saturating_add(1);
        Ok(())
    }

    #[must_use]
    pub fn next_slot(&self) -> u64 {
        self.next_slot
    }
}

/// Recover committed native COW transactions from a journal region.
pub fn recover_native_cow(
    cx: &Cx,
    dev: &dyn BlockDevice,
    region: JournalRegion,
) -> Result<Vec<RecoveredCommit>> {
    if region.is_empty() {
        return Ok(Vec::new());
    }

    let mut pending: BTreeMap<u64, Vec<CowWrite>> = BTreeMap::new();
    let mut commit_order = Vec::new();

    let mut slot = 0_u64;
    while slot < region.blocks {
        let block = resolve_region_block(region, slot)?;
        let raw = dev.read_block(cx, block)?;
        let Some(record) = decode_cow_record(raw.as_slice())? else {
            break;
        };

        match record {
            DecodedCowRecord::Write {
                commit_seq,
                block,
                payload,
            } => {
                pending.entry(commit_seq).or_default().push(CowWrite {
                    block,
                    bytes: payload,
                });
            }
            DecodedCowRecord::Commit { commit_seq } => commit_order.push(commit_seq),
        }

        slot = slot.saturating_add(1);
    }

    let mut recovered = Vec::new();
    for seq in commit_order {
        let writes = pending.remove(&seq).unwrap_or_default();
        recovered.push(RecoveredCommit {
            commit_seq: CommitSeq(seq),
            writes,
        });
    }

    Ok(recovered)
}

/// Apply recovered native COW commits to a block device.
pub fn replay_native_cow(
    cx: &Cx,
    dev: &dyn BlockDevice,
    commits: &[RecoveredCommit],
) -> Result<()> {
    let block_size = usize::try_from(dev.block_size())
        .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?;
    for commit in commits {
        for write in &commit.writes {
            if write.bytes.len() > block_size {
                return Err(FfsError::Format(format!(
                    "recovered payload too large for target block: {} > {block_size}",
                    write.bytes.len()
                )));
            }
            let mut full = vec![0_u8; block_size];
            full[..write.bytes.len()].copy_from_slice(&write.bytes);
            dev.write_block(cx, write.block, &full)?;
        }
    }
    Ok(())
}

fn resolve_region_block(region: JournalRegion, index: u64) -> Result<BlockNumber> {
    region.resolve(index).ok_or_else(|| {
        FfsError::Format(format!(
            "journal block index {index} out of range (region size={})",
            region.blocks
        ))
    })
}

fn resolve_segment_block(
    segments: &[JournalSegment],
    index: u64,
    total_blocks: u64,
) -> Result<BlockNumber> {
    let mut remaining = index;
    for segment in segments {
        if remaining < segment.blocks {
            return segment.resolve(remaining).ok_or_else(|| {
                FfsError::Format(format!(
                    "journal segment offset {remaining} out of range for segment size={}",
                    segment.blocks
                ))
            });
        }
        remaining = remaining.saturating_sub(segment.blocks);
    }

    Err(FfsError::Format(format!(
        "journal block index {index} out of range (region size={total_blocks})",
    )))
}

#[must_use]
fn parse_descriptor_tags(block: &[u8]) -> Vec<DescriptorTag> {
    let mut tags = Vec::new();
    let mut offset = JBD2_HEADER_SIZE;

    while offset.saturating_add(JBD2_TAG_SIZE) <= block.len() {
        let Some(target) = read_be_u32(block, offset) else {
            break;
        };
        let Some(flags) = read_be_u32(block, offset.saturating_add(4)) else {
            break;
        };

        let tag = DescriptorTag {
            target: BlockNumber(u64::from(target)),
            flags,
        };
        tags.push(tag);
        offset = offset.saturating_add(JBD2_TAG_SIZE);

        if tag.is_last() {
            break;
        }
    }

    tags
}

#[must_use]
fn parse_revoke_entries(block: &[u8]) -> Vec<BlockNumber> {
    let mut out = Vec::new();
    let Some(r_count) = read_be_u32(block, 12) else {
        return out;
    };
    let limit = (r_count as usize).min(block.len());
    let mut offset = JBD2_REVOKE_HEADER_SIZE;

    while offset.saturating_add(4) <= limit {
        let Some(raw) = read_be_u32(block, offset) else {
            break;
        };
        out.push(BlockNumber(u64::from(raw)));
        offset = offset.saturating_add(4);
    }

    out
}

#[must_use]
fn read_be_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    let arr: [u8; 4] = slice.try_into().ok()?;
    Some(u32::from_be_bytes(arr))
}

#[must_use]
fn read_le_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    let end = offset.checked_add(2)?;
    let slice = bytes.get(offset..end)?;
    let arr: [u8; 2] = slice.try_into().ok()?;
    Some(u16::from_le_bytes(arr))
}

#[must_use]
fn read_le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    let arr: [u8; 4] = slice.try_into().ok()?;
    Some(u32::from_le_bytes(arr))
}

#[must_use]
fn read_le_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    let end = offset.checked_add(8)?;
    let slice = bytes.get(offset..end)?;
    let arr: [u8; 8] = slice.try_into().ok()?;
    Some(u64::from_le_bytes(arr))
}

#[must_use]
fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| *b == 0)
}

#[must_use]
fn has_cow_magic(bytes: &[u8]) -> bool {
    read_le_u32(bytes, 0) == Some(COW_MAGIC)
}

enum CowRecord<'a> {
    Write {
        commit_seq: CommitSeq,
        block: BlockNumber,
        payload: &'a [u8],
    },
    Commit {
        commit_seq: CommitSeq,
    },
}

enum DecodedCowRecord {
    Write {
        commit_seq: u64,
        block: BlockNumber,
        payload: Vec<u8>,
    },
    Commit {
        commit_seq: u64,
    },
}

fn encode_cow_record(block_size: u32, record: &CowRecord<'_>) -> Result<Vec<u8>> {
    let block_size = usize::try_from(block_size)
        .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?;
    if block_size < COW_HEADER_SIZE {
        return Err(FfsError::Format(
            "block size too small for COW journal record".to_owned(),
        ));
    }

    let mut out = vec![0_u8; block_size];
    out[0..4].copy_from_slice(&COW_MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&COW_VERSION.to_le_bytes());

    match record {
        CowRecord::Write {
            commit_seq,
            block,
            payload,
        } => {
            let payload_capacity = block_size.saturating_sub(COW_HEADER_SIZE);
            if payload.len() > payload_capacity {
                return Err(FfsError::Format(format!(
                    "COW payload too large: {} bytes (capacity {payload_capacity})",
                    payload.len()
                )));
            }
            let payload_len = u32::try_from(payload.len())
                .map_err(|_| FfsError::Format("COW payload length does not fit u32".to_owned()))?;
            out[6..8].copy_from_slice(&COW_RECORD_WRITE.to_le_bytes());
            out[8..16].copy_from_slice(&commit_seq.0.to_le_bytes());
            out[16..24].copy_from_slice(&block.0.to_le_bytes());
            out[24..28].copy_from_slice(&payload_len.to_le_bytes());
            let checksum = crc32c::crc32c(payload);
            out[28..32].copy_from_slice(&checksum.to_le_bytes());
            let payload_end = COW_HEADER_SIZE.saturating_add(payload.len());
            out[COW_HEADER_SIZE..payload_end].copy_from_slice(payload);
        }
        CowRecord::Commit { commit_seq } => {
            out[6..8].copy_from_slice(&COW_RECORD_COMMIT.to_le_bytes());
            out[8..16].copy_from_slice(&commit_seq.0.to_le_bytes());
            out[16..24].copy_from_slice(&0_u64.to_le_bytes());
            out[24..28].copy_from_slice(&0_u32.to_le_bytes());
            out[28..32].copy_from_slice(&0_u32.to_le_bytes());
        }
    }

    Ok(out)
}

fn decode_cow_record(block: &[u8]) -> Result<Option<DecodedCowRecord>> {
    if is_all_zero(block) {
        return Ok(None);
    }

    let Some(magic) = read_le_u32(block, 0) else {
        return Ok(None);
    };
    if magic != COW_MAGIC {
        return Ok(None);
    }

    let version = read_le_u16(block, 4)
        .ok_or_else(|| FfsError::Format("truncated COW record version".to_owned()))?;
    if version != COW_VERSION {
        return Err(FfsError::Format(format!(
            "unsupported COW journal version: {version}"
        )));
    }

    let kind = read_le_u16(block, 6)
        .ok_or_else(|| FfsError::Format("truncated COW record kind".to_owned()))?;
    let commit_seq = read_le_u64(block, 8)
        .ok_or_else(|| FfsError::Format("truncated COW record commit_seq".to_owned()))?;
    let target_block = read_le_u64(block, 16)
        .ok_or_else(|| FfsError::Format("truncated COW record block".to_owned()))?;
    let payload_len = read_le_u32(block, 24)
        .ok_or_else(|| FfsError::Format("truncated COW record payload length".to_owned()))?;
    let payload_crc = read_le_u32(block, 28)
        .ok_or_else(|| FfsError::Format("truncated COW record checksum".to_owned()))?;

    match kind {
        COW_RECORD_WRITE => {
            let payload_len = usize::try_from(payload_len).map_err(|_| {
                FfsError::Format("COW payload length does not fit usize".to_owned())
            })?;
            let payload_end = COW_HEADER_SIZE
                .checked_add(payload_len)
                .ok_or_else(|| FfsError::Format("COW payload length overflow".to_owned()))?;
            if payload_end > block.len() {
                return Err(FfsError::Format(
                    "COW payload exceeds block boundary".to_owned(),
                ));
            }
            let payload = block[COW_HEADER_SIZE..payload_end].to_vec();
            let computed = crc32c::crc32c(&payload);
            if computed != payload_crc {
                return Err(FfsError::Corruption {
                    block: target_block,
                    detail: format!(
                        "COW payload CRC mismatch: expected {payload_crc:#010x}, got {computed:#010x}"
                    ),
                });
            }
            Ok(Some(DecodedCowRecord::Write {
                commit_seq,
                block: BlockNumber(target_block),
                payload,
            }))
        }
        COW_RECORD_COMMIT => Ok(Some(DecodedCowRecord::Commit { commit_seq })),
        other => Err(FfsError::Format(format!(
            "unknown COW record kind: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::collections::HashMap;

    #[derive(Debug)]
    struct MemBlockDevice {
        blocks: RwLock<HashMap<BlockNumber, Vec<u8>>>,
        block_size: u32,
        block_count: u64,
    }

    impl MemBlockDevice {
        fn new(block_size: u32, block_count: u64) -> Self {
            Self {
                blocks: RwLock::new(HashMap::new()),
                block_size,
                block_count,
            }
        }

        fn raw_write(&self, block: BlockNumber, data: Vec<u8>) {
            self.blocks.write().insert(block, data);
        }
    }

    impl BlockDevice for MemBlockDevice {
        fn read_block(&self, _cx: &Cx, block: BlockNumber) -> Result<ffs_block::BlockBuf> {
            if block.0 >= self.block_count {
                return Err(FfsError::Format(format!(
                    "block out of range: {} >= {}",
                    block.0, self.block_count
                )));
            }
            let bs = usize::try_from(self.block_size)
                .map_err(|_| FfsError::Format("block_size overflow".to_owned()))?;
            let data = self
                .blocks
                .read()
                .get(&block)
                .cloned()
                .unwrap_or_else(|| vec![0_u8; bs]);
            Ok(ffs_block::BlockBuf::new(data))
        }

        fn write_block(&self, _cx: &Cx, block: BlockNumber, data: &[u8]) -> Result<()> {
            if block.0 >= self.block_count {
                return Err(FfsError::Format(format!(
                    "block out of range: {} >= {}",
                    block.0, self.block_count
                )));
            }
            let expected = usize::try_from(self.block_size)
                .map_err(|_| FfsError::Format("block_size overflow".to_owned()))?;
            if data.len() != expected {
                return Err(FfsError::Format(format!(
                    "size mismatch: got {} expected {expected}",
                    data.len()
                )));
            }
            self.blocks.write().insert(block, data.to_vec());
            Ok(())
        }

        fn block_size(&self) -> u32 {
            self.block_size
        }

        fn block_count(&self) -> u64 {
            self.block_count
        }

        fn sync(&self, _cx: &Cx) -> Result<()> {
            Ok(())
        }
    }

    fn test_cx() -> Cx {
        Cx::for_testing()
    }

    fn jbd2_header(kind: u32, seq: u32) -> [u8; JBD2_HEADER_SIZE] {
        let mut h = [0_u8; JBD2_HEADER_SIZE];
        h[0..4].copy_from_slice(&JBD2_MAGIC.to_be_bytes());
        h[4..8].copy_from_slice(&kind.to_be_bytes());
        h[8..12].copy_from_slice(&seq.to_be_bytes());
        h
    }

    fn descriptor_block(block_size: usize, seq: u32, tags: &[(u32, u32)]) -> Vec<u8> {
        let mut out = vec![0_u8; block_size];
        out[0..JBD2_HEADER_SIZE].copy_from_slice(&jbd2_header(JBD2_BLOCKTYPE_DESCRIPTOR, seq));
        let mut off = JBD2_HEADER_SIZE;
        for (target, flags) in tags {
            out[off..off + 4].copy_from_slice(&target.to_be_bytes());
            out[off + 4..off + 8].copy_from_slice(&flags.to_be_bytes());
            off += JBD2_TAG_SIZE;
        }
        out
    }

    fn revoke_block(block_size: usize, seq: u32, targets: &[u32]) -> Vec<u8> {
        let mut out = vec![0_u8; block_size];
        out[0..JBD2_HEADER_SIZE].copy_from_slice(&jbd2_header(JBD2_BLOCKTYPE_REVOKE, seq));
        // r_count at offset 12: total bytes in the revoke record including header.
        // Header is 16 bytes, each 32-bit entry is 4 bytes.
        let r_count = u32::try_from(JBD2_REVOKE_HEADER_SIZE + targets.len() * 4)
            .expect("r_count should fit in u32");
        out[12..16].copy_from_slice(&r_count.to_be_bytes());
        // Revoke entries start at offset 16.
        let mut off = JBD2_REVOKE_HEADER_SIZE;
        for target in targets {
            out[off..off + 4].copy_from_slice(&target.to_be_bytes());
            off += 4;
        }
        out
    }

    fn commit_block(block_size: usize, seq: u32) -> Vec<u8> {
        let mut out = vec![0_u8; block_size];
        out[0..JBD2_HEADER_SIZE].copy_from_slice(&jbd2_header(JBD2_BLOCKTYPE_COMMIT, seq));
        out
    }

    #[test]
    fn replay_jbd2_committed_descriptor_replays_payload() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        let descriptor = descriptor_block(512, 11, &[(3, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xA5; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 11));

        let out = replay_jbd2(&cx, &dev, region).expect("replay should succeed");

        let target = dev
            .read_block(&cx, BlockNumber(3))
            .expect("target block should be readable");
        assert_eq!(target.as_slice(), &[0xA5; 512]);
        assert_eq!(out.committed_sequences, vec![11]);
        assert_eq!(out.stats.replayed_blocks, 1);
        assert_eq!(out.stats.skipped_revoked_blocks, 0);
    }

    #[test]
    fn replay_jbd2_revoke_skips_target_block() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(20),
            blocks: 8,
        };

        let descriptor = descriptor_block(512, 7, &[(5, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(20), descriptor);
        dev.raw_write(BlockNumber(21), vec![0xCC; 512]);
        dev.raw_write(BlockNumber(22), revoke_block(512, 7, &[5]));
        dev.raw_write(BlockNumber(23), commit_block(512, 7));

        let out = replay_jbd2(&cx, &dev, region).expect("replay should succeed");

        let target = dev
            .read_block(&cx, BlockNumber(5))
            .expect("target block should be readable");
        assert_eq!(target.as_slice(), &[0_u8; 512]);
        assert_eq!(out.stats.replayed_blocks, 0);
        assert_eq!(out.stats.skipped_revoked_blocks, 1);
        assert_eq!(out.committed_sequences, vec![7]);
    }

    #[test]
    fn replay_jbd2_uncommitted_transaction_is_ignored() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(30),
            blocks: 4,
        };

        let descriptor = descriptor_block(512, 3, &[(9, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(30), descriptor);
        dev.raw_write(BlockNumber(31), vec![0x77; 512]);
        // Missing commit block.

        let out = replay_jbd2(&cx, &dev, region).expect("replay should succeed");

        let target = dev
            .read_block(&cx, BlockNumber(9))
            .expect("target block should be readable");
        assert_eq!(target.as_slice(), &[0_u8; 512]);
        assert!(out.committed_sequences.is_empty());
        assert_eq!(out.stats.incomplete_transactions, 1);
    }

    #[test]
    fn native_cow_recovery_only_returns_committed_sequences() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(40),
            blocks: 16,
        };

        let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open journal");
        journal
            .append_write(&cx, &dev, CommitSeq(1), BlockNumber(2), &[0x11; 64])
            .expect("append write 1");
        journal
            .append_commit(&cx, &dev, CommitSeq(1))
            .expect("append commit 1");
        journal
            .append_write(&cx, &dev, CommitSeq(2), BlockNumber(3), &[0x22; 64])
            .expect("append write 2");
        // No commit for sequence 2.

        let recovered = recover_native_cow(&cx, &dev, region).expect("recover");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].commit_seq, CommitSeq(1));
        assert_eq!(recovered[0].writes.len(), 1);
        assert_eq!(recovered[0].writes[0].block, BlockNumber(2));
    }

    #[test]
    fn native_cow_replay_applies_recovered_writes() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(50),
            blocks: 16,
        };

        let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open journal");
        journal
            .append_write(&cx, &dev, CommitSeq(9), BlockNumber(6), &[0xAB; 128])
            .expect("append write");
        journal
            .append_commit(&cx, &dev, CommitSeq(9))
            .expect("append commit");

        let recovered = recover_native_cow(&cx, &dev, region).expect("recover");
        replay_native_cow(&cx, &dev, &recovered).expect("replay");

        let target = dev
            .read_block(&cx, BlockNumber(6))
            .expect("read target block");
        assert_eq!(&target.as_slice()[..128], &[0xAB; 128]);
    }

    #[test]
    fn native_cow_open_discovers_tail_after_existing_records() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(60),
            blocks: 16,
        };

        {
            let mut first = NativeCowJournal::open(&cx, &dev, region).expect("open first");
            first
                .append_write(&cx, &dev, CommitSeq(1), BlockNumber(1), &[0x01; 32])
                .expect("append write");
            first
                .append_commit(&cx, &dev, CommitSeq(1))
                .expect("append commit");
            assert_eq!(first.next_slot(), 2);
        }

        let second = NativeCowJournal::open(&cx, &dev, region).expect("open second");
        assert_eq!(second.next_slot(), 2);
    }

    #[test]
    fn clear_journal_zeros_region() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 4,
        };

        // Write journal data.
        let descriptor = descriptor_block(512, 1, &[(3, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xA5; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 1));

        // Replay first (to apply writes).
        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.stats.replayed_blocks, 1);

        // Clear the journal.
        clear_journal(&cx, &dev, region).expect("clear");

        // Verify all journal blocks are zeroed.
        for idx in 0..4 {
            let block = dev.read_block(&cx, BlockNumber(10 + idx)).expect("read");
            assert!(
                block.as_slice().iter().all(|b| *b == 0),
                "journal block {idx} not zeroed"
            );
        }

        // Replaying the cleared journal should produce no committed sequences.
        let second = replay_jbd2(&cx, &dev, region).expect("replay cleared journal");
        assert!(second.committed_sequences.is_empty());
        assert_eq!(second.stats.replayed_blocks, 0);
    }

    #[test]
    fn replay_jbd2_is_idempotent() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        let descriptor = descriptor_block(512, 5, &[(3, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 5));

        // First replay.
        let first = replay_jbd2(&cx, &dev, region).expect("first replay");
        assert_eq!(first.committed_sequences, vec![5]);
        let target_after_first = dev
            .read_block(&cx, BlockNumber(3))
            .expect("read")
            .as_slice()
            .to_vec();

        // Second replay (should produce identical result).
        let second = replay_jbd2(&cx, &dev, region).expect("second replay");
        assert_eq!(second.committed_sequences, vec![5]);
        let target_after_second = dev
            .read_block(&cx, BlockNumber(3))
            .expect("read")
            .as_slice()
            .to_vec();

        assert_eq!(target_after_first, target_after_second);
        assert_eq!(target_after_first, vec![0xBB; 512]);
    }

    #[test]
    fn apply_replay_writes_and_verifies() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);

        let writes = vec![(BlockNumber(5), vec![0xAA; 512])];
        let revoked = BTreeSet::new();
        let committed = vec![(writes, revoked)];

        let stats = apply_replay(&cx, &dev, &committed).expect("apply");
        assert_eq!(stats.blocks_written, 1);
        assert_eq!(stats.blocks_verified, 1);
        assert_eq!(stats.verify_mismatches, 0);

        let block = dev.read_block(&cx, BlockNumber(5)).expect("read");
        assert_eq!(block.as_slice(), &[0xAA; 512]);
    }

    #[test]
    fn apply_replay_respects_revoke_list() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);

        let writes = vec![
            (BlockNumber(5), vec![0xAA; 512]),
            (BlockNumber(6), vec![0xBB; 512]),
        ];
        let mut revoked = BTreeSet::new();
        revoked.insert(BlockNumber(6)); // Revoke block 6.
        let committed = vec![(writes, revoked)];

        let stats = apply_replay(&cx, &dev, &committed).expect("apply");
        assert_eq!(stats.blocks_written, 1); // Only block 5 written.

        let block5 = dev.read_block(&cx, BlockNumber(5)).expect("read 5");
        assert_eq!(block5.as_slice(), &[0xAA; 512]);

        let block6 = dev.read_block(&cx, BlockNumber(6)).expect("read 6");
        assert_eq!(block6.as_slice(), &[0_u8; 512]); // Block 6 not written.
    }

    // -----------------------------------------------------------------------
    // JBD2 Writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn jbd2_writer_single_write_replays_correctly() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(5), vec![0xAB; 512]);
        let (seq, stats) = writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        assert_eq!(seq, 1);
        assert_eq!(stats.descriptor_blocks, 1);
        assert_eq!(stats.data_blocks, 1);
        assert_eq!(stats.commit_blocks, 1);
        assert_eq!(stats.revoke_blocks, 0);

        // Replay and verify the target block.
        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.committed_sequences, vec![1]);
        assert_eq!(outcome.stats.replayed_blocks, 1);

        let target = dev.read_block(&cx, BlockNumber(5)).expect("read target");
        assert_eq!(target.as_slice(), &[0xAB; 512]);
    }

    #[test]
    fn jbd2_writer_multi_write_replays_all() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 10);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(2), vec![0x11; 512]);
        txn.add_write(BlockNumber(3), vec![0x22; 512]);
        txn.add_write(BlockNumber(4), vec![0x33; 512]);
        let (seq, stats) = writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        assert_eq!(seq, 10);
        assert_eq!(stats.descriptor_blocks, 1);
        assert_eq!(stats.data_blocks, 3);

        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.committed_sequences, vec![10]);
        assert_eq!(outcome.stats.replayed_blocks, 3);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(2)).unwrap().as_slice(),
            &[0x11; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0x22; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(4)).unwrap().as_slice(),
            &[0x33; 512]
        );
    }

    #[test]
    fn jbd2_writer_revoke_prevents_replay() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(7), vec![0xDD; 512]);
        txn.add_write(BlockNumber(8), vec![0xEE; 512]);
        txn.add_revoke(BlockNumber(7)); // Revoke block 7.
        let (_, stats) = writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        assert_eq!(stats.revoke_blocks, 1);

        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.committed_sequences, vec![1]);
        assert_eq!(outcome.stats.replayed_blocks, 1);
        assert_eq!(outcome.stats.skipped_revoked_blocks, 1);

        // Block 7 was revoked — should remain zeroed.
        assert_eq!(
            dev.read_block(&cx, BlockNumber(7)).unwrap().as_slice(),
            &[0_u8; 512]
        );
        // Block 8 was not revoked — should contain payload.
        assert_eq!(
            dev.read_block(&cx, BlockNumber(8)).unwrap().as_slice(),
            &[0xEE; 512]
        );
    }

    #[test]
    fn jbd2_writer_incomplete_txn_ignored_on_replay() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        // Simulate a crash: write descriptor + data but NO commit.
        let bs = 512_usize;
        let mut desc = vec![0_u8; bs];
        encode_jbd2_header(&mut desc, JBD2_BLOCKTYPE_DESCRIPTOR, 1);
        let target_u32 = 9_u32;
        desc[JBD2_HEADER_SIZE..JBD2_HEADER_SIZE + 4].copy_from_slice(&target_u32.to_be_bytes());
        desc[JBD2_HEADER_SIZE + 4..JBD2_HEADER_SIZE + 8]
            .copy_from_slice(&JBD2_TAG_FLAG_LAST.to_be_bytes());
        dev.raw_write(BlockNumber(100), desc);
        dev.raw_write(BlockNumber(101), vec![0xFF; 512]);
        // No commit block written — simulates crash.

        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert!(outcome.committed_sequences.is_empty());
        assert_eq!(outcome.stats.incomplete_transactions, 1);

        // Target block should remain untouched.
        assert_eq!(
            dev.read_block(&cx, BlockNumber(9)).unwrap().as_slice(),
            &[0_u8; 512]
        );
    }

    #[test]
    fn jbd2_writer_journal_full_returns_no_space() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        // Region with only 2 blocks: needs desc(1) + data(1) + commit(1) = 3.
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 2,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(5), vec![0xAA; 512]);

        let result = writer.commit_transaction(&cx, &dev, &txn);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::NoSpace => {}
            other => panic!("expected NoSpace, got: {other:?}"),
        }
    }

    #[test]
    fn jbd2_writer_multi_transaction_sequence() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 64,
        };

        let mut writer = Jbd2Writer::new(region, 1);

        // Transaction 1: write block 2.
        let mut txn1 = writer.begin_transaction();
        txn1.add_write(BlockNumber(2), vec![0x11; 512]);
        let (seq1, _) = writer
            .commit_transaction(&cx, &dev, &txn1)
            .expect("commit 1");
        assert_eq!(seq1, 1);

        // Transaction 2: write block 3.
        let mut txn2 = writer.begin_transaction();
        txn2.add_write(BlockNumber(3), vec![0x22; 512]);
        let (seq2, _) = writer
            .commit_transaction(&cx, &dev, &txn2)
            .expect("commit 2");
        assert_eq!(seq2, 2);

        // Transaction 3: write block 4.
        let mut txn3 = writer.begin_transaction();
        txn3.add_write(BlockNumber(4), vec![0x33; 512]);
        let (seq3, _) = writer
            .commit_transaction(&cx, &dev, &txn3)
            .expect("commit 3");
        assert_eq!(seq3, 3);

        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.committed_sequences, vec![1, 2, 3]);
        assert_eq!(outcome.stats.replayed_blocks, 3);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(2)).unwrap().as_slice(),
            &[0x11; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0x22; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(4)).unwrap().as_slice(),
            &[0x33; 512]
        );
    }

    #[test]
    fn jbd2_writer_open_discovers_existing_head() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        // Write a transaction.
        {
            let mut writer = Jbd2Writer::new(region, 1);
            let mut txn = writer.begin_transaction();
            txn.add_write(BlockNumber(5), vec![0xAA; 512]);
            writer.commit_transaction(&cx, &dev, &txn).expect("commit");
            assert_eq!(writer.head(), 3); // descriptor + data + commit
            assert_eq!(writer.next_seq(), 2);
        }

        // Re-open and verify head/seq discovery.
        let reopened = Jbd2Writer::open(&cx, &dev, region, 1).expect("open");
        assert_eq!(reopened.head(), 3);
        assert_eq!(reopened.next_seq(), 2);
    }

    #[test]
    fn jbd2_writer_descriptor_tag_encoding() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 42);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(7), vec![0x00; 512]);
        txn.add_write(BlockNumber(13), vec![0x00; 512]);
        writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        // Read back the descriptor block and verify tag encoding.
        let desc_raw = dev
            .read_block(&cx, BlockNumber(100))
            .expect("read descriptor");
        let header = Jbd2Header::parse(desc_raw.as_slice()).expect("parse header");
        assert_eq!(header.magic, JBD2_MAGIC);
        assert_eq!(header.block_type, JBD2_BLOCKTYPE_DESCRIPTOR);
        assert_eq!(header.sequence, 42);

        let tags = parse_descriptor_tags(desc_raw.as_slice());
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].target, BlockNumber(7));
        assert_eq!(tags[0].flags & JBD2_TAG_FLAG_LAST, 0); // Not last.
        assert_eq!(tags[1].target, BlockNumber(13));
        assert_ne!(tags[1].flags & JBD2_TAG_FLAG_LAST, 0); // Last.
    }

    #[test]
    fn jbd2_writer_blocks_needed_calculation() {
        // 512-byte blocks: (512-12)/8 = 62 tags per desc, (512-16)/4 = 124 entries per revoke.
        assert_eq!(Jbd2Writer::blocks_needed(512, 0, 0), 1); // Just commit.
        assert_eq!(Jbd2Writer::blocks_needed(512, 1, 0), 3); // desc + data + commit.
        assert_eq!(Jbd2Writer::blocks_needed(512, 3, 0), 5); // desc + 3 data + commit.
        assert_eq!(Jbd2Writer::blocks_needed(512, 0, 1), 2); // revoke + commit.
        assert_eq!(Jbd2Writer::blocks_needed(512, 1, 1), 4); // desc + data + revoke + commit.
        // 62 writes: fits in one descriptor.
        assert_eq!(Jbd2Writer::blocks_needed(512, 62, 0), 64); // 1 desc + 62 data + commit.
        // 63 writes: needs two descriptors.
        assert_eq!(Jbd2Writer::blocks_needed(512, 63, 0), 66); // 2 desc + 63 data + commit.
    }

    #[test]
    fn jbd2_writer_payload_padding() {
        // Payloads shorter than block_size should be zero-padded.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(5), vec![0xCC; 128]); // Only 128 bytes.
        writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        let outcome = replay_jbd2(&cx, &dev, region).expect("replay");
        assert_eq!(outcome.stats.replayed_blocks, 1);

        let target = dev.read_block(&cx, BlockNumber(5)).expect("read");
        assert_eq!(&target.as_slice()[..128], &[0xCC; 128]);
        assert_eq!(&target.as_slice()[128..], &[0_u8; 384]);
    }

    // ── bd-1xe.5: ext4 read path journal replay tests ───────────────────

    // Journal Replay Test 1: Replay empty journal — no-op
    #[test]
    fn readpath_replay_empty_journal_noop() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        // Journal region contains all zeros (no valid JBD2 headers).
        let out = replay_jbd2(&cx, &dev, region).expect("replay empty journal should succeed");

        assert!(
            out.committed_sequences.is_empty(),
            "empty journal should have no committed sequences"
        );
        assert_eq!(out.stats.replayed_blocks, 0);
        assert_eq!(out.stats.descriptor_blocks, 0);
        assert_eq!(out.stats.commit_blocks, 0);
    }

    // Journal Replay Test 2: Replay single committed transaction — blocks written back
    #[test]
    fn readpath_replay_single_committed_transaction() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        // Build a single committed transaction: descriptor + payload + commit
        let descriptor = descriptor_block(512, 42, &[(7, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xDE; 512]); // payload for target block 7
        dev.raw_write(BlockNumber(12), commit_block(512, 42));

        // Verify target block is initially zeros.
        let before = dev.read_block(&cx, BlockNumber(7)).expect("read before");
        assert_eq!(
            before.as_slice(),
            &[0_u8; 512],
            "target should be zero before replay"
        );

        let out = replay_jbd2(&cx, &dev, region).expect("replay");

        // Verify target block now contains the journal payload.
        let after = dev.read_block(&cx, BlockNumber(7)).expect("read after");
        assert_eq!(
            after.as_slice(),
            &[0xDE; 512],
            "target should have journal payload"
        );
        assert_eq!(out.committed_sequences, vec![42]);
        assert_eq!(out.stats.replayed_blocks, 1);
    }

    // Journal Replay Test 3: Replay aborted transaction — blocks discarded
    #[test]
    fn readpath_replay_aborted_transaction_discarded() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        // Descriptor + payload but NO commit block → incomplete/aborted transaction.
        let descriptor = descriptor_block(512, 99, &[(4, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xFF; 512]);
        // No commit block at BlockNumber(12).

        let out = replay_jbd2(&cx, &dev, region).expect("replay");

        // Target block should remain untouched.
        let target = dev.read_block(&cx, BlockNumber(4)).expect("read");
        assert_eq!(
            target.as_slice(),
            &[0_u8; 512],
            "aborted txn should not modify target"
        );
        assert!(out.committed_sequences.is_empty());
        assert_eq!(out.stats.incomplete_transactions, 1);
        assert_eq!(out.stats.replayed_blocks, 0);
    }

    // Journal Replay Test 4: Replay with torn write — partial transaction discarded
    #[test]
    fn readpath_replay_torn_write_partial_discarded() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // First transaction: complete (descriptor + 2 payloads + commit).
        let desc1 = descriptor_block(512, 10, &[(3, 0), (4, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), desc1);
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]); // payload for block 3
        dev.raw_write(BlockNumber(12), vec![0xBB; 512]); // payload for block 4
        dev.raw_write(BlockNumber(13), commit_block(512, 10));

        // Second transaction: torn (descriptor + payload, no commit).
        let desc2 = descriptor_block(512, 11, &[(5, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(14), desc2);
        dev.raw_write(BlockNumber(15), vec![0xCC; 512]);
        // Missing commit for seq=11.

        let out = replay_jbd2(&cx, &dev, region).expect("replay");

        // First transaction's blocks should be replayed.
        let b3 = dev.read_block(&cx, BlockNumber(3)).expect("read block 3");
        assert_eq!(
            b3.as_slice(),
            &[0xAA; 512],
            "complete txn block 3 should be written"
        );
        let b4 = dev.read_block(&cx, BlockNumber(4)).expect("read block 4");
        assert_eq!(
            b4.as_slice(),
            &[0xBB; 512],
            "complete txn block 4 should be written"
        );

        // Second transaction's block should NOT be replayed.
        let b5 = dev.read_block(&cx, BlockNumber(5)).expect("read block 5");
        assert_eq!(
            b5.as_slice(),
            &[0_u8; 512],
            "torn txn block 5 should be untouched"
        );

        assert_eq!(out.committed_sequences, vec![10]);
        assert_eq!(out.stats.replayed_blocks, 2);
        assert_eq!(out.stats.incomplete_transactions, 1);
    }

    // Journal Replay Test 5: Replay ordering — transactions applied in sequence order
    #[test]
    fn readpath_replay_ordering_sequence_order() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // Two committed transactions writing to the SAME target block.
        // Seq 20 writes 0xAA, Seq 21 writes 0xBB.
        // After replay, target should contain 0xBB (later sequence wins).
        let desc1 = descriptor_block(512, 20, &[(6, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), desc1);
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 20));

        let desc2 = descriptor_block(512, 21, &[(6, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(13), desc2);
        dev.raw_write(BlockNumber(14), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(15), commit_block(512, 21));

        let out = replay_jbd2(&cx, &dev, region).expect("replay");

        let target = dev.read_block(&cx, BlockNumber(6)).expect("read target");
        assert_eq!(
            target.as_slice(),
            &[0xBB; 512],
            "later sequence (21) should overwrite earlier sequence (20)"
        );
        assert_eq!(out.committed_sequences, vec![20, 21]);
        assert_eq!(out.stats.replayed_blocks, 2);
    }

    #[test]
    fn replay_jbd2_segments_handles_non_contiguous_layout() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let segments = [
            JournalSegment {
                start: BlockNumber(10),
                blocks: 2,
            },
            JournalSegment {
                start: BlockNumber(20),
                blocks: 2,
            },
        ];

        // Descriptor in first segment index 0, payload in first segment index 1.
        let descriptor = descriptor_block(512, 7, &[(6, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), descriptor);
        dev.raw_write(BlockNumber(11), vec![0xA6; 512]);
        // Commit in second segment index 0 (global journal index 2).
        dev.raw_write(BlockNumber(20), commit_block(512, 7));

        let out = replay_jbd2_segments(&cx, &dev, &segments).expect("replay");
        let target = dev.read_block(&cx, BlockNumber(6)).expect("read target");
        assert_eq!(target.as_slice(), &[0xA6; 512]);
        assert_eq!(out.committed_sequences, vec![7]);
        assert_eq!(out.stats.replayed_blocks, 1);
    }

    // ── Edge-case unit tests ──────────────────────────────────────────────

    #[test]
    fn journal_segment_resolve_in_range() {
        let segment = JournalSegment {
            start: BlockNumber(50),
            blocks: 4,
        };
        assert_eq!(segment.resolve(0), Some(BlockNumber(50)));
        assert_eq!(segment.resolve(3), Some(BlockNumber(53)));
        assert_eq!(segment.resolve(4), None);
    }

    #[test]
    fn journal_region_resolve_in_range() {
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 10,
        };
        assert_eq!(region.resolve(0), Some(BlockNumber(100)));
        assert_eq!(region.resolve(9), Some(BlockNumber(109)));
    }

    #[test]
    fn journal_region_resolve_out_of_range() {
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 10,
        };
        assert_eq!(region.resolve(10), None);
        assert_eq!(region.resolve(u64::MAX), None);
    }

    #[test]
    fn journal_region_is_empty() {
        let empty = JournalRegion {
            start: BlockNumber(0),
            blocks: 0,
        };
        assert!(empty.is_empty());

        let non_empty = JournalRegion {
            start: BlockNumber(0),
            blocks: 1,
        };
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn replay_jbd2_empty_region_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(0),
            blocks: 0,
        };

        let err = replay_jbd2(&cx, &dev, region).expect_err("empty region");
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn replay_jbd2_segments_rejects_empty_or_zero_length_segments() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);

        let err = replay_jbd2_segments(&cx, &dev, &[]).expect_err("empty segments");
        assert!(matches!(err, FfsError::Format(_)));

        let segments = [JournalSegment {
            start: BlockNumber(5),
            blocks: 0,
        }];
        let err = replay_jbd2_segments(&cx, &dev, &segments).expect_err("zero-length segment");
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn native_cow_open_empty_region_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(0),
            blocks: 0,
        };

        let err = NativeCowJournal::open(&cx, &dev, region).expect_err("empty region");
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn native_cow_multi_commit_recovery() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(40),
            blocks: 32,
        };

        let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open");

        journal
            .append_write(&cx, &dev, CommitSeq(1), BlockNumber(10), &[0x11; 64])
            .expect("write 1");
        journal
            .append_commit(&cx, &dev, CommitSeq(1))
            .expect("commit 1");

        journal
            .append_write(&cx, &dev, CommitSeq(2), BlockNumber(11), &[0x22; 64])
            .expect("write 2a");
        journal
            .append_write(&cx, &dev, CommitSeq(2), BlockNumber(12), &[0x33; 64])
            .expect("write 2b");
        journal
            .append_commit(&cx, &dev, CommitSeq(2))
            .expect("commit 2");

        let recovered = recover_native_cow(&cx, &dev, region).expect("recover");
        assert_eq!(recovered.len(), 2);

        assert_eq!(recovered[0].commit_seq, CommitSeq(1));
        assert_eq!(recovered[0].writes.len(), 1);

        assert_eq!(recovered[1].commit_seq, CommitSeq(2));
        assert_eq!(recovered[1].writes.len(), 2);
        assert_eq!(recovered[1].writes[0].block, BlockNumber(11));
        assert_eq!(recovered[1].writes[1].block, BlockNumber(12));
    }

    #[test]
    fn native_cow_crc_corruption_detected() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(40),
            blocks: 16,
        };

        let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open");
        journal
            .append_write(&cx, &dev, CommitSeq(1), BlockNumber(5), &[0xAA; 64])
            .expect("write");
        journal
            .append_commit(&cx, &dev, CommitSeq(1))
            .expect("commit");

        // Corrupt the payload bytes of the write record (first journal block).
        let write_block = BlockNumber(40);
        let mut raw = dev
            .read_block(&cx, write_block)
            .expect("read")
            .as_slice()
            .to_vec();
        // Payload starts at COW_HEADER_SIZE (32). Flip a byte.
        raw[COW_HEADER_SIZE] ^= 0xFF;
        dev.raw_write(write_block, raw);

        let err = recover_native_cow(&cx, &dev, region).expect_err("CRC mismatch");
        assert!(
            matches!(err, FfsError::Corruption { .. }),
            "expected Corruption, got {err:?}"
        );
    }

    #[test]
    fn native_cow_replay_rejects_oversized_payload() {
        let commits = vec![RecoveredCommit {
            commit_seq: CommitSeq(1),
            writes: vec![CowWrite {
                block: BlockNumber(5),
                bytes: vec![0xFF; 1024], // Larger than 512-byte blocks.
            }],
        }];

        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let err = replay_native_cow(&cx, &dev, &commits).expect_err("oversized payload");
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn native_cow_journal_full_returns_no_space() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 128);
        let region = JournalRegion {
            start: BlockNumber(40),
            blocks: 2,
        };

        let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open");
        journal
            .append_write(&cx, &dev, CommitSeq(1), BlockNumber(1), &[0x01; 32])
            .expect("write 1");
        journal
            .append_commit(&cx, &dev, CommitSeq(1))
            .expect("commit 1");

        // Region is now full (2 blocks used).
        let err = journal
            .append_write(&cx, &dev, CommitSeq(2), BlockNumber(2), &[0x02; 32])
            .expect_err("journal full");
        assert!(matches!(err, FfsError::NoSpace));
    }

    #[test]
    fn recover_native_cow_empty_region_returns_empty() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(0),
            blocks: 0,
        };

        let recovered = recover_native_cow(&cx, &dev, region).expect("recover empty");
        assert!(recovered.is_empty());
    }

    #[test]
    fn jbd2_writer_free_blocks_decreases_after_commit() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 256);
        let region = JournalRegion {
            start: BlockNumber(100),
            blocks: 32,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        assert_eq!(writer.free_blocks(), 32);

        let mut txn = writer.begin_transaction();
        txn.add_write(BlockNumber(5), vec![0xAA; 512]);
        writer.commit_transaction(&cx, &dev, &txn).expect("commit");

        // Used: desc(1) + data(1) + commit(1) = 3 blocks.
        assert_eq!(writer.free_blocks(), 29);
        assert_eq!(writer.head(), 3);
    }

    #[test]
    fn jbd2_writer_begin_transaction_increments_sequence() {
        let region = JournalRegion {
            start: BlockNumber(0),
            blocks: 100,
        };

        let mut writer = Jbd2Writer::new(region, 10);
        assert_eq!(writer.next_seq(), 10);

        let txn1 = writer.begin_transaction();
        assert_eq!(txn1.sequence(), 10);
        assert_eq!(writer.next_seq(), 11);

        let txn2 = writer.begin_transaction();
        assert_eq!(txn2.sequence(), 11);
        assert_eq!(writer.next_seq(), 12);
    }

    #[test]
    fn jbd2_transaction_accessors() {
        let region = JournalRegion {
            start: BlockNumber(0),
            blocks: 100,
        };

        let mut writer = Jbd2Writer::new(region, 1);
        let mut txn = writer.begin_transaction();

        assert_eq!(txn.write_count(), 0);
        assert_eq!(txn.revoke_count(), 0);

        txn.add_write(BlockNumber(1), vec![0; 512]);
        txn.add_write(BlockNumber(2), vec![0; 512]);
        txn.add_revoke(BlockNumber(3));

        assert_eq!(txn.write_count(), 2);
        assert_eq!(txn.revoke_count(), 1);
    }

    // ── Adversarial journal replay tests ─────────────────────────────────

    #[test]
    fn adversarial_truncated_descriptor_at_journal_end() {
        // Descriptor claims 3 data blocks but only 1 follows before the
        // journal region ends. Replay should treat this as an incomplete
        // transaction and succeed (not return an error).
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 3, // Only room for descriptor + 1 data block + 1 more
        };

        // Descriptor says 3 tags → needs 3 data blocks after it, but only
        // 2 slots remain in the region.
        let desc = descriptor_block(512, 1, &[(3, 0), (4, 0), (5, JBD2_TAG_FLAG_LAST)]);
        dev.raw_write(BlockNumber(10), desc);
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), vec![0xBB; 512]);

        let out =
            replay_jbd2(&cx, &dev, region).expect("truncated descriptor should not cause error");

        // No commit block present, so nothing should be replayed.
        assert!(out.committed_sequences.is_empty());
        // Target blocks should remain untouched.
        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0_u8; 512]
        );
    }

    #[test]
    fn adversarial_truncated_descriptor_after_valid_txn() {
        // First transaction is complete and committed. Second transaction
        // has a truncated descriptor at the journal end. The first
        // transaction should still be replayed successfully.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 6,
        };

        // Txn 1: complete (desc + data + commit = 3 blocks)
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 1, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 1));

        // Txn 2: truncated descriptor claiming 3 data blocks, only 2 slots
        // remain in region (blocks 13, 14, 15 → 3 slots but need desc + 3).
        dev.raw_write(
            BlockNumber(13),
            descriptor_block(512, 2, &[(4, 0), (5, 0), (6, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(14), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(15), vec![0xCC; 512]);

        let out =
            replay_jbd2(&cx, &dev, region).expect("should succeed despite truncated second txn");

        // First transaction should be replayed.
        assert_eq!(out.committed_sequences, vec![1]);
        assert_eq!(out.stats.replayed_blocks, 1);
        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0xAA; 512]
        );

        // Second transaction's targets should be untouched.
        assert_eq!(
            dev.read_block(&cx, BlockNumber(4)).unwrap().as_slice(),
            &[0_u8; 512]
        );
    }

    #[test]
    fn adversarial_orphaned_commit_no_descriptor() {
        // A commit block with no preceding descriptor for that sequence.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 4,
        };

        dev.raw_write(BlockNumber(10), commit_block(512, 42));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        // The commit is recorded but there are no writes to replay.
        assert_eq!(out.committed_sequences, vec![42]);
        assert_eq!(out.stats.commit_blocks, 1);
        assert_eq!(out.stats.replayed_blocks, 0);
        assert_eq!(out.stats.orphaned_commit_blocks, 1);
    }

    #[test]
    fn adversarial_unknown_block_type_skipped() {
        // A valid JBD2 header with an unknown block_type should be skipped
        // without causing an error.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        // Block 10: unknown type (99).
        let mut unknown = vec![0_u8; 512];
        encode_jbd2_header(&mut unknown, 99, 1);
        dev.raw_write(BlockNumber(10), unknown);

        // Block 11-13: valid committed transaction.
        dev.raw_write(
            BlockNumber(11),
            descriptor_block(512, 5, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(12), vec![0xDD; 512]);
        dev.raw_write(BlockNumber(13), commit_block(512, 5));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences, vec![5]);
        assert_eq!(out.stats.replayed_blocks, 1);
        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0xDD; 512]
        );
    }

    #[test]
    fn adversarial_out_of_order_sequence_numbers() {
        // Transactions committed out of sequence order (seq 20 then seq 10).
        // Both should be replayed. Later-in-journal wins for same target.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // First in journal: seq 20 writes 0xAA to block 3.
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 20, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 20));

        // Second in journal: seq 10 writes 0xBB to block 3.
        dev.raw_write(
            BlockNumber(13),
            descriptor_block(512, 10, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(14), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(15), commit_block(512, 10));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        // Both sequences committed. Replay processes in seq order (10, 20).
        // seq 10 writes 0xBB, then seq 20 writes 0xAA. Final value = 0xAA
        // because committed_sequences are iterated in sorted order (BTreeSet).
        assert_eq!(out.committed_sequences.len(), 2);
        assert!(out.committed_sequences.contains(&10));
        assert!(out.committed_sequences.contains(&20));

        let target = dev.read_block(&cx, BlockNumber(3)).unwrap();
        // seq 10 → 0xBB, seq 20 → 0xAA; sorted order means 20 is applied last.
        assert_eq!(target.as_slice(), &[0xAA; 512]);
    }

    #[test]
    fn adversarial_duplicate_commits_same_sequence() {
        // Two commit blocks for the same sequence number. Should not
        // cause duplication of writes.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 7, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xEE; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 7));
        dev.raw_write(BlockNumber(13), commit_block(512, 7)); // Duplicate commit.

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        // Sequence 7 committed (BTreeSet deduplicates).
        assert_eq!(out.committed_sequences, vec![7]);
        assert_eq!(out.stats.commit_blocks, 2);
        assert_eq!(out.stats.replayed_blocks, 1);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0xEE; 512]
        );
    }

    #[test]
    fn adversarial_cross_txn_revoke_supersedes_earlier_write() {
        // Seq 1 writes block 5. Seq 2 revokes block 5.
        // After replay, block 5 should NOT have seq 1's data because
        // the revoke in seq 2 supersedes seq 1's write (seq 2 > seq 1).
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // Seq 1: descriptor + data for block 5 + commit.
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 1, &[(5, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 1));

        // Seq 2: revoke block 5 + commit.
        dev.raw_write(BlockNumber(13), revoke_block(512, 2, &[5]));
        dev.raw_write(BlockNumber(14), commit_block(512, 2));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences.len(), 2);
        // The write from seq 1 to block 5 should be skipped because seq 2
        // revokes it (seq 1 <= revoke seq 2).
        assert_eq!(out.stats.skipped_revoked_blocks, 1);
        assert_eq!(out.stats.replayed_blocks, 0);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(5)).unwrap().as_slice(),
            &[0_u8; 512],
            "block 5 should remain untouched due to cross-txn revoke"
        );
    }

    #[test]
    fn adversarial_revoke_does_not_affect_later_write() {
        // Seq 1 revokes block 5. Seq 2 writes block 5.
        // The write in seq 2 should succeed because revoke in seq 1 only
        // affects writes with seq <= 1.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // Seq 1: revoke block 5 + commit.
        dev.raw_write(BlockNumber(10), revoke_block(512, 1, &[5]));
        dev.raw_write(BlockNumber(11), commit_block(512, 1));

        // Seq 2: write block 5 + commit.
        dev.raw_write(
            BlockNumber(12),
            descriptor_block(512, 2, &[(5, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(13), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(14), commit_block(512, 2));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences.len(), 2);
        assert_eq!(out.stats.replayed_blocks, 1);
        assert_eq!(out.stats.skipped_revoked_blocks, 0);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(5)).unwrap().as_slice(),
            &[0xBB; 512],
            "later write should not be affected by earlier revoke"
        );
    }

    #[test]
    fn adversarial_garbage_between_valid_transactions() {
        // Valid txn, then garbage block, then another valid txn.
        // Both valid transactions should be replayed.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // Txn 1: seq 1, writes block 3.
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 1, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);
        dev.raw_write(BlockNumber(12), commit_block(512, 1));

        // Garbage block.
        dev.raw_write(BlockNumber(13), vec![0xFF; 512]);

        // Txn 2: seq 2, writes block 4.
        dev.raw_write(
            BlockNumber(14),
            descriptor_block(512, 2, &[(4, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(15), vec![0xBB; 512]);
        dev.raw_write(BlockNumber(16), commit_block(512, 2));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences, vec![1, 2]);
        assert_eq!(out.stats.replayed_blocks, 2);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0xAA; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(4)).unwrap().as_slice(),
            &[0xBB; 512]
        );
    }

    #[test]
    fn adversarial_single_block_journal_region() {
        // A journal region with only 1 block. Anything there must be
        // a non-descriptor header or garbage.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 1,
        };

        // A lone commit block in a 1-block region.
        dev.raw_write(BlockNumber(10), commit_block(512, 1));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");
        assert_eq!(out.committed_sequences, vec![1]);
        assert_eq!(out.stats.orphaned_commit_blocks, 1);
        assert_eq!(out.stats.replayed_blocks, 0);
    }

    #[test]
    fn adversarial_descriptor_with_zero_tags() {
        // A descriptor block where all tag slots are zero (no valid tags).
        // Should be handled gracefully.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 4,
        };

        // Descriptor with no tags (all zeros after header).
        let mut desc = vec![0_u8; 512];
        encode_jbd2_header(&mut desc, JBD2_BLOCKTYPE_DESCRIPTOR, 1);
        // Tags area is all zeros → parse_descriptor_tags returns empty vec.
        dev.raw_write(BlockNumber(10), desc);
        dev.raw_write(BlockNumber(11), commit_block(512, 1));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");
        assert_eq!(out.committed_sequences, vec![1]);
        assert_eq!(out.stats.descriptor_blocks, 1);
        assert_eq!(out.stats.descriptor_tags, 0);
        assert_eq!(out.stats.replayed_blocks, 0);
    }

    #[test]
    fn adversarial_revoke_r_count_exceeds_block_size() {
        // A revoke block where r_count claims more entries than fit in the
        // block. parse_revoke_entries clamps to block.len().
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 8,
        };

        // Descriptor + data + revoke with inflated r_count + commit.
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 1, &[(5, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);

        // Craft a revoke block with r_count = 0xFFFF_FFFF (way more than
        // fits in 512 bytes). Only the entries that actually fit should be
        // parsed.
        let mut rev = vec![0_u8; 512];
        encode_jbd2_header(&mut rev, JBD2_BLOCKTYPE_REVOKE, 1);
        rev[12..16].copy_from_slice(&0xFFFF_FFFF_u32.to_be_bytes());
        // Write one actual revoke entry for block 5.
        rev[16..20].copy_from_slice(&5_u32.to_be_bytes());
        dev.raw_write(BlockNumber(12), rev);

        dev.raw_write(BlockNumber(13), commit_block(512, 1));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences, vec![1]);
        // Block 5 should be revoked despite the inflated r_count.
        assert_eq!(out.stats.skipped_revoked_blocks, 1);
        assert_eq!(out.stats.replayed_blocks, 0);
    }

    #[test]
    fn adversarial_many_descriptors_same_sequence() {
        // Multiple descriptor blocks for the same sequence number,
        // each contributing writes. All should be accumulated.
        let cx = test_cx();
        let dev = MemBlockDevice::new(512, 64);
        let region = JournalRegion {
            start: BlockNumber(10),
            blocks: 16,
        };

        // First descriptor for seq 1: writes block 3.
        dev.raw_write(
            BlockNumber(10),
            descriptor_block(512, 1, &[(3, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(11), vec![0xAA; 512]);

        // Second descriptor for seq 1: writes block 4.
        dev.raw_write(
            BlockNumber(12),
            descriptor_block(512, 1, &[(4, JBD2_TAG_FLAG_LAST)]),
        );
        dev.raw_write(BlockNumber(13), vec![0xBB; 512]);

        // Single commit for seq 1.
        dev.raw_write(BlockNumber(14), commit_block(512, 1));

        let out = replay_jbd2(&cx, &dev, region).expect("should succeed");

        assert_eq!(out.committed_sequences, vec![1]);
        assert_eq!(out.stats.descriptor_blocks, 2);
        assert_eq!(out.stats.replayed_blocks, 2);

        assert_eq!(
            dev.read_block(&cx, BlockNumber(3)).unwrap().as_slice(),
            &[0xAA; 512]
        );
        assert_eq!(
            dev.read_block(&cx, BlockNumber(4)).unwrap().as_slice(),
            &[0xBB; 512]
        );
    }

    // ── Property-based tests (proptest) ────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// JBD2 write → replay roundtrip: committed writes appear at target blocks.
        #[test]
        fn proptest_jbd2_write_replay_roundtrip(
            num_writes in 1_usize..8,
            payload_byte in any::<u8>(),
            start_seq in 1_u32..1000,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(512, 512);
            let region = JournalRegion {
                start: BlockNumber(100),
                blocks: 128,
            };

            let mut writer = Jbd2Writer::new(region, start_seq);
            let mut txn = writer.begin_transaction();

            // Each write targets a distinct block in [0, num_writes).
            for i in 0..num_writes {
                let target = BlockNumber(u64::try_from(i).unwrap());
                let byte = payload_byte.wrapping_add(u8::try_from(i % 256).unwrap());
                txn.add_write(target, vec![byte; 512]);
            }

            let (seq, stats) = writer.commit_transaction(&cx, &dev, &txn).expect("commit");
            prop_assert_eq!(seq, start_seq);
            prop_assert_eq!(stats.data_blocks, u64::try_from(num_writes).unwrap());

            // Replay from the same region.
            let out = replay_jbd2(&cx, &dev, region).expect("replay");
            prop_assert_eq!(out.committed_sequences, vec![start_seq]);
            prop_assert_eq!(out.stats.replayed_blocks, u64::try_from(num_writes).unwrap());

            // Verify each target block contains the correct data.
            for i in 0..num_writes {
                let target = BlockNumber(u64::try_from(i).unwrap());
                let expected_byte = payload_byte.wrapping_add(u8::try_from(i % 256).unwrap());
                let data = dev.read_block(&cx, target).expect("read target");
                prop_assert_eq!(
                    data.as_slice()[0], expected_byte,
                    "target block {} should contain {:#x}",
                    i, expected_byte,
                );
            }
        }

        /// JBD2 write with revokes → replay roundtrip: revoked blocks are skipped.
        #[test]
        fn proptest_jbd2_revoke_skips_writes(
            num_writes in 2_usize..8,
            revoke_idx in 0_usize..8,
        ) {
            let actual_revoke_idx = revoke_idx % num_writes;

            let cx = test_cx();
            let dev = MemBlockDevice::new(512, 512);
            let region = JournalRegion {
                start: BlockNumber(200),
                blocks: 128,
            };

            let mut writer = Jbd2Writer::new(region, 1);
            let mut txn = writer.begin_transaction();

            for i in 0..num_writes {
                txn.add_write(BlockNumber(u64::try_from(i).unwrap()), vec![0xAA; 512]);
            }
            // Revoke one of the writes.
            txn.add_revoke(BlockNumber(u64::try_from(actual_revoke_idx).unwrap()));

            writer.commit_transaction(&cx, &dev, &txn).expect("commit");

            let out = replay_jbd2(&cx, &dev, region).expect("replay");
            prop_assert_eq!(out.stats.skipped_revoked_blocks, 1);
            prop_assert_eq!(
                out.stats.replayed_blocks,
                u64::try_from(num_writes - 1).unwrap(),
            );

            // Revoked block should be untouched (zeros).
            let revoked = dev
                .read_block(&cx, BlockNumber(u64::try_from(actual_revoke_idx).unwrap()))
                .expect("read revoked");
            prop_assert_eq!(
                revoked.as_slice(),
                &[0_u8; 512],
                "revoked block {} should be untouched",
                actual_revoke_idx,
            );
        }

        /// Jbd2Writer::blocks_needed accurately predicts journal consumption.
        #[test]
        fn proptest_blocks_needed_matches_actual(
            num_writes in 0_usize..16,
            num_revokes in 0_usize..16,
        ) {
            prop_assume!(num_writes > 0 || num_revokes > 0);

            let cx = test_cx();
            let dev = MemBlockDevice::new(512, 512);
            let region = JournalRegion {
                start: BlockNumber(100),
                blocks: 256,
            };

            let predicted = Jbd2Writer::blocks_needed(512, num_writes, num_revokes);

            let mut writer = Jbd2Writer::new(region, 1);
            let head_before = writer.head();
            let mut txn = writer.begin_transaction();

            for i in 0..num_writes {
                txn.add_write(BlockNumber(u64::try_from(i + 300).unwrap()), vec![0xBB; 512]);
            }
            for i in 0..num_revokes {
                txn.add_revoke(BlockNumber(u64::try_from(i + 400).unwrap()));
            }

            writer.commit_transaction(&cx, &dev, &txn).expect("commit");
            let actual_used = writer.head() - head_before;

            prop_assert_eq!(
                actual_used, predicted,
                "blocks_needed({}, {}) predicted {}, actual {}",
                num_writes, num_revokes, predicted, actual_used,
            );
        }

        /// JournalRegion::resolve: in-range resolves correctly, out-of-range returns None.
        #[test]
        fn proptest_journal_region_resolve(
            start in 0_u64..10_000,
            blocks in 1_u64..1000,
            index in 0_u64..2000,
        ) {
            let region = JournalRegion {
                start: BlockNumber(start),
                blocks,
            };

            let result = region.resolve(index);
            if index < blocks {
                let expected = start.checked_add(index).map(BlockNumber);
                prop_assert_eq!(result, expected);
            } else {
                prop_assert_eq!(result, None);
            }
        }

        /// JournalSegment::resolve: in-range resolves correctly, out-of-range returns None.
        #[test]
        fn proptest_journal_segment_resolve(
            start in 0_u64..10_000,
            blocks in 1_u64..1000,
            index in 0_u64..2000,
        ) {
            let segment = JournalSegment {
                start: BlockNumber(start),
                blocks,
            };

            let result = segment.resolve(index);
            if index < blocks {
                let expected = start.checked_add(index).map(BlockNumber);
                prop_assert_eq!(result, expected);
            } else {
                prop_assert_eq!(result, None);
            }
        }

        /// parse_descriptor_tags never panics on arbitrary byte input.
        #[test]
        fn proptest_parse_descriptor_tags_no_panic(
            data in prop::collection::vec(any::<u8>(), 0..1024),
        ) {
            let _ = parse_descriptor_tags(&data);
        }

        /// parse_revoke_entries never panics on arbitrary byte input.
        #[test]
        fn proptest_parse_revoke_entries_no_panic(
            data in prop::collection::vec(any::<u8>(), 0..1024),
        ) {
            let _ = parse_revoke_entries(&data);
        }

        /// Native COW write → recover roundtrip: committed writes are recovered.
        #[test]
        fn proptest_native_cow_roundtrip(
            num_writes in 1_usize..6,
            payload_byte in any::<u8>(),
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(512, 512);
            let region = JournalRegion {
                start: BlockNumber(50),
                blocks: 64,
            };

            let mut journal = NativeCowJournal::open(&cx, &dev, region).expect("open");

            for i in 0..num_writes {
                let byte = payload_byte.wrapping_add(u8::try_from(i % 256).unwrap());
                journal
                    .append_write(
                        &cx,
                        &dev,
                        CommitSeq(1),
                        BlockNumber(u64::try_from(i).unwrap()),
                        &vec![byte; 64],
                    )
                    .expect("write");
            }
            journal.append_commit(&cx, &dev, CommitSeq(1)).expect("commit");

            let recovered = recover_native_cow(&cx, &dev, region).expect("recover");
            prop_assert_eq!(recovered.len(), 1);
            prop_assert_eq!(recovered[0].commit_seq, CommitSeq(1));
            prop_assert_eq!(recovered[0].writes.len(), num_writes);

            for i in 0..num_writes {
                let expected_byte = payload_byte.wrapping_add(u8::try_from(i % 256).unwrap());
                prop_assert_eq!(
                    recovered[0].writes[i].block,
                    BlockNumber(u64::try_from(i).unwrap()),
                );
                prop_assert_eq!(recovered[0].writes[i].bytes[0], expected_byte);
            }
        }

        /// apply_replay writes exactly the non-revoked blocks.
        #[test]
        fn proptest_apply_replay_writes_non_revoked(
            num_writes in 1_usize..6,
            revoke_idx in 0_usize..6,
        ) {
            let actual_revoke_idx = revoke_idx % num_writes;

            let cx = test_cx();
            let dev = MemBlockDevice::new(512, 512);

            let mut writes = Vec::new();
            let mut revoked = BTreeSet::new();

            for i in 0..num_writes {
                let target = BlockNumber(u64::try_from(i).unwrap());
                writes.push((target, vec![0xCC; 512]));
            }
            revoked.insert(BlockNumber(u64::try_from(actual_revoke_idx).unwrap()));

            let committed = vec![(writes, revoked)];
            let stats = apply_replay(&cx, &dev, &committed).expect("apply");

            prop_assert_eq!(
                stats.blocks_written,
                u64::try_from(num_writes - 1).unwrap(),
            );
            prop_assert_eq!(stats.blocks_verified, stats.blocks_written);
            prop_assert_eq!(stats.verify_mismatches, 0);

            // Revoked block should be zeros.
            let revoked_data = dev
                .read_block(&cx, BlockNumber(u64::try_from(actual_revoke_idx).unwrap()))
                .expect("read revoked");
            prop_assert_eq!(revoked_data.as_slice(), &[0_u8; 512]);

            // Non-revoked blocks should contain 0xCC.
            for i in 0..num_writes {
                if i == actual_revoke_idx {
                    continue;
                }
                let data = dev
                    .read_block(&cx, BlockNumber(u64::try_from(i).unwrap()))
                    .expect("read");
                prop_assert_eq!(
                    data.as_slice()[0], 0xCC,
                    "non-revoked block {} should contain 0xCC", i,
                );
            }
        }
    }
}
