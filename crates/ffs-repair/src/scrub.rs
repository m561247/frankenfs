//! Block-level scrub pipeline for detecting corruption signals.
//!
//! The scrub pipeline iterates over blocks in a device, running pluggable
//! validators to detect checksum mismatches, structural invariant violations,
//! and other corruption signals. Findings are collected into a [`ScrubReport`]
//! with per-block detail and summary statistics.
//!
//! # Design
//!
//! The scrub does not interpret filesystem-specific formats directly. Instead,
//! callers provide a [`BlockValidator`] implementation that knows how to check
//! a block's integrity (e.g., ext4 metadata_csum, btrfs csum tree). This keeps
//! the scrub engine format-agnostic while allowing precise corruption detection.
//!
//! # Usage
//!
//! ```text
//! let report = Scrubber::new(&device, &validator)
//!     .scrub_range(&cx, BlockNumber(0), 1000)?;
//! for finding in &report.findings {
//!     eprintln!("block {}: {:?} - {}", finding.block, finding.kind, finding.detail);
//! }
//! ```

use asupersync::Cx;
use ffs_block::{BlockBuf, BlockDevice};
use ffs_error::{FfsError, Result};
use ffs_ondisk::{
    BtrfsHeader, BtrfsSuperblock, Ext4Superblock, verify_btrfs_superblock_checksum,
    verify_btrfs_tree_block_checksum,
};
use ffs_types::{
    BTRFS_SUPER_INFO_OFFSET, BTRFS_SUPER_INFO_SIZE, BlockNumber, EXT4_SUPERBLOCK_OFFSET,
    EXT4_SUPERBLOCK_SIZE, ParseError,
};
use std::fmt;

// ── Corruption taxonomy ─────────────────────────────────────────────────────

/// Category of corruption detected during a scrub pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorruptionKind {
    /// Stored checksum does not match computed checksum (CRC32C, BLAKE3, etc.).
    ChecksumMismatch,
    /// A structural field is out of its valid range (e.g., block pointer beyond
    /// device bounds, negative free count, impossible inode size).
    StructuralInvariant,
    /// Magic number or signature bytes are wrong.
    BadMagic,
    /// Block is entirely zeroed when it should contain data.
    UnexpectedZeroes,
    /// Read I/O error (device returned an error for this block).
    IoError,
}

impl fmt::Display for CorruptionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChecksumMismatch => write!(f, "checksum_mismatch"),
            Self::StructuralInvariant => write!(f, "structural_invariant"),
            Self::BadMagic => write!(f, "bad_magic"),
            Self::UnexpectedZeroes => write!(f, "unexpected_zeroes"),
            Self::IoError => write!(f, "io_error"),
        }
    }
}

/// Severity of a scrub finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational — potential issue but not necessarily corruption.
    Info,
    /// Possible corruption that may affect data integrity.
    Warning,
    /// Confirmed corruption that affects data integrity.
    Error,
    /// Critical corruption in metadata essential for filesystem operation
    /// (superblock, group descriptors, journal).
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ── Findings ────────────────────────────────────────────────────────────────

/// A single corruption finding for one block.
#[derive(Debug, Clone)]
pub struct ScrubFinding {
    /// Block where corruption was detected.
    pub block: BlockNumber,
    /// Category of corruption.
    pub kind: CorruptionKind,
    /// How severe the finding is.
    pub severity: Severity,
    /// Human-readable detail (e.g., "CRC32C expected 0xABCD, got 0x1234").
    pub detail: String,
}

impl fmt::Display for ScrubFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "block {} [{}] {}: {}",
            self.block, self.severity, self.kind, self.detail
        )
    }
}

// ── Report ──────────────────────────────────────────────────────────────────

/// Aggregated results from a scrub pass.
#[derive(Debug, Clone)]
pub struct ScrubReport {
    /// All findings, ordered by block number.
    pub findings: Vec<ScrubFinding>,
    /// Total blocks scanned (including clean ones).
    pub blocks_scanned: u64,
    /// Blocks that had at least one finding.
    pub blocks_corrupt: u64,
    /// Blocks that returned I/O errors during read.
    pub blocks_io_error: u64,
}

impl ScrubReport {
    /// True if no corruption was found.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// Count of findings at or above the given severity.
    #[must_use]
    pub fn count_at_severity(&self, min: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity >= min).count()
    }
}

impl fmt::Display for ScrubReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "scanned {} blocks: {} corrupt, {} io_errors, {} findings",
            self.blocks_scanned,
            self.blocks_corrupt,
            self.blocks_io_error,
            self.findings.len(),
        )
    }
}

// ── Validator trait ─────────────────────────────────────────────────────────

/// Result of validating a single block.
#[derive(Debug)]
pub enum BlockVerdict {
    /// Block passes all checks.
    Clean,
    /// Block has one or more issues.
    Corrupt(Vec<(CorruptionKind, Severity, String)>),
    /// Validator does not know how to check this block (skip it).
    Skip,
}

/// Pluggable block validation strategy.
///
/// Implementations inspect raw block data and return a verdict. The scrub
/// engine calls `validate` for every block in the scan range and collects
/// non-clean verdicts into the report.
pub trait BlockValidator: Send + Sync {
    /// Validate a single block.
    ///
    /// `block` is the block number, `data` is the raw block content.
    /// Implementations should never panic on malformed data — return
    /// `BlockVerdict::Corrupt` with details instead.
    fn validate(&self, block: BlockNumber, data: &BlockBuf) -> BlockVerdict;
}

// ── Scrubber ────────────────────────────────────────────────────────────────

/// Block-level scrub engine.
///
/// Iterates over a range of blocks on a [`BlockDevice`], validates each using
/// a [`BlockValidator`], and collects findings into a [`ScrubReport`].
pub struct Scrubber<'a> {
    device: &'a dyn BlockDevice,
    validator: &'a dyn BlockValidator,
}

impl<'a> Scrubber<'a> {
    pub fn new(device: &'a dyn BlockDevice, validator: &'a dyn BlockValidator) -> Self {
        Self { device, validator }
    }

    /// Scrub a range of blocks `[start, start + count)`.
    ///
    /// Returns a report with all findings. Does not panic on I/O errors or
    /// corrupted data — those are recorded as findings.
    pub fn scrub_range(&self, cx: &Cx, start: BlockNumber, count: u64) -> Result<ScrubReport> {
        let device_blocks = self.device.block_count();
        let end = start.0.saturating_add(count).min(device_blocks);

        let mut findings = Vec::new();
        let mut blocks_scanned: u64 = 0;
        let mut blocks_corrupt: u64 = 0;
        let mut blocks_io_error: u64 = 0;

        let mut block_num = start.0;
        while block_num < end {
            let block = BlockNumber(block_num);

            // Cooperative cancellation check every 256 blocks.
            if blocks_scanned % 256 == 0 {
                cx.checkpoint().map_err(|_| FfsError::Cancelled)?;
            }

            blocks_scanned += 1;

            match self.device.read_block(cx, block) {
                Ok(buf) => match self.validator.validate(block, &buf) {
                    BlockVerdict::Clean | BlockVerdict::Skip => {}
                    BlockVerdict::Corrupt(issues) => {
                        blocks_corrupt += 1;
                        for (kind, severity, detail) in issues {
                            findings.push(ScrubFinding {
                                block,
                                kind,
                                severity,
                                detail,
                            });
                        }
                    }
                },
                Err(e) => {
                    blocks_io_error += 1;
                    findings.push(ScrubFinding {
                        block,
                        kind: CorruptionKind::IoError,
                        severity: Severity::Error,
                        detail: format!("read failed: {e}"),
                    });
                }
            }

            block_num += 1;
        }

        Ok(ScrubReport {
            findings,
            blocks_scanned,
            blocks_corrupt,
            blocks_io_error,
        })
    }

    /// Scrub the entire device.
    pub fn scrub_all(&self, cx: &Cx) -> Result<ScrubReport> {
        self.scrub_range(cx, BlockNumber(0), self.device.block_count())
    }
}

// ── Built-in validators ─────────────────────────────────────────────────────

/// A validator that checks whether a block is entirely zeroed.
///
/// Useful as a baseline or for detecting zeroed-out metadata blocks that
/// should contain data (e.g., superblock backups).
pub struct ZeroCheckValidator;

impl BlockValidator for ZeroCheckValidator {
    fn validate(&self, _block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
        if data.as_slice().iter().all(|&b| b == 0) {
            BlockVerdict::Corrupt(vec![(
                CorruptionKind::UnexpectedZeroes,
                Severity::Warning,
                "block is entirely zeroed".to_owned(),
            )])
        } else {
            BlockVerdict::Clean
        }
    }
}

/// Validator for the canonical ext4 primary superblock.
///
/// Checks parseability, basic geometry sanity, and metadata checksum integrity
/// (when `METADATA_CSUM` is enabled).
#[derive(Debug, Clone, Copy)]
pub struct Ext4SuperblockValidator {
    block_size: u32,
}

impl Ext4SuperblockValidator {
    #[must_use]
    pub fn new(block_size: u32) -> Self {
        Self { block_size }
    }

    fn target_block_and_offset(self) -> (u64, usize) {
        if self.block_size == 1024 {
            (1, 0)
        } else {
            (0, EXT4_SUPERBLOCK_OFFSET)
        }
    }
}

impl BlockValidator for Ext4SuperblockValidator {
    fn validate(&self, block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
        let (target_block, offset) = self.target_block_and_offset();
        if block.0 != target_block {
            return BlockVerdict::Skip;
        }

        let end = offset.saturating_add(EXT4_SUPERBLOCK_SIZE);
        let Some(region) = data.as_slice().get(offset..end) else {
            return BlockVerdict::Corrupt(vec![(
                CorruptionKind::StructuralInvariant,
                Severity::Critical,
                format!(
                    "ext4 superblock region out of bounds in block {block}: need bytes {offset}..{end}, got {}",
                    data.as_slice().len()
                ),
            )]);
        };

        let sb = match Ext4Superblock::parse_superblock_region(region) {
            Ok(sb) => sb,
            Err(err) => {
                return BlockVerdict::Corrupt(vec![parse_error_issue(
                    "ext4 superblock parse failed",
                    &err,
                )]);
            }
        };

        let mut issues = Vec::new();
        if let Err(err) = sb.validate_geometry() {
            issues.push(parse_error_issue("ext4 superblock geometry invalid", &err));
        }
        if let Err(err) = sb.validate_checksum(region) {
            issues.push(parse_error_issue("ext4 superblock checksum invalid", &err));
        }

        if issues.is_empty() {
            BlockVerdict::Clean
        } else {
            BlockVerdict::Corrupt(issues)
        }
    }
}

/// Validator for the canonical btrfs primary superblock.
///
/// Checks parseability and superblock checksum integrity.
#[derive(Debug, Clone, Copy)]
pub struct BtrfsSuperblockValidator {
    block_size: u32,
}

impl BtrfsSuperblockValidator {
    #[must_use]
    pub fn new(block_size: u32) -> Self {
        Self { block_size }
    }

    fn target_block_and_offset(self) -> (u64, usize) {
        if self.block_size == 0 {
            return (0, 0);
        }
        let block_size = u64::from(self.block_size);
        let byte_offset = BTRFS_SUPER_INFO_OFFSET as u64;
        (
            byte_offset / block_size,
            usize::try_from(byte_offset % block_size).unwrap_or(0),
        )
    }
}

impl BlockValidator for BtrfsSuperblockValidator {
    fn validate(&self, block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
        let (target_block, offset) = self.target_block_and_offset();
        if block.0 != target_block {
            return BlockVerdict::Skip;
        }

        let end = offset.saturating_add(BTRFS_SUPER_INFO_SIZE);
        let Some(region) = data.as_slice().get(offset..end) else {
            return BlockVerdict::Corrupt(vec![(
                CorruptionKind::StructuralInvariant,
                Severity::Critical,
                format!(
                    "btrfs superblock region out of bounds in block {block}: need bytes {offset}..{end}, got {}",
                    data.as_slice().len()
                ),
            )]);
        };

        let mut issues = Vec::new();
        if let Err(err) = BtrfsSuperblock::parse_superblock_region(region) {
            issues.push(parse_error_issue("btrfs superblock parse failed", &err));
        }
        if let Err(err) = verify_btrfs_superblock_checksum(region) {
            issues.push(parse_error_issue("btrfs superblock checksum invalid", &err));
        }

        if issues.is_empty() {
            BlockVerdict::Clean
        } else {
            BlockVerdict::Corrupt(issues)
        }
    }
}

/// Validator for btrfs tree block metadata integrity.
///
/// Applies to blocks that look like btrfs tree blocks for the current
/// filesystem (`fsid` match). For candidate blocks, validates header geometry
/// and tree-block checksum.
#[derive(Debug, Clone, Copy)]
pub struct BtrfsTreeBlockValidator {
    block_size: u32,
    fsid: [u8; 16],
    csum_type: u16,
}

impl BtrfsTreeBlockValidator {
    #[must_use]
    pub fn new(block_size: u32, fsid: [u8; 16], csum_type: u16) -> Self {
        Self {
            block_size,
            fsid,
            csum_type,
        }
    }

    fn superblock_block(self) -> u64 {
        if self.block_size == 0 {
            return 0;
        }
        (BTRFS_SUPER_INFO_OFFSET as u64) / u64::from(self.block_size)
    }
}

impl BlockValidator for BtrfsTreeBlockValidator {
    fn validate(&self, block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
        if block.0 == self.superblock_block() {
            return BlockVerdict::Skip;
        }

        let slice = data.as_slice();
        let Ok(header) = BtrfsHeader::parse_from_block(slice) else {
            return BlockVerdict::Skip;
        };

        // Treat only blocks that appear to belong to this filesystem as btrfs metadata candidates.
        if header.fsid != self.fsid {
            return BlockVerdict::Skip;
        }

        let Some(expected_bytenr) = block.0.checked_mul(u64::from(self.block_size)) else {
            return BlockVerdict::Corrupt(vec![(
                CorruptionKind::StructuralInvariant,
                Severity::Critical,
                format!(
                    "btrfs tree header invalid: expected bytenr overflow for block {block} and block_size {}",
                    self.block_size
                ),
            )]);
        };

        let mut issues = Vec::new();
        if let Err(err) = header.validate(slice.len(), Some(expected_bytenr)) {
            issues.push(parse_error_issue("btrfs tree header invalid", &err));
        }
        if let Err(err) = verify_btrfs_tree_block_checksum(slice, self.csum_type) {
            issues.push(parse_error_issue("btrfs tree block checksum invalid", &err));
        }

        if issues.is_empty() {
            BlockVerdict::Clean
        } else {
            BlockVerdict::Corrupt(issues)
        }
    }
}

fn parse_error_issue(prefix: &str, err: &ParseError) -> (CorruptionKind, Severity, String) {
    let (kind, severity) = match err {
        ParseError::InvalidMagic { .. } => (CorruptionKind::BadMagic, Severity::Critical),
        ParseError::InvalidField { field, .. }
            if field.contains("checksum") || field.contains("csum") =>
        {
            (CorruptionKind::ChecksumMismatch, Severity::Critical)
        }
        ParseError::InsufficientData { .. }
        | ParseError::InvalidField { .. }
        | ParseError::IntegerConversion { .. } => {
            (CorruptionKind::StructuralInvariant, Severity::Critical)
        }
    };
    (kind, severity, format!("{prefix}: {err}"))
}

/// A composite validator that runs multiple validators and merges their findings.
pub struct CompositeValidator {
    validators: Vec<Box<dyn BlockValidator>>,
}

impl CompositeValidator {
    #[must_use]
    pub fn new(validators: Vec<Box<dyn BlockValidator>>) -> Self {
        Self { validators }
    }
}

impl BlockValidator for CompositeValidator {
    fn validate(&self, block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
        let mut all_issues = Vec::new();
        let mut any_checked = false;

        for v in &self.validators {
            match v.validate(block, data) {
                BlockVerdict::Clean => {
                    any_checked = true;
                }
                BlockVerdict::Corrupt(issues) => {
                    any_checked = true;
                    all_issues.extend(issues);
                }
                BlockVerdict::Skip => {}
            }
        }

        if all_issues.is_empty() {
            if any_checked {
                BlockVerdict::Clean
            } else {
                BlockVerdict::Skip
            }
        } else {
            BlockVerdict::Corrupt(all_issues)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffs_block::BlockBuf;
    use ffs_ondisk::{Ext4IncompatFeatures, Ext4RoCompatFeatures};
    use ffs_types::EXT4_SB_CHECKSUM_OFFSET;
    use parking_lot::RwLock;
    use std::collections::HashMap;

    // ── Test block device ───────────────────────────────────────────────

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

        fn write(&self, block: BlockNumber, data: Vec<u8>) {
            self.blocks.write().insert(block, data);
        }
    }

    impl BlockDevice for MemBlockDevice {
        fn read_block(&self, _cx: &Cx, block: BlockNumber) -> Result<BlockBuf> {
            let bs = self.block_size as usize;
            let data = self
                .blocks
                .read()
                .get(&block)
                .cloned()
                .unwrap_or_else(|| vec![0_u8; bs]);
            Ok(BlockBuf::new(data))
        }

        fn write_block(&self, _cx: &Cx, block: BlockNumber, data: &[u8]) -> Result<()> {
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

    /// Block device that returns I/O errors for specific blocks.
    #[derive(Debug)]
    struct FaultyBlockDevice {
        inner: MemBlockDevice,
        faulty_blocks: Vec<BlockNumber>,
    }

    impl FaultyBlockDevice {
        fn new(inner: MemBlockDevice, faulty_blocks: Vec<BlockNumber>) -> Self {
            Self {
                inner,
                faulty_blocks,
            }
        }
    }

    impl BlockDevice for FaultyBlockDevice {
        fn read_block(&self, cx: &Cx, block: BlockNumber) -> Result<BlockBuf> {
            if self.faulty_blocks.contains(&block) {
                Err(FfsError::Io(std::io::Error::other(format!(
                    "simulated I/O error at block {}",
                    block.0
                ))))
            } else {
                self.inner.read_block(cx, block)
            }
        }

        fn write_block(&self, cx: &Cx, block: BlockNumber, data: &[u8]) -> Result<()> {
            self.inner.write_block(cx, block, data)
        }

        fn block_size(&self) -> u32 {
            self.inner.block_size()
        }

        fn block_count(&self) -> u64 {
            self.inner.block_count()
        }

        fn sync(&self, cx: &Cx) -> Result<()> {
            self.inner.sync(cx)
        }
    }

    fn test_cx() -> Cx {
        Cx::for_testing()
    }

    // ── Test validator: CRC32C of block data stored in first 4 bytes ────

    /// A simple validator that checks a CRC32C stored in the first 4 bytes
    /// of each block against the computed CRC of the remaining bytes.
    struct Crc32cBlockValidator;

    impl BlockValidator for Crc32cBlockValidator {
        fn validate(&self, _block: BlockNumber, data: &BlockBuf) -> BlockVerdict {
            let slice = data.as_slice();
            if slice.len() < 4 {
                return BlockVerdict::Skip;
            }
            let stored = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
            let computed = crc32c::crc32c(&slice[4..]);
            if stored == computed {
                BlockVerdict::Clean
            } else {
                BlockVerdict::Corrupt(vec![(
                    CorruptionKind::ChecksumMismatch,
                    Severity::Error,
                    format!("CRC32C expected {stored:#010x}, got {computed:#010x}"),
                )])
            }
        }
    }

    /// Create a block with CRC32C in first 4 bytes, payload in remainder.
    fn make_checksummed_block(block_size: u32, payload_byte: u8) -> Vec<u8> {
        let bs = block_size as usize;
        let mut block = vec![0_u8; bs];
        // Fill payload area (bytes 4..)
        for b in &mut block[4..] {
            *b = payload_byte;
        }
        // Compute and store CRC32C of payload
        let crc = crc32c::crc32c(&block[4..]);
        block[..4].copy_from_slice(&crc.to_le_bytes());
        block
    }

    fn make_valid_ext4_superblock_region() -> [u8; EXT4_SUPERBLOCK_SIZE] {
        let mut sb = [0_u8; EXT4_SUPERBLOCK_SIZE];
        sb[0x38..0x3A].copy_from_slice(&ffs_types::EXT4_SUPER_MAGIC.to_le_bytes()); // magic
        sb[0x18..0x1C].copy_from_slice(&2_u32.to_le_bytes()); // 4KiB blocks
        sb[0x1C..0x20].copy_from_slice(&2_u32.to_le_bytes()); // 4KiB clusters
        sb[0x00..0x04].copy_from_slice(&8192_u32.to_le_bytes()); // inodes_count
        sb[0x04..0x08].copy_from_slice(&32768_u32.to_le_bytes()); // blocks_count_lo
        sb[0x14..0x18].copy_from_slice(&0_u32.to_le_bytes()); // first_data_block
        sb[0x20..0x24].copy_from_slice(&32768_u32.to_le_bytes()); // blocks_per_group
        sb[0x24..0x28].copy_from_slice(&32768_u32.to_le_bytes()); // clusters_per_group
        sb[0x28..0x2C].copy_from_slice(&8192_u32.to_le_bytes()); // inodes_per_group
        sb[0x58..0x5A].copy_from_slice(&256_u16.to_le_bytes()); // inode_size
        let incompat =
            (Ext4IncompatFeatures::FILETYPE.0 | Ext4IncompatFeatures::EXTENTS.0).to_le_bytes();
        sb[0x60..0x64].copy_from_slice(&incompat);
        sb[0x64..0x68].copy_from_slice(&Ext4RoCompatFeatures::METADATA_CSUM.0.to_le_bytes());
        sb[0x175] = 1; // checksum_type=crc32c

        let checksum = ffs_ondisk::ext4_chksum(!0_u32, &sb[..EXT4_SB_CHECKSUM_OFFSET]);
        sb[EXT4_SB_CHECKSUM_OFFSET..EXT4_SB_CHECKSUM_OFFSET + 4]
            .copy_from_slice(&checksum.to_le_bytes());
        sb
    }

    fn make_valid_btrfs_superblock_region() -> Vec<u8> {
        let mut sb = vec![0_u8; BTRFS_SUPER_INFO_SIZE];
        sb[0x40..0x48].copy_from_slice(&ffs_types::BTRFS_MAGIC.to_le_bytes());
        sb[0x90..0x94].copy_from_slice(&4096_u32.to_le_bytes()); // sectorsize
        sb[0x94..0x98].copy_from_slice(&16384_u32.to_le_bytes()); // nodesize
        sb[0xC4..0xC6].copy_from_slice(&0_u16.to_le_bytes()); // csum_type=CRC32C
        let csum = crc32c::crc32c(&sb[0x20..BTRFS_SUPER_INFO_SIZE]);
        sb[0..4].copy_from_slice(&csum.to_le_bytes());
        sb
    }

    fn make_valid_btrfs_tree_block(block_size: usize, fsid: [u8; 16], bytenr: u64) -> Vec<u8> {
        let mut block = vec![0_u8; block_size];
        block[0x20..0x30].copy_from_slice(&fsid); // fsid
        block[0x30..0x38].copy_from_slice(&bytenr.to_le_bytes()); // bytenr
        block[0x50..0x58].copy_from_slice(&1_u64.to_le_bytes()); // generation
        block[0x58..0x60].copy_from_slice(&5_u64.to_le_bytes()); // owner (tree id)
        block[0x60..0x64].copy_from_slice(&0_u32.to_le_bytes()); // nritems
        block[0x64] = 0; // leaf level

        let csum = crc32c::crc32c(&block[0x20..]);
        block[0..4].copy_from_slice(&csum.to_le_bytes());
        block
    }

    // ── Tests ───────────────────────────────────────────────────────────

    #[test]
    fn clean_device_produces_clean_report() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 16);
        // Write valid checksummed blocks
        for i in 0..16 {
            dev.write(BlockNumber(i), make_checksummed_block(4096, 0xAA));
        }

        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert!(report.is_clean());
        assert_eq!(report.blocks_scanned, 16);
        assert_eq!(report.blocks_corrupt, 0);
        assert_eq!(report.blocks_io_error, 0);
    }

    #[test]
    fn bit_flip_detected_deterministically() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 16);
        // Write valid checksummed blocks
        for i in 0..16 {
            dev.write(BlockNumber(i), make_checksummed_block(4096, 0xBB));
        }
        // Inject bit flip in block 7 (flip a bit in the payload area)
        let mut corrupted = make_checksummed_block(4096, 0xBB);
        corrupted[100] ^= 0x01; // flip one bit
        dev.write(BlockNumber(7), corrupted);

        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert!(!report.is_clean());
        assert_eq!(report.blocks_scanned, 16);
        assert_eq!(report.blocks_corrupt, 1);
        assert_eq!(report.findings.len(), 1);

        let finding = &report.findings[0];
        assert_eq!(finding.block, BlockNumber(7));
        assert_eq!(finding.kind, CorruptionKind::ChecksumMismatch);
        assert_eq!(finding.severity, Severity::Error);
        assert!(finding.detail.contains("CRC32C"));

        // Run again — deterministic
        let report2 = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_all(&cx)
            .expect("second scrub should succeed");
        assert_eq!(report2.findings.len(), 1);
        assert_eq!(report2.findings[0].block, BlockNumber(7));
    }

    #[test]
    fn multiple_corrupt_blocks() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 32);
        for i in 0..32 {
            dev.write(BlockNumber(i), make_checksummed_block(4096, 0xCC));
        }
        // Corrupt blocks 3, 15, 31
        for &b in &[3, 15, 31] {
            let mut data = make_checksummed_block(4096, 0xCC);
            data[200] ^= 0xFF;
            dev.write(BlockNumber(b), data);
        }

        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_corrupt, 3);
        assert_eq!(report.findings.len(), 3);
        let corrupt_blocks: Vec<u64> = report.findings.iter().map(|f| f.block.0).collect();
        assert_eq!(corrupt_blocks, vec![3, 15, 31]);
    }

    #[test]
    fn io_error_recorded_as_finding() {
        let cx = test_cx();
        let inner = MemBlockDevice::new(4096, 16);
        for i in 0..16 {
            inner.write(BlockNumber(i), make_checksummed_block(4096, 0xDD));
        }
        let dev = FaultyBlockDevice::new(inner, vec![BlockNumber(5), BlockNumber(10)]);

        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_scanned, 16);
        assert_eq!(report.blocks_io_error, 2);
        let io_findings: Vec<&ScrubFinding> = report
            .findings
            .iter()
            .filter(|f| f.kind == CorruptionKind::IoError)
            .collect();
        assert_eq!(io_findings.len(), 2);
        assert_eq!(io_findings[0].block, BlockNumber(5));
        assert_eq!(io_findings[1].block, BlockNumber(10));
    }

    #[test]
    fn scrub_range_respects_bounds() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 100);
        // Write valid blocks
        for i in 0..100 {
            dev.write(BlockNumber(i), make_checksummed_block(4096, 0xEE));
        }
        // Corrupt block 50
        let mut bad = make_checksummed_block(4096, 0xEE);
        bad[500] ^= 0x80;
        dev.write(BlockNumber(50), bad);

        // Scrub only blocks 0..10 — should be clean
        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_range(&cx, BlockNumber(0), 10)
            .expect("scrub should succeed");
        assert!(report.is_clean());
        assert_eq!(report.blocks_scanned, 10);

        // Scrub blocks 45..55 — should find block 50
        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_range(&cx, BlockNumber(45), 10)
            .expect("scrub should succeed");
        assert_eq!(report.blocks_corrupt, 1);
        assert_eq!(report.findings[0].block, BlockNumber(50));
    }

    #[test]
    fn scrub_range_clamps_to_device_size() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 8);
        for i in 0..8 {
            dev.write(BlockNumber(i), make_checksummed_block(4096, 0xFF));
        }

        // Request more blocks than exist
        let report = Scrubber::new(&dev, &Crc32cBlockValidator)
            .scrub_range(&cx, BlockNumber(0), 1000)
            .expect("scrub should succeed");
        assert_eq!(report.blocks_scanned, 8);
        assert!(report.is_clean());
    }

    #[test]
    fn ext4_superblock_validator_accepts_valid_primary_superblock() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 8);
        let mut block0 = vec![0_u8; 4096];
        let sb = make_valid_ext4_superblock_region();
        block0[1024..2048].copy_from_slice(&sb);
        dev.write(BlockNumber(0), block0);

        let report = Scrubber::new(&dev, &Ext4SuperblockValidator::new(4096))
            .scrub_all(&cx)
            .expect("scrub should succeed");
        assert!(report.is_clean());
    }

    #[test]
    fn ext4_superblock_validator_detects_checksum_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 8);
        let mut block0 = vec![0_u8; 4096];
        let mut sb = make_valid_ext4_superblock_region();
        sb[0x50] ^= 0x01; // invalidate checksum without touching checksum field
        block0[1024..2048].copy_from_slice(&sb);
        dev.write(BlockNumber(0), block0);

        let report = Scrubber::new(&dev, &Ext4SuperblockValidator::new(4096))
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_corrupt, 1);
        assert_eq!(report.findings.len(), 1);
        let finding = &report.findings[0];
        assert_eq!(finding.block, BlockNumber(0));
        assert_eq!(finding.kind, CorruptionKind::ChecksumMismatch);
    }

    #[test]
    fn ext4_superblock_validator_handles_1k_block_layout() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(1024, 4);
        let mut sb = make_valid_ext4_superblock_region();
        sb[0x18..0x1C].copy_from_slice(&0_u32.to_le_bytes()); // log_block_size=0 -> 1KiB
        sb[0x1C..0x20].copy_from_slice(&0_u32.to_le_bytes()); // log_cluster_size=0 -> 1KiB
        sb[0x14..0x18].copy_from_slice(&1_u32.to_le_bytes()); // first_data_block=1 for 1KiB
        sb[0x20..0x24].copy_from_slice(&8192_u32.to_le_bytes()); // blocks_per_group
        sb[0x24..0x28].copy_from_slice(&8192_u32.to_le_bytes()); // clusters_per_group
        sb[0x04..0x08].copy_from_slice(&8193_u32.to_le_bytes()); // blocks_count_lo
        let checksum = ffs_ondisk::ext4_chksum(!0_u32, &sb[..EXT4_SB_CHECKSUM_OFFSET]);
        sb[EXT4_SB_CHECKSUM_OFFSET..EXT4_SB_CHECKSUM_OFFSET + 4]
            .copy_from_slice(&checksum.to_le_bytes());
        dev.write(BlockNumber(1), sb.to_vec());

        let report = Scrubber::new(&dev, &Ext4SuperblockValidator::new(1024))
            .scrub_all(&cx)
            .expect("scrub should succeed");
        assert!(
            report.is_clean(),
            "unexpected findings: {:?}",
            report.findings
        );
    }

    #[test]
    fn btrfs_superblock_validator_accepts_valid_superblock() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(16384, 8);
        let mut block4 = vec![0_u8; 16384];
        let sb = make_valid_btrfs_superblock_region();
        block4[..BTRFS_SUPER_INFO_SIZE].copy_from_slice(&sb);
        dev.write(BlockNumber(4), block4);

        let report = Scrubber::new(&dev, &BtrfsSuperblockValidator::new(16384))
            .scrub_all(&cx)
            .expect("scrub should succeed");
        assert!(report.is_clean());
    }

    #[test]
    fn btrfs_superblock_validator_detects_checksum_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(16384, 8);
        let mut block4 = vec![0_u8; 16384];
        let mut sb = make_valid_btrfs_superblock_region();
        sb[0x50] ^= 0x01;
        block4[..BTRFS_SUPER_INFO_SIZE].copy_from_slice(&sb);
        dev.write(BlockNumber(4), block4);

        let report = Scrubber::new(&dev, &BtrfsSuperblockValidator::new(16384))
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_corrupt, 1);
        assert_eq!(report.findings.len(), 1);
        let finding = &report.findings[0];
        assert_eq!(finding.block, BlockNumber(4));
        assert_eq!(finding.kind, CorruptionKind::ChecksumMismatch);
    }

    #[test]
    fn btrfs_tree_block_validator_accepts_valid_tree_block() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(16384, 8);

        let fsid = [0x11_u8; 16];
        let mut sb = make_valid_btrfs_superblock_region();
        sb[0x20..0x30].copy_from_slice(&fsid);
        let sb_csum = crc32c::crc32c(&sb[0x20..BTRFS_SUPER_INFO_SIZE]);
        sb[0..4].copy_from_slice(&sb_csum.to_le_bytes());

        let mut block4 = vec![0_u8; 16384];
        block4[..BTRFS_SUPER_INFO_SIZE].copy_from_slice(&sb);
        dev.write(BlockNumber(4), block4);

        let block5_bytenr = 5_u64 * 16384_u64;
        let tree_block = make_valid_btrfs_tree_block(16384, fsid, block5_bytenr);
        dev.write(BlockNumber(5), tree_block);

        let report = Scrubber::new(
            &dev,
            &CompositeValidator::new(vec![
                Box::new(BtrfsSuperblockValidator::new(16384)),
                Box::new(BtrfsTreeBlockValidator::new(16384, fsid, 0)),
            ]),
        )
        .scrub_all(&cx)
        .expect("scrub should succeed");
        assert!(
            report.is_clean(),
            "unexpected findings: {:?}",
            report.findings
        );
    }

    #[test]
    fn btrfs_tree_block_validator_detects_checksum_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(16384, 8);
        let fsid = [0x22_u8; 16];

        let block5_bytenr = 5_u64 * 16384_u64;
        let mut tree_block = make_valid_btrfs_tree_block(16384, fsid, block5_bytenr);
        tree_block[0x80] ^= 0x01;
        dev.write(BlockNumber(5), tree_block);

        let report = Scrubber::new(&dev, &BtrfsTreeBlockValidator::new(16384, fsid, 0))
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_corrupt, 1);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].block, BlockNumber(5));
        assert_eq!(report.findings[0].kind, CorruptionKind::ChecksumMismatch);
    }

    #[test]
    fn btrfs_tree_block_validator_skips_foreign_fsid_blocks() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(16384, 8);

        let block5_bytenr = 5_u64 * 16384_u64;
        let mut tree_block = make_valid_btrfs_tree_block(16384, [0x33_u8; 16], block5_bytenr);
        tree_block[0x80] ^= 0x01; // invalidate checksum, but fsid won't match validator
        dev.write(BlockNumber(5), tree_block);

        let report = Scrubber::new(&dev, &BtrfsTreeBlockValidator::new(16384, [0x44_u8; 16], 0))
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert!(report.is_clean());
    }

    #[test]
    fn zero_check_validator_detects_zeroed_blocks() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 4);
        // Block 0 and 2 have data, block 1 and 3 are all-zeroes (default)
        dev.write(BlockNumber(0), vec![1; 4096]);
        dev.write(BlockNumber(2), vec![2; 4096]);

        let report = Scrubber::new(&dev, &ZeroCheckValidator)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        assert_eq!(report.blocks_corrupt, 2);
        let zeroed: Vec<u64> = report.findings.iter().map(|f| f.block.0).collect();
        assert_eq!(zeroed, vec![1, 3]);
        assert!(
            report
                .findings
                .iter()
                .all(|f| f.kind == CorruptionKind::UnexpectedZeroes)
        );
    }

    #[test]
    fn composite_validator_merges_findings() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 4);
        // Block 0: all zeroes (caught by zero-check) + bad CRC (caught by crc32c check)
        // Block 1: valid checksummed, non-zero (clean for both)
        dev.write(BlockNumber(1), make_checksummed_block(4096, 0x42));

        let composite = CompositeValidator::new(vec![
            Box::new(ZeroCheckValidator),
            Box::new(Crc32cBlockValidator),
        ]);

        let report = Scrubber::new(&dev, &composite)
            .scrub_all(&cx)
            .expect("scrub should succeed");

        // Block 0 should have findings from both validators (zeroed + bad crc)
        let block_0_findings: Vec<&ScrubFinding> = report
            .findings
            .iter()
            .filter(|f| f.block == BlockNumber(0))
            .collect();
        assert!(block_0_findings.len() >= 2);

        let kinds: Vec<CorruptionKind> = block_0_findings.iter().map(|f| f.kind).collect();
        assert!(kinds.contains(&CorruptionKind::UnexpectedZeroes));
        assert!(kinds.contains(&CorruptionKind::ChecksumMismatch));
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn count_at_severity_filters_correctly() {
        let report = ScrubReport {
            findings: vec![
                ScrubFinding {
                    block: BlockNumber(0),
                    kind: CorruptionKind::UnexpectedZeroes,
                    severity: Severity::Warning,
                    detail: "zeroed".into(),
                },
                ScrubFinding {
                    block: BlockNumber(1),
                    kind: CorruptionKind::ChecksumMismatch,
                    severity: Severity::Error,
                    detail: "bad crc".into(),
                },
                ScrubFinding {
                    block: BlockNumber(2),
                    kind: CorruptionKind::BadMagic,
                    severity: Severity::Critical,
                    detail: "bad magic".into(),
                },
            ],
            blocks_scanned: 10,
            blocks_corrupt: 3,
            blocks_io_error: 0,
        };

        assert_eq!(report.count_at_severity(Severity::Info), 3);
        assert_eq!(report.count_at_severity(Severity::Warning), 3);
        assert_eq!(report.count_at_severity(Severity::Error), 2);
        assert_eq!(report.count_at_severity(Severity::Critical), 1);
    }

    #[test]
    fn display_formatting() {
        let finding = ScrubFinding {
            block: BlockNumber(42),
            kind: CorruptionKind::ChecksumMismatch,
            severity: Severity::Error,
            detail: "CRC32C expected 0xDEAD, got 0xBEEF".into(),
        };
        let s = finding.to_string();
        assert!(s.contains("42"));
        assert!(s.contains("error"));
        assert!(s.contains("checksum_mismatch"));
        assert!(s.contains("CRC32C expected"));

        let report = ScrubReport {
            findings: vec![finding],
            blocks_scanned: 100,
            blocks_corrupt: 1,
            blocks_io_error: 0,
        };
        let s = report.to_string();
        assert!(s.contains("100 blocks"));
        assert!(s.contains("1 corrupt"));
    }

    // ── Edge-case hardening tests ──────────────────────────────────────

    #[test]
    fn corruption_kind_display_all_variants() {
        assert_eq!(
            CorruptionKind::ChecksumMismatch.to_string(),
            "checksum_mismatch"
        );
        assert_eq!(
            CorruptionKind::StructuralInvariant.to_string(),
            "structural_invariant"
        );
        assert_eq!(CorruptionKind::BadMagic.to_string(), "bad_magic");
        assert_eq!(
            CorruptionKind::UnexpectedZeroes.to_string(),
            "unexpected_zeroes"
        );
        assert_eq!(CorruptionKind::IoError.to_string(), "io_error");
    }

    #[test]
    fn severity_display_all_variants() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Error.to_string(), "error");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn scrub_report_empty_is_clean() {
        let report = ScrubReport {
            findings: vec![],
            blocks_scanned: 0,
            blocks_corrupt: 0,
            blocks_io_error: 0,
        };
        assert!(report.is_clean());
        assert_eq!(report.count_at_severity(Severity::Info), 0);
    }

    #[test]
    fn corruption_kind_hash_consistent() {
        use std::collections::HashSet;
        let set: HashSet<CorruptionKind> = [
            CorruptionKind::ChecksumMismatch,
            CorruptionKind::ChecksumMismatch,
            CorruptionKind::BadMagic,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn severity_hash_consistent() {
        use std::collections::HashSet;
        let set: HashSet<Severity> = [Severity::Info, Severity::Info, Severity::Critical]
            .into_iter()
            .collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn scrub_finding_display_includes_all_parts() {
        let finding = ScrubFinding {
            block: BlockNumber(999),
            kind: CorruptionKind::BadMagic,
            severity: Severity::Critical,
            detail: "expected 0xEF53".into(),
        };
        let s = finding.to_string();
        assert!(s.contains("999"));
        assert!(s.contains("critical"));
        assert!(s.contains("bad_magic"));
        assert!(s.contains("expected 0xEF53"));
    }

    #[test]
    fn scrub_report_display_with_io_errors() {
        let report = ScrubReport {
            findings: vec![],
            blocks_scanned: 50,
            blocks_corrupt: 0,
            blocks_io_error: 3,
        };
        let s = report.to_string();
        assert!(s.contains("50 blocks"));
        assert!(s.contains("3 io_error"));
    }

    #[test]
    fn severity_total_ordering_equality() {
        assert_eq!(
            Severity::Info.cmp(&Severity::Info),
            std::cmp::Ordering::Equal
        );
        assert!(Severity::Critical >= Severity::Info);
    }

    #[test]
    fn corruption_kind_copy_semantics() {
        let a = CorruptionKind::IoError;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    struct AlwaysCorruptValidator;
    impl BlockValidator for AlwaysCorruptValidator {
        fn validate(&self, _block: BlockNumber, _data: &BlockBuf) -> BlockVerdict {
            BlockVerdict::Corrupt(vec![(
                CorruptionKind::StructuralInvariant,
                Severity::Error,
                "always corrupt".to_owned(),
            )])
        }
    }

    #[test]
    fn scrub_does_not_panic_on_corrupted_data() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096, 8);

        // Write various garbage patterns
        dev.write(BlockNumber(0), vec![0xFF; 4096]); // all 0xFF
        dev.write(BlockNumber(1), vec![0x00; 4096]); // all zeroes
        dev.write(BlockNumber(2), vec![0xDE, 0xAD]); // short (but MemBlockDevice pads)
        dev.write(BlockNumber(3), (0..=255).cycle().take(4096).collect()); // ramp pattern

        let report = Scrubber::new(&dev, &AlwaysCorruptValidator)
            .scrub_all(&cx)
            .expect("scrub must not panic");
        assert_eq!(report.blocks_scanned, 8);
        assert_eq!(report.blocks_corrupt, 8);
        assert_eq!(report.findings.len(), 8);
    }
}
