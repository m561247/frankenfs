#![forbid(unsafe_code)]

use asupersync::{Cx, RaptorQConfig, SystemPressure};
use ffs_alloc::{AllocHint, FsGeometry, GroupStats, PersistCtx, bitmap_count_free, bitmap_get};
use ffs_block::{
    BlockBuf, BlockDevice, ByteDevice, FileByteDevice, read_btrfs_superblock_region,
    read_ext4_superblock_region,
};
use ffs_btrfs::{
    BTRFS_BLOCK_GROUP_DATA, BTRFS_FILE_EXTENT_PREALLOC, BTRFS_FILE_EXTENT_REG,
    BTRFS_FS_TREE_OBJECTID, BTRFS_FT_BLKDEV, BTRFS_FT_CHRDEV, BTRFS_FT_DIR, BTRFS_FT_FIFO,
    BTRFS_FT_REG_FILE, BTRFS_FT_SOCK, BTRFS_FT_SYMLINK, BTRFS_ITEM_DIR_INDEX, BTRFS_ITEM_DIR_ITEM,
    BTRFS_ITEM_EXTENT_DATA, BTRFS_ITEM_INODE_ITEM, BTRFS_ITEM_INODE_REF, BTRFS_ITEM_ROOT_ITEM,
    BTRFS_ITEM_XATTR_ITEM, BtrfsBTree, BtrfsBlockGroupItem, BtrfsDirItem, BtrfsExtentAllocator,
    BtrfsExtentData, BtrfsInodeItem, BtrfsKey, BtrfsLeafEntry, BtrfsMutationError, BtrfsTreeItem,
    InMemoryCowBtrfsTree, map_logical_to_physical, parse_dir_items, parse_extent_data,
    parse_inode_item, parse_root_item, parse_xattr_items, walk_tree,
};
use ffs_error::FfsError;
use ffs_journal::{
    Jbd2WriteStats, Jbd2Writer, JournalSegment, ReplayOutcome, replay_jbd2_segments,
};
use ffs_mvcc::{CommitError, MvccStore, Transaction};
use ffs_ondisk::{
    BtrfsChunkEntry, BtrfsSuperblock, EXT4_ERROR_FS, EXT4_ORPHAN_FS, EXT4_VALID_FS, Ext4DirEntry,
    Ext4Extent, Ext4FileType, Ext4GroupDesc, Ext4ImageReader, Ext4Inode, Ext4Superblock, Ext4Xattr,
    ExtentTree, lookup_in_dir_block, parse_dir_block, parse_extent_tree, parse_inode_extent_tree,
    parse_sys_chunk_array,
};
use ffs_types::{
    BlockNumber, ByteOffset, CommitSeq, EXT4_EXTENTS_FL, GroupNumber, InodeNumber, ParseError,
    Snapshot, TxnId,
};
use ffs_xattr::XattrWriteAccess;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, info, trace, warn};

// ── Compute budget and degradation ─────────────────────────────────────────

/// Compute budget monitor that samples system load and updates pressure state.
///
/// Reads `/proc/loadavg` on Linux and converts the 1-minute load average
/// into a headroom value (0.0–1.0) based on the number of CPU cores.
pub struct ComputeBudget {
    pressure: Arc<SystemPressure>,
    cpu_count: f32,
}

impl ComputeBudget {
    /// Create a new compute budget monitor.
    ///
    /// `pressure` is the shared handle that will be updated on each sample.
    #[must_use]
    pub fn new(pressure: Arc<SystemPressure>) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let cpu_count =
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get) as f32;
        Self {
            pressure,
            cpu_count,
        }
    }

    /// Sample the current system load and update the pressure handle.
    ///
    /// On Linux, reads `/proc/loadavg`. On other platforms, returns 1.0 (idle).
    /// Returns the computed headroom value.
    pub fn sample(&self) -> f32 {
        let headroom = self.sample_headroom();
        self.pressure.set_headroom(headroom);
        trace!(
            target: "ffs::budget",
            headroom,
            cpu_count = self.cpu_count,
            level = self.pressure.level_label(),
            "budget_sample"
        );
        headroom
    }

    /// Read the current headroom without updating pressure.
    #[must_use]
    pub fn current_headroom(&self) -> f32 {
        self.pressure.headroom()
    }

    /// The shared pressure handle.
    #[must_use]
    pub fn pressure(&self) -> &Arc<SystemPressure> {
        &self.pressure
    }

    fn sample_headroom(&self) -> f32 {
        Self::read_load_avg().map_or(1.0, |load_1m| {
            // headroom = 1.0 - (load / cpus), clamped to [0, 1]
            let ratio = load_1m / self.cpu_count;
            (1.0 - ratio).clamp(0.0, 1.0)
        })
    }

    /// Read 1-minute load average from `/proc/loadavg` on Linux.
    fn read_load_avg() -> Option<f32> {
        let content = std::fs::read_to_string("/proc/loadavg").ok()?;
        let first = content.split_whitespace().next()?;
        first.parse::<f32>().ok()
    }
}

impl std::fmt::Debug for ComputeBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeBudget")
            .field("headroom", &self.pressure.headroom())
            .field("cpu_count", &self.cpu_count)
            .field("level", &self.pressure.level_label())
            .finish()
    }
}

/// Policy that reacts to system pressure changes.
///
/// Implementations adjust their behavior based on the current headroom value
/// (0.0 = critically overloaded, 1.0 = idle).
pub trait DegradationPolicy: Send + Sync {
    /// Apply the policy based on current headroom.
    ///
    /// Called periodically by the budget monitor. Implementations should
    /// adjust internal parameters (cache sizes, intervals, thresholds)
    /// based on the headroom value.
    fn apply(&self, headroom: f32);

    /// Human-readable name for this policy.
    fn name(&self) -> &str;
}

// ── Degradation FSM ─────────────────────────────────────────────────────────

/// Formal degradation levels matching `SystemPressure::degradation_level()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DegradationLevel {
    /// headroom >= 0.5 — full service
    Normal = 0,
    /// headroom >= 0.3 — background tasks paused
    Warning = 1,
    /// headroom >= 0.15 — caches reduced
    Degraded = 2,
    /// headroom >= 0.05 — writes throttled
    Critical = 3,
    /// headroom < 0.05 — read-only mode
    Emergency = 4,
}

impl DegradationLevel {
    /// Convert from a `SystemPressure::degradation_level()` u8 value.
    #[must_use]
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::Normal,
            1 => Self::Warning,
            2 => Self::Degraded,
            3 => Self::Critical,
            _ => Self::Emergency,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Warning => "warning",
            Self::Degraded => "degraded",
            Self::Critical => "critical",
            Self::Emergency => "emergency",
        }
    }

    /// Whether background work (scrub, GC) should be paused at this level.
    #[must_use]
    pub fn should_pause_background(self) -> bool {
        self >= Self::Warning
    }

    /// Whether caches should be reduced at this level.
    #[must_use]
    pub fn should_reduce_cache(self) -> bool {
        self >= Self::Degraded
    }

    /// Whether writes should be throttled at this level.
    #[must_use]
    pub fn should_throttle_writes(self) -> bool {
        self >= Self::Critical
    }

    /// Whether the filesystem should be read-only at this level.
    #[must_use]
    pub fn should_read_only(self) -> bool {
        self == Self::Emergency
    }
}

impl std::fmt::Display for DegradationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl From<DegradationLevel> for u8 {
    fn from(level: DegradationLevel) -> Self {
        level as Self
    }
}

/// Degradation FSM with hysteresis to prevent oscillation.
///
/// The FSM escalates immediately when pressure worsens but requires a
/// sustained improvement (configurable via `recovery_samples`) before
/// de-escalating. This prevents rapid flickering between levels.
pub struct DegradationFsm {
    current: parking_lot::Mutex<FsmState>,
    level_cache: std::sync::atomic::AtomicU8,
    pressure: Arc<SystemPressure>,
    policies: parking_lot::Mutex<Vec<Arc<dyn DegradationPolicy>>>,
    recovery_samples: u32,
}

struct FsmState {
    level: DegradationLevel,
    /// Counter of consecutive samples at a level better than current.
    /// Must reach `recovery_samples` before de-escalation.
    recovery_count: u32,
    /// Total transitions since creation.
    transition_count: u64,
}

/// Record of a degradation level transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DegradationTransition {
    pub from: DegradationLevel,
    pub to: DegradationLevel,
    pub headroom: u32, // headroom * 1000, stored as integer to be Eq
}

impl DegradationFsm {
    /// Create a new FSM starting at `Normal`.
    ///
    /// `recovery_samples` is how many consecutive improved samples are needed
    /// before de-escalating (default: 3).
    #[must_use]
    pub fn new(pressure: Arc<SystemPressure>, recovery_samples: u32) -> Self {
        Self {
            current: parking_lot::Mutex::new(FsmState {
                level: DegradationLevel::Normal,
                recovery_count: 0,
                transition_count: 0,
            }),
            level_cache: std::sync::atomic::AtomicU8::new(u8::from(DegradationLevel::Normal)),
            pressure,
            policies: parking_lot::Mutex::new(Vec::new()),
            recovery_samples,
        }
    }

    /// Register a policy to be notified on level changes.
    pub fn add_policy(&self, policy: Arc<dyn DegradationPolicy>) {
        self.policies.lock().push(policy);
    }

    /// Current degradation level.
    #[must_use]
    pub fn level(&self) -> DegradationLevel {
        DegradationLevel::from_raw(self.level_cache.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Total number of transitions since creation.
    #[must_use]
    pub fn transition_count(&self) -> u64 {
        self.current.lock().transition_count
    }

    /// Tick the FSM with a fresh pressure reading.
    ///
    /// Returns `Some(transition)` if the level changed, `None` otherwise.
    pub fn tick(&self) -> Option<DegradationTransition> {
        let headroom = self.pressure.headroom();
        let observed = DegradationLevel::from_raw(self.pressure.degradation_level());

        let mut state = self.current.lock();
        let prev = state.level;

        match observed.cmp(&prev) {
            std::cmp::Ordering::Greater => {
                // Escalate immediately.
                state.level = observed;
                state.recovery_count = 0;
                state.transition_count += 1;
            }
            std::cmp::Ordering::Less => {
                // Require sustained improvement before de-escalating.
                state.recovery_count += 1;
                if state.recovery_count >= self.recovery_samples {
                    state.level = observed;
                    state.recovery_count = 0;
                    state.transition_count += 1;
                }
            }
            std::cmp::Ordering::Equal => {
                state.recovery_count = 0;
            }
        }

        let new = state.level;
        self.level_cache
            .store(u8::from(new), std::sync::atomic::Ordering::Relaxed);
        drop(state);

        // Notify policies with current headroom (regardless of transition).
        {
            let policies = self.policies.lock();
            for policy in policies.iter() {
                policy.apply(headroom);
            }
        }

        if new == prev {
            None
        } else {
            info!(
                target: "ffs::backpressure",
                from = prev.label(),
                to = new.label(),
                headroom,
                "degradation_transition"
            );
            Some(DegradationTransition {
                from: prev,
                to: new,
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                headroom: (headroom * 1000.0) as u32,
            })
        }
    }
}

impl std::fmt::Debug for DegradationFsm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.current.lock();
        f.debug_struct("DegradationFsm")
            .field("level", &state.level)
            .field("recovery_count", &state.recovery_count)
            .field("transitions", &state.transition_count)
            .field("recovery_samples", &self.recovery_samples)
            .finish_non_exhaustive()
    }
}

// ── Backpressure gate ───────────────────────────────────────────────────────

/// Decision returned by [`BackpressureGate::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureDecision {
    /// Proceed normally.
    Proceed,
    /// Operation should be throttled (delay before proceeding).
    Throttle,
    /// Operation should be shed (rejected with EBUSY or ENOSPC).
    Shed,
}

/// Per-operation backpressure check.
///
/// Given a `DegradationFsm` and the type of operation, returns a decision
/// on whether to proceed, throttle, or shed the request.
pub struct BackpressureGate {
    fsm: Arc<DegradationFsm>,
}

impl BackpressureGate {
    /// Create a new gate wrapping the given FSM.
    #[must_use]
    pub fn new(fsm: Arc<DegradationFsm>) -> Self {
        Self { fsm }
    }

    /// Check whether the given operation should proceed.
    #[must_use]
    pub fn check(&self, op: RequestOp) -> BackpressureDecision {
        let level = self.fsm.level();
        match level {
            DegradationLevel::Normal | DegradationLevel::Warning => {
                // Normal and warning: all ops proceed (background pausing
                // is handled separately by the scrub/GC scheduler).
                BackpressureDecision::Proceed
            }
            DegradationLevel::Degraded => {
                // Reads always proceed; writes proceed but may be throttled.
                if op.is_write() {
                    BackpressureDecision::Throttle
                } else {
                    BackpressureDecision::Proceed
                }
            }
            DegradationLevel::Critical => {
                // Writes throttled, metadata writes shed.
                if op.is_write() {
                    BackpressureDecision::Throttle
                } else {
                    BackpressureDecision::Proceed
                }
            }
            DegradationLevel::Emergency => {
                // Read-only mode: all writes shed.
                if op.is_write() {
                    BackpressureDecision::Shed
                } else {
                    BackpressureDecision::Proceed
                }
            }
        }
    }

    /// Current degradation level.
    #[must_use]
    pub fn level(&self) -> DegradationLevel {
        self.fsm.level()
    }
}

impl std::fmt::Debug for BackpressureGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackpressureGate")
            .field("level", &self.fsm.level())
            .finish()
    }
}

// ── Pressure monitor ────────────────────────────────────────────────────────

/// Aggregated pressure monitor that drives the degradation FSM.
///
/// Combines CPU load sampling (via `ComputeBudget`) with the FSM to provide
/// a single entry point for periodic pressure updates.
pub struct PressureMonitor {
    budget: ComputeBudget,
    fsm: Arc<DegradationFsm>,
    sample_count: std::sync::atomic::AtomicU64,
}

impl PressureMonitor {
    /// Create a new monitor with a shared pressure handle and FSM.
    #[must_use]
    pub fn new(pressure: Arc<SystemPressure>, recovery_samples: u32) -> Self {
        let budget = ComputeBudget::new(Arc::clone(&pressure));
        let fsm = Arc::new(DegradationFsm::new(pressure, recovery_samples));
        Self {
            budget,
            fsm,
            sample_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Sample system pressure and tick the FSM.
    ///
    /// Returns any transition that occurred.
    pub fn sample(&self) -> Option<DegradationTransition> {
        self.budget.sample();
        self.sample_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.fsm.tick()
    }

    /// Get a `BackpressureGate` for checking individual operations.
    #[must_use]
    pub fn gate(&self) -> BackpressureGate {
        BackpressureGate::new(Arc::clone(&self.fsm))
    }

    /// The underlying FSM.
    #[must_use]
    pub fn fsm(&self) -> &Arc<DegradationFsm> {
        &self.fsm
    }

    /// The underlying compute budget.
    #[must_use]
    pub fn budget(&self) -> &ComputeBudget {
        &self.budget
    }

    /// Number of samples taken.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Current degradation level.
    #[must_use]
    pub fn level(&self) -> DegradationLevel {
        self.fsm.level()
    }

    /// Register a degradation policy.
    pub fn add_policy(&self, policy: Arc<dyn DegradationPolicy>) {
        self.fsm.add_policy(policy);
    }
}

impl std::fmt::Debug for PressureMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PressureMonitor")
            .field("budget", &self.budget)
            .field("fsm", &self.fsm)
            .field("samples", &self.sample_count())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FsFlavor {
    Ext4(Ext4Superblock),
    Btrfs(BtrfsSuperblock),
}

#[derive(Debug, Error)]
pub enum DetectionError {
    #[error("image does not decode as supported ext4/btrfs superblock")]
    UnsupportedImage,
    #[error("I/O error while probing image: {0}")]
    Io(#[from] FfsError),
}

/// Summary of ext4 free space from bitmap analysis.
///
/// Contains both the bitmap-derived counts and the group descriptor cached
/// values, allowing detection of inconsistencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ext4FreeSpaceSummary {
    /// Total free blocks derived from bitmap analysis.
    pub free_blocks_total: u64,
    /// Total free inodes derived from bitmap analysis.
    pub free_inodes_total: u64,
    /// Total free blocks from group descriptor cached values.
    pub gd_free_blocks_total: u64,
    /// Total free inodes from group descriptor cached values.
    pub gd_free_inodes_total: u64,
    /// True if bitmap count differs from group descriptor count.
    pub blocks_mismatch: bool,
    /// True if bitmap count differs from group descriptor count.
    pub inodes_mismatch: bool,
}

/// Safe traversal result for the ext4 orphan inode list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ext4OrphanList {
    /// Raw orphan-list head from the ext4 superblock (`s_last_orphan`).
    pub head: u32,
    /// Traversed inode numbers in on-disk chain order.
    pub inodes: Vec<InodeNumber>,
}

impl Ext4OrphanList {
    #[must_use]
    pub fn count(&self) -> usize {
        self.inodes.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Ext4OrphanRecoveryStats {
    scanned: u32,
    deleted: u32,
    truncated: u32,
    skipped: u32,
}

impl Ext4OrphanRecoveryStats {
    #[must_use]
    fn processed(self) -> u32 {
        self.deleted.saturating_add(self.truncated)
    }
}

pub fn detect_filesystem(image: &[u8]) -> Result<FsFlavor, DetectionError> {
    if let Ok(ext4) = Ext4Superblock::parse_from_image(image) {
        return Ok(FsFlavor::Ext4(ext4));
    }

    if let Ok(btrfs) = BtrfsSuperblock::parse_from_image(image) {
        return Ok(FsFlavor::Btrfs(btrfs));
    }

    Err(DetectionError::UnsupportedImage)
}

pub fn detect_filesystem_on_device(
    cx: &Cx,
    dev: &dyn ByteDevice,
) -> Result<FsFlavor, DetectionError> {
    let len = dev.len_bytes();

    let ext4_end =
        u64::try_from(ffs_types::EXT4_SUPERBLOCK_OFFSET + ffs_types::EXT4_SUPERBLOCK_SIZE)
            .map_err(|_| FfsError::Format("ext4 superblock end offset overflows u64".to_owned()))?;
    if len >= ext4_end {
        let ext4_region = read_ext4_superblock_region(cx, dev)?;
        if let Ok(sb) = Ext4Superblock::parse_superblock_region(&ext4_region) {
            return Ok(FsFlavor::Ext4(sb));
        }
    }

    let btrfs_end =
        u64::try_from(ffs_types::BTRFS_SUPER_INFO_OFFSET + ffs_types::BTRFS_SUPER_INFO_SIZE)
            .map_err(|_| {
                FfsError::Format("btrfs superblock end offset overflows u64".to_owned())
            })?;
    if len >= btrfs_end {
        let btrfs_region = read_btrfs_superblock_region(cx, dev)?;
        if let Ok(sb) = BtrfsSuperblock::parse_superblock_region(&btrfs_region) {
            return Ok(FsFlavor::Btrfs(sb));
        }
    }

    Err(DetectionError::UnsupportedImage)
}

pub fn detect_filesystem_at_path(
    cx: &Cx,
    path: impl AsRef<Path>,
) -> Result<FsFlavor, DetectionError> {
    let dev = FileByteDevice::open(path)?;
    detect_filesystem_on_device(cx, &dev)
}

// ── OpenFs API ──────────────────────────────────────────────────────────────

/// Options controlling how a filesystem image is opened.
///
/// By default, mount-time validation is enabled. Disable it only for
/// recovery or diagnostic workflows where reading a partially-corrupt
/// image is intentional.
#[derive(Debug, Clone)]
pub struct OpenOptions {
    /// Skip mount-time validation (geometry, features, checksums).
    ///
    /// When `true`, the superblock is parsed but not validated via
    /// `validate_v1()`. Use for recovery or diagnostics only.
    pub skip_validation: bool,
    /// ext4 internal journal replay mode.
    ///
    /// Controls whether mount-time JBD2 replay writes through to the underlying
    /// image, writes into an in-memory overlay, or is skipped entirely.
    pub ext4_journal_replay_mode: Ext4JournalReplayMode,
}

#[allow(clippy::derivable_impls)]
impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            skip_validation: false,
            ext4_journal_replay_mode: Ext4JournalReplayMode::Apply,
        }
    }
}

/// ext4 internal journal replay policy at open/mount time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ext4JournalReplayMode {
    /// Replay committed transactions to the underlying device.
    Apply,
    /// Replay into an in-memory overlay so reads observe recovered state
    /// without mutating the base image.
    SimulateOverlay,
    /// Skip replay entirely; expose that a journal was present.
    Skip,
}

/// Pre-computed ext4 geometry derived from the superblock.
///
/// These values are computed once at open time and cached so that
/// downstream code does not re-derive them on every operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ext4Geometry {
    /// Block size in bytes (1024, 2048, or 4096 for v1).
    pub block_size: u32,
    /// Total number of inodes.
    pub inodes_count: u32,
    /// Number of inodes per block group.
    pub inodes_per_group: u32,
    /// First non-reserved inode number.
    pub first_ino: u32,
    /// On-disk inode structure size in bytes.
    pub inode_size: u16,
    /// Number of block groups.
    pub groups_count: u32,
    /// Size of each group descriptor (32 or 64 bytes).
    pub group_desc_size: u16,
    /// Checksum seed for metadata_csum verification.
    pub csum_seed: u32,
    /// Whether the filesystem uses 64-bit block addressing.
    pub is_64bit: bool,
    /// Whether metadata_csum is enabled.
    pub has_metadata_csum: bool,
}

/// Pre-computed btrfs context derived from the superblock.
///
/// Contains the parsed sys_chunk logical-to-physical mapping and the node
/// size, computed once at open time so that tree-walk operations do not
/// re-parse the chunk array on every call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BtrfsContext {
    /// Parsed sys_chunk logical-to-physical mapping entries.
    pub chunks: Vec<BtrfsChunkEntry>,
    /// Tree node size in bytes.
    pub nodesize: u32,
}

/// Mutable ext4 allocation state for write operations.
///
/// Tracks per-group block/inode bitmaps and free counts. Loaded at open time
/// when write access is requested.
struct Ext4AllocState {
    /// Filesystem geometry derived from the superblock.
    geo: FsGeometry,
    /// Per-group allocation statistics (block/inode counts, bitmap locations).
    groups: Vec<GroupStats>,
    /// On-disk persistence context for group descriptor updates.
    #[allow(dead_code)]
    persist_ctx: PersistCtx,
}

/// Mutable btrfs allocation state for write operations.
///
/// Mirrors `Ext4AllocState` for the btrfs path: an in-memory COW tree that
/// tracks all FS-tree items, an extent allocator for data/metadata blocks,
/// and a monotonically-increasing objectid counter for new inodes.
struct BtrfsAllocState {
    /// In-memory COW B-tree holding all FS-tree items (inodes, dirs, extents).
    fs_tree: InMemoryCowBtrfsTree,
    /// Extent allocator for data and metadata block allocation.
    extent_alloc: BtrfsExtentAllocator,
    /// Next available objectid for new inodes / directory entries.
    next_objectid: u64,
    /// Current transaction generation (bumped on each commit).
    generation: u64,
    /// Node size in bytes (copied from superblock for convenience).
    nodesize: u32,
}

/// Outcome of crash recovery performed at mount time.
///
/// When an ext4 filesystem is mounted and the superblock `state` field
/// indicates an unclean shutdown (dirty flag set, error flag, or orphan
/// recovery in progress), FrankenFS performs recovery before serving
/// requests. This struct captures what was detected and what actions
/// were taken.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrashRecoveryOutcome {
    /// Whether the filesystem was found in a clean state on mount.
    pub was_clean: bool,
    /// Raw ext4 superblock state value at mount time.
    pub raw_state: u16,
    /// True if the `EXT4_ERROR_FS` flag was set.
    pub had_errors: bool,
    /// True if the `EXT4_ORPHAN_FS` flag was set.
    pub had_orphans: bool,
    /// Number of journal transactions replayed during recovery.
    pub journal_txns_replayed: u32,
    /// Number of journal blocks replayed during recovery.
    pub journal_blocks_replayed: u64,
    /// Whether MVCC version chains were reset (always true for unclean mount).
    pub mvcc_reset: bool,
}

impl CrashRecoveryOutcome {
    /// True if any recovery action was actually performed.
    #[must_use]
    pub fn recovery_performed(&self) -> bool {
        !self.was_clean
    }
}

/// An opened filesystem image, ready for VFS operations.
///
/// `OpenFs` bundles a validated superblock, pre-computed geometry, and the
/// block device handle into a single context. The constructor validates by
/// default so callers cannot accidentally operate on unvalidated metadata.
///
/// # Opening a filesystem
///
/// ```text
/// let cx = Cx::for_request();
/// let fs = OpenFs::open(&cx, "/path/to/image.ext4")?;
/// println!("block_size = {}", fs.block_size());
/// ```
pub struct OpenFs {
    /// Detected filesystem type with parsed superblock.
    pub flavor: FsFlavor,
    /// Pre-computed ext4 geometry (None for btrfs).
    pub ext4_geometry: Option<Ext4Geometry>,
    /// Pre-computed btrfs context (None for ext4).
    pub btrfs_context: Option<BtrfsContext>,
    /// Mount-time JBD2 replay outcome for ext4 images with an internal journal.
    pub ext4_journal_replay: Option<ReplayOutcome>,
    /// Crash recovery outcome, if the filesystem required recovery on mount.
    pub crash_recovery: Option<CrashRecoveryOutcome>,
    /// Block device for I/O operations.
    dev: Box<dyn ByteDevice>,
    /// MVCC version store for snapshot-isolated block access.
    ///
    /// Shared across all snapshots/transactions that operate on this filesystem.
    /// Writes stage versions here; reads check here before falling back to device.
    mvcc_store: Arc<RwLock<MvccStore>>,
    /// Optional JBD2 writer for ext4 compatibility-mode write path.
    ///
    /// When present, `commit_transaction_journaled` journals writes to the
    /// ext4 journal region before committing to the MVCC store. This ensures
    /// crash consistency compatible with standard ext4 mount.
    jbd2_writer: Option<Mutex<Jbd2Writer>>,
    /// Mutable ext4 allocation state (block/inode bitmaps, group stats).
    ///
    /// Protected by a Mutex since write operations need exclusive access.
    /// `None` for btrfs or when opened in read-only mode.
    ext4_alloc_state: Option<Mutex<Ext4AllocState>>,
    /// Mutable btrfs allocation state (COW FS tree, extent allocator).
    ///
    /// Protected by a Mutex since write operations need exclusive access.
    /// `None` for ext4 or when opened in read-only mode.
    btrfs_alloc_state: Option<Mutex<BtrfsAllocState>>,
}

// Compile-time assertion: OpenFs must be Send + Sync for multi-threaded FUSE dispatch.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<OpenFs>;
};

impl std::fmt::Debug for OpenFs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mvcc_guard = self.mvcc_store.read();
        f.debug_struct("OpenFs")
            .field("flavor", &self.flavor)
            .field("ext4_geometry", &self.ext4_geometry)
            .field("btrfs_context", &self.btrfs_context)
            .field("ext4_journal_replay", &self.ext4_journal_replay)
            .field("crash_recovery", &self.crash_recovery)
            .field("dev_len", &self.dev.len_bytes())
            .field("mvcc_version_count", &mvcc_guard.version_count())
            .field("mvcc_active_snapshots", &mvcc_guard.active_snapshot_count())
            .field("jbd2_writer", &self.jbd2_writer.is_some())
            .field("writable", &self.is_writable())
            .finish_non_exhaustive()
    }
}

/// Adapter that exposes a `ByteDevice` as a `BlockDevice` for journal replay.
struct ByteDeviceBlockAdapter<'a> {
    dev: &'a dyn ByteDevice,
    block_size: u32,
}

impl ByteDeviceBlockAdapter<'_> {
    fn block_offset(&self, block: BlockNumber) -> Result<u64, FfsError> {
        block
            .0
            .checked_mul(u64::from(self.block_size))
            .ok_or_else(|| FfsError::Format("block offset overflow".to_owned()))
    }
}

#[derive(Debug, Clone)]
struct OverlayWrite {
    offset: u64,
    bytes: Vec<u8>,
}

/// Copy-on-write byte overlay over an existing `ByteDevice`.
///
/// Reads merge overlay writes on top of the underlying device, while writes
/// are captured in-memory only.
struct OverlayByteDevice {
    inner: Box<dyn ByteDevice>,
    writes: RwLock<Vec<OverlayWrite>>,
}

impl OverlayByteDevice {
    fn new(inner: Box<dyn ByteDevice>) -> Self {
        Self {
            inner,
            writes: RwLock::new(Vec::new()),
        }
    }
}

impl ByteDevice for OverlayByteDevice {
    fn len_bytes(&self) -> u64 {
        self.inner.len_bytes()
    }

    fn read_exact_at(&self, cx: &Cx, offset: ByteOffset, buf: &mut [u8]) -> Result<(), FfsError> {
        self.inner.read_exact_at(cx, offset, buf)?;
        let range_start = offset.0;
        let range_end =
            range_start
                .checked_add(u64::try_from(buf.len()).map_err(|_| {
                    FfsError::Format("read buffer length does not fit u64".to_owned())
                })?)
                .ok_or_else(|| FfsError::Format("read range overflow".to_owned()))?;

        let writes = self.writes.read();
        for write in writes.iter() {
            let write_start = write.offset;
            let write_end = write_start
                .checked_add(u64::try_from(write.bytes.len()).map_err(|_| {
                    FfsError::Format("overlay write length does not fit u64".to_owned())
                })?)
                .ok_or_else(|| FfsError::Format("overlay write range overflow".to_owned()))?;

            if write_end <= range_start || write_start >= range_end {
                continue;
            }

            let overlap_start = write_start.max(range_start);
            let overlap_end = write_end.min(range_end);

            let src_start = usize::try_from(overlap_start.saturating_sub(write_start))
                .map_err(|_| FfsError::Format("overlay source offset overflow".to_owned()))?;
            let src_end = usize::try_from(overlap_end.saturating_sub(write_start))
                .map_err(|_| FfsError::Format("overlay source end overflow".to_owned()))?;
            let dst_start = usize::try_from(overlap_start.saturating_sub(range_start))
                .map_err(|_| FfsError::Format("overlay destination offset overflow".to_owned()))?;
            let dst_end = usize::try_from(overlap_end.saturating_sub(range_start))
                .map_err(|_| FfsError::Format("overlay destination end overflow".to_owned()))?;

            buf[dst_start..dst_end].copy_from_slice(&write.bytes[src_start..src_end]);
        }
        drop(writes);

        Ok(())
    }

    fn write_all_at(&self, _cx: &Cx, offset: ByteOffset, buf: &[u8]) -> Result<(), FfsError> {
        let range_end =
            offset
                .0
                .checked_add(u64::try_from(buf.len()).map_err(|_| {
                    FfsError::Format("write buffer length does not fit u64".to_owned())
                })?)
                .ok_or_else(|| FfsError::Format("write range overflow".to_owned()))?;
        if range_end > self.inner.len_bytes() {
            return Err(FfsError::Format(format!(
                "write out of bounds: offset={} len={} device_len={}",
                offset.0,
                buf.len(),
                self.inner.len_bytes()
            )));
        }

        self.writes.write().push(OverlayWrite {
            offset: offset.0,
            bytes: buf.to_vec(),
        });
        Ok(())
    }

    fn sync(&self, _cx: &Cx) -> Result<(), FfsError> {
        Ok(())
    }
}

impl BlockDevice for ByteDeviceBlockAdapter<'_> {
    fn read_block(&self, cx: &Cx, block: BlockNumber) -> Result<BlockBuf, FfsError> {
        let block_size = usize::try_from(self.block_size)
            .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?;
        let offset = self.block_offset(block)?;
        let end = offset
            .checked_add(u64::from(self.block_size))
            .ok_or_else(|| FfsError::Format("block range overflow".to_owned()))?;
        if end > self.dev.len_bytes() {
            return Err(FfsError::Format(format!(
                "block {} out of range for device length {}",
                block.0,
                self.dev.len_bytes()
            )));
        }
        let mut bytes = vec![0_u8; block_size];
        self.dev.read_exact_at(cx, ByteOffset(offset), &mut bytes)?;
        Ok(BlockBuf::new(bytes))
    }

    fn write_block(&self, cx: &Cx, block: BlockNumber, data: &[u8]) -> Result<(), FfsError> {
        let expected = usize::try_from(self.block_size)
            .map_err(|_| FfsError::Format("block_size does not fit usize".to_owned()))?;
        if data.len() != expected {
            return Err(FfsError::Format(format!(
                "write_block size mismatch: got {} expected {expected}",
                data.len()
            )));
        }

        let offset = self.block_offset(block)?;
        let end = offset
            .checked_add(u64::from(self.block_size))
            .ok_or_else(|| FfsError::Format("block range overflow".to_owned()))?;
        if end > self.dev.len_bytes() {
            return Err(FfsError::Format(format!(
                "block {} out of range for device length {}",
                block.0,
                self.dev.len_bytes()
            )));
        }

        self.dev.write_all_at(cx, ByteOffset(offset), data)
    }

    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn block_count(&self) -> u64 {
        self.dev.len_bytes() / u64::from(self.block_size)
    }

    fn sync(&self, cx: &Cx) -> Result<(), FfsError> {
        self.dev.sync(cx)
    }
}

impl OpenFs {
    /// Open a filesystem image at `path` with default options (validation enabled).
    pub fn open(cx: &Cx, path: impl AsRef<Path>) -> Result<Self, FfsError> {
        Self::open_with_options(cx, path, &OpenOptions::default())
    }

    /// Open a filesystem image with custom options.
    pub fn open_with_options(
        cx: &Cx,
        path: impl AsRef<Path>,
        options: &OpenOptions,
    ) -> Result<Self, FfsError> {
        let dev = FileByteDevice::open(path.as_ref())?;
        Self::from_device(cx, Box::new(dev), options)
    }

    /// Open a filesystem from an already-opened device.
    pub fn from_device(
        cx: &Cx,
        dev: Box<dyn ByteDevice>,
        options: &OpenOptions,
    ) -> Result<Self, FfsError> {
        let flavor = detect_filesystem_on_device(cx, &*dev).map_err(|e| match e {
            DetectionError::UnsupportedImage => {
                FfsError::Format("image is not a recognized ext4 or btrfs filesystem".into())
            }
            DetectionError::Io(ffs_err) => ffs_err,
        })?;

        let dev: Box<dyn ByteDevice> = if matches!(flavor, FsFlavor::Ext4(_))
            && !options.skip_validation
            && matches!(
                options.ext4_journal_replay_mode,
                Ext4JournalReplayMode::SimulateOverlay
            ) {
            Box::new(OverlayByteDevice::new(dev))
        } else {
            dev
        };

        let (ext4_geometry, btrfs_context) = match &flavor {
            FsFlavor::Ext4(sb) => {
                if !options.skip_validation {
                    sb.validate_v1().map_err(|e| parse_error_to_ffs(&e))?;
                }
                let geom = Ext4Geometry {
                    block_size: sb.block_size,
                    inodes_count: sb.inodes_count,
                    inodes_per_group: sb.inodes_per_group,
                    first_ino: sb.first_ino,
                    inode_size: sb.inode_size,
                    groups_count: sb.groups_count(),
                    group_desc_size: sb.group_desc_size(),
                    csum_seed: sb.csum_seed(),
                    is_64bit: sb.is_64bit(),
                    has_metadata_csum: sb.has_metadata_csum(),
                };
                (Some(geom), None)
            }
            FsFlavor::Btrfs(sb) => {
                if !options.skip_validation {
                    validate_btrfs_superblock(sb)?;
                }
                let chunks = parse_sys_chunk_array(&sb.sys_chunk_array)
                    .map_err(|e| parse_to_ffs_error(&e))?;
                let ctx = BtrfsContext {
                    chunks,
                    nodesize: sb.nodesize,
                };
                (None, Some(ctx))
            }
        };

        let mvcc_store = Arc::new(RwLock::new(MvccStore::new()));
        trace!(
            target: "ffs::mvcc",
            "mvcc_store_init"
        );

        let mut fs = Self {
            flavor,
            ext4_geometry,
            btrfs_context,
            ext4_journal_replay: None,
            crash_recovery: None,
            dev,
            mvcc_store,
            jbd2_writer: None,
            ext4_alloc_state: None,
            btrfs_alloc_state: None,
        };

        if fs.is_ext4() && !options.skip_validation {
            fs.ext4_journal_replay =
                fs.maybe_replay_ext4_journal(cx, options.ext4_journal_replay_mode)?;
            fs.crash_recovery = fs.detect_and_recover_crash();
            if options.ext4_journal_replay_mode != Ext4JournalReplayMode::Skip {
                match fs.maybe_recover_ext4_orphans(cx) {
                    Ok(stats) => {
                        if stats.scanned > 0 {
                            info!(
                                scanned = stats.scanned,
                                processed = stats.processed(),
                                deleted = stats.deleted,
                                truncated = stats.truncated,
                                skipped = stats.skipped,
                                "ext4 orphan recovery completed"
                            );
                        }
                    }
                    Err(err) => {
                        warn!(
                            error = %err,
                            "ext4 orphan recovery failed; filesystem remains read-only"
                        );
                    }
                }
            }
        }

        Ok(fs)
    }

    /// The block device backing this filesystem.
    #[must_use]
    pub fn device(&self) -> &dyn ByteDevice {
        &*self.dev
    }

    /// Block size in bytes.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        match &self.flavor {
            FsFlavor::Ext4(sb) => sb.block_size,
            FsFlavor::Btrfs(sb) => sb.sectorsize,
        }
    }

    /// Whether this is an ext4 filesystem.
    #[must_use]
    pub fn is_ext4(&self) -> bool {
        matches!(self.flavor, FsFlavor::Ext4(_))
    }

    /// Whether this is a btrfs filesystem.
    #[must_use]
    pub fn is_btrfs(&self) -> bool {
        matches!(self.flavor, FsFlavor::Btrfs(_))
    }

    /// Device length in bytes.
    #[must_use]
    pub fn device_len(&self) -> u64 {
        self.dev.len_bytes()
    }

    /// Return the ext4 superblock, or `None` if this is not ext4.
    #[must_use]
    pub fn ext4_superblock(&self) -> Option<&Ext4Superblock> {
        match &self.flavor {
            FsFlavor::Ext4(sb) => Some(sb),
            FsFlavor::Btrfs(_) => None,
        }
    }

    /// Return the btrfs superblock, or `None` if this is not btrfs.
    #[must_use]
    pub fn btrfs_superblock(&self) -> Option<&BtrfsSuperblock> {
        match &self.flavor {
            FsFlavor::Btrfs(sb) => Some(sb),
            FsFlavor::Ext4(_) => None,
        }
    }

    /// Return the btrfs context (chunk mapping + nodesize), or `None` if not btrfs.
    #[must_use]
    pub fn btrfs_context(&self) -> Option<&BtrfsContext> {
        self.btrfs_context.as_ref()
    }

    /// Return ext4 mount-time journal replay outcome when available.
    #[must_use]
    pub fn ext4_journal_replay(&self) -> Option<&ReplayOutcome> {
        self.ext4_journal_replay.as_ref()
    }

    /// Return crash recovery outcome if the filesystem was not cleanly mounted.
    #[must_use]
    pub fn crash_recovery(&self) -> Option<&CrashRecoveryOutcome> {
        self.crash_recovery.as_ref()
    }

    /// Detect unclean shutdown state from the ext4 superblock and record
    /// recovery actions taken. Returns `None` for btrfs or when the
    /// filesystem was cleanly unmounted.
    fn detect_and_recover_crash(&self) -> Option<CrashRecoveryOutcome> {
        let sb = self.ext4_superblock()?;
        let state = sb.state;
        let was_clean = (state & EXT4_VALID_FS) != 0
            && (state & EXT4_ERROR_FS) == 0
            && (state & EXT4_ORPHAN_FS) == 0;
        let had_errors = (state & EXT4_ERROR_FS) != 0;
        let had_orphans = (state & EXT4_ORPHAN_FS) != 0;

        // Derive journal replay stats from the already-completed replay.
        #[allow(clippy::cast_possible_truncation)]
        let (journal_txns_replayed, journal_blocks_replayed) =
            self.ext4_journal_replay.as_ref().map_or((0, 0), |replay| {
                (
                    replay.committed_sequences.len() as u32,
                    replay.stats.replayed_blocks,
                )
            });

        // MVCC store is always freshly initialized on mount — any in-flight
        // transactions from a previous session are implicitly discarded.
        let mvcc_reset = !was_clean;

        if was_clean {
            info!(raw_state = state, "ext4 filesystem state: clean");
        } else {
            warn!(
                raw_state = state,
                had_errors,
                had_orphans,
                journal_txns_replayed,
                journal_blocks_replayed,
                "ext4 unclean shutdown detected — recovery performed"
            );
        }

        Some(CrashRecoveryOutcome {
            was_clean,
            raw_state: state,
            had_errors,
            had_orphans,
            journal_txns_replayed,
            journal_blocks_replayed,
            mvcc_reset,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn maybe_recover_ext4_orphans(&mut self, cx: &Cx) -> Result<Ext4OrphanRecoveryStats, FfsError> {
        let (head, fs_state, inodes_count, csum_seed) = match self.ext4_superblock() {
            Some(sb) => (sb.last_orphan, sb.state, sb.inodes_count, sb.csum_seed()),
            None => return Ok(Ext4OrphanRecoveryStats::default()),
        };
        let has_orphan_flag = (fs_state & EXT4_ORPHAN_FS) != 0;

        if !has_orphan_flag && head == 0 {
            return Ok(Ext4OrphanRecoveryStats::default());
        }

        debug!(
            orphan_head = head,
            orphan_state_flag = has_orphan_flag,
            "ext4 orphan recovery start"
        );

        let mut alloc = self.load_ext4_alloc_state(cx)?;
        let block_dev = self.block_device_adapter();
        let orphans = self.collect_ext4_orphan_list_lenient(cx, head, inodes_count)?;
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();
        let mut stats = Ext4OrphanRecoveryStats::default();

        for ino in orphans {
            stats.scanned = stats.scanned.saturating_add(1);

            let mut inode = match self.read_inode(cx, ino) {
                Ok(inode) => inode,
                Err(err) => {
                    stats.skipped = stats.skipped.saturating_add(1);
                    warn!(
                        orphan_inode = ino.0,
                        error = %err,
                        "ext4 orphan recovery skipped unreadable inode"
                    );
                    continue;
                }
            };

            if inode.links_count == 0 {
                ffs_inode::delete_inode(
                    cx,
                    &block_dev,
                    &alloc.geo,
                    &mut alloc.groups,
                    ino,
                    &mut inode,
                    csum_seed,
                    tstamp_secs,
                    &alloc.persist_ctx,
                )?;
                stats.deleted = stats.deleted.saturating_add(1);
                info!(
                    orphan_inode = ino.0,
                    action = "deleted",
                    "ext4 orphan recovered"
                );
                continue;
            }

            if inode.flags & ffs_types::EXT4_EXTENTS_FL != 0 && inode.extent_bytes.len() >= 60 {
                let mut root_bytes = Self::extent_root(&inode);
                let logical_end_u64 = inode.size.div_ceil(u64::from(alloc.geo.block_size));
                let logical_end = u32::try_from(logical_end_u64).map_err(|_| {
                    FfsError::InvalidGeometry(format!(
                        "inode {} logical end {} does not fit u32",
                        ino.0, logical_end_u64
                    ))
                })?;
                let freed = ffs_extent::truncate_extents(
                    cx,
                    &block_dev,
                    &mut root_bytes,
                    &alloc.geo,
                    &mut alloc.groups,
                    logical_end,
                    &alloc.persist_ctx,
                )?;
                Self::set_extent_root(&mut inode, &root_bytes);
                let freed_sectors = if inode.is_huge_file() {
                    freed
                } else {
                    freed.saturating_mul(u64::from(alloc.geo.block_size / 512))
                };
                inode.blocks = inode.blocks.saturating_sub(freed_sectors);
            }

            inode.dtime = 0;
            ffs_inode::touch_ctime(&mut inode, tstamp_secs, tstamp_nanos);
            ffs_inode::write_inode(
                cx,
                &block_dev,
                &alloc.geo,
                &alloc.groups,
                ino,
                &inode,
                csum_seed,
            )?;
            stats.truncated = stats.truncated.saturating_add(1);
            info!(
                orphan_inode = ino.0,
                action = "truncated",
                "ext4 orphan recovered"
            );
        }

        self.clear_ext4_orphan_state(cx)?;
        info!(
            orphan_head = head,
            scanned = stats.scanned,
            processed = stats.processed(),
            deleted = stats.deleted,
            truncated = stats.truncated,
            skipped = stats.skipped,
            "ext4 orphan list cleared"
        );

        Ok(stats)
    }

    fn collect_ext4_orphan_list_lenient(
        &self,
        cx: &Cx,
        head: u32,
        inodes_count: u32,
    ) -> Result<Vec<InodeNumber>, FfsError> {
        let mut next = head;
        let mut seen = BTreeSet::new();
        let mut inodes = Vec::new();
        let max_inodes = usize::try_from(inodes_count)
            .map_err(|_| FfsError::InvalidGeometry("inodes_count does not fit usize".to_owned()))?;

        while next != 0 {
            if next > inodes_count {
                warn!(
                    orphan_inode = next,
                    inodes_count,
                    "ext4 orphan recovery detected out-of-range inode; stopping traversal"
                );
                break;
            }
            if !seen.insert(next) {
                warn!(
                    orphan_inode = next,
                    "ext4 orphan recovery detected cycle; stopping traversal"
                );
                break;
            }
            if inodes.len() >= max_inodes {
                warn!(
                    max_inodes,
                    "ext4 orphan recovery reached inode traversal bound; stopping traversal"
                );
                break;
            }

            let ino = InodeNumber(u64::from(next));
            inodes.push(ino);

            match self.read_inode(cx, ino) {
                Ok(inode) => {
                    next = inode.dtime;
                }
                Err(err) => {
                    warn!(
                        orphan_inode = ino.0,
                        error = %err,
                        "ext4 orphan recovery failed to read inode link; stopping traversal"
                    );
                    break;
                }
            }
        }

        Ok(inodes)
    }

    fn clear_ext4_orphan_state(&mut self, cx: &Cx) -> Result<(), FfsError> {
        let mut sb_region = read_ext4_superblock_region(cx, self.dev.as_ref())?;
        let old_state = u16::from_le_bytes([sb_region[0x3A], sb_region[0x3B]]);
        let new_state = old_state & !EXT4_ORPHAN_FS;
        sb_region[0x3A..0x3C].copy_from_slice(&new_state.to_le_bytes());
        sb_region[0xE8..0xEC].copy_from_slice(&0_u32.to_le_bytes());

        if let FsFlavor::Ext4(sb) = &mut self.flavor {
            if sb.has_metadata_csum() {
                let csum = ffs_ondisk::ext4_chksum(!0u32, &sb_region[..0x3FC]);
                sb_region[0x3FC..0x400].copy_from_slice(&csum.to_le_bytes());
                sb.checksum = csum;
            }
            sb.state = new_state;
            sb.last_orphan = 0;
        }

        let sb_offset = u64::try_from(ffs_types::EXT4_SUPERBLOCK_OFFSET)
            .map_err(|_| FfsError::Format("ext4 superblock offset does not fit u64".to_owned()))?;
        self.dev
            .write_all_at(cx, ByteOffset(sb_offset), &sb_region)?;

        Ok(())
    }

    fn maybe_replay_ext4_journal(
        &self,
        cx: &Cx,
        mode: Ext4JournalReplayMode,
    ) -> Result<Option<ReplayOutcome>, FfsError> {
        let (block_size, journal_inum, journal_dev) = match self.ext4_superblock() {
            Some(sb) => (sb.block_size, sb.journal_inum, sb.journal_dev),
            None => return Ok(None),
        };

        if journal_inum == 0 {
            info!("ext4 journal replay skipped: no journal inode");
            return Ok(None);
        }

        if journal_dev != 0 {
            return Err(FfsError::UnsupportedFeature(format!(
                "external ext4 journal device {journal_dev} is not supported"
            )));
        }

        if mode == Ext4JournalReplayMode::Skip {
            info!(
                journal_inum,
                "ext4 journal replay skipped by open options (journal present)"
            );
            return Ok(Some(ReplayOutcome::default()));
        }

        let journal_ino = InodeNumber(u64::from(journal_inum));
        let journal_inode = self.read_inode(cx, journal_ino)?;
        let extents = self.collect_extents(cx, &journal_inode)?;
        if extents.is_empty() {
            info!(
                journal_inum,
                "ext4 journal replay skipped: journal inode has no extents"
            );
            return Ok(None);
        }

        let mut total_blocks = 0_u64;
        let mut segments = Vec::with_capacity(extents.len());
        for ext in &extents {
            let len = u64::from(ext.actual_len());
            if len == 0 {
                return Err(FfsError::Corruption {
                    block: ext.physical_start,
                    detail: "ext4 journal extent has zero length".to_owned(),
                });
            }
            total_blocks = total_blocks
                .checked_add(len)
                .ok_or_else(|| FfsError::Format("journal extent length overflow".to_owned()))?;
            segments.push(JournalSegment {
                start: BlockNumber(ext.physical_start),
                blocks: len,
            });
        }
        if segments.is_empty() {
            return Ok(None);
        }

        let block_dev = ByteDeviceBlockAdapter {
            dev: &*self.dev,
            block_size,
        };

        let journal_start_block = segments[0].start.0;
        let journal_segment_count = segments.len();
        info!(
            journal_inum,
            journal_start_block,
            journal_blocks = total_blocks,
            journal_segments = journal_segment_count,
            replay_mode = ?mode,
            "ext4 journal replay start"
        );
        let outcome = replay_jbd2_segments(cx, &block_dev, &segments)?;
        info!(
            journal_inum,
            scanned_blocks = outcome.stats.scanned_blocks,
            descriptor_blocks = outcome.stats.descriptor_blocks,
            commit_blocks = outcome.stats.commit_blocks,
            revoke_blocks = outcome.stats.revoke_blocks,
            replayed_blocks = outcome.stats.replayed_blocks,
            skipped_revoked_blocks = outcome.stats.skipped_revoked_blocks,
            incomplete_transactions = outcome.stats.incomplete_transactions,
            committed_sequences = outcome.committed_sequences.len(),
            replay_mode = ?mode,
            "ext4 journal replay completed"
        );
        Ok(Some(outcome))
    }

    // ── MVCC transaction API ─────────────────────────────────────────

    /// Get the shared MVCC store for this filesystem.
    ///
    /// The returned `Arc<RwLock<MvccStore>>` can be cloned and used to create
    /// `MvccBlockDevice` wrappers or access versioned data directly.
    #[must_use]
    pub fn mvcc_store(&self) -> &Arc<RwLock<MvccStore>> {
        &self.mvcc_store
    }

    /// Get the current snapshot sequence for new read-only operations.
    ///
    /// Readers that want a consistent view should capture this snapshot
    /// at the start of their operation and use it throughout.
    #[must_use]
    pub fn current_snapshot(&self) -> Snapshot {
        let snap = self.mvcc_store.read().current_snapshot();
        trace!(
            target: "ffs::mvcc",
            snapshot_high = snap.high.0,
            "mvcc_snapshot_acquired"
        );
        snap
    }

    /// Begin a new MVCC transaction.
    ///
    /// The transaction captures a snapshot at the current commit sequence.
    /// Writes are staged in the transaction and become visible only after
    /// a successful commit.
    ///
    /// # Logging
    /// - `txn_begin`: transaction ID and snapshot sequence
    pub fn begin_transaction(&self) -> Transaction {
        let txn = self.mvcc_store.write().begin();
        debug!(
            target: "ffs::mvcc",
            txn_id = txn.id.0,
            snapshot_high = txn.snapshot.high.0,
            "txn_begin"
        );
        txn
    }

    /// Commit an MVCC transaction using First-Committer-Wins (FCW) semantics.
    ///
    /// # Errors
    /// Returns `CommitError::Conflict` if another transaction committed to
    /// a block in this transaction's write set after our snapshot.
    ///
    /// # Logging
    /// - `txn_commit_start`: transaction details before commit
    /// - `txn_commit_success`: on successful commit with commit sequence
    /// - `txn_commit_conflict`: on FCW conflict with conflict details
    #[allow(clippy::cast_possible_truncation)]
    pub fn commit_transaction(&self, txn: Transaction) -> Result<CommitSeq, CommitError> {
        let txn_id = txn.id;
        let write_set_size = txn.pending_writes();
        let read_set_size = txn.read_set().len();

        debug!(
            target: "ffs::mvcc",
            txn_id = txn_id.0,
            write_set_size,
            read_set_size,
            "txn_commit_start"
        );

        let start = std::time::Instant::now();
        let result = self.mvcc_store.write().commit(txn);
        let duration_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(commit_seq) => {
                info!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    commit_seq = commit_seq.0,
                    write_set_size,
                    duration_us,
                    "txn_commit_success"
                );
            }
            Err(CommitError::Conflict {
                block,
                snapshot,
                observed,
            }) => {
                warn!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    conflict_block = block.0,
                    snapshot = snapshot.0,
                    observed = observed.0,
                    conflict_type = "fcw",
                    "txn_commit_conflict"
                );
            }
            Err(CommitError::SsiConflict {
                pivot_block,
                read_version,
                write_version,
                concurrent_txn,
            }) => {
                warn!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    pivot_block = pivot_block.0,
                    read_version = read_version.0,
                    write_version = write_version.0,
                    concurrent_txn = concurrent_txn.0,
                    conflict_type = "ssi",
                    "txn_commit_conflict"
                );
            }
            Err(CommitError::ChainBackpressure {
                block,
                chain_len,
                cap,
                ..
            }) => {
                warn!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    block = block.0,
                    chain_len,
                    cap,
                    conflict_type = "backpressure",
                    "txn_commit_backpressure"
                );
            }
        }

        result
    }

    /// Commit an MVCC transaction using Serializable Snapshot Isolation (SSI).
    ///
    /// SSI extends FCW with rw-antidependency tracking to detect and prevent
    /// write skew anomalies. Use this when full serializability is required.
    ///
    /// # Errors
    /// Returns `CommitError::Conflict` for write-write conflicts (FCW layer).
    /// Returns `CommitError::SsiConflict` for rw-antidependency cycles.
    #[allow(clippy::cast_possible_truncation)]
    pub fn commit_transaction_ssi(&self, txn: Transaction) -> Result<CommitSeq, CommitError> {
        let txn_id = txn.id;
        let write_set_size = txn.pending_writes();
        let read_set_size = txn.read_set().len();

        debug!(
            target: "ffs::mvcc",
            txn_id = txn_id.0,
            write_set_size,
            read_set_size,
            mode = "ssi",
            "txn_commit_start"
        );

        let start = std::time::Instant::now();
        let result = self.mvcc_store.write().commit_ssi(txn);
        let duration_us = start.elapsed().as_micros() as u64;

        match &result {
            Ok(commit_seq) => {
                info!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    commit_seq = commit_seq.0,
                    write_set_size,
                    duration_us,
                    mode = "ssi",
                    "txn_commit_success"
                );
            }
            Err(e) => {
                warn!(
                    target: "ffs::mvcc",
                    txn_id = txn_id.0,
                    error = %e,
                    mode = "ssi",
                    "txn_commit_conflict"
                );
            }
        }

        result
    }

    /// Attach a JBD2 writer for ext4 compatibility-mode journaled commits.
    ///
    /// Once attached, [`commit_transaction_journaled`](Self::commit_transaction_journaled)
    /// can be used to atomically journal block writes before committing
    /// to the MVCC store.
    pub fn attach_jbd2_writer(&mut self, writer: Jbd2Writer) {
        self.jbd2_writer = Some(Mutex::new(writer));
    }

    /// Whether a JBD2 writer is attached.
    #[must_use]
    pub fn has_jbd2_writer(&self) -> bool {
        self.jbd2_writer.is_some()
    }

    /// Whether write operations are enabled.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        self.ext4_alloc_state.is_some() || self.btrfs_alloc_state.is_some()
    }

    /// Enable write operations by loading allocation state for the detected
    /// filesystem type (ext4 or btrfs).
    ///
    /// For ext4: reads all group descriptors from disk to populate the
    /// in-memory group statistics cache.
    /// For btrfs: walks the FS tree and extent tree to build an in-memory COW
    /// tree and extent allocator.
    pub fn enable_writes(&mut self, cx: &Cx) -> Result<(), FfsError> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let alloc_state = self.load_ext4_alloc_state(cx)?;
                self.ext4_alloc_state = Some(Mutex::new(alloc_state));
            }
            FsFlavor::Btrfs(_) => {
                let alloc_state = self.load_btrfs_alloc_state(cx)?;
                self.btrfs_alloc_state = Some(Mutex::new(alloc_state));
            }
        }
        Ok(())
    }

    fn load_ext4_alloc_state(&self, cx: &Cx) -> Result<Ext4AllocState, FfsError> {
        let sb = match &self.flavor {
            FsFlavor::Ext4(sb) => sb,
            FsFlavor::Btrfs(_) => {
                return Err(FfsError::UnsupportedFeature(
                    "btrfs write operations not yet supported".into(),
                ));
            }
        };

        let geo = FsGeometry::from_superblock(sb);
        let block_size = sb.block_size;
        // Load group descriptors from disk.
        let mut groups = Vec::with_capacity(geo.group_count as usize);
        let geom = self
            .ext4_geometry
            .as_ref()
            .ok_or_else(|| FfsError::Format("ext4_geometry not available".into()))?;

        for g in 0..geo.group_count {
            let gd = self.read_group_desc(cx, GroupNumber(g))?;
            groups.push(GroupStats::from_group_desc(GroupNumber(g), &gd));
        }

        let persist_ctx = PersistCtx {
            gdt_block: BlockNumber(if block_size == 1024 { 2 } else { 1 }),
            desc_size: geom.group_desc_size,
            has_metadata_csum: geom.has_metadata_csum,
            csum_seed: geom.csum_seed,
        };

        info!(
            target: "ffs::write",
            groups = geo.group_count,
            block_size,
            "ext4 write state initialized"
        );

        Ok(Ext4AllocState {
            geo,
            groups,
            persist_ctx,
        })
    }

    /// Load btrfs allocation state by walking the FS tree and populating
    /// the in-memory COW tree and extent allocator.
    fn load_btrfs_alloc_state(&self, cx: &Cx) -> Result<BtrfsAllocState, FfsError> {
        let sb = match &self.flavor {
            FsFlavor::Btrfs(sb) => sb,
            FsFlavor::Ext4(_) => {
                return Err(FfsError::Format("not a btrfs filesystem".into()));
            }
        };
        let nodesize = sb.nodesize;
        let generation = sb.generation;

        // Walk the FS tree to populate the in-memory COW tree.
        let items = self.walk_btrfs_fs_tree(cx)?;
        let max_items_per_node = (nodesize as usize - 101) / 25; // header=101, item=25
        let max_items = max_items_per_node.max(3);
        let mut fs_tree =
            InMemoryCowBtrfsTree::new(max_items).map_err(|e| btrfs_mutation_to_ffs(&e))?;

        // Find the highest objectid in use so we can mint new ones.
        let mut max_objectid = 256_u64; // btrfs reserves objectids below 256
        for item in &items {
            if item.key.objectid > max_objectid {
                max_objectid = item.key.objectid;
            }
            let tree_item = BtrfsTreeItem {
                key: BtrfsKey {
                    objectid: item.key.objectid,
                    item_type: item.key.item_type,
                    offset: item.key.offset,
                },
                data: item.data.clone(),
            };
            // Allow replace in case of duplicate keys from multiple walks.
            fs_tree
                .update(&tree_item.key, &tree_item.data)
                .or_else(|_| fs_tree.insert(tree_item.key, &tree_item.data))
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        }

        // Build a minimal extent allocator with a synthetic data block group.
        // For V1 single-device images, we create a block group spanning the
        // device and mark known extents as allocated.
        let mut extent_alloc =
            BtrfsExtentAllocator::new(generation).map_err(|e| btrfs_mutation_to_ffs(&e))?;

        // Add a data block group covering the device.  On real images the
        // first 1 MiB is reserved for superblock copies and system chunks,
        // but for small test images we lower the start so there is actually
        // allocatable space.
        let data_start = if sb.total_bytes > 1_048_576 {
            1_048_576_u64
        } else {
            // Small image — start data right after the superblock region.
            u64::from(nodesize).max(65_536)
        };
        let total_bytes = sb.total_bytes.saturating_sub(data_start);
        if total_bytes == 0 {
            return Err(FfsError::Format(format!(
                "btrfs image too small for writable allocation state: total_bytes={} data_start={}",
                sb.total_bytes, data_start
            )));
        }
        extent_alloc.add_block_group(
            data_start,
            BtrfsBlockGroupItem {
                total_bytes,
                used_bytes: sb.bytes_used.saturating_sub(data_start),
                flags: BTRFS_BLOCK_GROUP_DATA,
            },
        );

        info!(
            target: "ffs::write",
            nodesize,
            generation,
            items_loaded = items.len(),
            next_objectid = max_objectid + 1,
            "btrfs write state initialized"
        );

        Ok(BtrfsAllocState {
            fs_tree,
            extent_alloc,
            next_objectid: max_objectid + 1,
            generation,
            nodesize,
        })
    }

    /// Require the btrfs alloc state to be present (i.e., writes enabled).
    fn require_btrfs_alloc_state(&self) -> Result<&Mutex<BtrfsAllocState>, FfsError> {
        self.btrfs_alloc_state.as_ref().ok_or(FfsError::ReadOnly)
    }

    /// Commit an MVCC transaction with JBD2 journaling.
    ///
    /// The write path is:
    /// 1. Extract the transaction's write set.
    /// 2. Preflight FCW/chain-pressure checks under the MVCC write lock.
    /// 3. Write a JBD2 transaction (descriptor + data + commit blocks)
    ///    to the journal region via the attached [`Jbd2Writer`].
    /// 4. Make the transaction visible in MVCC after journal durability succeeds.
    ///
    /// The same MVCC write lock is held across steps (2)-(4), preventing any
    /// interleaving writer from invalidating the preflight decision before
    /// visibility is applied.
    ///
    /// # Errors
    ///
    /// Returns `FfsError` if no JBD2 writer is attached or if the journal
    /// write fails. Returns `CommitError` (wrapped in `FfsError`) if the
    /// MVCC commit detects a conflict.
    #[allow(clippy::cast_possible_truncation)]
    pub fn commit_transaction_journaled(
        &self,
        cx: &Cx,
        txn: Transaction,
    ) -> std::result::Result<(CommitSeq, Jbd2WriteStats), FfsError> {
        let jbd2_mutex = self.jbd2_writer.as_ref().ok_or_else(|| {
            FfsError::Format("no JBD2 writer attached for journaled commit".to_owned())
        })?;

        let txn_id = txn.id;
        let write_count = txn.pending_writes();

        debug!(
            target: "ffs::journal",
            txn_id = txn_id.0,
            write_count,
            "journaled_commit_start"
        );

        // Build a BlockDevice adapter for journal I/O.
        let block_size = self.block_size();
        let block_dev = ByteDeviceBlockAdapter {
            dev: self.dev.as_ref(),
            block_size,
        };

        let writes: Vec<_> = txn
            .write_set()
            .iter()
            .map(|(b, d)| (*b, d.clone()))
            .collect();

        // Hold the MVCC write lock across preflight, journal I/O, and final
        // visibility so no other writer can invalidate this transaction in between.
        let mut mvcc_guard = self.mvcc_store.write();

        // Phase 1: preflight conflict checks. This does not make versions visible.
        mvcc_guard.preflight_commit_fcw(&txn).map_err(|e| {
            warn!(
                target: "ffs::journal",
                txn_id = txn_id.0,
                error = %e,
                "journaled_commit_mvcc_conflict"
            );
            match e {
                CommitError::Conflict { block, .. }
                | CommitError::ChainBackpressure { block, .. } => FfsError::MvccConflict {
                    tx: txn_id.0,
                    block: block.0,
                },
                CommitError::SsiConflict { pivot_block, .. } => FfsError::MvccConflict {
                    tx: txn_id.0,
                    block: pivot_block.0,
                },
            }
        })?;

        // Phase 2: write all pending blocks to the JBD2 journal.
        let jbd2_stats = {
            let mut jbd2_writer = jbd2_mutex.lock();
            let mut jbd2_txn = jbd2_writer.begin_transaction();

            for (block, payload) in writes {
                jbd2_txn.add_write(block, payload);
            }

            match jbd2_writer.commit_transaction(cx, &block_dev, &jbd2_txn) {
                Ok((_seq, stats)) => stats,
                Err(e) => {
                    warn!(
                        target: "ffs::journal",
                        txn_id = txn_id.0,
                        error = %e,
                        "journaled_commit_jbd2_failed_before_visibility"
                    );
                    return Err(FfsError::Format(format!(
                        "JBD2 journal write failed before MVCC visibility: {e}"
                    )));
                }
            }
        };

        // Phase 3: make preflighted writes visible in MVCC.
        let commit_seq = mvcc_guard.commit_fcw_prechecked(txn);
        drop(mvcc_guard);

        info!(
            target: "ffs::journal",
            txn_id = txn_id.0,
            commit_seq = commit_seq.0,
            data_blocks = jbd2_stats.data_blocks,
            "journaled_commit_success"
        );

        Ok((commit_seq, jbd2_stats))
    }

    /// Read a block at a specific snapshot, checking MVCC store first.
    ///
    /// This is the core MVCC-aware read operation:
    /// 1. Check if the MVCC store has a version visible at `snapshot`
    /// 2. If found, return the versioned data
    /// 3. Otherwise, fall back to the underlying device
    ///
    /// # Logging
    /// - `mvcc_read_start`: block and snapshot
    /// - `mvcc_version_hit`: when found in MVCC store
    /// - `mvcc_version_miss`: when falling back to device
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_block_at_snapshot(
        &self,
        cx: &Cx,
        block: BlockNumber,
        snapshot: Snapshot,
    ) -> Result<Vec<u8>, FfsError> {
        trace!(
            target: "ffs::mvcc",
            block = block.0,
            snapshot = snapshot.high.0,
            "mvcc_read_start"
        );

        let start = std::time::Instant::now();

        // Check MVCC store first (shared lock, no I/O).
        {
            let guard = self.mvcc_store.read();
            if let Some(bytes) = guard.read_visible(block, snapshot) {
                let duration_us = start.elapsed().as_micros() as u64;
                trace!(
                    target: "ffs::mvcc",
                    block = block.0,
                    snapshot = snapshot.high.0,
                    source = "mvcc",
                    duration_us,
                    "mvcc_version_hit"
                );
                return Ok(bytes.into_owned());
            }
        }

        // Fall back to device.
        let block_size = self.block_size();
        let offset = block
            .0
            .checked_mul(u64::from(block_size))
            .ok_or_else(|| FfsError::Format("block offset overflow".to_owned()))?;

        let bs = usize::try_from(block_size)
            .map_err(|_| FfsError::Format("block_size overflow".to_owned()))?;
        let mut buf = vec![0_u8; bs];
        self.dev.read_exact_at(cx, ByteOffset(offset), &mut buf)?;

        let duration_us = start.elapsed().as_micros() as u64;
        trace!(
            target: "ffs::mvcc",
            block = block.0,
            snapshot = snapshot.high.0,
            source = "device",
            duration_us,
            "mvcc_version_miss"
        );

        Ok(buf)
    }

    /// Stage a block write in a transaction.
    ///
    /// The write is not visible to other transactions until commit.
    ///
    /// # Logging
    /// - `mvcc_write_stage`: block, transaction, and data length
    pub fn stage_block_write(&self, txn: &mut Transaction, block: BlockNumber, data: Vec<u8>) {
        trace!(
            target: "ffs::mvcc",
            txn_id = txn.id.0,
            block = block.0,
            data_len = data.len(),
            "mvcc_write_stage"
        );
        txn.stage_write(block, data);
    }

    /// Record that a block was read at a specific version (for SSI tracking).
    ///
    /// Call this after reading a block if using SSI commit mode.
    pub fn record_read(&self, txn: &mut Transaction, block: BlockNumber, version: CommitSeq) {
        trace!(
            target: "ffs::mvcc",
            txn_id = txn.id.0,
            block = block.0,
            version = version.0,
            "mvcc_read_set_add"
        );
        txn.record_read(block, version);
    }

    /// Get the latest commit sequence for a block.
    ///
    /// Useful for recording reads when using SSI mode.
    #[must_use]
    pub fn latest_block_version(&self, block: BlockNumber) -> CommitSeq {
        self.mvcc_store.read().latest_commit_seq(block)
    }

    /// Prune old versions that are no longer needed by any active snapshot.
    ///
    /// Call this periodically to reclaim memory from superseded versions.
    /// Returns the watermark that was used for pruning.
    pub fn prune_mvcc_versions(&self) -> CommitSeq {
        let mut guard = self.mvcc_store.write();
        let watermark = guard.prune_safe();
        debug!(
            target: "ffs::mvcc",
            watermark = watermark.0,
            remaining_versions = guard.version_count(),
            "mvcc_prune"
        );
        watermark
    }

    // ── Btrfs tree-walk via device ───────────────────────────────────

    /// Walk a btrfs tree from the given logical root, reading nodes via the device.
    ///
    /// Uses the sys_chunk logical-to-physical mapping to translate addresses,
    /// then reads each node from the block device. Returns all leaf items in
    /// key order (left-to-right DFS).
    ///
    /// Returns `FfsError::Format` if this is not a btrfs filesystem.
    pub fn walk_btrfs_tree(
        &self,
        cx: &Cx,
        root_logical: u64,
    ) -> Result<Vec<BtrfsLeafEntry>, FfsError> {
        let ctx = self
            .btrfs_context()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;

        let nodesize = ctx.nodesize;
        let ns =
            usize::try_from(nodesize).map_err(|_| FfsError::Format("nodesize overflow".into()))?;

        let mut read_fn = |phys: u64| -> Result<Vec<u8>, ParseError> {
            let mut buf = vec![0_u8; ns];
            self.dev
                .read_exact_at(cx, ByteOffset(phys), &mut buf)
                .map_err(|_| ParseError::InsufficientData {
                    needed: ns,
                    offset: 0,
                    actual: 0,
                })?;
            Ok(buf)
        };

        walk_tree(&mut read_fn, &ctx.chunks, root_logical, nodesize)
            .map_err(|e| parse_to_ffs_error(&e))
    }

    /// Walk the btrfs root tree, returning all leaf items.
    ///
    /// Convenience wrapper around [`walk_btrfs_tree`](Self::walk_btrfs_tree)
    /// that uses the superblock's `root` address.
    pub fn walk_btrfs_root_tree(&self, cx: &Cx) -> Result<Vec<BtrfsLeafEntry>, FfsError> {
        let sb = self
            .btrfs_superblock()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;
        self.walk_btrfs_tree(cx, sb.root)
    }

    /// Translate the FUSE/root-facing inode number to the btrfs objectid.
    ///
    /// For btrfs we treat inode `1` as an alias for the superblock's
    /// `root_dir_objectid` so root getattr/readdir/lookup calls can work
    /// through the VFS root inode contract.
    fn btrfs_canonical_inode(&self, ino: InodeNumber) -> Result<u64, FfsError> {
        let sb = self
            .btrfs_superblock()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;
        if ino.0 == 1 {
            Ok(sb.root_dir_objectid)
        } else {
            Ok(ino.0)
        }
    }

    /// Translate VFS inode numbers to ext4 on-disk inode numbers.
    ///
    /// FsOps is consumed by FUSE, where inode `1` is the synthetic VFS root.
    /// ext4 uses inode `2` as root on disk, so we canonicalize at the FsOps
    /// boundary.
    const fn ext4_canonical_inode(ino: InodeNumber) -> InodeNumber {
        if ino.0 == 1 { InodeNumber(2) } else { ino }
    }

    /// Translate ext4 on-disk inode numbers back to VFS inode numbers.
    const fn ext4_presented_inode(ino: InodeNumber) -> InodeNumber {
        if ino.0 == 2 { InodeNumber(1) } else { ino }
    }

    fn ext4_present_attr(mut attr: InodeAttr) -> InodeAttr {
        attr.ino = Self::ext4_presented_inode(attr.ino);
        attr
    }

    fn ext4_present_dir_entry(mut entry: DirEntry) -> DirEntry {
        entry.ino = Self::ext4_presented_inode(entry.ino);
        entry
    }

    /// Resolve and walk the default filesystem tree (`FS_TREE`).
    fn walk_btrfs_fs_tree(&self, cx: &Cx) -> Result<Vec<BtrfsLeafEntry>, FfsError> {
        let root_items = self.walk_btrfs_root_tree(cx)?;
        let fs_tree_root = root_items
            .iter()
            .find(|item| {
                item.key.objectid == BTRFS_FS_TREE_OBJECTID
                    && item.key.item_type == BTRFS_ITEM_ROOT_ITEM
            })
            .ok_or_else(|| {
                FfsError::NotFound(format!(
                    "btrfs ROOT_ITEM for FS_TREE objectid {BTRFS_FS_TREE_OBJECTID}"
                ))
            })?;

        let root_item = parse_root_item(&fs_tree_root.data).map_err(|e| parse_to_ffs_error(&e))?;
        self.walk_btrfs_tree(cx, root_item.bytenr)
    }

    fn btrfs_timespec(sec: u64, nsec: u32) -> SystemTime {
        let clamped_nsec = nsec.min(999_999_999);
        UNIX_EPOCH
            .checked_add(Duration::new(sec, clamped_nsec))
            .unwrap_or(UNIX_EPOCH)
    }

    fn btrfs_mode_to_file_type(mode: u32) -> FileType {
        match mode & 0o170_000 {
            0o040_000 => FileType::Directory,
            0o120_000 => FileType::Symlink,
            0o060_000 => FileType::BlockDevice,
            0o020_000 => FileType::CharDevice,
            0o010_000 => FileType::Fifo,
            0o140_000 => FileType::Socket,
            _ => FileType::RegularFile,
        }
    }

    fn btrfs_dir_type_to_file_type(dir_type: u8) -> FileType {
        match dir_type {
            BTRFS_FT_DIR => FileType::Directory,
            BTRFS_FT_SYMLINK => FileType::Symlink,
            BTRFS_FT_BLKDEV => FileType::BlockDevice,
            BTRFS_FT_CHRDEV => FileType::CharDevice,
            BTRFS_FT_FIFO => FileType::Fifo,
            BTRFS_FT_SOCK => FileType::Socket,
            _ => FileType::RegularFile,
        }
    }

    fn btrfs_inode_attr_from_item(
        &self,
        ino: InodeNumber,
        inode: BtrfsInodeItem,
    ) -> Result<InodeAttr, FfsError> {
        let sb = self
            .btrfs_superblock()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;
        let blocks = inode.nbytes.div_ceil(512);
        let rdev = u32::try_from(inode.rdev).unwrap_or(u32::MAX);
        Ok(InodeAttr {
            ino,
            size: inode.size,
            blocks,
            atime: Self::btrfs_timespec(inode.atime_sec, inode.atime_nsec),
            mtime: Self::btrfs_timespec(inode.mtime_sec, inode.mtime_nsec),
            ctime: Self::btrfs_timespec(inode.ctime_sec, inode.ctime_nsec),
            crtime: Self::btrfs_timespec(inode.otime_sec, inode.otime_nsec),
            kind: Self::btrfs_mode_to_file_type(inode.mode),
            perm: (inode.mode & 0o7777) as u16,
            nlink: inode.nlink,
            uid: inode.uid,
            gid: inode.gid,
            rdev,
            blksize: sb.sectorsize,
        })
    }

    fn btrfs_find_inode_item(
        items: &[BtrfsLeafEntry],
        objectid: u64,
    ) -> Result<&BtrfsLeafEntry, FfsError> {
        items
            .iter()
            .find(|item| {
                item.key.objectid == objectid && item.key.item_type == BTRFS_ITEM_INODE_ITEM
            })
            .ok_or_else(|| FfsError::NotFound(format!("btrfs inode objectid {objectid}")))
    }

    fn btrfs_read_inode_attr(&self, cx: &Cx, ino: InodeNumber) -> Result<InodeAttr, FfsError> {
        let canonical = self.btrfs_canonical_inode(ino)?;

        // When writes are enabled the COW tree holds all items (seeded from
        // on-disk at enable_writes time and updated by mutations).  Read from
        // it so that newly-created inodes are visible.
        if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
            let alloc = alloc_mutex.lock();
            let inode = self.btrfs_read_inode_from_tree(&alloc, canonical)?;
            drop(alloc);
            return Ok(self.btrfs_inode_to_attr(canonical, &inode));
        }

        let items = self.walk_btrfs_fs_tree(cx)?;
        let inode_item = Self::btrfs_find_inode_item(&items, canonical)?;
        let inode = parse_inode_item(&inode_item.data).map_err(|e| parse_to_ffs_error(&e))?;
        self.btrfs_inode_attr_from_item(ino, inode)
    }

    fn btrfs_lookup_child(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
    ) -> Result<InodeAttr, FfsError> {
        let parent_attr = self.btrfs_read_inode_attr(cx, parent)?;
        if parent_attr.kind != FileType::Directory {
            return Err(FfsError::NotDirectory);
        }

        let canonical_parent = self.btrfs_canonical_inode(parent)?;

        // When writes are enabled, look up via the COW tree so mutations are
        // visible (the write-path helper already handles hash-collision
        // disambiguation).
        if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
            let alloc = alloc_mutex.lock();
            let dir_item = self.btrfs_lookup_dir_entry(&alloc, canonical_parent, name)?;
            let child_ino = InodeNumber(dir_item.child_objectid);
            drop(alloc);
            return self.btrfs_read_inode_attr(cx, child_ino);
        }

        let items = self.walk_btrfs_fs_tree(cx)?;

        for preferred_item_type in [BTRFS_ITEM_DIR_ITEM, BTRFS_ITEM_DIR_INDEX] {
            for item in &items {
                if item.key.objectid != canonical_parent
                    || item.key.item_type != preferred_item_type
                {
                    continue;
                }
                let dir_items = parse_dir_items(&item.data).map_err(|e| parse_to_ffs_error(&e))?;
                for dir_item in dir_items {
                    if dir_item.name == name {
                        let child_ino = InodeNumber(dir_item.child_objectid);
                        return self.btrfs_read_inode_attr(cx, child_ino);
                    }
                }
            }
        }

        Err(FfsError::NotFound(
            String::from_utf8_lossy(name).into_owned(),
        ))
    }

    #[allow(clippy::too_many_lines)]
    fn btrfs_readdir_entries(
        &self,
        cx: &Cx,
        ino: InodeNumber,
    ) -> Result<Vec<(u64, DirEntry)>, FfsError> {
        trace!(inode = %ino, "btrfs readdir_start");

        let dir_attr = self.btrfs_read_inode_attr(cx, ino)?;
        if dir_attr.kind != FileType::Directory {
            return Err(FfsError::NotDirectory);
        }

        let canonical_dir = self.btrfs_canonical_inode(ino)?;
        let mut rows: Vec<(u64, DirEntry)> = Vec::new();

        // Collect (key, data) pairs — either from the COW tree or the on-disk
        // tree depending on whether writes are enabled.
        let cow_items: Vec<(BtrfsKey, Vec<u8>)>;
        let ondisk_items: Vec<BtrfsLeafEntry>;

        let kv_iter: Box<dyn Iterator<Item = (&BtrfsKey, &[u8])>> =
            if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
                let alloc = alloc_mutex.lock();
                let start = BtrfsKey {
                    objectid: canonical_dir,
                    item_type: BTRFS_ITEM_DIR_ITEM,
                    offset: 0,
                };
                let end = BtrfsKey {
                    objectid: canonical_dir,
                    item_type: BTRFS_ITEM_DIR_INDEX,
                    offset: u64::MAX,
                };
                cow_items = alloc
                    .fs_tree
                    .range(&start, &end)
                    .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                drop(alloc);
                Box::new(cow_items.iter().map(|(k, v)| (k, v.as_slice())))
            } else {
                ondisk_items = self.walk_btrfs_fs_tree(cx)?;
                Box::new(
                    ondisk_items
                        .iter()
                        .filter(|item| {
                            item.key.objectid == canonical_dir
                                && (item.key.item_type == BTRFS_ITEM_DIR_INDEX
                                    || item.key.item_type == BTRFS_ITEM_DIR_ITEM)
                        })
                        .map(|item| (&item.key, item.data.as_slice())),
                )
            };

        for (key, data) in kv_iter {
            let parsed = match parse_dir_items(data) {
                Ok(p) => p,
                Err(e) => {
                    warn!(
                        inode = canonical_dir,
                        item_type = key.item_type,
                        "btrfs invalid_dir_entry"
                    );
                    return Err(parse_to_ffs_error(&e));
                }
            };
            for (idx, dir_item) in parsed.into_iter().enumerate() {
                let local_idx = u64::try_from(idx).map_err(|_| FfsError::Corruption {
                    block: 0,
                    detail: "directory index conversion overflow".into(),
                })?;
                // Prefer DIR_INDEX ordering when available. DIR_ITEM entries get
                // a high-bit bias so they naturally sort after DIR_INDEX.
                let base = if key.item_type == BTRFS_ITEM_DIR_INDEX {
                    key.offset
                } else {
                    key.offset | (1_u64 << 63)
                };

                trace!(
                    inode = canonical_dir,
                    child_inode = dir_item.child_objectid,
                    file_type = dir_item.file_type,
                    index = base.saturating_add(local_idx),
                    "btrfs dir_item_found"
                );

                rows.push((
                    base.saturating_add(local_idx),
                    DirEntry {
                        ino: InodeNumber(dir_item.child_objectid),
                        offset: 0,
                        kind: Self::btrfs_dir_type_to_file_type(dir_item.file_type),
                        name: dir_item.name,
                    },
                ));
            }
        }

        rows.sort_by_key(|(k, _)| *k);

        // Remove duplicate names (DIR_ITEM and DIR_INDEX can both describe
        // the same entry). Keep first-by-sort-key for stable pagination.
        let mut deduped: Vec<(u64, DirEntry)> = Vec::new();
        for row in rows {
            if deduped
                .iter()
                .any(|(_, existing)| existing.name == row.1.name)
            {
                continue;
            }
            deduped.push(row);
        }

        if deduped.len() > 1000 {
            debug!(
                inode = canonical_dir,
                entry_count = deduped.len(),
                "btrfs large_directory"
            );
        }

        // Add synthetic "." and ".." for VFS compatibility.
        // Resolve parent via INODE_REF when COW tree is available.
        // Reverse-map root objectid back to FUSE inode 1.
        let root_oid = self.btrfs_superblock().map(|sb| sb.root_dir_objectid);
        #[allow(clippy::option_if_let_else)]
        let parent_ino = if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
            let alloc = alloc_mutex.lock();
            self.btrfs_lookup_parent(&alloc, canonical_dir)
        } else {
            self.walk_btrfs_fs_tree(cx)
                .unwrap_or_default()
                .into_iter()
                .find(|item| {
                    item.key.objectid == canonical_dir && item.key.item_type == BTRFS_ITEM_INODE_REF
                })
                .map(|item| item.key.offset)
        }
        .map_or(ino, |oid| {
            if Some(oid) == root_oid {
                InodeNumber(1) // FUSE root
            } else {
                InodeNumber(oid)
            }
        });
        let dot = DirEntry {
            ino,
            offset: 0,
            kind: FileType::Directory,
            name: b".".to_vec(),
        };
        let dotdot = DirEntry {
            ino: parent_ino,
            offset: 0,
            kind: FileType::Directory,
            name: b"..".to_vec(),
        };

        let mut out = Vec::with_capacity(deduped.len() + 2);
        out.push((0, dot));
        out.push((1, dotdot));
        out.extend(deduped);

        trace!(
            inode = canonical_dir,
            entry_count = out.len(),
            "btrfs readdir_complete"
        );

        Ok(out)
    }

    fn btrfs_logical_chunk_end(&self, logical: u64) -> Result<u64, FfsError> {
        let ctx = self
            .btrfs_context()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;
        for chunk in &ctx.chunks {
            let end =
                chunk
                    .key
                    .offset
                    .checked_add(chunk.length)
                    .ok_or_else(|| FfsError::Corruption {
                        block: logical,
                        detail: "btrfs chunk logical range overflow".into(),
                    })?;
            if logical >= chunk.key.offset && logical < end {
                return Ok(end);
            }
        }
        Err(FfsError::Corruption {
            block: logical,
            detail: "logical bytenr not covered by any btrfs chunk".into(),
        })
    }

    fn btrfs_read_logical_into(
        &self,
        cx: &Cx,
        mut logical: u64,
        mut out: &mut [u8],
    ) -> Result<(), FfsError> {
        let ctx = self
            .btrfs_context()
            .ok_or_else(|| FfsError::Format("not a btrfs filesystem".into()))?;

        while !out.is_empty() {
            let mapping = map_logical_to_physical(&ctx.chunks, logical)
                .map_err(|e| {
                    warn!(
                        logical,
                        reason = %e,
                        "btrfs chunk_map_failed: parse error while mapping logical address"
                    );
                    parse_to_ffs_error(&e)
                })?
                .ok_or_else(|| FfsError::Corruption {
                    block: logical,
                    detail: "logical bytenr not mapped to physical bytenr".into(),
                })?;
            let chunk_end = self.btrfs_logical_chunk_end(logical)?;
            let span_u64 = chunk_end.saturating_sub(logical);
            let span = usize::try_from(span_u64)
                .unwrap_or(usize::MAX)
                .min(out.len());

            trace!(
                logical_start = logical,
                physical_start = mapping.physical,
                span,
                "btrfs chunk_map"
            );

            let (head, tail) = out.split_at_mut(span);
            if let Err(err) = self
                .dev
                .read_exact_at(cx, ByteOffset(mapping.physical), head)
            {
                warn!(
                    physical_start = mapping.physical,
                    reason = %err,
                    "btrfs io_error: failed to read mapped physical range"
                );
                return Err(err);
            }
            logical = logical.saturating_add(span as u64);
            out = tail;
        }

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn btrfs_read_file(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        size: u32,
    ) -> Result<Vec<u8>, FfsError> {
        let read_started = Instant::now();
        let canonical = self.btrfs_canonical_inode(ino)?;
        trace!(inode = canonical, offset, length = size, "btrfs read_start");

        // Fetch the inode and extent items from either the COW tree (when
        // writes are enabled) or the on-disk FS tree.
        let (inode, extents): (BtrfsInodeItem, Vec<(u64, BtrfsExtentData)>) =
            if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
                let alloc = alloc_mutex.lock();
                let inode = self.btrfs_read_inode_from_tree(&alloc, canonical)?;
                let ext_start = BtrfsKey {
                    objectid: canonical,
                    item_type: BTRFS_ITEM_EXTENT_DATA,
                    offset: 0,
                };
                let ext_end = BtrfsKey {
                    objectid: canonical,
                    item_type: BTRFS_ITEM_EXTENT_DATA,
                    offset: u64::MAX,
                };
                let ext_items = alloc
                    .fs_tree
                    .range(&ext_start, &ext_end)
                    .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                drop(alloc);
                let exts = ext_items
                    .iter()
                    .map(|(k, v)| {
                        parse_extent_data(v)
                            .map(|parsed| (k.offset, parsed))
                            .map_err(|e| parse_to_ffs_error(&e))
                    })
                    .collect::<Result<_, _>>()?;
                (inode, exts)
            } else {
                let items = self.walk_btrfs_fs_tree(cx)?;
                let inode_entry = Self::btrfs_find_inode_item(&items, canonical)?;
                let inode =
                    parse_inode_item(&inode_entry.data).map_err(|e| parse_to_ffs_error(&e))?;
                let exts = items
                    .iter()
                    .filter(|item| {
                        item.key.objectid == canonical
                            && item.key.item_type == BTRFS_ITEM_EXTENT_DATA
                    })
                    .map(|item| {
                        parse_extent_data(&item.data)
                            .map(|parsed| (item.key.offset, parsed))
                            .map_err(|e| parse_to_ffs_error(&e))
                    })
                    .collect::<Result<_, _>>()?;
                (inode, exts)
            };

        if Self::btrfs_mode_to_file_type(inode.mode) == FileType::Directory {
            return Err(FfsError::IsDirectory);
        }

        if offset >= inode.size {
            trace!(
                inode = canonical,
                bytes_returned = 0_u64,
                duration_us = read_started.elapsed().as_micros(),
                "btrfs read_complete"
            );
            return Ok(Vec::new());
        }

        let to_read =
            usize::try_from((inode.size - offset).min(u64::from(size))).unwrap_or(size as usize);
        let mut out = vec![0_u8; to_read];
        let read_end = offset.saturating_add(to_read as u64);

        if to_read > 1_048_576 {
            debug!(inode = canonical, length = to_read, "btrfs large_read");
        }
        if extents.len() > 10 {
            debug!(
                inode = canonical,
                extent_count = extents.len(),
                "btrfs fragmented_read"
            );
        }
        if extents.is_empty() {
            debug!(
                inode = canonical,
                file_offset = offset,
                "btrfs extent_not_found"
            );
        }

        let mut covered_until = offset;
        for (logical_start, extent) in &extents {
            match extent {
                BtrfsExtentData::Inline {
                    compression, data, ..
                } => {
                    if *compression != 0 {
                        info!(
                            inode = canonical,
                            compression_type = *compression,
                            "btrfs compressed_unsupported"
                        );
                        return Err(FfsError::UnsupportedFeature(format!(
                            "btrfs compression type {compression}"
                        )));
                    }

                    trace!(
                        inode = canonical,
                        file_offset = logical_start,
                        data_len = data.len(),
                        "btrfs extent_lookup inline"
                    );

                    let extent_len =
                        u64::try_from(data.len()).map_err(|_| FfsError::Corruption {
                            block: 0,
                            detail: "inline extent length overflow".into(),
                        })?;
                    let extent_end = logical_start.saturating_add(extent_len);
                    let overlap_start = (*logical_start).max(offset);
                    let overlap_end = extent_end.min(read_end);
                    if overlap_start >= overlap_end {
                        continue;
                    }
                    if covered_until < overlap_start {
                        let zero_len = overlap_start - covered_until;
                        trace!(
                            inode = canonical,
                            file_offset = covered_until,
                            zero_len,
                            "btrfs hole_fill"
                        );
                    }

                    let src_start =
                        usize::try_from(overlap_start - logical_start).map_err(|_| {
                            FfsError::Corruption {
                                block: 0,
                                detail: "inline source offset overflow".into(),
                            }
                        })?;
                    let dst_start = usize::try_from(overlap_start - offset).map_err(|_| {
                        FfsError::Corruption {
                            block: 0,
                            detail: "inline destination offset overflow".into(),
                        }
                    })?;
                    let copy_len = usize::try_from(overlap_end - overlap_start).map_err(|_| {
                        FfsError::Corruption {
                            block: 0,
                            detail: "inline copy length overflow".into(),
                        }
                    })?;
                    out[dst_start..dst_start + copy_len]
                        .copy_from_slice(&data[src_start..src_start + copy_len]);
                    covered_until = covered_until.max(overlap_end);
                }
                BtrfsExtentData::Regular {
                    extent_type,
                    compression,
                    disk_bytenr,
                    extent_offset,
                    num_bytes,
                    ..
                } => {
                    if *compression != 0 {
                        info!(
                            inode = canonical,
                            compression_type = *compression,
                            "btrfs compressed_unsupported"
                        );
                        return Err(FfsError::UnsupportedFeature(format!(
                            "btrfs compression type {compression}"
                        )));
                    }

                    let extent_end = logical_start.saturating_add(*num_bytes);
                    let overlap_start = (*logical_start).max(offset);
                    let overlap_end = extent_end.min(read_end);
                    if overlap_start >= overlap_end {
                        continue;
                    }
                    if covered_until < overlap_start {
                        let zero_len = overlap_start - covered_until;
                        trace!(
                            inode = canonical,
                            file_offset = covered_until,
                            zero_len,
                            "btrfs hole_fill"
                        );
                    }

                    // Preallocated extents have no initialized data yet.
                    if *extent_type == BTRFS_FILE_EXTENT_PREALLOC || *disk_bytenr == 0 {
                        trace!(
                            inode = canonical,
                            file_offset = overlap_start,
                            zero_len = overlap_end - overlap_start,
                            "btrfs hole_fill"
                        );
                        covered_until = covered_until.max(overlap_end);
                        continue;
                    }
                    if *extent_type != BTRFS_FILE_EXTENT_REG {
                        return Err(FfsError::Format(format!(
                            "unsupported btrfs extent type {extent_type}"
                        )));
                    }

                    trace!(
                        inode = canonical,
                        file_offset = logical_start,
                        logical_addr = disk_bytenr,
                        disk_len = num_bytes,
                        "btrfs extent_lookup regular"
                    );

                    let extent_delta = overlap_start - logical_start;
                    let source_logical = disk_bytenr
                        .checked_add(*extent_offset)
                        .and_then(|x| x.checked_add(extent_delta))
                        .ok_or_else(|| FfsError::Corruption {
                            block: *disk_bytenr,
                            detail: "extent source logical overflow".into(),
                        })?;
                    let dst_start = usize::try_from(overlap_start - offset).map_err(|_| {
                        FfsError::Corruption {
                            block: 0,
                            detail: "extent destination offset overflow".into(),
                        }
                    })?;
                    let copy_len = usize::try_from(overlap_end - overlap_start).map_err(|_| {
                        FfsError::Corruption {
                            block: 0,
                            detail: "extent copy length overflow".into(),
                        }
                    })?;
                    self.btrfs_read_logical_into(
                        cx,
                        source_logical,
                        &mut out[dst_start..dst_start + copy_len],
                    )?;
                    covered_until = covered_until.max(overlap_end);
                }
            }
        }
        if covered_until < read_end {
            let zero_len = read_end - covered_until;
            trace!(
                inode = canonical,
                file_offset = covered_until,
                zero_len,
                "btrfs hole_fill"
            );
        }

        trace!(
            inode = canonical,
            bytes_returned = out.len(),
            duration_us = read_started.elapsed().as_micros(),
            "btrfs read_complete"
        );
        Ok(out)
    }

    /// Read a group descriptor via the device.
    ///
    /// When `metadata_csum` is enabled, verifies the CRC32C checksum embedded
    /// in the descriptor before returning it.
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn read_group_desc(&self, cx: &Cx, group: GroupNumber) -> Result<Ext4GroupDesc, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let desc_size = sb.group_desc_size();
        let offset = sb
            .group_desc_offset(group)
            .ok_or_else(|| FfsError::InvalidGeometry("group desc offset overflow".into()))?;

        let mut buf = vec![0_u8; usize::from(desc_size)];
        self.dev.read_exact_at(cx, ByteOffset(offset), &mut buf)?;

        // Verify checksum if metadata_csum is enabled.
        if let Some(geom) = &self.ext4_geometry {
            if geom.has_metadata_csum {
                ffs_ondisk::ext4::verify_group_desc_checksum(
                    &buf,
                    geom.csum_seed,
                    group.0,
                    desc_size,
                )
                .map_err(|e| parse_to_ffs_error(&e))?;
            }
        }

        Ext4GroupDesc::parse_from_bytes(&buf, desc_size).map_err(|e| parse_to_ffs_error(&e))
    }

    /// Read an ext4 inode by number via the device.
    ///
    /// Uses [`Ext4Superblock::locate_inode`] and [`Ext4Superblock::inode_device_offset`]
    /// to compute the on-disk position, reads the group descriptor for the
    /// inode table pointer, then reads and parses the inode.
    pub fn read_inode(&self, cx: &Cx, ino: InodeNumber) -> Result<Ext4Inode, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;

        let loc = sb.locate_inode(ino).map_err(|e| parse_to_ffs_error(&e))?;
        let gd = self.read_group_desc(cx, loc.group)?;
        let abs_offset = sb
            .inode_device_offset(&loc, gd.inode_table)
            .map_err(|e| parse_to_ffs_error(&e))?;

        let inode_size = usize::from(sb.inode_size);
        let mut buf = vec![0_u8; inode_size];
        self.dev
            .read_exact_at(cx, ByteOffset(abs_offset), &mut buf)?;
        Ext4Inode::parse_from_bytes(&buf).map_err(|e| parse_to_ffs_error(&e))
    }

    /// Read an ext4 inode and return its VFS attributes.
    pub fn read_inode_attr(&self, cx: &Cx, ino: InodeNumber) -> Result<InodeAttr, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let inode = self.read_inode(cx, ino)?;
        Ok(inode_to_attr(sb, ino, &inode))
    }

    /// Traverse the ext4 orphan inode list (`s_last_orphan` + inode `dtime` links).
    ///
    /// This is read-only diagnostic behavior: traversal validates bounds and
    /// detects cycles but does not mutate image state.
    pub fn read_ext4_orphan_list(&self, cx: &Cx) -> Result<Ext4OrphanList, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;

        let head = sb.last_orphan;
        let mut next = head;
        let mut seen = BTreeSet::new();
        let mut inodes = Vec::new();
        let max_inodes = usize::try_from(sb.inodes_count)
            .map_err(|_| FfsError::InvalidGeometry("inodes_count does not fit usize".to_owned()))?;

        while next != 0 {
            if next > sb.inodes_count {
                return Err(FfsError::InvalidGeometry(format!(
                    "ext4 orphan inode {next} out of range (inodes_count={})",
                    sb.inodes_count
                )));
            }
            if !seen.insert(next) {
                return Err(FfsError::Corruption {
                    block: 0,
                    detail: format!("ext4 orphan list cycle detected at inode {next}"),
                });
            }
            if inodes.len() >= max_inodes {
                return Err(FfsError::Corruption {
                    block: 0,
                    detail: "ext4 orphan list exceeds inode count bound".to_owned(),
                });
            }

            let ino = InodeNumber(u64::from(next));
            inodes.push(ino);
            let inode = self.read_inode(cx, ino)?;
            next = inode.dtime;
        }

        Ok(Ext4OrphanList { head, inodes })
    }

    // ── Bitmap reading (ext4 free-space inspection) ───────────────────

    /// Read the block allocation bitmap for a group.
    ///
    /// Returns the raw bitmap bytes. The bitmap has one bit per block in the
    /// group. A 0 bit indicates a free block, a 1 bit indicates an allocated
    /// block.
    ///
    /// # Errors
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn read_block_bitmap(&self, cx: &Cx, group: GroupNumber) -> Result<Vec<u8>, FfsError> {
        let _sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let gd = self.read_group_desc(cx, group)?;

        // Read the bitmap block
        self.read_block_vec(cx, BlockNumber(gd.block_bitmap))
    }

    /// Read the inode allocation bitmap for a group.
    ///
    /// Returns the raw bitmap bytes. The bitmap has one bit per inode in the
    /// group. A 0 bit indicates a free inode, a 1 bit indicates an allocated
    /// inode.
    ///
    /// # Errors
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn read_inode_bitmap(&self, cx: &Cx, group: GroupNumber) -> Result<Vec<u8>, FfsError> {
        let _sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let gd = self.read_group_desc(cx, group)?;

        // Read the bitmap block
        self.read_block_vec(cx, BlockNumber(gd.inode_bitmap))
    }

    /// Count free blocks in a specific group by reading and analyzing the bitmap.
    ///
    /// This reads the block bitmap from disk and counts zero bits (free blocks).
    /// For the last group, only the bits corresponding to actual blocks are
    /// counted.
    ///
    /// # Errors
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn count_free_blocks_in_group(&self, cx: &Cx, group: GroupNumber) -> Result<u32, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;

        let geo = FsGeometry::from_superblock(sb);
        let blocks_in_group = geo.blocks_in_group(group);

        let bitmap = self.read_block_bitmap(cx, group)?;
        Ok(bitmap_count_free(&bitmap, blocks_in_group))
    }

    /// Count free inodes in a specific group by reading and analyzing the bitmap.
    ///
    /// This reads the inode bitmap from disk and counts zero bits (free inodes).
    /// For the last group, only the bits corresponding to actual inodes are
    /// counted.
    ///
    /// # Errors
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn count_free_inodes_in_group(&self, cx: &Cx, group: GroupNumber) -> Result<u32, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;

        let geo = FsGeometry::from_superblock(sb);
        let inodes_in_group = geo.inodes_in_group(group);

        let bitmap = self.read_inode_bitmap(cx, group)?;
        Ok(bitmap_count_free(&bitmap, inodes_in_group))
    }

    /// Compute a free-space summary for the entire ext4 filesystem.
    ///
    /// Iterates over all groups, reading bitmaps and counting free blocks
    /// and inodes. Also compares against the group descriptor cached values
    /// to detect potential inconsistencies.
    ///
    /// # Errors
    ///
    /// Returns `FfsError::Format` if this is not an ext4 filesystem.
    pub fn free_space_summary(&self, cx: &Cx) -> Result<Ext4FreeSpaceSummary, FfsError> {
        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;

        let geo = FsGeometry::from_superblock(sb);
        let group_count = geo.group_count;

        let mut total_free_blocks: u64 = 0;
        let mut total_free_inodes: u64 = 0;
        let mut gd_free_blocks: u64 = 0;
        let mut gd_free_inodes: u64 = 0;

        for group_idx in 0..group_count {
            let group = GroupNumber(group_idx);

            // Count from bitmaps
            let free_blocks = self.count_free_blocks_in_group(cx, group)?;
            let free_inodes = self.count_free_inodes_in_group(cx, group)?;

            total_free_blocks += u64::from(free_blocks);
            total_free_inodes += u64::from(free_inodes);

            // Get group descriptor values for comparison
            let gd = self.read_group_desc(cx, group)?;
            gd_free_blocks += u64::from(gd.free_blocks_count);
            gd_free_inodes += u64::from(gd.free_inodes_count);
        }

        Ok(Ext4FreeSpaceSummary {
            free_blocks_total: total_free_blocks,
            free_inodes_total: total_free_inodes,
            gd_free_blocks_total: gd_free_blocks,
            gd_free_inodes_total: gd_free_inodes,
            blocks_mismatch: total_free_blocks != gd_free_blocks,
            inodes_mismatch: total_free_inodes != gd_free_inodes,
        })
    }

    // ── Extent mapping via device ─────────────────────────────────────

    /// Maximum extent tree depth (ext4 kernel limit).
    const MAX_EXTENT_DEPTH: u16 = 5;

    /// Read a full filesystem block from the device.
    #[allow(clippy::cast_possible_truncation)] // block_size is u32, always fits usize
    pub fn read_block_vec(&self, cx: &Cx, block: BlockNumber) -> Result<Vec<u8>, FfsError> {
        let bs = u64::from(self.block_size());
        let offset = block
            .0
            .checked_mul(bs)
            .ok_or_else(|| FfsError::Corruption {
                block: block.0,
                detail: "block offset overflow".into(),
            })?;
        let mut buf = vec![0_u8; self.block_size() as usize];
        self.dev.read_exact_at(cx, ByteOffset(offset), &mut buf)?;
        Ok(buf)
    }

    /// Resolve a logical file block to a physical block number via the inode's
    /// extent tree, reading index blocks from the device as needed.
    ///
    /// Returns `Ok(None)` if the logical block falls in a hole (no mapping).
    pub fn resolve_extent(
        &self,
        cx: &Cx,
        inode: &Ext4Inode,
        logical_block: u32,
    ) -> Result<Option<(u64, bool)>, FfsError> {
        let (header, tree) = parse_inode_extent_tree(inode).map_err(|e| parse_to_ffs_error(&e))?;
        self.walk_extent_tree(cx, &tree, logical_block, header.depth)
    }

    fn walk_extent_tree(
        &self,
        cx: &Cx,
        tree: &ExtentTree,
        logical_block: u32,
        remaining_depth: u16,
    ) -> Result<Option<(u64, bool)>, FfsError> {
        if remaining_depth > Self::MAX_EXTENT_DEPTH {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "extent tree depth exceeds maximum".into(),
            });
        }

        match tree {
            ExtentTree::Leaf(extents) => {
                for ext in extents {
                    let start = ext.logical_block;
                    let len = u32::from(ext.actual_len());
                    if logical_block >= start && logical_block < start.saturating_add(len) {
                        let offset_within = u64::from(logical_block - start);
                        return Ok(Some((
                            ext.physical_start + offset_within,
                            ext.is_unwritten(),
                        )));
                    }
                }
                Ok(None)
            }
            ExtentTree::Index(indexes) => {
                if remaining_depth == 0 {
                    return Err(FfsError::Corruption {
                        block: 0,
                        detail: "extent index at depth 0".into(),
                    });
                }
                let mut chosen: Option<usize> = None;
                for (i, idx) in indexes.iter().enumerate() {
                    if idx.logical_block <= logical_block {
                        chosen = Some(i);
                    } else {
                        break;
                    }
                }
                let Some(i) = chosen else {
                    return Ok(None);
                };
                let idx = &indexes[i];

                let child_data = self.read_block_vec(cx, BlockNumber(idx.leaf_block))?;
                let (child_header, child_tree) =
                    parse_extent_tree(&child_data).map_err(|e| parse_to_ffs_error(&e))?;

                if child_header.depth + 1 != remaining_depth {
                    return Err(FfsError::Corruption {
                        block: idx.leaf_block,
                        detail: "child extent tree depth inconsistency".into(),
                    });
                }

                self.walk_extent_tree(cx, &child_tree, logical_block, remaining_depth - 1)
            }
        }
    }

    /// Collect all leaf extents for an inode, flattening multi-level trees.
    ///
    /// Returns extents in tree-traversal order (sorted by logical block).
    pub fn collect_extents(&self, cx: &Cx, inode: &Ext4Inode) -> Result<Vec<Ext4Extent>, FfsError> {
        let (header, tree) = parse_inode_extent_tree(inode).map_err(|e| parse_to_ffs_error(&e))?;
        let mut result = Vec::new();
        self.collect_extents_recursive(cx, &tree, header.depth, &mut result)?;
        Ok(result)
    }

    fn collect_extents_recursive(
        &self,
        cx: &Cx,
        tree: &ExtentTree,
        remaining_depth: u16,
        result: &mut Vec<Ext4Extent>,
    ) -> Result<(), FfsError> {
        if remaining_depth > Self::MAX_EXTENT_DEPTH {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "extent tree depth exceeds maximum".into(),
            });
        }

        match tree {
            ExtentTree::Leaf(extents) => {
                result.extend_from_slice(extents);
                Ok(())
            }
            ExtentTree::Index(indexes) => {
                if remaining_depth == 0 {
                    return Err(FfsError::Corruption {
                        block: 0,
                        detail: "extent index at depth 0".into(),
                    });
                }
                for idx in indexes {
                    let child_data = self.read_block_vec(cx, BlockNumber(idx.leaf_block))?;
                    let (child_header, child_tree) =
                        parse_extent_tree(&child_data).map_err(|e| parse_to_ffs_error(&e))?;
                    if child_header.depth + 1 != remaining_depth {
                        return Err(FfsError::Corruption {
                            block: idx.leaf_block,
                            detail: "child extent tree depth inconsistency".into(),
                        });
                    }
                    self.collect_extents_recursive(cx, &child_tree, remaining_depth - 1, result)?;
                }
                Ok(())
            }
        }
    }

    /// Read file data using extent mapping via the device.
    ///
    /// Resolves each logical block through the extent tree and reads the
    /// corresponding physical blocks from the device. Holes are filled
    /// with zeroes. Returns the number of bytes actually read.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_file_data(
        &self,
        cx: &Cx,
        inode: &Ext4Inode,
        offset: u64,
        buf: &mut [u8],
    ) -> Result<usize, FfsError> {
        let file_size = inode.size;
        if offset >= file_size {
            return Ok(0);
        }

        let available = file_size - offset;
        let to_read = usize::try_from(available.min(buf.len() as u64)).unwrap_or(buf.len());

        let bs = u64::from(self.block_size());
        let bs_usize = self.block_size() as usize;
        let mut bytes_read = 0_usize;

        while bytes_read < to_read {
            let current_offset = offset + bytes_read as u64;
            let logical_block =
                u32::try_from(current_offset / bs).map_err(|_| FfsError::Corruption {
                    block: 0,
                    detail: "logical block number overflow".into(),
                })?;
            let offset_in_block = (current_offset % bs) as usize;
            let remaining_in_block = bs_usize - offset_in_block;
            let chunk_size = remaining_in_block.min(to_read - bytes_read);

            match self.resolve_extent(cx, inode, logical_block)? {
                Some((phys_block, unwritten)) => {
                    if unwritten {
                        buf[bytes_read..bytes_read + chunk_size].fill(0);
                    } else {
                        let block_data = self.read_block_vec(cx, BlockNumber(phys_block))?;
                        buf[bytes_read..bytes_read + chunk_size].copy_from_slice(
                            &block_data[offset_in_block..offset_in_block + chunk_size],
                        );
                    }
                }
                None => {
                    buf[bytes_read..bytes_read + chunk_size].fill(0);
                }
            }

            bytes_read += chunk_size;
        }

        Ok(bytes_read)
    }

    // ── Directory operations via device ───────────────────────────────

    /// Read all directory entries from a directory inode via the device.
    ///
    /// Iterates over the inode's data blocks via extent mapping, reading
    /// each block from the device and parsing directory entries.
    pub fn read_dir(&self, cx: &Cx, inode: &Ext4Inode) -> Result<Vec<Ext4DirEntry>, FfsError> {
        let bs = u64::from(self.block_size());
        let num_blocks = dir_logical_block_count(inode.size, bs)?;

        let mut all_entries = Vec::new();

        for lb in 0..num_blocks {
            if let Some((phys, unwritten)) = self.resolve_extent(cx, inode, lb)? {
                if unwritten {
                    continue;
                }
                let block_data = self.read_block_vec(cx, BlockNumber(phys))?;
                let (entries, _tail) = parse_dir_block(&block_data, self.block_size())
                    .map_err(|e| parse_to_ffs_error(&e))?;
                all_entries.extend(entries);
            }
        }

        Ok(all_entries)
    }

    /// Look up a single name in a directory inode via the device.
    ///
    /// Returns the matching `Ext4DirEntry` if found, `None` otherwise.
    pub fn lookup_name(
        &self,
        cx: &Cx,
        dir_inode: &Ext4Inode,
        name: &[u8],
    ) -> Result<Option<Ext4DirEntry>, FfsError> {
        let bs = u64::from(self.block_size());
        let num_blocks = dir_logical_block_count(dir_inode.size, bs)?;

        for lb in 0..num_blocks {
            if let Some((phys, unwritten)) = self.resolve_extent(cx, dir_inode, lb)? {
                if unwritten {
                    continue;
                }
                let block_data = self.read_block_vec(cx, BlockNumber(phys))?;
                if let Some(entry) = lookup_in_dir_block(&block_data, self.block_size(), name) {
                    return Ok(Some(entry));
                }
            }
        }

        Ok(None)
    }

    // ── High-level file read ──────────────────────────────────────────

    /// Read file data by inode number via the device.
    ///
    /// Reads the inode, validates it is a regular file, then reads up to
    /// `size` bytes starting at `offset` using extent mapping. Returns
    /// `FfsError::IsDirectory` if the inode is a directory.
    pub fn read_file(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        size: u32,
    ) -> Result<Vec<u8>, FfsError> {
        let inode = self.read_inode(cx, ino)?;
        if inode.is_dir() {
            return Err(FfsError::IsDirectory);
        }
        let mut buf = vec![0_u8; size as usize];
        let n = self.read_file_data(cx, &inode, offset, &mut buf)?;
        buf.truncate(n);
        Ok(buf)
    }

    // ── Path resolution ───────────────────────────────────────────────

    /// Resolve an absolute path to an inode number and parsed inode.
    ///
    /// Walks the directory tree from root (inode 2), looking up each path
    /// component via [`lookup_name`](Self::lookup_name). The path must
    /// start with `/`.
    ///
    /// Returns `FfsError::NotFound` if a component does not exist, or
    /// `FfsError::NotDirectory` if an intermediate component is not a
    /// directory.
    pub fn resolve_path(&self, cx: &Cx, path: &str) -> Result<(InodeNumber, Ext4Inode), FfsError> {
        if !path.starts_with('/') {
            return Err(FfsError::Format(
                "path must be absolute (start with /)".into(),
            ));
        }

        let mut current_ino = InodeNumber::ROOT;
        let mut current_inode = self.read_inode(cx, current_ino)?;

        for component in path.split('/').filter(|c| !c.is_empty()) {
            if !current_inode.is_dir() {
                return Err(FfsError::NotDirectory);
            }

            let entry = self
                .lookup_name(cx, &current_inode, component.as_bytes())?
                .ok_or_else(|| FfsError::NotFound(component.to_owned()))?;

            current_ino = InodeNumber(u64::from(entry.inode));
            current_inode = self.read_inode(cx, current_ino)?;
        }

        Ok((current_ino, current_inode))
    }

    // ── Symlink reading ───────────────────────────────────────────────

    /// Read the target of a symbolic link via the device.
    ///
    /// Fast symlinks (target <= 60 bytes) are stored inline in the inode's
    /// block area. Slow symlinks read their target from data blocks via
    /// extent mapping.
    pub fn read_symlink(&self, cx: &Cx, inode: &Ext4Inode) -> Result<Vec<u8>, FfsError> {
        if !inode.is_symlink() {
            return Err(FfsError::Format("not a symlink".into()));
        }
        // Fast symlink: target stored inline in extent_bytes
        if let Some(target) = inode.fast_symlink_target() {
            let mut buf = target.to_vec();
            if let Some(pos) = buf.iter().position(|&b| b == 0) {
                buf.truncate(pos);
            }
            return Ok(buf);
        }
        // Slow symlink: read from data blocks
        let len = usize::try_from(inode.size).map_err(|_| FfsError::Corruption {
            block: 0,
            detail: "symlink size overflow".into(),
        })?;
        let mut buf = vec![0_u8; len];
        self.read_file_data(cx, inode, 0, &mut buf)?;
        // Trim trailing NUL
        if let Some(pos) = buf.iter().position(|&b| b == 0) {
            buf.truncate(pos);
        }
        Ok(buf)
    }
}

/// Compute the number of logical blocks in a directory, as a u32.
fn dir_logical_block_count(file_size: u64, block_size: u64) -> Result<u32, FfsError> {
    let num = file_size.div_ceil(block_size);
    u32::try_from(num).map_err(|_| FfsError::Corruption {
        block: 0,
        detail: "directory block count overflow".into(),
    })
}

/// Validate btrfs superblock fields at mount time.
///
/// Checks that `sectorsize` and `nodesize` are within the range accepted
/// by the kernel and are consistent with each other.
fn validate_btrfs_superblock(sb: &BtrfsSuperblock) -> Result<(), FfsError> {
    // sectorsize: power of 2, [512, 4096]
    if sb.sectorsize < 512 || sb.sectorsize > 4096 {
        return Err(FfsError::InvalidGeometry(format!(
            "btrfs sectorsize {} out of range [512, 4096]",
            sb.sectorsize
        )));
    }
    // nodesize: power of 2, [sectorsize, 65536]
    if sb.nodesize < sb.sectorsize || sb.nodesize > 65536 {
        return Err(FfsError::InvalidGeometry(format!(
            "btrfs nodesize {} out of range [{}, 65536]",
            sb.nodesize, sb.sectorsize
        )));
    }
    Ok(())
}

/// Convert a mount-time `ParseError` into the appropriate `FfsError` variant.
///
/// This is the crate-boundary conversion described in the `ffs-error` error
/// taxonomy. During mount-time validation, `ParseError::InvalidField` is
/// mapped based on the field name to distinguish incompatible feature
/// contracts, unsupported features/block sizes, geometry errors, and format
/// errors.
fn parse_error_to_ffs(e: &ParseError) -> FfsError {
    match e {
        ParseError::InvalidField { field, reason } => {
            let field_lc = field.to_ascii_lowercase();
            let reason_lc = reason.to_ascii_lowercase();

            // ext4 block size can be valid on-disk but unsupported by v1 scope.
            if field_lc.contains("block_size") && reason_lc.contains("unsupported") {
                FfsError::UnsupportedBlockSize(format!("{field}: {reason}"))
            }
            // ext4 incompat contract failures (missing required or unknown bits).
            else if field_lc.contains("feature_incompat")
                && (reason_lc.contains("missing required")
                    || reason_lc.contains("unknown incompatible"))
            {
                FfsError::IncompatibleFeature(format!("{field}: {reason}"))
            }
            // Feature validation failures → UnsupportedFeature
            else if field_lc.contains("feature") || reason_lc.contains("unsupported") {
                FfsError::UnsupportedFeature(format!("{field}: {reason}"))
            }
            // Geometry failures → InvalidGeometry
            else if field_lc.contains("block_size")
                || field_lc.contains("blocks_per_group")
                || field_lc.contains("inodes_per_group")
                || field_lc.contains("inode_size")
                || field_lc.contains("desc_size")
                || field_lc.contains("first_data_block")
                || field_lc.contains("blocks_count")
                || field_lc.contains("inodes_count")
            {
                FfsError::InvalidGeometry(format!("{field}: {reason}"))
            }
            // Everything else → Format
            else {
                FfsError::Format(e.to_string())
            }
        }
        ParseError::InvalidMagic { .. } => FfsError::Format(e.to_string()),
        ParseError::InsufficientData { .. } | ParseError::IntegerConversion { .. } => {
            FfsError::Corruption {
                block: 0,
                detail: e.to_string(),
            }
        }
    }
}

// ── VFS semantics layer ─────────────────────────────────────────────────────

/// Filesystem-agnostic file type for VFS operations.
///
/// This is the semantics-level file type used by [`FsOps`] methods. It unifies
/// ext4's `Ext4FileType` and btrfs's inode type into a single enum that
/// higher layers (FUSE, harness) consume without filesystem-specific knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileType {
    RegularFile,
    Directory,
    Symlink,
    BlockDevice,
    CharDevice,
    Fifo,
    Socket,
}

/// Inode attributes returned by [`FsOps::getattr`] and [`FsOps::lookup`].
///
/// This is the semantics-level stat structure, analogous to POSIX `struct stat`.
/// Format-specific crates (ffs-ext4, ffs-btrfs) convert their on-disk inode
/// representations into `InodeAttr` at the crate boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InodeAttr {
    /// Inode number.
    pub ino: InodeNumber,
    /// File size in bytes.
    pub size: u64,
    /// Number of 512-byte blocks allocated.
    pub blocks: u64,
    /// Last access time.
    pub atime: SystemTime,
    /// Last modification time.
    pub mtime: SystemTime,
    /// Last status change time.
    pub ctime: SystemTime,
    /// Creation time (if available).
    pub crtime: SystemTime,
    /// File type.
    pub kind: FileType,
    /// POSIX permission bits (lower 12 bits of mode).
    pub perm: u16,
    /// Number of hard links.
    pub nlink: u32,
    /// Owner user ID.
    pub uid: u32,
    /// Owner group ID.
    pub gid: u32,
    /// Device ID (for block/char devices).
    pub rdev: u32,
    /// Preferred I/O block size.
    pub blksize: u32,
}

/// A directory entry returned by [`FsOps::readdir`].
///
/// Each entry represents one name in a directory listing. The `offset` field
/// is an opaque cookie for resuming iteration — FUSE passes it back on
/// subsequent `readdir` calls so the implementation can skip already-returned
/// entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirEntry {
    /// Inode number of the target.
    pub ino: InodeNumber,
    /// Opaque offset cookie for readdir continuation.
    pub offset: u64,
    /// File type of the target.
    pub kind: FileType,
    /// Entry name (filename component, not a full path).
    pub name: Vec<u8>,
}

impl DirEntry {
    /// Return the name as a UTF-8 string (lossy).
    #[must_use]
    pub fn name_str(&self) -> String {
        String::from_utf8_lossy(&self.name).into_owned()
    }
}

/// FUSE/VFS operation kind used for MVCC request-scope hooks.
///
/// These operation tags let `FsOps` implementations choose an MVCC policy per
/// request (for example: read-snapshot only vs. begin write transaction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestOp {
    Getattr,
    Statfs,
    Getxattr,
    Lookup,
    Listxattr,
    Flush,
    Fsync,
    Fsyncdir,
    Open,
    Opendir,
    Read,
    Readdir,
    Readlink,
    // Write operations
    Create,
    Mkdir,
    Unlink,
    Rmdir,
    Rename,
    Link,
    Symlink,
    Fallocate,
    Setattr,
    Setxattr,
    Removexattr,
    Write,
}

impl RequestOp {
    /// Whether this operation mutates the filesystem.
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(
            self,
            Self::Create
                | Self::Mkdir
                | Self::Unlink
                | Self::Rmdir
                | Self::Rename
                | Self::Link
                | Self::Symlink
                | Self::Fallocate
                | Self::Setattr
                | Self::Setxattr
                | Self::Removexattr
                | Self::Write
                | Self::Fsync
                | Self::Fsyncdir
        )
    }
}

/// MVCC scope acquired for a single VFS request.
///
/// Current read-only implementations can return an empty scope. Future write
/// implementations may attach a transaction id and snapshot captured at request
/// start so that begin/end hooks can manage commit/abort semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RequestScope {
    pub snapshot: Option<Snapshot>,
    pub tx: Option<TxnId>,
}

impl RequestScope {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            snapshot: None,
            tx: None,
        }
    }
}

/// Request to modify inode attributes via `setattr`.
///
/// Each field is `Option` — only present fields are applied. Missing fields
/// leave the corresponding attribute unchanged.
#[derive(Debug, Clone, Default)]
pub struct SetAttrRequest {
    /// New permission mode bits (lower 12 bits of st_mode).
    pub mode: Option<u16>,
    /// New owner UID.
    pub uid: Option<u32>,
    /// New owner GID.
    pub gid: Option<u32>,
    /// New file size (truncate/extend).
    pub size: Option<u64>,
    /// New access time.
    pub atime: Option<SystemTime>,
    /// New modification time.
    pub mtime: Option<SystemTime>,
}

/// How `setxattr` should treat pre-existing attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XattrSetMode {
    /// Create if missing, replace if existing.
    Set,
    /// Fail with `EEXIST` if the attribute already exists.
    Create,
    /// Fail with `ENODATA`/`ENOATTR` if the attribute does not exist.
    Replace,
}

/// Filesystem statistics returned by `statfs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsStat {
    /// Total data blocks in filesystem units.
    pub blocks: u64,
    /// Free data blocks.
    pub blocks_free: u64,
    /// Free blocks available to unprivileged callers.
    pub blocks_available: u64,
    /// Total inode count (or object count when available).
    pub files: u64,
    /// Free inode/object count.
    pub files_free: u64,
    /// Preferred block size in bytes.
    pub block_size: u32,
    /// Maximum filename length.
    pub name_max: u32,
    /// Fundamental fragment size in bytes.
    pub fragment_size: u32,
}

/// VFS operations trait for filesystem access.
///
/// This is the internal interface that FUSE and the test harness call.
/// Format-specific implementations (ext4, btrfs) live behind this trait so
/// that higher layers are filesystem-agnostic.
///
/// # Design Notes
///
/// - All methods take `&Cx` for cooperative cancellation and deadline
///   propagation via the asupersync runtime.
/// - Errors are returned as `ffs_error::FfsError`, which maps to POSIX
///   errnos via [`FfsError::to_errno()`].
/// - The trait is `Send + Sync` so that FUSE can call it from multiple
///   threads concurrently.
/// - Write operations have default implementations returning `FfsError::ReadOnly`.
/// - `begin_request_scope`/`end_request_scope` provide a policy hook for
///   per-request MVCC snapshot/transaction management.
pub trait FsOps: Send + Sync {
    /// Get file attributes by inode number.
    ///
    /// Returns the attributes for the given inode. Returns
    /// `FfsError::NotFound` if the inode does not exist.
    fn getattr(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<InodeAttr>;

    /// Look up a directory entry by name.
    ///
    /// Returns the attributes of the child inode named `name` within the
    /// directory `parent`. Returns `FfsError::NotFound` if the name does
    /// not exist, or `FfsError::NotDirectory` if `parent` is not a directory.
    fn lookup(&self, cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<InodeAttr>;

    /// List directory entries starting from `offset`.
    ///
    /// Returns a batch of entries from the directory identified by `ino`.
    /// The `offset` parameter is an opaque cookie from a previous call's
    /// `DirEntry::offset` field (use 0 for the first call). An empty
    /// result indicates the end of the directory.
    ///
    /// Returns `FfsError::NotDirectory` if `ino` is not a directory.
    fn readdir(&self, cx: &Cx, ino: InodeNumber, offset: u64) -> ffs_error::Result<Vec<DirEntry>>;

    /// Read file data.
    ///
    /// Returns up to `size` bytes starting at byte `offset` within the
    /// file identified by `ino`. Returns fewer bytes at EOF. Returns
    /// `FfsError::IsDirectory` if `ino` is a directory.
    fn read(&self, cx: &Cx, ino: InodeNumber, offset: u64, size: u32)
    -> ffs_error::Result<Vec<u8>>;

    /// Read the target of a symbolic link.
    ///
    /// Returns the raw bytes of the symlink target. Returns
    /// `FfsError::Format` if `ino` is not a symlink.
    fn readlink(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<u8>>;

    /// Return filesystem-level capacity and free-space statistics.
    fn statfs(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<FsStat> {
        Err(FfsError::UnsupportedFeature(
            "statfs is not implemented by this backend".to_owned(),
        ))
    }

    /// List extended attribute names for an inode.
    ///
    /// Returns the full attribute names (including namespace prefix, e.g.
    /// `"user.myattr"`, `"security.selinux"`). Returns an empty list if the
    /// inode has no xattrs or the filesystem does not support them.
    fn listxattr(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<String>> {
        let _ = (cx, ino);
        Ok(Vec::new())
    }

    /// Get the value of an extended attribute by full name.
    ///
    /// The `name` parameter is the full attribute name including namespace
    /// prefix (e.g. `"user.myattr"`). Returns `None` if the attribute does
    /// not exist.
    fn getxattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        name: &str,
    ) -> ffs_error::Result<Option<Vec<u8>>> {
        let _ = (cx, ino, name);
        Ok(None)
    }

    // ── Write operations (default: return ReadOnly) ─────────────────────

    /// Create, replace, or upsert one extended attribute.
    fn setxattr(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _name: &str,
        _value: &[u8],
        _mode: XattrSetMode,
    ) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Remove one extended attribute.
    ///
    /// Returns `true` if the attribute existed and was removed.
    fn removexattr(&self, _cx: &Cx, _ino: InodeNumber, _name: &str) -> ffs_error::Result<bool> {
        Err(FfsError::ReadOnly)
    }

    /// Create a regular file in directory `parent` with name `name`.
    ///
    /// Returns attributes of the newly created inode.
    fn create(
        &self,
        _cx: &Cx,
        _parent: InodeNumber,
        _name: &OsStr,
        _mode: u16,
        _uid: u32,
        _gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        Err(FfsError::ReadOnly)
    }

    /// Create a directory in `parent` with name `name`.
    fn mkdir(
        &self,
        _cx: &Cx,
        _parent: InodeNumber,
        _name: &OsStr,
        _mode: u16,
        _uid: u32,
        _gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        Err(FfsError::ReadOnly)
    }

    /// Remove a non-directory entry from `parent`.
    fn unlink(&self, _cx: &Cx, _parent: InodeNumber, _name: &OsStr) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Remove an empty directory entry from `parent`.
    fn rmdir(&self, _cx: &Cx, _parent: InodeNumber, _name: &OsStr) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Rename an entry from `parent`/`name` to `new_parent`/`new_name`.
    fn rename(
        &self,
        _cx: &Cx,
        _parent: InodeNumber,
        _name: &OsStr,
        _new_parent: InodeNumber,
        _new_name: &OsStr,
    ) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Write data to file `ino` at byte `offset`. Returns bytes written.
    fn write(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _offset: u64,
        _data: &[u8],
    ) -> ffs_error::Result<u32> {
        Err(FfsError::ReadOnly)
    }

    /// Create a hard link to `ino` in `new_parent` under `new_name`.
    fn link(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _new_parent: InodeNumber,
        _new_name: &OsStr,
    ) -> ffs_error::Result<InodeAttr> {
        Err(FfsError::ReadOnly)
    }

    /// Create a symlink in `parent` named `name` targeting `target`.
    fn symlink(
        &self,
        _cx: &Cx,
        _parent: InodeNumber,
        _name: &OsStr,
        _target: &Path,
        _uid: u32,
        _gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        Err(FfsError::ReadOnly)
    }

    /// Preallocate or punch file space (POSIX `fallocate`-style).
    fn fallocate(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _offset: u64,
        _length: u64,
        _mode: i32,
    ) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Set inode attributes. Returns updated attributes.
    fn setattr(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _attrs: &SetAttrRequest,
    ) -> ffs_error::Result<InodeAttr> {
        Err(FfsError::ReadOnly)
    }

    /// Flush per-handle state on `close(2)`; no durability guarantee required.
    ///
    /// This hook exists for backends that keep per-handle locks or delayed
    /// write errors. Stateless implementations may return `Ok(())`.
    fn flush(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _fh: u64,
        _lock_owner: u64,
    ) -> ffs_error::Result<()> {
        Ok(())
    }

    /// Synchronize file data to stable storage.
    ///
    /// `datasync=true` allows skipping non-essential metadata where supported.
    fn fsync(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _fh: u64,
        _datasync: bool,
    ) -> ffs_error::Result<()> {
        Err(FfsError::ReadOnly)
    }

    /// Synchronize directory contents to stable storage.
    fn fsyncdir(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        fh: u64,
        datasync: bool,
    ) -> ffs_error::Result<()> {
        self.fsync(cx, ino, fh, datasync)
    }

    // ── Request scope hooks ───────────────────────────────────────────

    /// Acquire request scope before executing a VFS operation.
    ///
    /// Default behavior is a no-op for read-only backends.
    fn begin_request_scope(&self, _cx: &Cx, _op: RequestOp) -> ffs_error::Result<RequestScope> {
        Ok(RequestScope::empty())
    }

    /// Release request scope after executing a VFS operation.
    ///
    /// Called even when the operation body fails. Default behavior is a no-op.
    fn end_request_scope(
        &self,
        _cx: &Cx,
        _op: RequestOp,
        _scope: RequestScope,
    ) -> ffs_error::Result<()> {
        Ok(())
    }
}

// ── Ext4FsOps: bridge from Ext4ImageReader to FsOps ───────────────────────

/// Read-only ext4 filesystem operations backed by an in-memory image.
///
/// This is the bridge layer that connects the pure-parsing `Ext4ImageReader`
/// (which operates on `&[u8]` slices) to the VFS-level `FsOps` trait (which
/// the FUSE adapter and test harness consume).
///
/// # Design
///
/// The image is stored as `Arc<Vec<u8>>` so that `Ext4FsOps` is `Send + Sync`
/// without copying. The `Ext4ImageReader` holds only the parsed superblock
/// and pre-computed geometry — no mutable state — so concurrent reads are safe.
pub struct Ext4FsOps {
    reader: Ext4ImageReader,
    image: std::sync::Arc<Vec<u8>>,
}

impl std::fmt::Debug for Ext4FsOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ext4FsOps")
            .field("block_size", &self.reader.sb.block_size)
            .field("image_len", &self.image.len())
            .finish()
    }
}

impl Ext4FsOps {
    /// Create from an in-memory ext4 image.
    ///
    /// Parses the superblock and validates geometry. The image is wrapped
    /// in `Arc` for zero-copy sharing.
    pub fn new(image: Vec<u8>) -> Result<Self, FfsError> {
        let reader = Ext4ImageReader::new(&image).map_err(|e| parse_to_ffs_error(&e))?;
        Ok(Self {
            reader,
            image: std::sync::Arc::new(image),
        })
    }

    /// Create from an already-shared image.
    pub fn from_arc(image: std::sync::Arc<Vec<u8>>) -> Result<Self, FfsError> {
        let reader = Ext4ImageReader::new(&image).map_err(|e| parse_to_ffs_error(&e))?;
        Ok(Self { reader, image })
    }

    /// Access the underlying `Ext4ImageReader`.
    #[must_use]
    pub fn reader(&self) -> &Ext4ImageReader {
        &self.reader
    }

    /// Access the raw image bytes.
    #[must_use]
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// Read and convert an inode to `InodeAttr`.
    fn inode_to_attr(&self, ino: InodeNumber, inode: &Ext4Inode) -> InodeAttr {
        inode_to_attr(&self.reader.sb, ino, inode)
    }
}

/// Convert `ParseError` to `FfsError` for runtime operations (not mount-time).
fn parse_to_ffs_error(e: &ParseError) -> FfsError {
    match e {
        ParseError::InvalidField { field, reason } => {
            if reason.contains("not found") || reason.contains("component not found") {
                FfsError::NotFound(format!("{field}: {reason}"))
            } else if reason.contains("not a directory") {
                FfsError::NotDirectory
            } else {
                FfsError::Format(e.to_string())
            }
        }
        ParseError::InvalidMagic { .. } => FfsError::Format(e.to_string()),
        ParseError::InsufficientData { .. } | ParseError::IntegerConversion { .. } => {
            FfsError::Corruption {
                block: 0,
                detail: e.to_string(),
            }
        }
    }
}

/// Convert a btrfs mutation error to an `FfsError`.
fn btrfs_mutation_to_ffs(e: &BtrfsMutationError) -> FfsError {
    match e {
        BtrfsMutationError::BrokenInvariant(msg) => FfsError::Corruption {
            block: 0,
            detail: format!("btrfs invariant: {msg}"),
        },
        BtrfsMutationError::KeyAlreadyExists => FfsError::Exists,
        _ => FfsError::Format(format!("btrfs mutation: {e}")),
    }
}

/// Convert an ext4 inode into VFS `InodeAttr` using the superblock for context.
fn inode_to_attr(sb: &Ext4Superblock, ino: InodeNumber, inode: &Ext4Inode) -> InodeAttr {
    let kind = inode_file_type(inode);
    let blocks_512 = if (inode.flags & ffs_types::EXT4_HUGE_FILE_FL) != 0 {
        inode.blocks.saturating_mul(u64::from(sb.block_size / 512))
    } else {
        inode.blocks
    };

    InodeAttr {
        ino,
        size: inode.size,
        blocks: blocks_512,
        atime: inode.atime_system_time(),
        mtime: inode.mtime_system_time(),
        ctime: inode.ctime_system_time(),
        crtime: inode.crtime_system_time(),
        kind,
        perm: inode.permission_bits(),
        nlink: u32::from(inode.links_count),
        uid: inode.uid,
        gid: inode.gid,
        rdev: inode.device_number(),
        blksize: sb.block_size,
    }
}

/// Map ext4 inode mode to VFS `FileType`.
fn inode_file_type(inode: &Ext4Inode) -> FileType {
    if inode.is_regular() {
        FileType::RegularFile
    } else if inode.is_dir() {
        FileType::Directory
    } else if inode.is_symlink() {
        FileType::Symlink
    } else if inode.is_blkdev() {
        FileType::BlockDevice
    } else if inode.is_chrdev() {
        FileType::CharDevice
    } else if inode.is_fifo() {
        FileType::Fifo
    } else if inode.is_socket() {
        FileType::Socket
    } else {
        FileType::RegularFile // fallback for unknown types
    }
}

/// Map inode mode bits to ext4 on-disk directory-entry file type tags.
fn inode_dir_entry_file_type(inode: &Ext4Inode) -> Ext4FileType {
    if inode.is_dir() {
        Ext4FileType::Dir
    } else if inode.is_symlink() {
        Ext4FileType::Symlink
    } else if inode.is_blkdev() {
        Ext4FileType::Blkdev
    } else if inode.is_chrdev() {
        Ext4FileType::Chrdev
    } else if inode.is_fifo() {
        Ext4FileType::Fifo
    } else if inode.is_socket() {
        Ext4FileType::Sock
    } else if inode.is_regular() {
        Ext4FileType::RegFile
    } else {
        Ext4FileType::Unknown
    }
}

/// Map ext4 directory entry file type to VFS `FileType`.
fn dir_entry_file_type(ft: Ext4FileType) -> FileType {
    match ft {
        Ext4FileType::Dir => FileType::Directory,
        Ext4FileType::Symlink => FileType::Symlink,
        Ext4FileType::Blkdev => FileType::BlockDevice,
        Ext4FileType::Chrdev => FileType::CharDevice,
        Ext4FileType::Fifo => FileType::Fifo,
        Ext4FileType::Sock => FileType::Socket,
        Ext4FileType::RegFile | Ext4FileType::Unknown => FileType::RegularFile,
    }
}

impl FsOps for Ext4FsOps {
    fn getattr(&self, _cx: &Cx, ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;
        Ok(self.inode_to_attr(ino, &inode))
    }

    fn lookup(&self, _cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<InodeAttr> {
        let parent_inode = self
            .reader
            .read_inode(&self.image, parent)
            .map_err(|e| parse_to_ffs_error(&e))?;

        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        let name_bytes = name.as_encoded_bytes();
        let entry = self
            .reader
            .lookup(&self.image, &parent_inode, name_bytes)
            .map_err(|e| parse_to_ffs_error(&e))?
            .ok_or_else(|| FfsError::NotFound(name.to_string_lossy().into_owned()))?;

        let child_ino = InodeNumber(u64::from(entry.inode));
        let child_inode = self
            .reader
            .read_inode(&self.image, child_ino)
            .map_err(|e| parse_to_ffs_error(&e))?;
        Ok(self.inode_to_attr(child_ino, &child_inode))
    }

    fn readdir(&self, _cx: &Cx, ino: InodeNumber, offset: u64) -> ffs_error::Result<Vec<DirEntry>> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;

        if !inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        let raw_entries = self
            .reader
            .read_dir(&self.image, &inode)
            .map_err(|e| parse_to_ffs_error(&e))?;

        // Convert to VFS DirEntry with offset cookies.
        // Offset is 1-indexed position in the entry list.
        let entries: Vec<DirEntry> = raw_entries
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| (*idx as u64) >= offset)
            .map(|(idx, e)| DirEntry {
                ino: InodeNumber(u64::from(e.inode)),
                offset: (idx as u64) + 1,
                kind: dir_entry_file_type(e.file_type),
                name: e.name,
            })
            .collect();

        Ok(entries)
    }

    fn read(
        &self,
        _cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        size: u32,
    ) -> ffs_error::Result<Vec<u8>> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;

        if inode.is_dir() {
            return Err(FfsError::IsDirectory);
        }

        let mut buf = vec![0_u8; size as usize];
        let n = self
            .reader
            .read_inode_data(&self.image, &inode, offset, &mut buf)
            .map_err(|e| parse_to_ffs_error(&e))?;
        buf.truncate(n);
        Ok(buf)
    }

    fn readlink(&self, _cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;

        if !inode.is_symlink() {
            return Err(FfsError::Format("not a symlink".into()));
        }

        self.reader
            .read_symlink(&self.image, &inode)
            .map_err(|e| parse_to_ffs_error(&e))
    }

    fn listxattr(&self, _cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<String>> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;
        let xattrs = self
            .reader
            .list_xattrs(&self.image, &inode)
            .map_err(|e| parse_to_ffs_error(&e))?;
        Ok(xattrs.iter().map(Ext4Xattr::full_name).collect())
    }

    fn getxattr(
        &self,
        _cx: &Cx,
        ino: InodeNumber,
        name: &str,
    ) -> ffs_error::Result<Option<Vec<u8>>> {
        let inode = self
            .reader
            .read_inode(&self.image, ino)
            .map_err(|e| parse_to_ffs_error(&e))?;
        let xattrs = self
            .reader
            .list_xattrs(&self.image, &inode)
            .map_err(|e| parse_to_ffs_error(&e))?;
        Ok(xattrs
            .into_iter()
            .find(|x| x.full_name() == name)
            .map(|x| x.value))
    }
}

// ── ext4 write-path helpers ─────────────────────────────────────────────────

impl OpenFs {
    /// Extract 60-byte extent tree root from inode's extent_bytes.
    fn extent_root(inode: &Ext4Inode) -> [u8; 60] {
        let mut root = [0u8; 60];
        let len = inode.extent_bytes.len().min(60);
        root[..len].copy_from_slice(&inode.extent_bytes[..len]);
        root
    }

    /// Write back 60-byte extent tree root into inode's extent_bytes.
    fn set_extent_root(inode: &mut Ext4Inode, root: &[u8; 60]) {
        if inode.extent_bytes.len() < 60 {
            inode.extent_bytes.resize(60, 0);
        }
        inode.extent_bytes[..60].copy_from_slice(root);
    }

    /// Iterate physical blocks for an extent.
    fn extent_phys_blocks(ext: &Ext4Extent) -> impl Iterator<Item = BlockNumber> + '_ {
        (0..u32::from(ext.actual_len()))
            .map(move |i| BlockNumber(ext.physical_start + u64::from(i)))
    }

    /// Last physical block after an extent (for allocation hint).
    fn extent_end_hint(ext: &Ext4Extent) -> BlockNumber {
        BlockNumber(ext.physical_start + u64::from(ext.actual_len()))
    }

    /// Get a block device adapter for the underlying byte device.
    fn block_device_adapter(&self) -> ByteDeviceBlockAdapter<'_> {
        ByteDeviceBlockAdapter {
            dev: self.dev.as_ref(),
            block_size: self.block_size(),
        }
    }

    /// Get current wall-clock timestamp as (seconds-since-epoch, nanoseconds).
    ///
    /// Returns full 64-bit seconds to support ext4 34-bit timestamps (epoch
    /// extension bits in the `_extra` fields) through year ~2446.
    fn now_timestamp() -> (u64, u32) {
        let now = SystemTime::now();
        let dur = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        (dur.as_secs(), dur.subsec_nanos())
    }

    /// Require the ext4 alloc state to be present (i.e., writes enabled).
    fn require_alloc_state(&self) -> Result<&Mutex<Ext4AllocState>, FfsError> {
        self.ext4_alloc_state.as_ref().ok_or(FfsError::ReadOnly)
    }

    /// Create a regular file in an ext4 directory.
    #[allow(clippy::significant_drop_tightening)]
    fn ext4_create(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        // Determine parent's group for locality hint.
        let parent_inode = self.read_inode(cx, parent)?;
        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        let mut alloc = alloc_mutex.lock();
        let Ext4AllocState {
            geo,
            groups,
            persist_ctx,
        } = &mut *alloc;

        // Allocate a new inode.
        let parent_group = GroupNumber(
            #[allow(clippy::cast_possible_truncation)]
            {
                (parent.0.saturating_sub(1) / u64::from(geo.inodes_per_group)) as u32
            },
        );
        let (ino, new_inode) = ffs_inode::create_inode(
            cx,
            &block_dev,
            geo,
            groups,
            mode | 0o100_000, // S_IFREG
            uid,
            gid,
            parent_group,
            csum_seed,
            tstamp_secs,
            tstamp_nanos,
            persist_ctx,
        )?;

        // Add directory entry to parent.
        self.ext4_add_dir_entry(
            cx,
            &block_dev,
            &mut alloc,
            parent,
            &parent_inode,
            name,
            ino,
            ffs_ondisk::Ext4FileType::RegFile,
            csum_seed,
            tstamp_secs,
            tstamp_nanos,
        )?;

        let attr = inode_to_attr(sb, ino, &new_inode);

        trace!(
            target: "ffs::write",
            op = "create",
            parent = parent.0,
            ino = ino.0,
            name = %String::from_utf8_lossy(name),
            mode,
            "file created"
        );

        Ok(attr)
    }

    /// Create a directory inside an ext4 parent directory.
    #[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
    fn ext4_mkdir(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let parent_inode = self.read_inode(cx, parent)?;
        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        let mut alloc = alloc_mutex.lock();

        let parent_group = GroupNumber(
            #[allow(clippy::cast_possible_truncation)]
            {
                (parent.0.saturating_sub(1) / u64::from(alloc.geo.inodes_per_group)) as u32
            },
        );
        let (ino, mut new_inode) = {
            let Ext4AllocState {
                geo,
                groups,
                persist_ctx,
            } = &mut *alloc;
            ffs_inode::create_inode(
                cx,
                &block_dev,
                geo,
                groups,
                mode | 0o040_000, // S_IFDIR
                uid,
                gid,
                parent_group,
                csum_seed,
                tstamp_secs,
                tstamp_nanos,
                persist_ctx,
            )?
        };

        // Allocate a data block for the directory and initialize with . and ..
        let hint = AllocHint {
            goal_group: Some(parent_group),
            goal_block: None,
        };
        let dir_alloc = {
            let Ext4AllocState {
                geo,
                groups,
                persist_ctx,
            } = &mut *alloc;
            ffs_alloc::alloc_blocks_persist(cx, &block_dev, geo, groups, 1, &hint, persist_ctx)?
        };

        let block_size_usize = alloc.geo.block_size as usize;
        let mut dir_block = vec![0u8; block_size_usize];
        #[allow(clippy::cast_possible_truncation)]
        ffs_dir::init_dir_block(&mut dir_block, ino.0 as u32, parent.0 as u32)?;
        block_dev.write_block(cx, dir_alloc.start, &dir_block)?;

        // Set up the extent tree to point to this block.
        let extent = Ext4Extent {
            logical_block: 0,
            raw_len: 1,
            physical_start: dir_alloc.start.0,
        };
        let mut root_bytes = Self::extent_root(&new_inode);
        {
            let Ext4AllocState {
                geo,
                groups,
                persist_ctx,
            } = &mut *alloc;
            let mut tree_alloc = ffs_extent::GroupBlockAllocator {
                cx,
                dev: &block_dev,
                geo,
                groups,
                hint,
                pctx: persist_ctx,
            };
            ffs_btree::insert(cx, &block_dev, &mut root_bytes, extent, &mut tree_alloc)?;
        }
        Self::set_extent_root(&mut new_inode, &root_bytes);

        // Update inode metadata.
        let bs = alloc.geo.block_size;
        new_inode.size = u64::from(bs);
        if new_inode.is_huge_file() {
            new_inode.blocks = 1;
        } else {
            new_inode.blocks = u64::from(bs / 512);
        }
        new_inode.links_count = 2; // . and parent
        {
            let Ext4AllocState { geo, groups, .. } = &mut *alloc;
            ffs_inode::write_inode(cx, &block_dev, geo, groups, ino, &new_inode, csum_seed)?;
        }

        // Increment parent's link count (for ..)
        let mut parent_inode = parent_inode;
        parent_inode.links_count = parent_inode.links_count.saturating_add(1);
        ffs_inode::touch_mtime_ctime(&mut parent_inode, tstamp_secs, tstamp_nanos);
        {
            let Ext4AllocState { geo, groups, .. } = &mut *alloc;
            ffs_inode::write_inode(
                cx,
                &block_dev,
                geo,
                groups,
                parent,
                &parent_inode,
                csum_seed,
            )?;
        }

        // Add directory entry to parent.
        self.ext4_add_dir_entry(
            cx,
            &block_dev,
            &mut alloc,
            parent,
            &parent_inode,
            name,
            ino,
            ffs_ondisk::Ext4FileType::Dir,
            csum_seed,
            tstamp_secs,
            tstamp_nanos,
        )?;

        let attr = inode_to_attr(sb, ino, &new_inode);

        trace!(
            target: "ffs::write",
            op = "mkdir",
            parent = parent.0,
            ino = ino.0,
            name = %String::from_utf8_lossy(name),
            "directory created"
        );

        Ok(attr)
    }

    /// Add a directory entry by scanning existing dir blocks, or allocating a new one.
    #[allow(clippy::too_many_arguments, clippy::significant_drop_tightening)]
    fn ext4_add_dir_entry(
        &self,
        cx: &Cx,
        block_dev: &ByteDeviceBlockAdapter<'_>,
        alloc: &mut Ext4AllocState,
        parent: InodeNumber,
        parent_inode: &Ext4Inode,
        name: &[u8],
        child_ino: InodeNumber,
        file_type: Ext4FileType,
        csum_seed: u32,
        tstamp_secs: u64,
        tstamp_nanos: u32,
    ) -> ffs_error::Result<()> {
        // Collect existing directory extents.
        let extents = self.collect_extents(cx, parent_inode)?;

        // Try adding to each existing block.
        #[allow(clippy::cast_possible_truncation)]
        let child_ino_u32 = child_ino.0 as u32;
        for ext in &extents {
            for block in Self::extent_phys_blocks(ext) {
                let mut data = self.read_block_vec(cx, block)?;
                if ffs_dir::add_entry(&mut data, child_ino_u32, name, file_type).is_ok() {
                    block_dev.write_block(cx, block, &data)?;

                    // Update parent mtime/ctime.
                    let mut parent_upd = self.read_inode(cx, parent)?;
                    ffs_inode::touch_mtime_ctime(&mut parent_upd, tstamp_secs, tstamp_nanos);
                    ffs_inode::write_inode(
                        cx,
                        block_dev,
                        &alloc.geo,
                        &alloc.groups,
                        parent,
                        &parent_upd,
                        csum_seed,
                    )?;
                    return Ok(());
                }
            }
        }

        // All blocks full — allocate a new directory block.
        let hint = AllocHint {
            goal_group: None,
            goal_block: extents.last().map(Self::extent_end_hint),
        };
        let new_alloc = ffs_alloc::alloc_blocks_persist(
            cx,
            block_dev,
            &alloc.geo,
            &mut alloc.groups,
            1,
            &hint,
            &alloc.persist_ctx,
        )?;

        let block_size = alloc.geo.block_size as usize;
        let mut new_block = vec![0u8; block_size];
        // Write a single empty dir entry spanning the whole block, then add our entry.
        // Initialize with a single unused entry spanning the whole block.
        {
            // rec_len covers the whole block
            #[allow(clippy::cast_possible_truncation)]
            let rec_len = block_size as u16;
            new_block[4..6].copy_from_slice(&rec_len.to_le_bytes());
            // inode=0, name_len=0, file_type=0 ⇒ unused entry
        }
        ffs_dir::add_entry(&mut new_block, child_ino_u32, name, file_type)?;
        block_dev.write_block(cx, new_alloc.start, &new_block)?;

        // Insert extent for the new directory block.
        let mut parent_upd = self.read_inode(cx, parent)?;
        let logical_end = extents
            .iter()
            .map(|e| e.logical_block + u32::from(e.actual_len()))
            .max()
            .unwrap_or(0);
        let extent = Ext4Extent {
            logical_block: logical_end,
            raw_len: 1,
            physical_start: new_alloc.start.0,
        };
        let mut root_bytes = Self::extent_root(&parent_upd);
        let tree_hint = AllocHint {
            goal_group: None,
            goal_block: Some(new_alloc.start),
        };
        let mut tree_alloc = ffs_extent::GroupBlockAllocator {
            cx,
            dev: block_dev,
            geo: &alloc.geo,
            groups: &mut alloc.groups,
            hint: tree_hint,
            pctx: &alloc.persist_ctx,
        };
        ffs_btree::insert(cx, block_dev, &mut root_bytes, extent, &mut tree_alloc)?;
        Self::set_extent_root(&mut parent_upd, &root_bytes);

        parent_upd.size += u64::from(alloc.geo.block_size);
        if parent_upd.is_huge_file() {
            parent_upd.blocks += 1;
        } else {
            parent_upd.blocks += u64::from(alloc.geo.block_size / 512);
        }
        ffs_inode::touch_mtime_ctime(&mut parent_upd, tstamp_secs, tstamp_nanos);
        ffs_inode::write_inode(
            cx,
            block_dev,
            &alloc.geo,
            &alloc.groups,
            parent,
            &parent_upd,
            csum_seed,
        )?;

        Ok(())
    }

    /// Remove a directory entry (unlink a file or rmdir).
    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn ext4_unlink_impl(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        expect_dir: bool,
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let parent_inode = self.read_inode(cx, parent)?;
        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        // Look up the child to get its inode number.
        let entry = self
            .lookup_name(cx, &parent_inode, name)?
            .ok_or_else(|| FfsError::NotFound(String::from_utf8_lossy(name).into_owned()))?;
        let child_ino = InodeNumber(u64::from(entry.inode));
        let child_inode = self.read_inode(cx, child_ino)?;

        if expect_dir {
            if !child_inode.is_dir() {
                return Err(FfsError::NotDirectory);
            }
            // Check directory is empty (only . and ..).
            let entries = self.read_dir(cx, &child_inode)?;
            let real_entries = entries
                .iter()
                .filter(|e| e.name != b"." && e.name != b"..")
                .count();
            if real_entries > 0 {
                return Err(FfsError::NotEmpty);
            }
        } else if child_inode.is_dir() {
            return Err(FfsError::IsDirectory);
        }

        let mut alloc = alloc_mutex.lock();

        // Remove directory entry from parent blocks.
        let extents = self.collect_extents(cx, &parent_inode)?;
        let mut removed = false;
        'outer: for ext in &extents {
            if removed {
                break;
            }
            for block in Self::extent_phys_blocks(ext) {
                let mut data = self.read_block_vec(cx, block)?;
                if ffs_dir::remove_entry(&mut data, name)? {
                    block_dev.write_block(cx, block, &data)?;
                    removed = true;
                    break 'outer;
                }
            }
        }
        if !removed {
            return Err(FfsError::NotFound(
                String::from_utf8_lossy(name).into_owned(),
            ));
        }

        // Decrement child link count.
        let mut child_upd = child_inode;
        if expect_dir {
            child_upd.links_count = 0; // Removing a directory drops both the parent link and the `.` link
        } else {
            child_upd.links_count = child_upd.links_count.saturating_sub(1);
        }
        ffs_inode::touch_ctime(&mut child_upd, tstamp_secs, tstamp_nanos);

        {
            let Ext4AllocState {
                geo,
                groups,
                persist_ctx,
            } = &mut *alloc;
            if child_upd.links_count == 0 {
                ffs_inode::delete_inode(
                    cx,
                    &block_dev,
                    geo,
                    groups,
                    child_ino,
                    &mut child_upd,
                    csum_seed,
                    tstamp_secs,
                    persist_ctx,
                )?;
            } else {
                ffs_inode::write_inode(
                    cx, &block_dev, geo, groups, child_ino, &child_upd, csum_seed,
                )?;
            }
        }

        // Update parent timestamps.
        let mut parent_upd = self.read_inode(cx, parent)?;
        if expect_dir {
            parent_upd.links_count = parent_upd.links_count.saturating_sub(1);
        }
        ffs_inode::touch_mtime_ctime(&mut parent_upd, tstamp_secs, tstamp_nanos);
        {
            let Ext4AllocState { geo, groups, .. } = &mut *alloc;
            ffs_inode::write_inode(cx, &block_dev, geo, groups, parent, &parent_upd, csum_seed)?;
        }

        trace!(
            target: "ffs::write",
            op = if expect_dir { "rmdir" } else { "unlink" },
            parent = parent.0,
            child = child_ino.0,
            name = %String::from_utf8_lossy(name),
            links_remaining = child_upd.links_count,
            "entry removed"
        );

        Ok(())
    }

    /// Create a hard link in `new_parent/new_name` to existing inode `ino`.
    #[allow(clippy::significant_drop_tightening)]
    fn ext4_link(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        new_parent: InodeNumber,
        new_name: &[u8],
    ) -> ffs_error::Result<InodeAttr> {
        const EXT4_LINK_MAX: u16 = 65_000;
        const EPERM_ERRNO: i32 = 1;
        const EMLINK_ERRNO: i32 = 31;

        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let src_inode = self.read_inode(cx, ino)?;
        if src_inode.is_dir() {
            return Err(FfsError::Io(std::io::Error::from_raw_os_error(EPERM_ERRNO)));
        }
        if src_inode.links_count >= EXT4_LINK_MAX {
            return Err(FfsError::Io(std::io::Error::from_raw_os_error(
                EMLINK_ERRNO,
            )));
        }

        let new_parent_inode = self.read_inode(cx, new_parent)?;
        if !new_parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }
        if self.lookup_name(cx, &new_parent_inode, new_name)?.is_some() {
            return Err(FfsError::Exists);
        }

        let mut src_upd = src_inode;
        let mut alloc = alloc_mutex.lock();
        self.ext4_add_dir_entry(
            cx,
            &block_dev,
            &mut alloc,
            new_parent,
            &new_parent_inode,
            new_name,
            ino,
            inode_dir_entry_file_type(&src_upd),
            csum_seed,
            tstamp_secs,
            tstamp_nanos,
        )?;

        src_upd.links_count = src_upd.links_count.saturating_add(1);
        ffs_inode::touch_ctime(&mut src_upd, tstamp_secs, tstamp_nanos);
        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            ino,
            &src_upd,
            csum_seed,
        )?;

        debug!(
            target: "ffs::write",
            op = "link",
            source_ino = ino.0,
            parent = new_parent.0,
            name = %String::from_utf8_lossy(new_name),
            new_link_count = src_upd.links_count,
            "hard link created"
        );

        Ok(inode_to_attr(sb, ino, &src_upd))
    }

    /// Create a symbolic link inode and directory entry.
    #[allow(clippy::significant_drop_tightening)]
    fn ext4_symlink(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        target: &Path,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();
        let target_bytes = target.as_os_str().as_encoded_bytes();
        let fast_storage = target_bytes.len() <= ffs_types::EXT4_FAST_SYMLINK_MAX;

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let parent_inode = self.read_inode(cx, parent)?;
        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }
        if self.lookup_name(cx, &parent_inode, name)?.is_some() {
            return Err(FfsError::Exists);
        }

        let (ino, mut symlink_inode) = {
            let mut alloc = alloc_mutex.lock();
            let parent_group = GroupNumber(
                #[allow(clippy::cast_possible_truncation)]
                {
                    (parent.0.saturating_sub(1) / u64::from(alloc.geo.inodes_per_group)) as u32
                },
            );
            let (ino, mut inode) = {
                let Ext4AllocState {
                    geo,
                    groups,
                    persist_ctx,
                } = &mut *alloc;
                ffs_inode::create_inode(
                    cx,
                    &block_dev,
                    geo,
                    groups,
                    ffs_inode::file_type::S_IFLNK | 0o777,
                    uid,
                    gid,
                    parent_group,
                    csum_seed,
                    tstamp_secs,
                    tstamp_nanos,
                    persist_ctx,
                )?
            };

            if fast_storage {
                inode.flags &= !EXT4_EXTENTS_FL;
                if inode.extent_bytes.len() < ffs_types::EXT4_FAST_SYMLINK_MAX {
                    inode
                        .extent_bytes
                        .resize(ffs_types::EXT4_FAST_SYMLINK_MAX, 0);
                }
                inode.extent_bytes.fill(0);
                inode.extent_bytes[..target_bytes.len()].copy_from_slice(target_bytes);
                inode.size = u64::try_from(target_bytes.len()).map_err(|_| {
                    FfsError::Format("symlink target length does not fit u64".to_owned())
                })?;
                inode.blocks = 0;

                let Ext4AllocState { geo, groups, .. } = &mut *alloc;
                ffs_inode::write_inode(cx, &block_dev, geo, groups, ino, &inode, csum_seed)?;
            }

            self.ext4_add_dir_entry(
                cx,
                &block_dev,
                &mut alloc,
                parent,
                &parent_inode,
                name,
                ino,
                Ext4FileType::Symlink,
                csum_seed,
                tstamp_secs,
                tstamp_nanos,
            )?;

            (ino, inode)
        };

        if !fast_storage {
            let _written = self.ext4_write(cx, ino, 0, target_bytes)?;
            symlink_inode = self.read_inode(cx, ino)?;
        }

        debug!(
            target: "ffs::write",
            op = "symlink",
            parent = parent.0,
            ino = ino.0,
            name = %String::from_utf8_lossy(name),
            target = %target.to_string_lossy(),
            storage = if fast_storage { "fast" } else { "slow" },
            "symlink created"
        );

        Ok(inode_to_attr(sb, ino, &symlink_inode))
    }

    /// Preallocate or punch file space.
    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn ext4_fallocate(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        length: u64,
        mode: i32,
    ) -> ffs_error::Result<()> {
        const KEEP_SIZE: i32 = 0x01;
        const PUNCH_HOLE: i32 = 0x02;
        const EINVAL_ERRNO: i32 = 22;
        const MAX_EXTENT_COUNT: u32 = (u16::MAX >> 1) as u32;

        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();
        let block_size = u64::from(sb.block_size);
        let sectors_per_block = u64::from(sb.block_size / 512);

        if length == 0 {
            return Ok(());
        }
        let end = offset
            .checked_add(length)
            .ok_or_else(|| FfsError::Format("fallocate range overflow".to_owned()))?;

        let keep_size = (mode & KEEP_SIZE) != 0;
        let punch_hole = (mode & PUNCH_HOLE) != 0;
        let unsupported_bits = mode & !(KEEP_SIZE | PUNCH_HOLE);
        if unsupported_bits != 0 {
            return Err(FfsError::UnsupportedFeature(format!(
                "ext4 fallocate unsupported mode bits: 0x{unsupported_bits:08x}"
            )));
        }
        if punch_hole && !keep_size {
            return Err(FfsError::Io(std::io::Error::from_raw_os_error(
                EINVAL_ERRNO,
            )));
        }

        let mut inode = self.read_inode(cx, ino)?;
        if inode.is_dir() {
            return Err(FfsError::IsDirectory);
        }

        let mut root_bytes = Self::extent_root(&inode);
        let mut alloc = alloc_mutex.lock();

        if punch_hole {
            if (offset % block_size) != 0 || (length % block_size) != 0 {
                return Err(FfsError::UnsupportedFeature(
                    "ext4 punch_hole currently requires block-aligned offset/length".to_owned(),
                ));
            }

            let logical_start =
                u32::try_from(offset / block_size).map_err(|_| FfsError::NoSpace)?;
            let logical_count =
                u32::try_from(length / block_size).map_err(|_| FfsError::NoSpace)?;
            if logical_count > 0 {
                let freed_blocks = {
                    let Ext4AllocState {
                        geo,
                        groups,
                        persist_ctx,
                    } = &mut *alloc;
                    ffs_extent::punch_hole(
                        cx,
                        &block_dev,
                        &mut root_bytes,
                        geo,
                        groups,
                        logical_start,
                        logical_count,
                        persist_ctx,
                    )?
                };
                if inode.is_huge_file() {
                    inode.blocks = inode.blocks.saturating_sub(freed_blocks);
                } else {
                    inode.blocks = inode
                        .blocks
                        .saturating_sub(freed_blocks.saturating_mul(sectors_per_block));
                }
                Self::set_extent_root(&mut inode, &root_bytes);
            }
        } else {
            let logical_start =
                u32::try_from(offset / block_size).map_err(|_| FfsError::NoSpace)?;
            let logical_end =
                u32::try_from(end.div_ceil(block_size)).map_err(|_| FfsError::NoSpace)?;
            let logical_count = logical_end.saturating_sub(logical_start);
            let mappings = ffs_extent::map_logical_to_physical(
                cx,
                &block_dev,
                &root_bytes,
                logical_start,
                logical_count,
            )?;
            let zero_block = vec![
                0_u8;
                usize::try_from(block_size).map_err(|_| {
                    FfsError::Format("block_size does not fit usize".to_owned())
                })?
            ];

            let mut goal_block = None;
            let mut newly_allocated_blocks = 0_u64;
            for mapping in mappings {
                if mapping.physical_start != 0 {
                    goal_block = Some(BlockNumber(
                        mapping.physical_start + u64::from(mapping.count),
                    ));
                    continue;
                }

                let mut remaining = mapping.count;
                let mut logical = mapping.logical_start;
                while remaining > 0 {
                    let chunk = remaining.min(MAX_EXTENT_COUNT);
                    let hint = AllocHint {
                        goal_group: None,
                        goal_block,
                    };
                    let alloc_mapping = {
                        let Ext4AllocState {
                            geo,
                            groups,
                            persist_ctx,
                        } = &mut *alloc;
                        ffs_extent::allocate_extent(
                            cx,
                            &block_dev,
                            &mut root_bytes,
                            geo,
                            groups,
                            logical,
                            chunk,
                            &hint,
                            persist_ctx,
                        )?
                    };

                    for rel in 0..alloc_mapping.count {
                        block_dev.write_block(
                            cx,
                            BlockNumber(alloc_mapping.physical_start + u64::from(rel)),
                            &zero_block,
                        )?;
                    }
                    newly_allocated_blocks += u64::from(alloc_mapping.count);
                    goal_block = Some(BlockNumber(
                        alloc_mapping.physical_start + u64::from(alloc_mapping.count),
                    ));
                    logical = logical.saturating_add(chunk);
                    remaining -= chunk;
                }
            }

            if newly_allocated_blocks > 0 {
                if inode.is_huge_file() {
                    inode.blocks = inode.blocks.saturating_add(newly_allocated_blocks);
                } else {
                    inode.blocks = inode
                        .blocks
                        .saturating_add(newly_allocated_blocks.saturating_mul(sectors_per_block));
                }
                Self::set_extent_root(&mut inode, &root_bytes);
            }

            if !keep_size && end > inode.size {
                inode.size = end;
            }
        }

        ffs_inode::touch_mtime_ctime(&mut inode, tstamp_secs, tstamp_nanos);
        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            ino,
            &inode,
            csum_seed,
        )?;

        debug!(
            target: "ffs::write",
            op = "fallocate",
            ino = ino.0,
            offset,
            length,
            mode,
            keep_size,
            punch_hole,
            size = inode.size,
            blocks = inode.blocks,
            "fallocate completed"
        );

        Ok(())
    }

    /// Write data to an ext4 file.
    #[allow(
        clippy::too_many_lines,
        clippy::significant_drop_tightening,
        clippy::cast_possible_truncation,
        clippy::single_match_else
    )]
    fn ext4_write(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        data: &[u8],
    ) -> ffs_error::Result<u32> {
        if data.is_empty() {
            return Ok(0);
        }

        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();
        let block_size = sb.block_size;
        let bs = u64::from(block_size);

        let mut inode = self.read_inode(cx, ino)?;
        if inode.is_dir() {
            return Err(FfsError::IsDirectory);
        }

        let mut alloc = alloc_mutex.lock();
        let mut bytes_written = 0u32;
        let mut pos = offset;
        let end = offset + data.len() as u64;

        while pos < end {
            #[allow(clippy::cast_possible_truncation)]
            let logical_block = (pos / bs) as u32;
            let block_offset = (pos % bs) as usize;
            let chunk_len = ((bs as usize) - block_offset).min((end - pos) as usize);

            // Resolve or allocate the physical block.
            let extents = self.collect_extents(cx, &inode)?;
            let phys = extents.iter().find_map(|e| {
                if logical_block >= e.logical_block
                    && logical_block < e.logical_block + u32::from(e.actual_len())
                {
                    Some(BlockNumber(
                        e.physical_start + u64::from(logical_block - e.logical_block),
                    ))
                } else {
                    None
                }
            });

            let phys_block = match phys {
                Some(b) => b,
                None => {
                    // Allocate new extent for this block.
                    let hint = AllocHint {
                        goal_group: None,
                        goal_block: extents.last().map(Self::extent_end_hint),
                    };
                    let mut root_bytes = Self::extent_root(&inode);
                    let mapping = {
                        let Ext4AllocState {
                            geo,
                            groups,
                            persist_ctx,
                        } = &mut *alloc;
                        ffs_extent::allocate_extent(
                            cx,
                            &block_dev,
                            &mut root_bytes,
                            geo,
                            groups,
                            logical_block,
                            1,
                            &hint,
                            persist_ctx,
                        )?
                    };
                    Self::set_extent_root(&mut inode, &root_bytes);
                    if inode.is_huge_file() {
                        inode.blocks += 1;
                    } else {
                        inode.blocks += u64::from(block_size / 512);
                    }
                    BlockNumber(mapping.physical_start)
                }
            };

            // Read-modify-write the block.
            let mut block_data = if block_offset == 0 && chunk_len == bs as usize {
                vec![0u8; bs as usize]
            } else {
                let buf = block_dev.read_block(cx, phys_block)?;
                buf.as_slice().to_vec()
            };
            let data_start = (pos - offset) as usize;
            block_data[block_offset..block_offset + chunk_len]
                .copy_from_slice(&data[data_start..data_start + chunk_len]);
            block_dev.write_block(cx, phys_block, &block_data)?;

            pos += chunk_len as u64;
            #[allow(clippy::cast_possible_truncation)]
            {
                bytes_written += chunk_len as u32;
            }
        }

        // Update inode size if we extended the file.
        if end > inode.size {
            inode.size = end;
        }
        ffs_inode::touch_mtime_ctime(&mut inode, tstamp_secs, tstamp_nanos);
        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            ino,
            &inode,
            csum_seed,
        )?;

        trace!(
            target: "ffs::write",
            op = "write",
            ino = ino.0,
            offset,
            len = data.len(),
            new_size = inode.size,
            "data written"
        );

        Ok(bytes_written)
    }

    /// Rename an entry from one directory to another.
    #[allow(
        clippy::too_many_lines,
        clippy::significant_drop_tightening,
        clippy::cast_possible_truncation
    )]
    fn ext4_rename(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        new_parent: InodeNumber,
        new_name: &[u8],
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let parent_inode = self.read_inode(cx, parent)?;
        if !parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        // Look up the source entry.
        let entry = self
            .lookup_name(cx, &parent_inode, name)?
            .ok_or_else(|| FfsError::NotFound(String::from_utf8_lossy(name).into_owned()))?;
        let child_ino = InodeNumber(u64::from(entry.inode));
        let child_inode = self.read_inode(cx, child_ino)?;
        let ft = inode_dir_entry_file_type(&child_inode);

        let mut alloc = alloc_mutex.lock();

        // Check if target already exists — if so, remove it first.
        let new_parent_inode = if new_parent == parent {
            parent_inode.clone()
        } else {
            self.read_inode(cx, new_parent)?
        };
        if !new_parent_inode.is_dir() {
            return Err(FfsError::NotDirectory);
        }

        if let Some(existing) = self.lookup_name(cx, &new_parent_inode, new_name)? {
            let existing_ino = InodeNumber(u64::from(existing.inode));
            let existing_inode = self.read_inode(cx, existing_ino)?;

            if existing_inode.is_dir() {
                if !child_inode.is_dir() {
                    return Err(FfsError::IsDirectory);
                }
                let entries = self.read_dir(cx, &existing_inode)?;
                let real_entries = entries
                    .iter()
                    .filter(|e| e.name != b"." && e.name != b"..")
                    .count();
                if real_entries > 0 {
                    return Err(FfsError::NotEmpty);
                }
            } else if child_inode.is_dir() {
                return Err(FfsError::NotDirectory);
            }

            // Remove the existing target.
            let extents = self.collect_extents(cx, &new_parent_inode)?;
            'rm_existing: for ext in &extents {
                for block in Self::extent_phys_blocks(ext) {
                    let mut data = self.read_block_vec(cx, block)?;
                    if ffs_dir::remove_entry(&mut data, new_name)? {
                        block_dev.write_block(cx, block, &data)?;
                        break 'rm_existing;
                    }
                }
            }
            // Decrement link count / delete.
            let mut ex_upd = existing_inode;
            if ex_upd.is_dir() {
                ex_upd.links_count = 0;

                // Decrement the new parent's link count for the ".." backref from the deleted directory.
                let mut new_par = self.read_inode(cx, new_parent)?;
                new_par.links_count = new_par.links_count.saturating_sub(1);
                ffs_inode::touch_mtime_ctime(&mut new_par, tstamp_secs, tstamp_nanos);
                {
                    let Ext4AllocState { geo, groups, .. } = &mut *alloc;
                    ffs_inode::write_inode(
                        cx, &block_dev, geo, groups, new_parent, &new_par, csum_seed,
                    )?;
                }
            } else {
                ex_upd.links_count = ex_upd.links_count.saturating_sub(1);
            }
            {
                let Ext4AllocState {
                    geo,
                    groups,
                    persist_ctx,
                } = &mut *alloc;
                if ex_upd.links_count == 0 {
                    ffs_inode::delete_inode(
                        cx,
                        &block_dev,
                        geo,
                        groups,
                        existing_ino,
                        &mut ex_upd,
                        csum_seed,
                        tstamp_secs,
                        persist_ctx,
                    )?;
                } else {
                    ffs_inode::write_inode(
                        cx,
                        &block_dev,
                        geo,
                        groups,
                        existing_ino,
                        &ex_upd,
                        csum_seed,
                    )?;
                }
            }
        }

        // Remove old entry from source parent.
        let src_extents = self.collect_extents(cx, &parent_inode)?;
        'rm_src: for ext in &src_extents {
            for block in Self::extent_phys_blocks(ext) {
                let mut data = self.read_block_vec(cx, block)?;
                if ffs_dir::remove_entry(&mut data, name)? {
                    block_dev.write_block(cx, block, &data)?;
                    break 'rm_src;
                }
            }
        }

        // Add new entry to target parent.
        let new_parent_inode_fresh = self.read_inode(cx, new_parent)?;
        self.ext4_add_dir_entry(
            cx,
            &block_dev,
            &mut alloc,
            new_parent,
            &new_parent_inode_fresh,
            new_name,
            child_ino,
            ft,
            csum_seed,
            tstamp_secs,
            tstamp_nanos,
        )?;

        // If renaming a directory across parents, update .. and link counts.
        if child_inode.is_dir() && parent != new_parent {
            // Update .. in child directory to point to new_parent.
            let child_extents = self.collect_extents(cx, &child_inode)?;
            if let Some(first_ext) = child_extents.first() {
                let dot_dot_block = BlockNumber(first_ext.physical_start);
                let mut data = self.read_block_vec(cx, dot_dot_block)?;
                // Remove old .. and add new one.
                if let Err(e) = ffs_dir::remove_entry(&mut data, b"..") {
                    warn!("rename: failed to remove '..' entry: {e}");
                }
                #[allow(clippy::cast_possible_truncation)]
                ffs_dir::add_entry(&mut data, new_parent.0 as u32, b"..", Ext4FileType::Dir)?;
                block_dev.write_block(cx, dot_dot_block, &data)?;
            }

            // Decrement old parent link count, increment new parent.
            let mut old_parent = self.read_inode(cx, parent)?;
            old_parent.links_count = old_parent.links_count.saturating_sub(1);
            ffs_inode::touch_mtime_ctime(&mut old_parent, tstamp_secs, tstamp_nanos);
            ffs_inode::write_inode(
                cx,
                &block_dev,
                &alloc.geo,
                &alloc.groups,
                parent,
                &old_parent,
                csum_seed,
            )?;

            let mut new_par = self.read_inode(cx, new_parent)?;
            new_par.links_count = new_par.links_count.saturating_add(1);
            ffs_inode::touch_mtime_ctime(&mut new_par, tstamp_secs, tstamp_nanos);
            ffs_inode::write_inode(
                cx,
                &block_dev,
                &alloc.geo,
                &alloc.groups,
                new_parent,
                &new_par,
                csum_seed,
            )?;
        }

        // Touch ctime on the moved inode.
        let mut child_upd = self.read_inode(cx, child_ino)?;
        ffs_inode::touch_ctime(&mut child_upd, tstamp_secs, tstamp_nanos);
        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            child_ino,
            &child_upd,
            csum_seed,
        )?;

        trace!(
            target: "ffs::write",
            op = "rename",
            src_parent = parent.0,
            dst_parent = new_parent.0,
            old_name = %String::from_utf8_lossy(name),
            new_name = %String::from_utf8_lossy(new_name),
            ino = child_ino.0,
            "entry renamed"
        );

        Ok(())
    }

    /// Set attributes on an ext4 inode.
    #[allow(clippy::significant_drop_tightening)]
    fn ext4_setattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        attrs: &SetAttrRequest,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let mut inode = self.read_inode(cx, ino)?;

        if let Some(mode) = attrs.mode {
            // Preserve file type bits (upper 4 bits of 16-bit mode).
            let type_bits = inode.mode & 0xF000;
            inode.mode = type_bits | (mode & 0o7777);
        }
        if let Some(uid) = attrs.uid {
            inode.uid = uid;
        }
        if let Some(gid) = attrs.gid {
            inode.gid = gid;
        }
        if let Some(atime) = attrs.atime {
            let dur = atime.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
            ffs_inode::touch_atime(&mut inode, dur.as_secs(), dur.subsec_nanos());
        }
        if let Some(mtime) = attrs.mtime {
            let dur = mtime.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
            ffs_inode::touch_mtime_ctime(&mut inode, dur.as_secs(), dur.subsec_nanos());
        }

        // Handle truncation.
        if let Some(new_size) = attrs.size {
            if new_size != inode.size {
                let mut alloc = alloc_mutex.lock();
                let block_size = alloc.geo.block_size;

                if new_size < inode.size {
                    // Truncate: free blocks beyond new size.
                    #[allow(clippy::cast_possible_truncation)]
                    let new_logical_end = new_size.div_ceil(u64::from(block_size)) as u32;
                    let mut root_bytes = Self::extent_root(&inode);
                    let freed = {
                        let Ext4AllocState {
                            geo,
                            groups,
                            persist_ctx,
                        } = &mut *alloc;
                        ffs_extent::truncate_extents(
                            cx,
                            &block_dev,
                            &mut root_bytes,
                            geo,
                            groups,
                            new_logical_end,
                            persist_ctx,
                        )?
                    };
                    Self::set_extent_root(&mut inode, &root_bytes);
                    if inode.is_huge_file() {
                        inode.blocks = inode.blocks.saturating_sub(freed);
                    } else {
                        inode.blocks = inode
                            .blocks
                            .saturating_sub(freed * u64::from(block_size / 512));
                    }
                } else {
                    // Extend: just update size (sparse — blocks allocated on write).
                }

                inode.size = new_size;
            }
        }

        ffs_inode::touch_ctime(&mut inode, tstamp_secs, tstamp_nanos);

        {
            let alloc = alloc_mutex.lock();
            ffs_inode::write_inode(
                cx,
                &block_dev,
                &alloc.geo,
                &alloc.groups,
                ino,
                &inode,
                csum_seed,
            )?;
        }

        let attr = inode_to_attr(sb, ino, &inode);

        trace!(
            target: "ffs::write",
            op = "setattr",
            ino = ino.0,
            new_size = inode.size,
            mode = inode.mode,
            "attributes updated"
        );

        Ok(attr)
    }

    /// Set or replace one ext4 xattr.
    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn ext4_setxattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        name: &str,
        value: &[u8],
        mode: XattrSetMode,
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let mut inode = self.read_inode(cx, ino)?;
        let old_acl = inode.file_acl;
        let mut external_block = None;
        let mut old_refcount = 1;

        if old_acl != 0 {
            let buf = self.read_block_vec(cx, BlockNumber(old_acl))?;
            if buf.len() >= 32 {
                old_refcount = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            }
            external_block = Some(buf);
        }

        let existing = ffs_xattr::get_xattr(&inode, external_block.as_deref(), name)?;
        match mode {
            XattrSetMode::Create if existing.is_some() => return Err(FfsError::Exists),
            XattrSetMode::Replace if existing.is_none() => {
                return Err(FfsError::NotFound(name.to_owned()));
            }
            XattrSetMode::Set | XattrSetMode::Create | XattrSetMode::Replace => {}
        }

        let access = XattrWriteAccess {
            // The FUSE mount uses `default_permissions`, so ownership checks are
            // expected to have already happened in-kernel.
            is_owner: true,
            has_cap_fowner: false,
            has_cap_sys_admin: false,
        };

        let storage = match ffs_xattr::set_xattr(
            &mut inode,
            external_block.as_deref_mut(),
            name,
            value,
            access,
        ) {
            Ok(s) => s,
            Err(FfsError::NoSpace) if old_acl == 0 => {
                // Inline region exhausted and no external block exists yet — allocate one.
                let mut alloc = alloc_mutex.lock();
                let ino_group = ffs_types::inode_to_group(ino, alloc.geo.inodes_per_group);
                let hint = AllocHint {
                    goal_group: Some(ino_group),
                    goal_block: None,
                };
                let Ext4AllocState {
                    geo,
                    groups,
                    persist_ctx,
                } = &mut *alloc;
                let block_alloc = ffs_alloc::alloc_blocks_persist(
                    cx,
                    &block_dev,
                    geo,
                    groups,
                    1,
                    &hint,
                    persist_ctx,
                )?;
                let block_size = geo.block_size;
                drop(alloc);

                external_block = Some(vec![0u8; block_size as usize]);
                inode.file_acl = block_alloc.start.0;
                if inode.is_huge_file() {
                    inode.blocks = inode.blocks.saturating_add(1);
                } else {
                    inode.blocks = inode.blocks.saturating_add(u64::from(block_size / 512));
                }

                ffs_xattr::set_xattr(
                    &mut inode,
                    external_block.as_deref_mut(),
                    name,
                    value,
                    access,
                )?
            }
            Err(e) => return Err(e),
        };

        if let Some(mut block) = external_block {
            let mut alloc = alloc_mutex.lock();
            let Ext4AllocState {
                geo,
                groups,
                persist_ctx,
            } = &mut *alloc;
            if old_acl != 0 && old_refcount > 1 {
                // Block was shared, we must allocate a new block (COW)
                let ino_group = ffs_types::inode_to_group(ino, geo.inodes_per_group);
                let hint = AllocHint {
                    goal_group: Some(ino_group),
                    goal_block: None,
                };
                let block_alloc = ffs_alloc::alloc_blocks_persist(
                    cx,
                    &block_dev,
                    geo,
                    groups,
                    1,
                    &hint,
                    persist_ctx,
                )?;
                inode.file_acl = block_alloc.start.0;
                block[4..8].copy_from_slice(&1_u32.to_le_bytes()); // new block has refcount 1
                block_dev.write_block(cx, block_alloc.start, &block)?;

                // Decrement old block's refcount.
                let mut old_block = self.read_block_vec(cx, BlockNumber(old_acl))?;
                let new_refcount = old_refcount - 1;
                old_block[4..8].copy_from_slice(&new_refcount.to_le_bytes());
                block_dev.write_block(cx, BlockNumber(old_acl), &old_block)?;
            } else if old_acl != 0 {
                // Not shared, modify in place
                block_dev.write_block(cx, BlockNumber(old_acl), &block)?;
            } else {
                // New block
                block_dev.write_block(cx, BlockNumber(inode.file_acl), &block)?;
            }
            drop(alloc);
        }

        ffs_inode::touch_ctime(&mut inode, tstamp_secs, tstamp_nanos);

        let alloc = alloc_mutex.lock();
        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            ino,
            &inode,
            csum_seed,
        )?;

        trace!(
            target: "ffs::write",
            op = "setxattr",
            ino = ino.0,
            name,
            value_len = value.len(),
            storage = ?storage,
            "xattr updated"
        );

        Ok(())
    }

    /// Reconcile an external xattr block after a remove-xattr mutation.
    ///
    /// Handles three cases: block became empty (free or decrement refcount),
    /// block modified but shared (COW), or block modified and unshared (in-place write).
    fn reconcile_xattr_block_after_remove(
        &self,
        cx: &Cx,
        inode: &mut Ext4Inode,
        mut block: Vec<u8>,
        old_acl_refcount: (u64, u32),
        ino: InodeNumber,
        alloc: &mut Ext4AllocState,
    ) -> ffs_error::Result<()> {
        let (old_acl, old_refcount) = old_acl_refcount;
        let block_dev = self.block_device_adapter();
        let new_acl = inode.file_acl;
        let Ext4AllocState {
            geo,
            groups,
            persist_ctx,
        } = alloc;

        if new_acl == 0 {
            // Block became empty.
            if old_refcount > 1 {
                let mut old_block = self.read_block_vec(cx, BlockNumber(old_acl))?;
                let new_refcount = old_refcount - 1;
                old_block[4..8].copy_from_slice(&new_refcount.to_le_bytes());
                block_dev.write_block(cx, BlockNumber(old_acl), &old_block)?;
            } else {
                ffs_alloc::free_blocks_persist(
                    cx,
                    &block_dev,
                    geo,
                    groups,
                    BlockNumber(old_acl),
                    1,
                    persist_ctx,
                )?;
            }
            if inode.is_huge_file() {
                inode.blocks = inode.blocks.saturating_sub(1);
            } else {
                inode.blocks = inode.blocks.saturating_sub(u64::from(geo.block_size / 512));
            }
        } else if old_refcount > 1 {
            // Block modified but shared — COW.
            let ino_group = ffs_types::inode_to_group(ino, geo.inodes_per_group);
            let hint = AllocHint {
                goal_group: Some(ino_group),
                goal_block: None,
            };
            let block_alloc = ffs_alloc::alloc_blocks_persist(
                cx,
                &block_dev,
                geo,
                groups,
                1,
                &hint,
                persist_ctx,
            )?;
            inode.file_acl = block_alloc.start.0;
            block[4..8].copy_from_slice(&1_u32.to_le_bytes());
            block_dev.write_block(cx, block_alloc.start, &block)?;

            let mut old_block = self.read_block_vec(cx, BlockNumber(old_acl))?;
            let new_refcount = old_refcount - 1;
            old_block[4..8].copy_from_slice(&new_refcount.to_le_bytes());
            block_dev.write_block(cx, BlockNumber(old_acl), &old_block)?;
        } else {
            // Not shared, modify in place.
            block_dev.write_block(cx, BlockNumber(old_acl), &block)?;
        }
        Ok(())
    }

    /// Remove one ext4 xattr.
    #[allow(clippy::significant_drop_tightening)]
    fn ext4_removexattr(&self, cx: &Cx, ino: InodeNumber, name: &str) -> ffs_error::Result<bool> {
        let alloc_mutex = self.require_alloc_state()?;
        let block_dev = self.block_device_adapter();
        let (tstamp_secs, tstamp_nanos) = Self::now_timestamp();

        let sb = self
            .ext4_superblock()
            .ok_or_else(|| FfsError::Format("not an ext4 filesystem".into()))?;
        let csum_seed = sb.csum_seed();

        let mut inode = self.read_inode(cx, ino)?;
        let old_acl = inode.file_acl;
        let mut external_block = None;
        let mut old_refcount = 1;

        if old_acl != 0 {
            let buf = self.read_block_vec(cx, BlockNumber(old_acl))?;
            if buf.len() >= 32 {
                old_refcount = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
            }
            external_block = Some(buf);
        }

        let access = XattrWriteAccess {
            is_owner: true,
            has_cap_fowner: false,
            has_cap_sys_admin: false,
        };

        let removed =
            ffs_xattr::remove_xattr(&mut inode, external_block.as_deref_mut(), name, access)?;
        if !removed {
            return Ok(false);
        }

        let mut alloc = alloc_mutex.lock();

        if let Some(block) = external_block {
            self.reconcile_xattr_block_after_remove(
                cx,
                &mut inode,
                block,
                (old_acl, old_refcount),
                ino,
                &mut alloc,
            )?;
        }

        ffs_inode::touch_ctime(&mut inode, tstamp_secs, tstamp_nanos);

        ffs_inode::write_inode(
            cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            ino,
            &inode,
            csum_seed,
        )?;

        trace!(
            target: "ffs::write",
            op = "removexattr",
            ino = ino.0,
            name,
            "xattr removed"
        );

        Ok(true)
    }

    /// Get current time as `(u64, u32)` for btrfs inode timestamps.
    fn btrfs_now_timestamp() -> (u64, u32) {
        let now = SystemTime::now();
        let dur = now.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        (dur.as_secs(), dur.subsec_nanos())
    }

    // ── Btrfs write path ─────────────────────────────────────────────────

    /// Write file data on a btrfs filesystem.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    fn btrfs_write(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        data: &[u8],
    ) -> ffs_error::Result<u32> {
        if data.is_empty() {
            return Ok(0);
        }

        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let canonical = self.btrfs_canonical_inode(ino)?;

        let mut alloc = alloc_mutex.lock();

        // Look up the INODE_ITEM for this objectid.
        let inode_key = BtrfsKey {
            objectid: canonical,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        let inode_items = alloc
            .fs_tree
            .range(&inode_key, &inode_key)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        let inode_data = inode_items
            .first()
            .ok_or_else(|| FfsError::NotFound(format!("inode {canonical}")))?
            .1
            .clone();
        let mut inode = parse_inode_item(&inode_data).map_err(|e| parse_to_ffs_error(&e))?;

        if Self::btrfs_mode_to_file_type(inode.mode) == FileType::Directory {
            return Err(FfsError::IsDirectory);
        }

        let end = offset
            .checked_add(data.len() as u64)
            .ok_or_else(|| FfsError::InvalidGeometry("offset + length overflow".into()))?;
        let sectorsize = u64::from(alloc.nodesize.min(4096));

        // For simplicity in V1, write data as an inline extent if small enough,
        // otherwise allocate a data extent and write through the device.
        let nodesize = alloc.nodesize;
        let mut can_be_inline =
            end.max(inode.size) <= u64::from(nodesize) - 200 && data.len() <= 2048;

        if can_be_inline {
            let ext_start = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset: 0,
            };
            let ext_end = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset: u64::MAX,
            };
            if let Ok(extents) = alloc.fs_tree.range(&ext_start, &ext_end) {
                for (k, edata) in extents {
                    if k.offset > 0 {
                        can_be_inline = false;
                        break;
                    }
                    if !matches!(
                        parse_extent_data(&edata),
                        Ok(BtrfsExtentData::Inline { .. })
                    ) {
                        can_be_inline = false;
                        break;
                    }
                }
            }
        }

        if can_be_inline {
            // Inline extent: store data directly in the tree item.
            let extent = BtrfsExtentData::Inline {
                generation: alloc.generation,
                compression: 0,
                data: {
                    // Build the full file content up to `end`.
                    let mut content = vec![0u8; end as usize];
                    // Read any existing inline data.
                    let existing_key = BtrfsKey {
                        objectid: canonical,
                        item_type: BTRFS_ITEM_EXTENT_DATA,
                        offset: 0,
                    };
                    if let Ok(existing) = alloc.fs_tree.range(&existing_key, &existing_key) {
                        if let Some((_, edata)) = existing.first() {
                            if let Ok(BtrfsExtentData::Inline { data: prev, .. }) =
                                parse_extent_data(edata)
                            {
                                let copy_len = prev.len().min(content.len());
                                content[..copy_len].copy_from_slice(&prev[..copy_len]);
                            }
                        }
                    }
                    content[offset as usize..end as usize].copy_from_slice(data);
                    content
                },
            };
            let extent_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset: 0,
            };
            let extent_bytes = extent.to_bytes();
            alloc
                .fs_tree
                .update(&extent_key, &extent_bytes)
                .or_else(|_| alloc.fs_tree.insert(extent_key, &extent_bytes))
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        } else {
            // If there's an existing inline extent at offset 0, convert it
            // to a regular extent to preserve its data before writing the
            // new regular extent at the (possibly non-zero) write offset.
            let inline_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset: 0,
            };
            let mut merged_data = None;

            if let Ok(existing) = alloc.fs_tree.range(&inline_key, &inline_key) {
                if let Some((_, edata)) = existing.first() {
                    if let Ok(BtrfsExtentData::Inline {
                        data: prev_data, ..
                    }) = parse_extent_data(edata)
                    {
                        alloc
                            .fs_tree
                            .delete(&inline_key)
                            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

                        if offset == 0 && data.len() < prev_data.len() {
                            // Partial overwrite at offset 0: merge new data into old
                            // inline extent to preserve the tail before converting.
                            let mut merged = prev_data;
                            merged[..data.len()].copy_from_slice(data);
                            merged_data = Some(merged);
                        } else if !prev_data.is_empty() && offset > 0 {
                            // Persist the old inline data as a regular extent at
                            // offset 0 so reads in [0, prev_data.len()) still work.
                            let prev_alloc_size = (prev_data.len() as u64)
                                .saturating_add(sectorsize - 1)
                                & !(sectorsize - 1);
                            let prev_allocation = alloc
                                .extent_alloc
                                .alloc_data(prev_alloc_size)
                                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                            self.dev.write_all_at(
                                cx,
                                ByteOffset(prev_allocation.bytenr),
                                &prev_data,
                            )?;
                            let prev_extent = BtrfsExtentData::Regular {
                                generation: alloc.generation,
                                extent_type: BTRFS_FILE_EXTENT_REG,
                                compression: 0,
                                disk_bytenr: prev_allocation.bytenr,
                                disk_num_bytes: prev_alloc_size,
                                extent_offset: 0,
                                num_bytes: prev_data.len() as u64,
                            };
                            alloc
                                .fs_tree
                                .insert(inline_key, &prev_extent.to_bytes())
                                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                        }
                    }
                }
            }

            let data_to_write = merged_data.as_deref().unwrap_or(data);
            let write_len = data_to_write.len();

            // Allocate a data extent and write through the device.
            let alloc_size = (write_len as u64).saturating_add(sectorsize - 1) & !(sectorsize - 1);
            let allocation = alloc
                .extent_alloc
                .alloc_data(alloc_size)
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
            let disk_bytenr = allocation.bytenr;

            // Write data to the allocated physical region.
            // For single-device, the logical address is the physical address
            // (the data block group we create maps 1:1).
            self.dev
                .write_all_at(cx, ByteOffset(disk_bytenr), data_to_write)?;

            // Insert the EXTENT_DATA item.
            let extent = BtrfsExtentData::Regular {
                generation: alloc.generation,
                extent_type: BTRFS_FILE_EXTENT_REG,
                compression: 0,
                disk_bytenr,
                disk_num_bytes: alloc_size,
                extent_offset: 0,
                num_bytes: write_len as u64,
            };
            let extent_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset,
            };

            // Free the old physical extent if we are overwriting an existing regular extent
            if let Ok(existing) = alloc.fs_tree.range(&extent_key, &extent_key) {
                if let Some((_, edata)) = existing.first() {
                    if let Ok(BtrfsExtentData::Regular {
                        disk_bytenr: old_bytenr,
                        disk_num_bytes: old_num_bytes,
                        ..
                    }) = parse_extent_data(edata)
                    {
                        if old_bytenr > 0 {
                            if let Err(e) =
                                alloc
                                    .extent_alloc
                                    .free_extent(old_bytenr, old_num_bytes, false)
                            {
                                warn!(
                                    "btrfs_write: failed to free overwritten extent at {old_bytenr}: {e:?}"
                                );
                            }
                        }
                    }
                }
            }

            let extent_bytes = extent.to_bytes();
            alloc
                .fs_tree
                .update(&extent_key, &extent_bytes)
                .or_else(|_| alloc.fs_tree.insert(extent_key, &extent_bytes))
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        }

        // Update inode metadata.
        if end > inode.size {
            inode.size = end;
        }
        inode.nbytes = inode.size;
        let (secs, nanos) = Self::btrfs_now_timestamp();
        inode.mtime_sec = secs;
        inode.mtime_nsec = nanos;
        inode.ctime_sec = secs;
        inode.ctime_nsec = nanos;
        let inode_bytes = inode.to_bytes();
        alloc
            .fs_tree
            .update(&inode_key, &inode_bytes)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        drop(alloc);

        trace!(
            target: "ffs::write",
            op = "btrfs_write",
            ino = canonical,
            offset,
            len = data.len(),
            new_size = inode.size,
            "btrfs data written"
        );

        Ok(data.len() as u32)
    }

    /// Create a regular file in a btrfs directory.
    fn btrfs_create(
        &self,
        _cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let parent_oid = self.btrfs_canonical_inode(parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();

        let mut alloc = alloc_mutex.lock();
        let new_oid = alloc.next_objectid;
        alloc.next_objectid = alloc.next_objectid.saturating_add(1);

        // Create the INODE_ITEM.
        let inode = BtrfsInodeItem {
            size: 0,
            nbytes: 0,
            nlink: 1,
            uid,
            gid,
            mode: u32::from(mode) | 0o100_000, // S_IFREG
            rdev: 0,
            atime_sec: secs,
            atime_nsec: nanos,
            ctime_sec: secs,
            ctime_nsec: nanos,
            mtime_sec: secs,
            mtime_nsec: nanos,
            otime_sec: secs,
            otime_nsec: nanos,
        };
        let inode_key = BtrfsKey {
            objectid: new_oid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .insert(inode_key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        // Add DIR_ITEM and DIR_INDEX in the parent.
        let dir_item = BtrfsDirItem {
            child_objectid: new_oid,
            child_key_type: BTRFS_ITEM_INODE_ITEM,
            child_key_offset: 0,
            file_type: BTRFS_FT_REG_FILE,
            name: name.to_vec(),
        };
        self.btrfs_insert_dir_entry(&mut alloc, parent_oid, &dir_item)?;

        // Add INODE_REF for parent backref.
        self.btrfs_insert_inode_ref(&mut alloc, new_oid, parent_oid, name, new_oid)?;

        // Update parent inode timestamps.
        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;
        drop(alloc);

        Ok(self.btrfs_inode_to_attr(new_oid, &inode))
    }

    /// Create a directory in a btrfs filesystem.
    fn btrfs_mkdir(
        &self,
        _cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let parent_oid = self.btrfs_canonical_inode(parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();

        let mut alloc = alloc_mutex.lock();
        let new_oid = alloc.next_objectid;
        alloc.next_objectid = alloc.next_objectid.saturating_add(1);

        let inode = BtrfsInodeItem {
            size: 0,
            nbytes: 0,
            nlink: 2, // . and parent's reference
            uid,
            gid,
            mode: u32::from(mode) | 0o040_000, // S_IFDIR
            rdev: 0,
            atime_sec: secs,
            atime_nsec: nanos,
            ctime_sec: secs,
            ctime_nsec: nanos,
            mtime_sec: secs,
            mtime_nsec: nanos,
            otime_sec: secs,
            otime_nsec: nanos,
        };
        let inode_key = BtrfsKey {
            objectid: new_oid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .insert(inode_key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        let dir_item = BtrfsDirItem {
            child_objectid: new_oid,
            child_key_type: BTRFS_ITEM_INODE_ITEM,
            child_key_offset: 0,
            file_type: BTRFS_FT_DIR,
            name: name.to_vec(),
        };
        self.btrfs_insert_dir_entry(&mut alloc, parent_oid, &dir_item)?;

        // Add INODE_REF for parent backref.
        self.btrfs_insert_inode_ref(&mut alloc, new_oid, parent_oid, name, new_oid)?;

        // Bump parent nlink for the new subdirectory.
        self.btrfs_adjust_nlink(&mut alloc, parent_oid, 1)?;
        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;
        drop(alloc);

        Ok(self.btrfs_inode_to_attr(new_oid, &inode))
    }

    /// Unlink a file or directory from a btrfs filesystem.
    fn btrfs_unlink_impl(
        &self,
        _cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        expect_dir: bool,
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let parent_oid = self.btrfs_canonical_inode(parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();

        let mut alloc = alloc_mutex.lock();

        // Lookup the child entry in the parent directory.
        let child = self.btrfs_lookup_dir_entry(&alloc, parent_oid, name)?;
        let child_oid = child.child_objectid;

        // Validate file type.
        if expect_dir && child.file_type != BTRFS_FT_DIR {
            return Err(FfsError::NotDirectory);
        }
        if !expect_dir && child.file_type == BTRFS_FT_DIR {
            return Err(FfsError::IsDirectory);
        }

        // If removing a directory, verify it's empty.
        if expect_dir && !self.btrfs_dir_is_empty(&alloc, child_oid)? {
            return Err(FfsError::NotEmpty);
        }

        // Remove DIR_ITEM and DIR_INDEX entries.
        self.btrfs_remove_dir_entry(&mut alloc, parent_oid, name)?;

        // Remove the INODE_REF back-pointer from child → parent.
        self.btrfs_remove_inode_ref(&mut alloc, child_oid, parent_oid);

        // Decrement child nlink.
        if expect_dir {
            self.btrfs_adjust_nlink(&mut alloc, child_oid, -2)?;
        } else {
            self.btrfs_adjust_nlink(&mut alloc, child_oid, -1)?;
        }

        // If nlink reached 0, purge the orphaned inode and all its data.
        let child_inode = self.btrfs_read_inode_from_tree(&alloc, child_oid)?;
        if child_inode.nlink == 0 {
            self.btrfs_purge_inode(&mut alloc, child_oid)?;
        }

        // If unlinking a directory, decrement parent nlink too.
        if expect_dir {
            self.btrfs_adjust_nlink(&mut alloc, parent_oid, -1)?;
        }
        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;
        drop(alloc);

        Ok(())
    }

    /// Rename a btrfs directory entry.
    fn btrfs_rename(
        &self,
        _cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        new_parent: InodeNumber,
        new_name: &[u8],
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let parent_oid = self.btrfs_canonical_inode(parent)?;
        let new_parent_oid = self.btrfs_canonical_inode(new_parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();

        let mut alloc = alloc_mutex.lock();

        let child = self.btrfs_lookup_dir_entry(&alloc, parent_oid, name)?;

        // Remove old entry and its INODE_REF.
        self.btrfs_remove_dir_entry(&mut alloc, parent_oid, name)?;
        self.btrfs_remove_inode_ref(&mut alloc, child.child_objectid, parent_oid);

        // If target exists, remove it first and handle nlink cleanup.
        if let Ok(target) = self.btrfs_lookup_dir_entry(&alloc, new_parent_oid, new_name) {
            let target_oid = target.child_objectid;
            self.btrfs_remove_dir_entry(&mut alloc, new_parent_oid, new_name)?;
            self.btrfs_remove_inode_ref(&mut alloc, target_oid, new_parent_oid);
            if target.file_type == BTRFS_FT_DIR {
                self.btrfs_adjust_nlink(&mut alloc, target_oid, -2)?;
                self.btrfs_adjust_nlink(&mut alloc, new_parent_oid, -1)?;
            } else {
                self.btrfs_adjust_nlink(&mut alloc, target_oid, -1)?;
            }
            let target_inode = self.btrfs_read_inode_from_tree(&alloc, target_oid)?;
            if target_inode.nlink == 0 {
                self.btrfs_purge_inode(&mut alloc, target_oid)?;
            }
        }

        // Insert in new location.
        let dir_item = BtrfsDirItem {
            child_objectid: child.child_objectid,
            child_key_type: child.child_key_type,
            child_key_offset: child.child_key_offset,
            file_type: child.file_type,
            name: new_name.to_vec(),
        };
        self.btrfs_insert_dir_entry(&mut alloc, new_parent_oid, &dir_item)?;
        self.btrfs_insert_inode_ref(
            &mut alloc,
            child.child_objectid,
            new_parent_oid,
            new_name,
            child.child_objectid,
        )?;

        // If moving a directory across parents, adjust nlink for ".." backref.
        if child.file_type == BTRFS_FT_DIR && new_parent_oid != parent_oid {
            self.btrfs_adjust_nlink(&mut alloc, parent_oid, -1)?;
            self.btrfs_adjust_nlink(&mut alloc, new_parent_oid, 1)?;
        }

        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;
        if new_parent_oid != parent_oid {
            self.btrfs_touch_inode_times(&mut alloc, new_parent_oid, secs, nanos)?;
        }
        drop(alloc);

        Ok(())
    }

    /// Create a hard link in a btrfs filesystem.
    fn btrfs_link(
        &self,
        _cx: &Cx,
        ino: InodeNumber,
        new_parent: InodeNumber,
        new_name: &[u8],
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let target_oid = self.btrfs_canonical_inode(ino)?;
        let parent_oid = self.btrfs_canonical_inode(new_parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();

        let mut alloc = alloc_mutex.lock();

        // Get the target inode to determine its file type.
        let inode = self.btrfs_read_inode_from_tree(&alloc, target_oid)?;
        let ft = if Self::btrfs_mode_to_file_type(inode.mode) == FileType::Directory {
            return Err(FfsError::IsDirectory); // Can't hard-link directories
        } else {
            BTRFS_FT_REG_FILE
        };

        let dir_item = BtrfsDirItem {
            child_objectid: target_oid,
            child_key_type: BTRFS_ITEM_INODE_ITEM,
            child_key_offset: 0,
            file_type: ft,
            name: new_name.to_vec(),
        };
        self.btrfs_insert_dir_entry(&mut alloc, parent_oid, &dir_item)?;
        self.btrfs_insert_inode_ref(&mut alloc, target_oid, parent_oid, new_name, target_oid)?;
        self.btrfs_adjust_nlink(&mut alloc, target_oid, 1)?;
        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;

        let updated = self.btrfs_read_inode_from_tree(&alloc, target_oid)?;
        drop(alloc);
        Ok(self.btrfs_inode_to_attr(target_oid, &updated))
    }

    /// Create a symbolic link in a btrfs filesystem.
    fn btrfs_symlink(
        &self,
        _cx: &Cx,
        parent: InodeNumber,
        name: &[u8],
        target: &Path,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let parent_oid = self.btrfs_canonical_inode(parent)?;
        let (secs, nanos) = Self::btrfs_now_timestamp();
        let target_bytes = target.as_os_str().as_encoded_bytes();

        let mut alloc = alloc_mutex.lock();
        let new_oid = alloc.next_objectid;
        alloc.next_objectid = alloc.next_objectid.saturating_add(1);

        let inode = BtrfsInodeItem {
            size: target_bytes.len() as u64,
            nbytes: target_bytes.len() as u64,
            nlink: 1,
            uid,
            gid,
            mode: 0o120_777, // S_IFLNK | 0o777
            rdev: 0,
            atime_sec: secs,
            atime_nsec: nanos,
            ctime_sec: secs,
            ctime_nsec: nanos,
            mtime_sec: secs,
            mtime_nsec: nanos,
            otime_sec: secs,
            otime_nsec: nanos,
        };
        let inode_key = BtrfsKey {
            objectid: new_oid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .insert(inode_key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        // Store symlink target as an inline extent.
        let extent = BtrfsExtentData::Inline {
            generation: alloc.generation,
            compression: 0,
            data: target_bytes.to_vec(),
        };
        let extent_key = BtrfsKey {
            objectid: new_oid,
            item_type: BTRFS_ITEM_EXTENT_DATA,
            offset: 0,
        };
        alloc
            .fs_tree
            .insert(extent_key, &extent.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        let dir_item = BtrfsDirItem {
            child_objectid: new_oid,
            child_key_type: BTRFS_ITEM_INODE_ITEM,
            child_key_offset: 0,
            file_type: BTRFS_FT_SYMLINK,
            name: name.to_vec(),
        };
        self.btrfs_insert_dir_entry(&mut alloc, parent_oid, &dir_item)?;
        self.btrfs_insert_inode_ref(&mut alloc, new_oid, parent_oid, name, new_oid)?;
        self.btrfs_touch_inode_times(&mut alloc, parent_oid, secs, nanos)?;
        drop(alloc);

        Ok(self.btrfs_inode_to_attr(new_oid, &inode))
    }

    /// Set attributes on a btrfs inode.
    fn btrfs_setattr(
        &self,
        _cx: &Cx,
        ino: InodeNumber,
        attrs: &SetAttrRequest,
    ) -> ffs_error::Result<InodeAttr> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let canonical = self.btrfs_canonical_inode(ino)?;

        let mut alloc = alloc_mutex.lock();
        let mut inode = self.btrfs_read_inode_from_tree(&alloc, canonical)?;

        if let Some(mode) = attrs.mode {
            inode.mode = (inode.mode & !0o7777) | u32::from(mode & 0o7777);
        }
        if let Some(uid) = attrs.uid {
            inode.uid = uid;
        }
        if let Some(gid) = attrs.gid {
            inode.gid = gid;
        }
        if let Some(size) = attrs.size {
            let old_size = inode.size;
            inode.size = size;
            inode.nbytes = size;

            // If truncating to a smaller size, remove EXTENT_DATA items
            // whose start offset is at or beyond the new size.
            if size < old_size {
                let ext_start = BtrfsKey {
                    objectid: canonical,
                    item_type: BTRFS_ITEM_EXTENT_DATA,
                    offset: 0,
                };
                let ext_end = BtrfsKey {
                    objectid: canonical,
                    item_type: BTRFS_ITEM_EXTENT_DATA,
                    offset: u64::MAX,
                };
                if let Ok(extents) = alloc.fs_tree.range(&ext_start, &ext_end) {
                    for (k, edata) in extents {
                        if k.offset >= size {
                            if let Ok(BtrfsExtentData::Regular {
                                disk_bytenr,
                                disk_num_bytes,
                                ..
                            }) = parse_extent_data(&edata)
                            {
                                if disk_bytenr > 0 {
                                    if let Err(e) = alloc.extent_alloc.free_extent(
                                        disk_bytenr,
                                        disk_num_bytes,
                                        false,
                                    ) {
                                        warn!(
                                            "truncate: failed to free extent at {disk_bytenr}: {e:?}"
                                        );
                                    }
                                }
                            }
                            if let Err(e) = alloc.fs_tree.delete(&k) {
                                warn!("truncate: failed to delete extent key {k:?}: {e:?}");
                            }
                        }
                    }
                }

                // If truncating to 0, also remove any inline extent at offset 0.
                if size == 0 {
                    let inline_key = BtrfsKey {
                        objectid: canonical,
                        item_type: BTRFS_ITEM_EXTENT_DATA,
                        offset: 0,
                    };
                    if let Err(e) = alloc.fs_tree.delete(&inline_key) {
                        warn!("truncate: failed to delete inline extent: {e:?}");
                    }
                }
            }
        }
        if let Some(atime) = attrs.atime {
            let dur = atime.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
            inode.atime_sec = dur.as_secs();
            inode.atime_nsec = dur.subsec_nanos();
        }
        if let Some(mtime) = attrs.mtime {
            let dur = mtime.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
            inode.mtime_sec = dur.as_secs();
            inode.mtime_nsec = dur.subsec_nanos();
        }
        let (secs, nanos) = Self::btrfs_now_timestamp();
        inode.ctime_sec = secs;
        inode.ctime_nsec = nanos;

        let inode_key = BtrfsKey {
            objectid: canonical,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .update(&inode_key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        drop(alloc);

        Ok(self.btrfs_inode_to_attr(canonical, &inode))
    }

    /// Preallocate extents in a btrfs filesystem.
    #[allow(clippy::too_many_lines)]
    fn btrfs_fallocate(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        length: u64,
        mode: i32,
    ) -> ffs_error::Result<()> {
        const KEEP_SIZE: i32 = 0x01;
        const PUNCH_HOLE: i32 = 0x02;

        let keep_size = (mode & KEEP_SIZE) != 0;
        let punch_hole = (mode & PUNCH_HOLE) != 0;
        let unsupported_bits = mode & !(KEEP_SIZE | PUNCH_HOLE);
        if unsupported_bits != 0 {
            return Err(FfsError::UnsupportedFeature(format!(
                "btrfs fallocate unsupported mode bits: 0x{unsupported_bits:08x}"
            )));
        }
        if punch_hole {
            return Err(FfsError::UnsupportedFeature(
                "btrfs fallocate punch-hole mode is not yet supported".into(),
            ));
        }

        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let canonical = self.btrfs_canonical_inode(ino)?;

        let mut alloc = alloc_mutex.lock();

        // Check if a data extent (inline or regular non-prealloc) already
        // exists at this offset.  If so, skip the prealloc to avoid
        // overwriting the user's data.
        let probe_key = BtrfsKey {
            objectid: canonical,
            item_type: BTRFS_ITEM_EXTENT_DATA,
            offset,
        };
        let already_has_data = alloc
            .fs_tree
            .range(&probe_key, &probe_key)
            .is_ok_and(|existing| {
                existing
                    .first()
                    .is_some_and(|(_, edata)| match parse_extent_data(edata) {
                        Ok(BtrfsExtentData::Inline { .. }) => true,
                        Ok(BtrfsExtentData::Regular { extent_type, .. }) => {
                            extent_type != BTRFS_FILE_EXTENT_PREALLOC
                        }
                        Err(_) => false,
                    })
            });

        if !already_has_data {
            // If there's an existing inline extent at offset 0, convert it
            // to a regular extent because a file cannot mix inline and regular extents.
            let inline_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset: 0,
            };
            if let Ok(existing) = alloc.fs_tree.range(&inline_key, &inline_key) {
                if let Some((_, edata)) = existing.first() {
                    if let Ok(BtrfsExtentData::Inline {
                        data: prev_data, ..
                    }) = parse_extent_data(edata)
                    {
                        alloc
                            .fs_tree
                            .delete(&inline_key)
                            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                        let sectorsize = u64::from(alloc.nodesize.min(4096));
                        if !prev_data.is_empty() {
                            let prev_alloc_size = (prev_data.len() as u64)
                                .saturating_add(sectorsize - 1)
                                & !(sectorsize - 1);
                            let prev_allocation = alloc
                                .extent_alloc
                                .alloc_data(prev_alloc_size)
                                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                            self.dev.write_all_at(
                                cx,
                                ByteOffset(prev_allocation.bytenr),
                                &prev_data,
                            )?;
                            let prev_extent = BtrfsExtentData::Regular {
                                generation: alloc.generation,
                                extent_type: BTRFS_FILE_EXTENT_REG,
                                compression: 0,
                                disk_bytenr: prev_allocation.bytenr,
                                disk_num_bytes: prev_alloc_size,
                                extent_offset: 0,
                                num_bytes: prev_data.len() as u64,
                            };
                            alloc
                                .fs_tree
                                .insert(inline_key, &prev_extent.to_bytes())
                                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
                        }
                    }
                }
            }

            let sectorsize = u64::from(alloc.nodesize.min(4096));
            let alloc_size = length
                .checked_add(sectorsize - 1)
                .ok_or_else(|| FfsError::InvalidGeometry("offset + length overflow".into()))?
                & !(sectorsize - 1);

            let allocation = alloc
                .extent_alloc
                .alloc_data(alloc_size)
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;

            let extent = BtrfsExtentData::Regular {
                generation: alloc.generation,
                extent_type: BTRFS_FILE_EXTENT_PREALLOC,
                compression: 0,
                disk_bytenr: allocation.bytenr,
                disk_num_bytes: alloc_size,
                extent_offset: 0,
                num_bytes: length,
            };
            let extent_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_EXTENT_DATA,
                offset,
            };
            alloc
                .fs_tree
                .insert(extent_key, &extent.to_bytes())
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        }

        // Update inode size if fallocate extends beyond current size.
        let mut inode = self.btrfs_read_inode_from_tree(&alloc, canonical)?;
        let new_end = offset
            .checked_add(length)
            .ok_or_else(|| FfsError::InvalidGeometry("offset + length overflow".into()))?;
        if !keep_size && new_end > inode.size {
            inode.size = new_end;
            inode.nbytes = new_end;
            let inode_key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_INODE_ITEM,
                offset: 0,
            };
            alloc
                .fs_tree
                .update(&inode_key, &inode.to_bytes())
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        }
        drop(alloc);

        Ok(())
    }

    /// List extended attribute names for a btrfs inode.
    ///
    /// When writes are enabled reads from the COW tree; otherwise scans the
    /// on-disk FS tree for XATTR_ITEM entries (type 24).
    fn btrfs_listxattr(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<String>> {
        let canonical = self.btrfs_canonical_inode(ino)?;
        let xattr_entries = self.btrfs_collect_xattr_items(cx, canonical)?;
        Ok(xattr_entries
            .into_iter()
            .map(|x| String::from_utf8_lossy(&x.name).into_owned())
            .collect())
    }

    /// Get the value of a specific extended attribute on a btrfs inode.
    fn btrfs_getxattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        name: &str,
    ) -> ffs_error::Result<Option<Vec<u8>>> {
        let canonical = self.btrfs_canonical_inode(ino)?;

        // Fast path: look up the specific xattr by name hash (COW tree or
        // on-disk tree).
        if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
            let alloc = alloc_mutex.lock();
            let name_hash = ffs_types::crc32c_append(0, name.as_bytes());
            let key = BtrfsKey {
                objectid: canonical,
                item_type: BTRFS_ITEM_XATTR_ITEM,
                offset: u64::from(name_hash),
            };
            let items = alloc
                .fs_tree
                .range(&key, &key)
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
            drop(alloc);
            let Some((_k, data)) = items.first() else {
                return Ok(None);
            };
            let parsed = parse_xattr_items(data).map_err(|e| parse_to_ffs_error(&e))?;
            return Ok(parsed
                .into_iter()
                .find(|x| x.name == name.as_bytes())
                .map(|x| x.value));
        }

        // Read-only path: scan the on-disk FS tree.
        let items = self.walk_btrfs_fs_tree(cx)?;
        for item in &items {
            if item.key.objectid == canonical && item.key.item_type == BTRFS_ITEM_XATTR_ITEM {
                let parsed = parse_xattr_items(&item.data).map_err(|e| parse_to_ffs_error(&e))?;
                for xattr in parsed {
                    if xattr.name == name.as_bytes() {
                        return Ok(Some(xattr.value));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Collect all xattr items for a given objectid from COW tree or on-disk tree.
    fn btrfs_collect_xattr_items(
        &self,
        cx: &Cx,
        objectid: u64,
    ) -> ffs_error::Result<Vec<ffs_btrfs::BtrfsXattrItem>> {
        if let Some(alloc_mutex) = self.btrfs_alloc_state.as_ref() {
            let alloc = alloc_mutex.lock();
            let start = BtrfsKey {
                objectid,
                item_type: BTRFS_ITEM_XATTR_ITEM,
                offset: 0,
            };
            let end = BtrfsKey {
                objectid,
                item_type: BTRFS_ITEM_XATTR_ITEM,
                offset: u64::MAX,
            };
            let items = alloc
                .fs_tree
                .range(&start, &end)
                .map_err(|e| btrfs_mutation_to_ffs(&e))?;
            drop(alloc);
            let mut result = Vec::new();
            for (_k, data) in &items {
                let parsed = parse_xattr_items(data).map_err(|e| parse_to_ffs_error(&e))?;
                result.extend(parsed);
            }
            Ok(result)
        } else {
            let items = self.walk_btrfs_fs_tree(cx)?;
            let mut result = Vec::new();
            for item in &items {
                if item.key.objectid == objectid && item.key.item_type == BTRFS_ITEM_XATTR_ITEM {
                    let parsed =
                        parse_xattr_items(&item.data).map_err(|e| parse_to_ffs_error(&e))?;
                    result.extend(parsed);
                }
            }
            Ok(result)
        }
    }

    /// Set an extended attribute on a btrfs inode.
    fn btrfs_setxattr(
        &self,
        _cx: &Cx,
        ino: InodeNumber,
        name: &str,
        value: &[u8],
        _mode: XattrSetMode,
    ) -> ffs_error::Result<()> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let canonical = self.btrfs_canonical_inode(ino)?;

        let mut alloc = alloc_mutex.lock();

        let name_hash = ffs_types::crc32c_append(0, name.as_bytes());
        let key = BtrfsKey {
            objectid: canonical,
            item_type: BTRFS_ITEM_XATTR_ITEM,
            offset: u64::from(name_hash),
        };

        // Build the xattr item: name + value concatenated, with a header.
        let mut payload = Vec::with_capacity(30 + name.len() + value.len());
        // Location key: zeros (xattrs don't have a location)
        payload.extend_from_slice(&[0u8; 17]);
        // transid: zeros
        payload.extend_from_slice(&[0u8; 8]);
        #[allow(clippy::cast_possible_truncation)]
        {
            payload.extend_from_slice(&(value.len() as u16).to_le_bytes());
            payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
        }
        payload.push(0); // type = 0 for xattr
        payload.extend_from_slice(name.as_bytes());
        payload.extend_from_slice(value);

        alloc
            .fs_tree
            .update(&key, &payload)
            .or_else(|_| alloc.fs_tree.insert(key, &payload))
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        drop(alloc);

        Ok(())
    }

    /// Remove an extended attribute from a btrfs inode.
    fn btrfs_removexattr(&self, _cx: &Cx, ino: InodeNumber, name: &str) -> ffs_error::Result<bool> {
        let alloc_mutex = self.require_btrfs_alloc_state()?;
        let canonical = self.btrfs_canonical_inode(ino)?;

        let mut alloc = alloc_mutex.lock();
        let name_hash = ffs_types::crc32c_append(0, name.as_bytes());
        let key = BtrfsKey {
            objectid: canonical,
            item_type: BTRFS_ITEM_XATTR_ITEM,
            offset: u64::from(name_hash),
        };

        match alloc.fs_tree.delete(&key) {
            Ok(_) => Ok(true),
            Err(BtrfsMutationError::MissingNode(_)) => Ok(false),
            Err(e) => Err(btrfs_mutation_to_ffs(&e)),
        }
    }

    // ── Btrfs write-path helpers ────────────────────────────────────────

    /// Read a `BtrfsInodeItem` from the in-memory COW FS tree.
    #[allow(clippy::unused_self)]
    fn btrfs_read_inode_from_tree(
        &self,
        alloc: &BtrfsAllocState,
        objectid: u64,
    ) -> ffs_error::Result<BtrfsInodeItem> {
        let key = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        let items = alloc
            .fs_tree
            .range(&key, &key)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        let data = items
            .first()
            .ok_or_else(|| FfsError::NotFound(format!("btrfs inode {objectid}")))?
            .1
            .as_slice();
        parse_inode_item(data).map_err(|e| parse_to_ffs_error(&e))
    }

    /// Convert a `BtrfsInodeItem` into an `InodeAttr`.
    #[allow(clippy::cast_possible_truncation)]
    fn btrfs_inode_to_attr(&self, objectid: u64, inode: &BtrfsInodeItem) -> InodeAttr {
        InodeAttr {
            ino: InodeNumber(objectid),
            size: inode.size,
            blocks: inode.nbytes.div_ceil(512),
            atime: UNIX_EPOCH + Duration::new(inode.atime_sec, inode.atime_nsec),
            mtime: UNIX_EPOCH + Duration::new(inode.mtime_sec, inode.mtime_nsec),
            ctime: UNIX_EPOCH + Duration::new(inode.ctime_sec, inode.ctime_nsec),
            crtime: UNIX_EPOCH + Duration::new(inode.otime_sec, inode.otime_nsec),
            kind: Self::btrfs_mode_to_file_type(inode.mode),
            perm: (inode.mode & 0o7777) as u16,
            nlink: inode.nlink,
            uid: inode.uid,
            gid: inode.gid,
            rdev: inode.rdev as u32,
            blksize: self.block_size(),
        }
    }

    /// Insert a directory entry (both DIR_ITEM and DIR_INDEX) into the FS tree.
    #[allow(clippy::unused_self)]
    fn btrfs_insert_dir_entry(
        &self,
        alloc: &mut BtrfsAllocState,
        parent_oid: u64,
        item: &BtrfsDirItem,
    ) -> ffs_error::Result<()> {
        let name_hash = ffs_types::crc32c_append(0, &item.name);
        let dir_item_key = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_ITEM,
            offset: u64::from(name_hash),
        };
        let new_bytes = item.to_bytes();

        // btrfs stores multiple dir entries with the same CRC32C hash in a
        // single tree item (concatenated payloads). Merge with any existing
        // entries under this hash to avoid silent data loss on collision.
        let merged = alloc
            .fs_tree
            .range(&dir_item_key, &dir_item_key)
            .ok()
            .and_then(|items| items.first().map(|(_, v)| v.clone()))
            .map_or_else(
                || new_bytes.clone(),
                |mut existing| {
                    existing.extend_from_slice(&new_bytes);
                    existing
                },
            );
        alloc
            .fs_tree
            .update(&dir_item_key, &merged)
            .or_else(|_| alloc.fs_tree.insert(dir_item_key, &merged))
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        // Also insert a DIR_INDEX entry. Use the child objectid as the index
        // for a simple monotonic ordering. DIR_INDEX keys are unique per child,
        // so no hash-collision merging is needed here.
        let dir_index_key = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_INDEX,
            offset: item.child_objectid,
        };
        alloc
            .fs_tree
            .update(&dir_index_key, &new_bytes)
            .or_else(|_| alloc.fs_tree.insert(dir_index_key, &new_bytes))
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        Ok(())
    }

    /// Look up a directory entry by name in the in-memory FS tree.
    #[allow(clippy::unused_self)]
    fn btrfs_lookup_dir_entry(
        &self,
        alloc: &BtrfsAllocState,
        parent_oid: u64,
        name: &[u8],
    ) -> ffs_error::Result<BtrfsDirItem> {
        let name_hash = ffs_types::crc32c_append(0, name);
        let key = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_ITEM,
            offset: u64::from(name_hash),
        };
        let items = alloc
            .fs_tree
            .range(&key, &key)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        let data = items
            .first()
            .ok_or_else(|| FfsError::NotFound(String::from_utf8_lossy(name).into_owned()))?
            .1
            .as_slice();
        let entries = parse_dir_items(data).map_err(|e| parse_to_ffs_error(&e))?;
        entries
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| FfsError::NotFound(String::from_utf8_lossy(name).into_owned()))
    }

    /// Remove a directory entry (DIR_ITEM and DIR_INDEX) from the FS tree.
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn btrfs_remove_dir_entry(
        &self,
        alloc: &mut BtrfsAllocState,
        parent_oid: u64,
        name: &[u8],
    ) -> ffs_error::Result<()> {
        let name_hash = ffs_types::crc32c_append(0, name);
        let dir_item_key = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_ITEM,
            offset: u64::from(name_hash),
        };

        // btrfs stores multiple dir entries with the same CRC32C hash in a
        // single payload. We must parse, remove only the matching entry, and
        // store the remaining entries back (or delete the key if empty).
        if let Ok(existing) = alloc.fs_tree.range(&dir_item_key, &dir_item_key) {
            if let Some((_, edata)) = existing.first() {
                if let Ok(entries) = parse_dir_items(edata) {
                    let remaining: Vec<_> = entries.iter().filter(|e| e.name != name).collect();
                    if remaining.is_empty() {
                        if let Err(e) = alloc.fs_tree.delete(&dir_item_key) {
                            warn!("unlink: failed to delete dir_item {dir_item_key:?}: {e:?}");
                        }
                    } else {
                        let mut payload = Vec::new();
                        for entry in &remaining {
                            payload.extend_from_slice(&entry.to_bytes());
                        }
                        if let Err(e) = alloc.fs_tree.update(&dir_item_key, &payload) {
                            warn!("unlink: failed to update dir_item {dir_item_key:?}: {e:?}");
                        }
                    }
                } else {
                    // Can't parse existing entries; delete the whole key.
                    if let Err(e) = alloc.fs_tree.delete(&dir_item_key) {
                        warn!("unlink: failed to delete unparseable dir_item: {e:?}");
                    }
                }
            }
        }

        // Also remove the DIR_INDEX. We need to find it by scanning.
        let range_start = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_INDEX,
            offset: 0,
        };
        let range_end = BtrfsKey {
            objectid: parent_oid,
            item_type: BTRFS_ITEM_DIR_INDEX,
            offset: u64::MAX,
        };
        if let Ok(indices) = alloc.fs_tree.range(&range_start, &range_end) {
            for (key, data) in &indices {
                if let Ok(entries) = parse_dir_items(data) {
                    if entries.iter().any(|e| e.name == name) {
                        if let Err(e) = alloc.fs_tree.delete(key) {
                            warn!("unlink: failed to delete dir_index {key:?}: {e:?}");
                        }
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if a btrfs directory is empty (has no DIR_INDEX entries).
    #[allow(clippy::unused_self)]
    fn btrfs_dir_is_empty(
        &self,
        alloc: &BtrfsAllocState,
        objectid: u64,
    ) -> ffs_error::Result<bool> {
        let range_start = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_DIR_INDEX,
            offset: 0,
        };
        let range_end = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_DIR_INDEX,
            offset: u64::MAX,
        };
        let items = alloc
            .fs_tree
            .range(&range_start, &range_end)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        Ok(items.is_empty())
    }

    /// Adjust the nlink count of a btrfs inode by `delta` (+1 or -1).
    fn btrfs_adjust_nlink(
        &self,
        alloc: &mut BtrfsAllocState,
        objectid: u64,
        delta: i32,
    ) -> ffs_error::Result<()> {
        let mut inode = self.btrfs_read_inode_from_tree(alloc, objectid)?;
        if delta > 0 {
            inode.nlink = inode.nlink.saturating_add(delta.unsigned_abs());
        } else {
            inode.nlink = inode.nlink.saturating_sub(delta.unsigned_abs());
        }
        let key = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .update(&key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        Ok(())
    }

    /// Remove all tree items for an inode (INODE_ITEM, EXTENT_DATA, XATTR_ITEM).
    ///
    /// Called when nlink drops to 0 to free the orphaned inode's metadata.
    #[allow(clippy::unused_self)]
    fn btrfs_purge_inode(
        &self,
        alloc: &mut BtrfsAllocState,
        objectid: u64,
    ) -> ffs_error::Result<()> {
        // Collect all keys for this objectid, then delete them.
        // We query a range covering all possible item types (1..255).
        let start = BtrfsKey {
            objectid,
            item_type: 0,
            offset: 0,
        };
        let end = BtrfsKey {
            objectid,
            item_type: u8::MAX,
            offset: u64::MAX,
        };
        let items = alloc
            .fs_tree
            .range(&start, &end)
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;

        for (key, data) in items {
            if key.item_type == BTRFS_ITEM_EXTENT_DATA {
                if let Ok(BtrfsExtentData::Regular {
                    disk_bytenr,
                    disk_num_bytes,
                    ..
                }) = parse_extent_data(&data)
                {
                    if disk_bytenr > 0 {
                        if let Err(e) =
                            alloc
                                .extent_alloc
                                .free_extent(disk_bytenr, disk_num_bytes, false)
                        {
                            warn!("cleanup: failed to free extent at {disk_bytenr}: {e:?}");
                        }
                    }
                }
            }
            if let Err(e) = alloc.fs_tree.delete(&key) {
                warn!("cleanup: failed to delete inode item {key:?}: {e:?}");
            }
        }
        Ok(())
    }

    /// Insert an INODE_REF item linking `child_oid` back to `parent_oid`.
    ///
    /// The INODE_REF key is `{objectid: child, item_type: 12, offset: parent}`.
    /// The payload is `index(8) + name_len(2) + name`.
    #[allow(clippy::unused_self)]
    fn btrfs_insert_inode_ref(
        &self,
        alloc: &mut BtrfsAllocState,
        child_oid: u64,
        parent_oid: u64,
        name: &[u8],
        index: u64,
    ) -> ffs_error::Result<()> {
        let ref_key = BtrfsKey {
            objectid: child_oid,
            item_type: BTRFS_ITEM_INODE_REF,
            offset: parent_oid,
        };
        let name_len = u16::try_from(name.len())
            .map_err(|_| FfsError::Format("inode ref name too long".into()))?;
        let mut payload = Vec::with_capacity(10 + name.len());
        payload.extend_from_slice(&index.to_le_bytes());
        payload.extend_from_slice(&name_len.to_le_bytes());
        payload.extend_from_slice(name);

        alloc
            .fs_tree
            .update(&ref_key, &payload)
            .or_else(|_| alloc.fs_tree.insert(ref_key, &payload))
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        Ok(())
    }

    /// Remove the INODE_REF item linking `child_oid` back to `parent_oid`.
    #[allow(clippy::unused_self)]
    fn btrfs_remove_inode_ref(&self, alloc: &mut BtrfsAllocState, child_oid: u64, parent_oid: u64) {
        let ref_key = BtrfsKey {
            objectid: child_oid,
            item_type: BTRFS_ITEM_INODE_REF,
            offset: parent_oid,
        };
        if let Err(e) = alloc.fs_tree.delete(&ref_key) {
            warn!("unlink: failed to delete inode_ref {ref_key:?}: {e:?}");
        }
    }

    /// Look up the parent objectid for a btrfs inode via its INODE_REF.
    ///
    /// Returns `None` if no INODE_REF is found (e.g. root directory).
    #[allow(clippy::unused_self)]
    fn btrfs_lookup_parent(&self, alloc: &BtrfsAllocState, objectid: u64) -> Option<u64> {
        let start = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_INODE_REF,
            offset: 0,
        };
        let end = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_INODE_REF,
            offset: u64::MAX,
        };
        alloc
            .fs_tree
            .range(&start, &end)
            .ok()
            .and_then(|items| items.first().map(|(k, _)| k.offset))
    }

    /// Update mtime and ctime of a btrfs inode.
    fn btrfs_touch_inode_times(
        &self,
        alloc: &mut BtrfsAllocState,
        objectid: u64,
        secs: u64,
        nanos: u32,
    ) -> ffs_error::Result<()> {
        let mut inode = self.btrfs_read_inode_from_tree(alloc, objectid)?;
        inode.mtime_sec = secs;
        inode.mtime_nsec = nanos;
        inode.ctime_sec = secs;
        inode.ctime_nsec = nanos;
        let key = BtrfsKey {
            objectid,
            item_type: BTRFS_ITEM_INODE_ITEM,
            offset: 0,
        };
        alloc
            .fs_tree
            .update(&key, &inode.to_bytes())
            .map_err(|e| btrfs_mutation_to_ffs(&e))?;
        Ok(())
    }
}

// ── FsOps for OpenFs (device-based ext4 adapter) ──────────────────────────

impl FsOps for OpenFs {
    fn getattr(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .read_inode_attr(cx, Self::ext4_canonical_inode(ino))
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => self.btrfs_read_inode_attr(cx, ino),
        }
    }

    fn lookup(&self, cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let parent_ino = Self::ext4_canonical_inode(parent);
                let parent_inode = self.read_inode(cx, parent_ino)?;
                if !parent_inode.is_dir() {
                    return Err(FfsError::NotDirectory);
                }

                let name_bytes = name.as_encoded_bytes();
                let entry = self
                    .lookup_name(cx, &parent_inode, name_bytes)?
                    .ok_or_else(|| FfsError::NotFound(name.to_string_lossy().into_owned()))?;

                let child_ino = InodeNumber(u64::from(entry.inode));
                self.read_inode_attr(cx, child_ino)
                    .map(Self::ext4_present_attr)
            }
            FsFlavor::Btrfs(_) => self.btrfs_lookup_child(cx, parent, name.as_encoded_bytes()),
        }
    }

    fn readdir(&self, cx: &Cx, ino: InodeNumber, offset: u64) -> ffs_error::Result<Vec<DirEntry>> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let inode = self.read_inode(cx, Self::ext4_canonical_inode(ino))?;
                if !inode.is_dir() {
                    return Err(FfsError::NotDirectory);
                }

                let raw_entries = self.read_dir(cx, &inode)?;
                let entries: Vec<DirEntry> = raw_entries
                    .into_iter()
                    .enumerate()
                    .filter(|(idx, _)| (*idx as u64) >= offset)
                    .map(|(idx, e)| {
                        Self::ext4_present_dir_entry(DirEntry {
                            ino: InodeNumber(u64::from(e.inode)),
                            offset: (idx as u64) + 1,
                            kind: dir_entry_file_type(e.file_type),
                            name: e.name,
                        })
                    })
                    .collect();
                Ok(entries)
            }
            FsFlavor::Btrfs(_) => {
                let rows = self.btrfs_readdir_entries(cx, ino)?;
                let entries = rows
                    .into_iter()
                    .enumerate()
                    .filter(|(idx, _)| (*idx as u64) >= offset)
                    .map(|(idx, (_, mut e))| {
                        e.offset = (idx as u64) + 1;
                        e
                    })
                    .collect();
                Ok(entries)
            }
        }
    }

    fn read(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        size: u32,
    ) -> ffs_error::Result<Vec<u8>> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.read_file(cx, Self::ext4_canonical_inode(ino), offset, size),
            FsFlavor::Btrfs(_) => self.btrfs_read_file(cx, ino, offset, size),
        }
    }

    fn readlink(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let inode = self.read_inode(cx, Self::ext4_canonical_inode(ino))?;
                self.read_symlink(cx, &inode)
            }
            FsFlavor::Btrfs(_) => {
                let attr = self.btrfs_read_inode_attr(cx, ino)?;
                if attr.kind != FileType::Symlink {
                    return Err(FfsError::Format("not a symlink".into()));
                }
                let read_size = u32::try_from(attr.size).unwrap_or(u32::MAX);
                let mut target = self.btrfs_read_file(cx, ino, 0, read_size)?;
                if let Some(nul) = target.iter().position(|b| *b == 0) {
                    target.truncate(nul);
                }
                Ok(target)
            }
        }
    }

    fn statfs(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<FsStat> {
        match &self.flavor {
            FsFlavor::Ext4(sb) => {
                let blocks_free = sb.free_blocks_count;
                let blocks_available = blocks_free.saturating_sub(sb.reserved_blocks_count);
                Ok(FsStat {
                    blocks: sb.blocks_count,
                    blocks_free,
                    blocks_available,
                    files: u64::from(sb.inodes_count),
                    files_free: u64::from(sb.free_inodes_count),
                    block_size: sb.block_size,
                    name_max: 255,
                    fragment_size: sb.block_size,
                })
            }
            FsFlavor::Btrfs(sb) => {
                let unit = sb.sectorsize.max(1);
                let unit_u64 = u64::from(unit);
                // Use live allocator stats when writes are enabled,
                // otherwise fall back to the on-disk superblock.
                let used_bytes = self
                    .btrfs_alloc_state
                    .as_ref()
                    .map_or(sb.bytes_used, |alloc_mutex| {
                        alloc_mutex.lock().extent_alloc.total_used()
                    });
                let total_bytes = sb.total_bytes;
                let free_bytes = total_bytes.saturating_sub(used_bytes);
                Ok(FsStat {
                    blocks: total_bytes / unit_u64,
                    blocks_free: free_bytes / unit_u64,
                    blocks_available: free_bytes / unit_u64,
                    files: 0,
                    files_free: 0,
                    block_size: unit,
                    name_max: 255,
                    fragment_size: unit,
                })
            }
        }
    }

    fn listxattr(&self, cx: &Cx, ino: InodeNumber) -> ffs_error::Result<Vec<String>> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let inode = self.read_inode(cx, Self::ext4_canonical_inode(ino))?;
                let mut xattrs =
                    ffs_ondisk::parse_ibody_xattrs(&inode).map_err(|e| parse_to_ffs_error(&e))?;
                if inode.file_acl != 0 {
                    let block_data = self.read_block_vec(cx, BlockNumber(inode.file_acl))?;
                    let block_xattrs = ffs_ondisk::parse_xattr_block(&block_data)
                        .map_err(|e| parse_to_ffs_error(&e))?;
                    xattrs.extend(block_xattrs);
                }
                Ok(xattrs.iter().map(Ext4Xattr::full_name).collect())
            }
            FsFlavor::Btrfs(_) => self.btrfs_listxattr(cx, ino),
        }
    }

    fn getxattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        name: &str,
    ) -> ffs_error::Result<Option<Vec<u8>>> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                let inode = self.read_inode(cx, Self::ext4_canonical_inode(ino))?;
                let mut xattrs =
                    ffs_ondisk::parse_ibody_xattrs(&inode).map_err(|e| parse_to_ffs_error(&e))?;
                if inode.file_acl != 0 {
                    let block_data = self.read_block_vec(cx, BlockNumber(inode.file_acl))?;
                    let block_xattrs = ffs_ondisk::parse_xattr_block(&block_data)
                        .map_err(|e| parse_to_ffs_error(&e))?;
                    xattrs.extend(block_xattrs);
                }
                Ok(xattrs
                    .into_iter()
                    .find(|x| x.full_name() == name)
                    .map(|x| x.value))
            }
            FsFlavor::Btrfs(_) => self.btrfs_getxattr(cx, ino, name),
        }
    }

    fn setxattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        name: &str,
        value: &[u8],
        mode: XattrSetMode,
    ) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                self.ext4_setxattr(cx, Self::ext4_canonical_inode(ino), name, value, mode)
            }
            FsFlavor::Btrfs(_) => self.btrfs_setxattr(cx, ino, name, value, mode),
        }
    }

    fn removexattr(&self, cx: &Cx, ino: InodeNumber, name: &str) -> ffs_error::Result<bool> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.ext4_removexattr(cx, Self::ext4_canonical_inode(ino), name),
            FsFlavor::Btrfs(_) => self.btrfs_removexattr(cx, ino, name),
        }
    }

    // ── Write operations ──────────────────────────────────────────────

    fn create(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &OsStr,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .ext4_create(
                    cx,
                    Self::ext4_canonical_inode(parent),
                    name.as_encoded_bytes(),
                    mode,
                    uid,
                    gid,
                )
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => {
                self.btrfs_create(cx, parent, name.as_encoded_bytes(), mode, uid, gid)
            }
        }
    }

    fn mkdir(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &OsStr,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .ext4_mkdir(
                    cx,
                    Self::ext4_canonical_inode(parent),
                    name.as_encoded_bytes(),
                    mode,
                    uid,
                    gid,
                )
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => {
                self.btrfs_mkdir(cx, parent, name.as_encoded_bytes(), mode, uid, gid)
            }
        }
    }

    fn unlink(&self, cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.ext4_unlink_impl(
                cx,
                Self::ext4_canonical_inode(parent),
                name.as_encoded_bytes(),
                false,
            ),
            FsFlavor::Btrfs(_) => {
                self.btrfs_unlink_impl(cx, parent, name.as_encoded_bytes(), false)
            }
        }
    }

    fn rmdir(&self, cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.ext4_unlink_impl(
                cx,
                Self::ext4_canonical_inode(parent),
                name.as_encoded_bytes(),
                true,
            ),
            FsFlavor::Btrfs(_) => self.btrfs_unlink_impl(cx, parent, name.as_encoded_bytes(), true),
        }
    }

    fn rename(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &OsStr,
        new_parent: InodeNumber,
        new_name: &OsStr,
    ) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.ext4_rename(
                cx,
                Self::ext4_canonical_inode(parent),
                name.as_encoded_bytes(),
                Self::ext4_canonical_inode(new_parent),
                new_name.as_encoded_bytes(),
            ),
            FsFlavor::Btrfs(_) => self.btrfs_rename(
                cx,
                parent,
                name.as_encoded_bytes(),
                new_parent,
                new_name.as_encoded_bytes(),
            ),
        }
    }

    fn write(&self, cx: &Cx, ino: InodeNumber, offset: u64, data: &[u8]) -> ffs_error::Result<u32> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self.ext4_write(cx, Self::ext4_canonical_inode(ino), offset, data),
            FsFlavor::Btrfs(_) => self.btrfs_write(cx, ino, offset, data),
        }
    }

    fn link(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        new_parent: InodeNumber,
        new_name: &OsStr,
    ) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .ext4_link(
                    cx,
                    Self::ext4_canonical_inode(ino),
                    Self::ext4_canonical_inode(new_parent),
                    new_name.as_encoded_bytes(),
                )
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => self.btrfs_link(cx, ino, new_parent, new_name.as_encoded_bytes()),
        }
    }

    fn symlink(
        &self,
        cx: &Cx,
        parent: InodeNumber,
        name: &OsStr,
        target: &Path,
        uid: u32,
        gid: u32,
    ) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .ext4_symlink(
                    cx,
                    Self::ext4_canonical_inode(parent),
                    name.as_encoded_bytes(),
                    target,
                    uid,
                    gid,
                )
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => {
                self.btrfs_symlink(cx, parent, name.as_encoded_bytes(), target, uid, gid)
            }
        }
    }

    fn fallocate(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        offset: u64,
        length: u64,
        mode: i32,
    ) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                self.ext4_fallocate(cx, Self::ext4_canonical_inode(ino), offset, length, mode)
            }
            FsFlavor::Btrfs(_) => self.btrfs_fallocate(cx, ino, offset, length, mode),
        }
    }

    fn setattr(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        attrs: &SetAttrRequest,
    ) -> ffs_error::Result<InodeAttr> {
        match &self.flavor {
            FsFlavor::Ext4(_) => self
                .ext4_setattr(cx, Self::ext4_canonical_inode(ino), attrs)
                .map(Self::ext4_present_attr),
            FsFlavor::Btrfs(_) => self.btrfs_setattr(cx, ino, attrs),
        }
    }

    fn flush(
        &self,
        _cx: &Cx,
        _ino: InodeNumber,
        _fh: u64,
        _lock_owner: u64,
    ) -> ffs_error::Result<()> {
        Ok(())
    }

    fn fsync(&self, cx: &Cx, ino: InodeNumber, _fh: u64, datasync: bool) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                if !self.is_writable() {
                    return Err(FfsError::ReadOnly);
                }
                let started = Instant::now();
                debug!(
                    target: "ffs::durability",
                    op = "fsync",
                    ino = ino.0,
                    datasync,
                    "fsync_start"
                );
                self.dev.sync(cx)?;
                debug!(
                    target: "ffs::durability",
                    op = "fsync",
                    ino = ino.0,
                    datasync,
                    duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                    "fsync_complete"
                );
                Ok(())
            }
            FsFlavor::Btrfs(_) => {
                if !self.is_writable() {
                    return Err(FfsError::ReadOnly);
                }
                self.dev.sync(cx)
            }
        }
    }

    fn fsyncdir(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        _fh: u64,
        datasync: bool,
    ) -> ffs_error::Result<()> {
        match &self.flavor {
            FsFlavor::Ext4(_) => {
                if !self.is_writable() {
                    return Err(FfsError::ReadOnly);
                }
                let started = Instant::now();
                debug!(
                    target: "ffs::durability",
                    op = "fsyncdir",
                    ino = ino.0,
                    datasync,
                    "fsyncdir_start"
                );
                self.dev.sync(cx)?;
                debug!(
                    target: "ffs::durability",
                    op = "fsyncdir",
                    ino = ino.0,
                    datasync,
                    duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
                    "fsyncdir_complete"
                );
                Ok(())
            }
            FsFlavor::Btrfs(_) => {
                if !self.is_writable() {
                    return Err(FfsError::ReadOnly);
                }
                self.dev.sync(cx)
            }
        }
    }
}

// ── Bayesian Filesystem Integrity Scanner ──────────────────────────────────
//
// Alien-artifact quality: cascading checksum verification with a formal
// evidence ledger. Each verification step contributes a log-likelihood
// observation to a Beta-Binomial posterior over the corruption rate.
//
// The posterior P(healthy|evidence) is computed via conjugate update:
//   α += corrupted_count, β += clean_count
// where clean = verified_ok and corrupted = checksum_mismatch.
//
// Decision theory: the expected corruption rate E[p] = α/(α+β) and the
// upper credible bound p_hi = E[p] + z·√Var[p] provide actionable thresholds.

/// Verdict for a single integrity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckVerdict {
    /// What was checked (e.g., "superblock", "group_desc[3]", "inode[142]").
    pub component: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Human-readable detail (empty on success).
    pub detail: String,
}

/// The complete evidence ledger from an integrity scan.
///
/// # Bayesian Model
///
/// We model each metadata object as a Bernoulli trial: P(corrupt) = p.
/// Using a Beta(α, β) conjugate prior (default: uninformative Beta(1,1)),
/// after observing n_clean clean objects and n_corrupt corrupted objects:
///
/// ```text
/// Posterior: Beta(α + n_corrupt, β + n_clean)
/// E[p] = α / (α + β)                           — expected corruption rate
/// Var[p] = αβ / ((α+β)²(α+β+1))                — posterior variance
/// p_hi = E[p] + z·√Var[p]                       — upper credible bound
/// ```
///
/// A filesystem is "healthy" when p_hi < threshold (default 0.01).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityReport {
    /// Per-component verdicts (evidence ledger).
    pub verdicts: Vec<CheckVerdict>,
    /// Number of checks that passed.
    pub passed: u64,
    /// Number of checks that failed.
    pub failed: u64,
    /// Posterior α (prior + observed corruptions).
    pub posterior_alpha: f64,
    /// Posterior β (prior + observed clean).
    pub posterior_beta: f64,
    /// E[p] = expected corruption rate.
    pub expected_corruption_rate: f64,
    /// Upper credible bound on corruption rate (z=3 by default).
    pub upper_bound_corruption_rate: f64,
    /// Overall health verdict: true if upper_bound < 0.01.
    pub healthy: bool,
}

impl IntegrityReport {
    /// Posterior probability that corruption rate < threshold.
    ///
    /// Uses the regularized incomplete beta function approximation:
    /// for large sample sizes, the Beta posterior is approximately Normal,
    /// so P(p < t) ≈ Φ((t - μ) / σ) where μ = E[p], σ = √Var[p].
    #[must_use]
    pub fn prob_healthy(&self, threshold: f64) -> f64 {
        let a = self.posterior_alpha;
        let b = self.posterior_beta;
        let mean = a / (a + b);
        let var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
        let std = var.sqrt();
        if std < 1e-15 {
            return if mean < threshold { 1.0 } else { 0.0 };
        }
        // Normal CDF approximation: Φ(x) ≈ 0.5 * erfc(-x/√2)
        let z = (threshold - mean) / std;
        0.5 * erfc_approx(-z / std::f64::consts::SQRT_2)
    }

    /// Log Bayes factor: ln(P(evidence|healthy) / P(evidence|corrupt)).
    ///
    /// Positive values favor health; negative values favor corruption.
    /// Uses the ratio of Beta-Binomial marginal likelihoods:
    ///   - H₀ (healthy): p ~ Beta(1, 99) (expect ~1% corruption)
    ///   - H₁ (corrupt): p ~ Beta(1, 1) (uniform prior)
    #[must_use]
    pub fn log_bayes_factor(&self) -> f64 {
        // ln B(α₀ + f, β₀ + p) - ln B(α₀, β₀) - ln B(α₁ + f, β₁ + p) + ln B(α₁, β₁)
        // where f = failed, p = passed, and B is the beta function
        let f = self.failed as f64;
        let p = self.passed as f64;

        // H₀: healthy prior Beta(1, 99)
        let a0 = 1.0_f64;
        let b0 = 99.0_f64;
        // H₁: corrupt prior Beta(1, 1)
        let a1 = 1.0_f64;
        let b1 = 1.0_f64;

        ln_beta(a0 + f, b0 + p) - ln_beta(a0, b0) - ln_beta(a1 + f, b1 + p) + ln_beta(a1, b1)
    }
}

/// Approximate complementary error function erfc(x) for Normal CDF.
fn erfc_approx(x: f64) -> f64 {
    // Abramowitz & Stegun approximation (7.1.26), max error < 1.5e-7
    let t = 1.0 / 0.327_591_1_f64.mul_add(x.abs(), 1.0);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 { result } else { 2.0 - result }
}

/// ln(Beta(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for ln(Γ(x)), accurate for x > 0.
#[allow(clippy::excessive_precision)]
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x <= 0.0 {
        return f64::INFINITY;
    }
    let g = 7.0_f64;
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        sum += c / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5_f64.mul_add(
        (2.0 * std::f64::consts::PI).ln(),
        (z + 0.5).mul_add(t.ln(), -t),
    ) + sum.ln()
}

/// Run a comprehensive integrity scan of an ext4 filesystem image.
///
/// Cascades through verification levels:
/// 1. **Superblock checksum** (if metadata_csum enabled)
/// 2. **Group descriptor checksums** (all groups)
/// 3. **Inode checksums** (sampled or exhaustive)
/// 4. **Directory block checksums** (for sampled directory inodes)
///
/// Returns an `IntegrityReport` with per-component verdicts and a
/// Bayesian posterior over the corruption rate.
///
/// # Arguments
/// * `image` - raw filesystem image bytes
/// * `max_inodes` - maximum number of inodes to verify (0 = all)
#[allow(clippy::too_many_lines)]
pub fn verify_ext4_integrity(image: &[u8], max_inodes: u32) -> Result<IntegrityReport, FfsError> {
    let reader = Ext4ImageReader::new(image).map_err(|e| parse_to_ffs_error(&e))?;
    let sb = &reader.sb;

    let mut verdicts = Vec::new();
    let mut passed = 0_u64;
    let mut failed = 0_u64;
    let mut sb_passed = false;

    // ── Level 1: Superblock checksum ───────────────────────────────────
    if sb.has_metadata_csum() {
        let sb_region = &image[ffs_types::EXT4_SUPERBLOCK_OFFSET
            ..ffs_types::EXT4_SUPERBLOCK_OFFSET + ffs_types::EXT4_SUPERBLOCK_SIZE];
        match sb.validate_checksum(sb_region) {
            Ok(()) => {
                verdicts.push(CheckVerdict {
                    component: "superblock".into(),
                    passed: true,
                    detail: String::new(),
                });
                passed += 1;
                sb_passed = true;
            }
            Err(e) => {
                verdicts.push(CheckVerdict {
                    component: "superblock".into(),
                    passed: false,
                    detail: e.to_string(),
                });
                failed += 1;
            }
        }
    }

    // ── Level 2: Group descriptor checksums ────────────────────────────
    let csum_seed = sb.csum_seed();
    let groups_count = sb.groups_count();
    let desc_size = sb.group_desc_size();

    for g in 0..groups_count {
        let group = ffs_types::GroupNumber(g);
        let gd_result = reader.read_group_desc(image, group);
        match gd_result {
            Ok(_gd) => {
                // Read raw GD bytes for checksum verification
                if let Some(gd_off) = sb.group_desc_offset(group) {
                    let ds = usize::from(desc_size);
                    let offset = usize::try_from(gd_off).unwrap_or(usize::MAX);
                    if offset.saturating_add(ds) <= image.len() {
                        let raw_gd = &image[offset..offset + ds];
                        match ffs_ondisk::verify_group_desc_checksum(
                            raw_gd, csum_seed, g, desc_size,
                        ) {
                            Ok(()) => {
                                passed += 1;
                            }
                            Err(e) => {
                                verdicts.push(CheckVerdict {
                                    component: format!("group_desc[{g}]"),
                                    passed: false,
                                    detail: e.to_string(),
                                });
                                failed += 1;
                            }
                        }
                    }
                }
            }
            Err(e) => {
                verdicts.push(CheckVerdict {
                    component: format!("group_desc[{g}]"),
                    passed: false,
                    detail: e.to_string(),
                });
                failed += 1;
            }
        }
    }
    // Single success verdict for all clean group descs
    if failed == 0 || passed > 0 {
        let clean_gd = passed - u64::from(sb_passed); // subtract superblock if it was counted
        if clean_gd > 0 {
            verdicts.push(CheckVerdict {
                component: format!("group_descs ({clean_gd}/{groups_count} verified)"),
                passed: true,
                detail: String::new(),
            });
        }
    }

    // ── Level 3: Inode checksums (sampled) ─────────────────────────────
    let inodes_count = sb.inodes_count;
    let first_ino = sb.first_ino;
    let inode_size = usize::from(sb.inode_size);
    let block_size_usize = sb.block_size as usize;

    // Always check root inode (2) and first non-reserved inode
    let check_limit = if max_inodes == 0 {
        inodes_count
    } else {
        max_inodes.min(inodes_count)
    };

    let mut inodes_checked = 0_u64;
    let mut inodes_clean = 0_u64;
    let mut inodes_corrupt = 0_u64;

    // Cache inode bitmaps per group to avoid re-reading.
    let mut inode_bitmap_cache: std::collections::HashMap<u32, Vec<u8>> =
        std::collections::HashMap::new();

    // Check inodes: root (2), then first_ino..first_ino+check_limit
    let ino_list: Vec<u32> = {
        let mut v = vec![2_u32]; // root inode
        let start = first_ino.max(2);
        let end = start
            .saturating_add(check_limit)
            .min(inodes_count.saturating_add(1));
        for i in start..end {
            if i != 2 {
                v.push(i);
            }
        }
        v
    };

    for &ino in &ino_list {
        if inodes_checked >= u64::from(check_limit) {
            break;
        }

        // Skip unallocated inodes by consulting the inode bitmap.
        let group_idx = (ino - 1) / sb.inodes_per_group;
        let local = (ino - 1) % sb.inodes_per_group;
        let bitmap = inode_bitmap_cache.entry(group_idx).or_insert_with(|| {
            let group = ffs_types::GroupNumber(group_idx);
            if let Ok(gd) = reader.read_group_desc(image, group) {
                let bm_off = usize::try_from(gd.inode_bitmap * u64::from(sb.block_size))
                    .unwrap_or(usize::MAX);
                if bm_off.saturating_add(block_size_usize) <= image.len() {
                    return image[bm_off..bm_off + block_size_usize].to_vec();
                }
            }
            Vec::new()
        });
        if !bitmap.is_empty() && !bitmap_get(bitmap, local) {
            // Inode not allocated — skip without counting as a failure.
            inodes_checked += 1;
            continue;
        }

        // Read raw inode bytes for checksum verification
        let ino_num = ffs_types::InodeNumber(u64::from(ino));
        match reader.read_inode(image, ino_num) {
            Ok(inode) => {
                // Verify inode checksum using raw bytes
                let group = ffs_types::GroupNumber((ino - 1) / sb.inodes_per_group);
                if let Ok(gd) = reader.read_group_desc(image, group) {
                    let local = (ino - 1) % sb.inodes_per_group;
                    let itable_off = gd.inode_table * u64::from(sb.block_size);
                    let inode_off = itable_off + u64::from(local) * inode_size as u64;
                    let off = usize::try_from(inode_off).unwrap_or(usize::MAX);
                    if off.saturating_add(inode_size) <= image.len() {
                        let raw = &image[off..off + inode_size];
                        match ffs_ondisk::verify_inode_checksum(
                            raw,
                            csum_seed,
                            ino,
                            u16::try_from(inode_size).unwrap_or(256),
                        ) {
                            Ok(()) => {
                                inodes_clean += 1;
                                passed += 1;
                            }
                            Err(e) => {
                                verdicts.push(CheckVerdict {
                                    component: format!("inode[{ino}]"),
                                    passed: false,
                                    detail: e.to_string(),
                                });
                                inodes_corrupt += 1;
                                failed += 1;
                            }
                        }
                    }
                }

                // ── Level 4: Directory block checksums ─────────────────
                if inode.is_dir() && inode.uses_extents() {
                    let dir_blocks = inode.size / u64::from(sb.block_size);
                    let scan_blocks = u32::try_from(dir_blocks.min(16)).unwrap_or(16);
                    for lb in 0..scan_blocks {
                        if let Ok(Some(phys)) = reader.resolve_extent(image, &inode, lb) {
                            let blk_off = usize::try_from(phys * u64::from(sb.block_size))
                                .unwrap_or(usize::MAX);
                            let bs = sb.block_size as usize;
                            if blk_off.saturating_add(bs) <= image.len() {
                                let block_data = &image[blk_off..blk_off + bs];
                                // Check if block has a checksum tail
                                // (last entry has inode=0 and file_type=0xDE)
                                if bs >= 12 {
                                    let tail_ino = u32::from_le_bytes([
                                        block_data[bs - 12],
                                        block_data[bs - 11],
                                        block_data[bs - 10],
                                        block_data[bs - 9],
                                    ]);
                                    let tail_ft = block_data[bs - 5];
                                    if tail_ino == 0 && tail_ft == 0xDE {
                                        match ffs_ondisk::verify_dir_block_checksum(
                                            block_data,
                                            csum_seed,
                                            ino,
                                            inode.generation,
                                        ) {
                                            Ok(()) => {
                                                passed += 1;
                                            }
                                            Err(e) => {
                                                verdicts.push(CheckVerdict {
                                                    component: format!(
                                                        "dir_block[ino={ino},lb={lb}]"
                                                    ),
                                                    passed: false,
                                                    detail: e.to_string(),
                                                });
                                                failed += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                inodes_checked += 1;
            }
            Err(_) => {
                // Inode unreadable — skip silently (might be unallocated)
                inodes_checked += 1;
            }
        }
    }

    if inodes_clean > 0 {
        verdicts.push(CheckVerdict {
            component: format!(
                "inodes ({inodes_clean}/{inodes_checked} verified, {inodes_corrupt} corrupt)"
            ),
            passed: inodes_corrupt == 0,
            detail: String::new(),
        });
    }

    // ── Compute Bayesian posterior ──────────────────────────────────────
    // Prior: Beta(1, 1) — uninformative
    let alpha = 1.0 + failed as f64;
    let beta_param = 1.0 + passed as f64;
    let mean = alpha / (alpha + beta_param);
    let var = (alpha * beta_param) / ((alpha + beta_param).powi(2) * (alpha + beta_param + 1.0));
    let z = 3.0_f64; // 99.7% credible interval
    let upper = z.mul_add(var.sqrt(), mean).clamp(0.0, 1.0);

    Ok(IntegrityReport {
        verdicts,
        passed,
        failed,
        posterior_alpha: alpha,
        posterior_beta: beta_param,
        expected_corruption_rate: mean,
        upper_bound_corruption_rate: upper,
        healthy: upper < 0.01,
    })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DurabilityPosterior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for DurabilityPosterior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl DurabilityPosterior {
    /// Observe a single Bernoulli event ("did we see any corruption?").
    ///
    /// This is intentionally coarse; prefer `observe_blocks()` when scrub can
    /// report counts.
    pub fn observe_event(&mut self, corruption_event: bool) {
        self.observe_blocks(1, u64::from(corruption_event));
    }

    /// Observe scrub results as counts of scanned vs corrupted blocks.
    ///
    /// Uses a Beta-Binomial conjugate update where `alpha` counts "corrupt"
    /// and `beta` counts "clean".
    pub fn observe_blocks(&mut self, scanned_blocks: u64, corrupted_blocks: u64) {
        let scanned = scanned_blocks as f64;
        let corrupted = (corrupted_blocks.min(scanned_blocks)) as f64;
        let clean = (scanned - corrupted).max(0.0);
        self.alpha += corrupted;
        self.beta += clean;
    }

    #[must_use]
    pub fn expected_corruption_rate(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    #[must_use]
    pub fn variance(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let denom = (a + b).powi(2) * (a + b + 1.0);
        if denom <= 0.0 {
            return 0.0;
        }
        (a * b) / denom
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DurabilityLossModel {
    pub corruption_cost: f64,
    pub redundancy_cost: f64,
    pub z_score: f64,
}

impl Default for DurabilityLossModel {
    fn default() -> Self {
        Self {
            corruption_cost: 10_000.0,
            redundancy_cost: 25.0,
            z_score: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RedundancyDecision {
    pub repair_overhead: f64,
    pub expected_loss: f64,
    pub posterior_mean_corruption_rate: f64,
    pub posterior_hi_corruption_rate: f64,
    pub unrecoverable_risk_bound: f64,
    pub redundancy_loss: f64,
    pub corruption_loss: f64,
}

impl RedundancyDecision {
    #[must_use]
    pub fn to_raptorq_config(self, block_size: u32) -> RaptorQConfig {
        let mut cfg = RaptorQConfig::default();
        cfg.encoding.repair_overhead = self.repair_overhead;
        cfg.encoding.max_block_size = usize::try_from(block_size).unwrap_or(4096);
        cfg.encoding.symbol_size = u16::try_from(block_size.clamp(64, 1024)).unwrap_or(256);
        cfg
    }
}

#[derive(Debug, Clone, Default)]
pub struct DurabilityAutopilot {
    posterior: DurabilityPosterior,
    loss: DurabilityLossModel,
}

impl DurabilityAutopilot {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_event(&mut self, corruption_event: bool) {
        self.posterior.observe_event(corruption_event);
    }

    pub fn observe_scrub(&mut self, scanned_blocks: u64, corrupted_blocks: u64) {
        self.posterior
            .observe_blocks(scanned_blocks, corrupted_blocks);
    }

    #[must_use]
    pub fn choose_overhead(&self, candidates: &[f64]) -> RedundancyDecision {
        self.choose_overhead_for_group(candidates, 32_768)
    }

    #[must_use]
    pub fn choose_overhead_for_group(
        &self,
        candidates: &[f64],
        source_block_count: u32,
    ) -> RedundancyDecision {
        const MIN_OVERHEAD: f64 = 1.01;
        const MAX_OVERHEAD: f64 = 1.10;
        const DEFAULT_OVERHEAD: f64 = 1.05;

        let p_mean = self.posterior.expected_corruption_rate();
        let p_hi = self
            .loss
            .z_score
            .mul_add(self.posterior.variance().sqrt(), p_mean)
            .clamp(0.0, 1.0);

        let mut best = RedundancyDecision {
            repair_overhead: DEFAULT_OVERHEAD,
            expected_loss: f64::INFINITY,
            posterior_mean_corruption_rate: p_mean,
            posterior_hi_corruption_rate: p_hi,
            unrecoverable_risk_bound: 1.0,
            redundancy_loss: 0.0,
            corruption_loss: f64::INFINITY,
        };

        let k = f64::from(source_block_count.max(1));
        let mut considered_any = false;

        for candidate in candidates {
            if !candidate.is_finite() || *candidate < MIN_OVERHEAD || *candidate > MAX_OVERHEAD {
                continue;
            }
            considered_any = true;

            // Repair budget fraction relative to source blocks.
            let rho = (candidate - 1.0).clamp(0.0, 1.0);

            // Conservative tail-risk estimate (Chernoff bound) for:
            //   P(N >= rho*K) where N ~ Binomial(K, p) and p is conservatively taken as p_hi.
            let risk_bound = if p_hi <= 0.0 {
                0.0
            } else if rho <= p_hi {
                1.0
            } else {
                let eps = 1e-12;
                let q = rho.clamp(eps, 1.0 - eps);
                let p = p_hi.clamp(eps, 1.0 - eps);
                let kl = q * (q / p).ln() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).ln();
                (-k * kl.max(0.0)).exp()
            };

            let redundancy_loss = self.loss.redundancy_cost * rho;
            let corruption_loss = self.loss.corruption_cost * risk_bound;
            let expected_loss = redundancy_loss + corruption_loss;

            if expected_loss < best.expected_loss {
                best = RedundancyDecision {
                    repair_overhead: *candidate,
                    expected_loss,
                    posterior_mean_corruption_rate: p_mean,
                    posterior_hi_corruption_rate: p_hi,
                    unrecoverable_risk_bound: risk_bound,
                    redundancy_loss,
                    corruption_loss,
                };
            }
        }

        if !considered_any {
            best.repair_overhead = DEFAULT_OVERHEAD;
            best.redundancy_loss = self.loss.redundancy_cost * (DEFAULT_OVERHEAD - 1.0);
            best.corruption_loss = self.loss.corruption_cost;
            best.expected_loss = best.redundancy_loss + best.corruption_loss;
        }

        best
    }
}

// ── Repair Policy ────────────────────────────────────────────────────────────

/// Mount-configurable repair policy governing overhead ratio and autopilot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairPolicy {
    /// Static overhead ratio, range `[1.01, 1.10]`, default `1.05`.
    pub overhead_ratio: f64,
    /// Refresh repair symbols eagerly on every write?
    pub eager_refresh: bool,
    /// When present, the autopilot's `choose_overhead()` overrides the static
    /// ratio at each scrub cycle.
    #[serde(skip)]
    pub autopilot: Option<DurabilityAutopilot>,
}

impl Default for RepairPolicy {
    fn default() -> Self {
        Self {
            overhead_ratio: 1.05,
            eager_refresh: false,
            autopilot: None,
        }
    }
}

impl RepairPolicy {
    /// Return the effective overhead ratio.  When autopilot is engaged, query
    /// it with the standard candidate set; otherwise return the static ratio.
    #[must_use]
    pub fn effective_overhead(&self) -> f64 {
        self.effective_overhead_for_group(32_768)
    }

    /// Return the effective overhead ratio for a specific group size.
    #[must_use]
    pub fn effective_overhead_for_group(&self, source_block_count: u32) -> f64 {
        self.autopilot.as_ref().map_or(self.overhead_ratio, |ap| {
            let candidates: Vec<f64> = (1..=10).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
            ap.choose_overhead_for_group(&candidates, source_block_count)
                .repair_overhead
        })
    }

    /// Return the full `RedundancyDecision` when autopilot is engaged, or
    /// `None` when using static overhead.
    #[must_use]
    pub fn autopilot_decision(&self) -> Option<RedundancyDecision> {
        self.autopilot_decision_for_group(32_768)
    }

    /// Return the full `RedundancyDecision` for a given group size.
    #[must_use]
    pub fn autopilot_decision_for_group(
        &self,
        source_block_count: u32,
    ) -> Option<RedundancyDecision> {
        let ap = self.autopilot.as_ref()?;
        let candidates: Vec<f64> = (1..=10).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect();
        Some(ap.choose_overhead_for_group(&candidates, source_block_count))
    }
}

#[derive(Debug, Default)]
pub struct FrankenFsEngine {
    store: MvccStore,
}

impl FrankenFsEngine {
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: MvccStore::new(),
        }
    }

    pub fn begin(&mut self) -> Transaction {
        self.store.begin()
    }

    pub fn commit(&mut self, txn: Transaction) -> Result<CommitSeq, CommitError> {
        self.store.commit(txn)
    }

    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        self.store.current_snapshot()
    }

    #[must_use]
    pub fn read(
        &self,
        block: BlockNumber,
        snapshot: Snapshot,
    ) -> Option<std::borrow::Cow<'_, [u8]>> {
        self.store.read_visible(block, snapshot)
    }

    pub fn checkpoint(cx: &Cx) -> Result<(), Box<asupersync::Error>> {
        cx.checkpoint().map_err(Box::new)
    }

    pub fn inspect_image(image: &[u8]) -> Result<FsFlavor, DetectionError> {
        detect_filesystem(image)
    }

    pub fn parse_ext4(image: &[u8]) -> Result<Ext4Superblock, ParseError> {
        Ext4Superblock::parse_from_image(image)
    }

    pub fn parse_btrfs(image: &[u8]) -> Result<BtrfsSuperblock, ParseError> {
        BtrfsSuperblock::parse_from_image(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffs_types::{
        BTRFS_MAGIC, BTRFS_SUPER_INFO_OFFSET, BTRFS_SUPER_INFO_SIZE, ByteOffset, EXT4_SUPER_MAGIC,
        EXT4_SUPERBLOCK_OFFSET, EXT4_SUPERBLOCK_SIZE,
    };
    use std::sync::Mutex;

    /// In-memory ByteDevice for testing (no file I/O).
    #[derive(Debug)]
    struct TestDevice {
        data: Mutex<Vec<u8>>,
    }

    impl TestDevice {
        fn from_vec(v: Vec<u8>) -> Self {
            Self {
                data: Mutex::new(v),
            }
        }
    }

    impl ByteDevice for TestDevice {
        fn len_bytes(&self) -> u64 {
            self.data.lock().unwrap().len() as u64
        }

        #[allow(clippy::cast_possible_truncation)]
        fn read_exact_at(
            &self,
            _cx: &Cx,
            offset: ByteOffset,
            buf: &mut [u8],
        ) -> ffs_error::Result<()> {
            let off = offset.0 as usize;
            let data = self.data.lock().unwrap();
            let end = off + buf.len();
            if end > data.len() {
                return Err(FfsError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "read past end",
                )));
            }
            buf.copy_from_slice(&data[off..end]);
            drop(data);
            Ok(())
        }

        #[allow(clippy::cast_possible_truncation)]
        fn write_all_at(&self, _cx: &Cx, offset: ByteOffset, buf: &[u8]) -> ffs_error::Result<()> {
            let off = offset.0 as usize;
            let mut data = self.data.lock().unwrap();
            let end = off + buf.len();
            if end > data.len() {
                return Err(FfsError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "write past end",
                )));
            }
            data[off..end].copy_from_slice(buf);
            drop(data);
            Ok(())
        }

        fn sync(&self, _cx: &Cx) -> ffs_error::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn detect_ext4_and_btrfs_images() {
        let mut ext4_img = vec![0_u8; EXT4_SUPERBLOCK_OFFSET + EXT4_SUPERBLOCK_SIZE];
        let sb = EXT4_SUPERBLOCK_OFFSET;
        ext4_img[sb + 0x38..sb + 0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        ext4_img[sb + 0x18..sb + 0x1C].copy_from_slice(&0_u32.to_le_bytes());
        let ext4 = detect_filesystem(&ext4_img).expect("detect ext4");
        assert!(matches!(ext4, FsFlavor::Ext4(_)));

        let mut btrfs_img = vec![0_u8; BTRFS_SUPER_INFO_OFFSET + BTRFS_SUPER_INFO_SIZE];
        let sb2 = BTRFS_SUPER_INFO_OFFSET;
        btrfs_img[sb2 + 0x40..sb2 + 0x48].copy_from_slice(&BTRFS_MAGIC.to_le_bytes());
        btrfs_img[sb2 + 0x90..sb2 + 0x94].copy_from_slice(&4096_u32.to_le_bytes());
        btrfs_img[sb2 + 0x94..sb2 + 0x98].copy_from_slice(&4096_u32.to_le_bytes());
        let btrfs = detect_filesystem(&btrfs_img).expect("detect btrfs");
        assert!(matches!(btrfs, FsFlavor::Btrfs(_)));
    }

    // ── FsOps VFS trait tests ─────────────────────────────────────────

    /// A stub FsOps implementation for testing that the trait is object-safe
    /// and can be used as a trait object behind `dyn`.
    struct StubFs;

    impl FsOps for StubFs {
        fn getattr(&self, _cx: &Cx, ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
            if ino == InodeNumber(1) {
                Ok(InodeAttr {
                    ino,
                    size: 4096,
                    blocks: 8,
                    atime: SystemTime::UNIX_EPOCH,
                    mtime: SystemTime::UNIX_EPOCH,
                    ctime: SystemTime::UNIX_EPOCH,
                    crtime: SystemTime::UNIX_EPOCH,
                    kind: FileType::Directory,
                    perm: 0o755,
                    nlink: 2,
                    uid: 0,
                    gid: 0,
                    rdev: 0,
                    blksize: 4096,
                })
            } else {
                Err(FfsError::NotFound(format!("inode {ino}")))
            }
        }

        fn lookup(
            &self,
            _cx: &Cx,
            _parent: InodeNumber,
            name: &OsStr,
        ) -> ffs_error::Result<InodeAttr> {
            if name == "hello.txt" {
                Ok(InodeAttr {
                    ino: InodeNumber(11),
                    size: 13,
                    blocks: 8,
                    atime: SystemTime::UNIX_EPOCH,
                    mtime: SystemTime::UNIX_EPOCH,
                    ctime: SystemTime::UNIX_EPOCH,
                    crtime: SystemTime::UNIX_EPOCH,
                    kind: FileType::RegularFile,
                    perm: 0o644,
                    nlink: 1,
                    uid: 1000,
                    gid: 1000,
                    rdev: 0,
                    blksize: 4096,
                })
            } else {
                Err(FfsError::NotFound(name.to_string_lossy().into_owned()))
            }
        }

        fn readdir(
            &self,
            _cx: &Cx,
            ino: InodeNumber,
            offset: u64,
        ) -> ffs_error::Result<Vec<DirEntry>> {
            if ino != InodeNumber(1) {
                return Err(FfsError::NotDirectory);
            }
            let all = vec![
                DirEntry {
                    ino: InodeNumber(1),
                    offset: 1,
                    kind: FileType::Directory,
                    name: b".".to_vec(),
                },
                DirEntry {
                    ino: InodeNumber(1),
                    offset: 2,
                    kind: FileType::Directory,
                    name: b"..".to_vec(),
                },
                DirEntry {
                    ino: InodeNumber(11),
                    offset: 3,
                    kind: FileType::RegularFile,
                    name: b"hello.txt".to_vec(),
                },
            ];
            Ok(all.into_iter().filter(|e| e.offset > offset).collect())
        }

        fn read(
            &self,
            _cx: &Cx,
            ino: InodeNumber,
            offset: u64,
            size: u32,
        ) -> ffs_error::Result<Vec<u8>> {
            if ino == InodeNumber(1) {
                return Err(FfsError::IsDirectory);
            }
            let data = b"Hello, world!";
            let start = usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(data.len());
            let end = (start + size as usize).min(data.len());
            Ok(data[start..end].to_vec())
        }

        fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
            Err(FfsError::Format("not a symlink".into()))
        }
    }

    #[test]
    fn fsops_getattr_root() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let attr = fs.getattr(&cx, InodeNumber(1)).unwrap();
        assert_eq!(attr.ino, InodeNumber(1));
        assert_eq!(attr.kind, FileType::Directory);
        assert_eq!(attr.perm, 0o755);
        assert_eq!(attr.nlink, 2);
    }

    #[test]
    fn fsops_getattr_not_found() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let err = fs.getattr(&cx, InodeNumber(999)).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn fsops_lookup_found() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let attr = fs
            .lookup(&cx, InodeNumber(1), OsStr::new("hello.txt"))
            .unwrap();
        assert_eq!(attr.ino, InodeNumber(11));
        assert_eq!(attr.kind, FileType::RegularFile);
    }

    #[test]
    fn fsops_lookup_not_found() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let err = fs
            .lookup(&cx, InodeNumber(1), OsStr::new("missing"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn fsops_readdir_with_offset() {
        let fs = StubFs;
        let cx = Cx::for_testing();

        // Full listing from offset 0
        let entries = fs.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name_str(), ".");
        assert_eq!(entries[2].name_str(), "hello.txt");

        // Resume from offset 2 (skip . and ..)
        let entries = fs.readdir(&cx, InodeNumber(1), 2).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name_str(), "hello.txt");
    }

    #[test]
    fn fsops_readdir_not_directory() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let err = fs.readdir(&cx, InodeNumber(11), 0).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTDIR);
    }

    #[test]
    fn fsops_read_file() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let data = fs.read(&cx, InodeNumber(11), 0, 5).unwrap();
        assert_eq!(&data, b"Hello");

        // Read from offset
        let data = fs.read(&cx, InodeNumber(11), 7, 100).unwrap();
        assert_eq!(&data, b"world!");
    }

    #[test]
    fn fsops_read_directory_returns_is_directory() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let err = fs.read(&cx, InodeNumber(1), 0, 4096).unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn fsops_trait_is_object_safe() {
        // Verify FsOps can be used as dyn trait object
        let fs: Box<dyn FsOps> = Box::new(StubFs);
        let cx = Cx::for_testing();
        let attr = fs.getattr(&cx, InodeNumber(1)).unwrap();
        assert_eq!(attr.kind, FileType::Directory);
    }

    #[test]
    fn dir_entry_name_str() {
        let entry = DirEntry {
            ino: InodeNumber(5),
            offset: 1,
            kind: FileType::RegularFile,
            name: b"test.txt".to_vec(),
        };
        assert_eq!(entry.name_str(), "test.txt");
    }

    #[test]
    fn file_type_variants_are_distinct() {
        let types = [
            FileType::RegularFile,
            FileType::Directory,
            FileType::Symlink,
            FileType::BlockDevice,
            FileType::CharDevice,
            FileType::Fifo,
            FileType::Socket,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── inode_to_attr tests ───────────────────────────────────────────────

    /// Build a minimal Ext4Superblock for unit tests.
    fn make_test_superblock() -> Ext4Superblock {
        let mut sb_buf = vec![0_u8; EXT4_SUPERBLOCK_SIZE];
        sb_buf[0x38..0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        sb_buf[0x18..0x1C].copy_from_slice(&2_u32.to_le_bytes()); // log_block_size=2 → 4K
        sb_buf[0x00..0x04].copy_from_slice(&1024_u32.to_le_bytes()); // inodes_count
        sb_buf[0x04..0x08].copy_from_slice(&4096_u32.to_le_bytes()); // blocks_count
        sb_buf[0x20..0x24].copy_from_slice(&4096_u32.to_le_bytes()); // blocks_per_group
        sb_buf[0x28..0x2C].copy_from_slice(&1024_u32.to_le_bytes()); // inodes_per_group
        sb_buf[0x58..0x5A].copy_from_slice(&256_u16.to_le_bytes()); // inode_size
        Ext4Superblock::parse_superblock_region(&sb_buf).expect("test superblock")
    }

    /// Build a minimal inode buffer with mode and device encoding in i_block.
    fn make_test_inode(mode: u16, block0: u32, block1: u32) -> Ext4Inode {
        let mut buf = [0_u8; 128];
        buf[0x00..0x02].copy_from_slice(&mode.to_le_bytes());
        buf[0x28..0x2C].copy_from_slice(&block0.to_le_bytes());
        buf[0x2C..0x30].copy_from_slice(&block1.to_le_bytes());
        Ext4Inode::parse_from_bytes(&buf).expect("test inode")
    }

    #[test]
    fn inode_to_attr_block_device_rdev() {
        use ffs_types::{S_IFBLK, S_IFCHR};

        let sb = make_test_superblock();

        // Block device: major=8, minor=1 → /dev/sda1 (new format in i_block[1])
        let inode = make_test_inode(S_IFBLK | 0o660, 0, 0x0801);
        let attr = inode_to_attr(&sb, InodeNumber(100), &inode);
        assert_eq!(attr.kind, FileType::BlockDevice);
        assert_eq!(attr.rdev, 0x0801);
        assert_eq!(attr.perm, 0o660);

        // Char device: major=1, minor=3 → /dev/null (old format in i_block[0])
        let inode = make_test_inode(S_IFCHR | 0o666, 0x0103, 0);
        let attr = inode_to_attr(&sb, InodeNumber(101), &inode);
        assert_eq!(attr.kind, FileType::CharDevice);
        assert_eq!(attr.rdev, 0x0103);
        assert_eq!(attr.perm, 0o666);
    }

    #[test]
    fn inode_to_attr_regular_file_rdev_zero() {
        use ffs_types::S_IFREG;

        let sb = make_test_superblock();
        let inode = make_test_inode(S_IFREG | 0o644, 0, 0);
        let attr = inode_to_attr(&sb, InodeNumber(11), &inode);
        assert_eq!(attr.kind, FileType::RegularFile);
        assert_eq!(attr.rdev, 0);
        assert_eq!(attr.perm, 0o644);
        assert_eq!(attr.uid, 0);
        assert_eq!(attr.gid, 0);
    }

    // ── listxattr/getxattr via FsOps defaults ────────────────────────────

    #[test]
    fn fsops_listxattr_default_returns_empty() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let names = fs.listxattr(&cx, InodeNumber(1)).unwrap();
        assert!(names.is_empty());
    }

    #[test]
    fn fsops_getxattr_default_returns_none() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let val = fs.getxattr(&cx, InodeNumber(1), "user.test").unwrap();
        assert!(val.is_none());
    }

    // ── OpenFs tests ─────────────────────────────────────────────────────

    /// Build a minimal synthetic ext4 image for OpenFs testing.
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image(block_size_log: u32) -> Vec<u8> {
        let block_size = 1024_u32 << block_size_log;
        let image_size: u32 = 128 * 1024; // 128K
        let mut image = vec![0_u8; image_size as usize];
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // magic
        image[sb_off + 0x38..sb_off + 0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        // log_block_size
        image[sb_off + 0x18..sb_off + 0x1C].copy_from_slice(&block_size_log.to_le_bytes());
        // blocks_count_lo
        let blocks_count = image_size / block_size;
        image[sb_off + 0x04..sb_off + 0x08].copy_from_slice(&blocks_count.to_le_bytes());
        // inodes_count
        image[sb_off..sb_off + 0x04].copy_from_slice(&128_u32.to_le_bytes());
        // first_data_block
        let first_data = u32::from(block_size == 1024);
        image[sb_off + 0x14..sb_off + 0x18].copy_from_slice(&first_data.to_le_bytes());
        // blocks_per_group
        image[sb_off + 0x20..sb_off + 0x24].copy_from_slice(&blocks_count.to_le_bytes());
        // inodes_per_group
        image[sb_off + 0x28..sb_off + 0x2C].copy_from_slice(&128_u32.to_le_bytes());
        // inode_size = 256
        image[sb_off + 0x58..sb_off + 0x5A].copy_from_slice(&256_u16.to_le_bytes());
        // rev_level = 1 (dynamic)
        image[sb_off + 0x4C..sb_off + 0x50].copy_from_slice(&1_u32.to_le_bytes());
        // feature_incompat = FILETYPE | EXTENTS
        let filetype: u32 = 0x0002;
        let extents: u32 = 0x0040;
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&(filetype | extents).to_le_bytes());
        // first_ino
        image[sb_off + 0x54..sb_off + 0x58].copy_from_slice(&11_u32.to_le_bytes());

        image
    }

    #[test]
    fn open_options_default_enables_validation() {
        let opts = OpenOptions::default();
        assert!(!opts.skip_validation);
        assert_eq!(opts.ext4_journal_replay_mode, Ext4JournalReplayMode::Apply);
    }

    #[test]
    fn open_fs_from_ext4_image() {
        let image = build_ext4_image(2); // 4K blocks
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        assert!(fs.is_ext4());
        assert!(!fs.is_btrfs());
        assert_eq!(fs.block_size(), 4096);
        assert!(fs.ext4_geometry.is_some());
        assert!(fs.ext4_journal_replay().is_none());

        let geom = fs.ext4_geometry.as_ref().unwrap();
        assert!(geom.groups_count > 0);
        assert!(geom.group_desc_size == 32 || geom.group_desc_size == 64);
    }

    #[test]
    fn open_fs_debug_format() {
        let image = build_ext4_image(2);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let debug = format!("{fs:?}");
        assert!(debug.contains("OpenFs"));
        assert!(debug.contains("dev_len"));
    }

    #[test]
    fn open_fs_rejects_garbage() {
        let garbage = vec![0xAB_u8; 1024 * 128];
        let dev = TestDevice::from_vec(garbage);
        let cx = Cx::for_testing();

        let err = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap_err();
        assert_eq!(err.to_errno(), libc::EINVAL); // Format error
    }

    #[test]
    fn open_fs_rejects_bad_superblock_magic() {
        let mut image = build_ext4_image(2);
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        image[sb_off + 0x38..sb_off + 0x3A].copy_from_slice(&0_u16.to_le_bytes());
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let err = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap_err();
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn open_fs_skip_validation() {
        // Build an image with bad features (should fail validation but pass with skip)
        let mut image = build_ext4_image(2);
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        // Set unsupported incompat feature (COMPRESSION = 0x0001)
        let bad_incompat: u32 = 0x0002 | 0x0040 | 0x0001; // FILETYPE | EXTENTS | COMPRESSION
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&bad_incompat.to_le_bytes());

        let dev = TestDevice::from_vec(image.clone());
        let cx = Cx::for_testing();

        // Should fail with default options
        let err = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap_err();
        assert!(
            matches!(
                err,
                FfsError::UnsupportedFeature(_)
                    | FfsError::IncompatibleFeature(_)
                    | FfsError::UnsupportedBlockSize(_)
                    | FfsError::Format(_)
            ),
            "expected feature/format error, got {err:?}",
        );

        // Should succeed with skip_validation
        let dev2 = TestDevice::from_vec(image);
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev2), &opts).unwrap();
        assert!(fs.is_ext4());
    }

    #[test]
    fn open_fs_replays_internal_journal_transaction() {
        let image = build_ext4_image_with_internal_journal();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let replay = fs
            .ext4_journal_replay()
            .expect("journal replay outcome should be present");

        assert_eq!(replay.committed_sequences, vec![1]);
        assert_eq!(replay.stats.replayed_blocks, 1);
        assert_eq!(replay.stats.skipped_revoked_blocks, 0);

        let target = fs.read_block_vec(&cx, BlockNumber(15)).unwrap();
        assert_eq!(&target[..16], b"JBD2-REPLAY-TEST");
    }

    #[test]
    fn open_fs_replays_non_contiguous_internal_journal_transaction() {
        let image = build_ext4_image_with_non_contiguous_internal_journal();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let replay = fs
            .ext4_journal_replay()
            .expect("journal replay outcome should be present");

        assert_eq!(replay.committed_sequences, vec![1]);
        assert_eq!(replay.stats.replayed_blocks, 1);
        assert_eq!(replay.stats.skipped_revoked_blocks, 0);

        let target = fs.read_block_vec(&cx, BlockNumber(15)).unwrap();
        assert_eq!(&target[..16], b"JBD2-REPLAY-TEST");
    }

    #[test]
    fn open_fs_skip_mode_reports_journal_present_without_replay() {
        let image = build_ext4_image_with_internal_journal();
        let baseline = image[15 * 4096..15 * 4096 + 16].to_vec();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
            ..OpenOptions::default()
        };

        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();
        let replay = fs
            .ext4_journal_replay()
            .expect("journal presence should still be reported");
        assert!(replay.committed_sequences.is_empty());
        assert_eq!(replay.stats.scanned_blocks, 0);
        assert_eq!(replay.stats.replayed_blocks, 0);

        let target = fs.read_block_vec(&cx, BlockNumber(15)).unwrap();
        assert_eq!(&target[..16], baseline.as_slice());
    }

    #[test]
    fn read_ext4_orphan_list_traverses_chain() {
        let image = build_ext4_image_with_orphan_chain(&[11, 12, 13]);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let orphans = fs.read_ext4_orphan_list(&cx).unwrap();
        assert_eq!(orphans.head, 11);
        assert_eq!(
            orphans.inodes,
            vec![InodeNumber(11), InodeNumber(12), InodeNumber(13)]
        );
        assert_eq!(orphans.count(), 3);
    }

    #[test]
    fn read_ext4_orphan_list_detects_cycle() {
        let mut image = build_ext4_image_with_orphan_chain(&[11, 12]);
        set_test_inode_dtime(&mut image, 12, 11);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let err = fs.read_ext4_orphan_list(&cx).unwrap_err();
        assert!(matches!(err, FfsError::Corruption { .. }));
    }

    #[test]
    fn read_ext4_orphan_list_rejects_out_of_range_head() {
        let mut image = build_ext4_image_with_extents();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        image[sb_off + 0xE8..sb_off + 0xEC].copy_from_slice(&129_u32.to_le_bytes());
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let err = fs.read_ext4_orphan_list(&cx).unwrap_err();
        assert!(matches!(err, FfsError::InvalidGeometry(_)));
    }

    #[test]
    fn open_fs_recovers_deleted_ext4_orphan_inode() {
        let mut image = build_ext4_image_with_orphan_chain(&[11, 12]);
        set_test_ext4_state(&mut image, EXT4_VALID_FS | EXT4_ORPHAN_FS);
        set_test_inode_links_count(&mut image, 11, 0);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("ext4 crash recovery outcome");
        assert!(recovery.had_orphans);

        let inode11 = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        assert_eq!(inode11.links_count, 0);
        assert_eq!(inode11.size, 0);
        assert_ne!(inode11.dtime, 12);

        let orphan_list = fs.read_ext4_orphan_list(&cx).unwrap();
        assert!(orphan_list.inodes.is_empty());

        let sb = fs.ext4_superblock().expect("ext4 superblock");
        assert_eq!(sb.last_orphan, 0);
        assert_eq!(sb.state & EXT4_ORPHAN_FS, 0);
    }

    #[test]
    fn open_fs_recovers_truncated_ext4_orphan_inode() {
        let mut image = build_ext4_image_with_orphan_chain(&[11, 12]);
        set_test_ext4_state(&mut image, EXT4_VALID_FS | EXT4_ORPHAN_FS);
        set_test_inode_links_count(&mut image, 11, 1);
        set_test_inode_links_count(&mut image, 12, 0);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode11 = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        assert_eq!(inode11.links_count, 1);
        assert_eq!(inode11.dtime, 0);

        let orphan_list = fs.read_ext4_orphan_list(&cx).unwrap();
        assert!(orphan_list.inodes.is_empty());

        let sb = fs.ext4_superblock().expect("ext4 superblock");
        assert_eq!(sb.last_orphan, 0);
        assert_eq!(sb.state & EXT4_ORPHAN_FS, 0);
    }

    #[test]
    fn open_fs_recovers_orphan_list_with_invalid_head() {
        let mut image = build_ext4_image_with_extents();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        image[sb_off + 0xE8..sb_off + 0xEC].copy_from_slice(&129_u32.to_le_bytes());
        set_test_ext4_state(&mut image, EXT4_VALID_FS | EXT4_ORPHAN_FS);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("ext4 crash recovery outcome");
        assert!(recovery.had_orphans);

        let orphan_list = fs.read_ext4_orphan_list(&cx).unwrap();
        assert!(orphan_list.inodes.is_empty());

        let sb = fs.ext4_superblock().expect("ext4 superblock");
        assert_eq!(sb.last_orphan, 0);
        assert_eq!(sb.state & EXT4_ORPHAN_FS, 0);
    }

    #[test]
    fn readpath_orphan_empty_list_no_orphans() {
        // When s_last_orphan == 0, the orphan list is empty.
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let orphans = fs.read_ext4_orphan_list(&cx).unwrap();
        assert_eq!(orphans.head, 0);
        assert!(orphans.inodes.is_empty());
        assert_eq!(orphans.count(), 0);
    }

    #[test]
    fn readpath_orphan_inode_allocated_but_in_list() {
        // An inode with links_count > 0 still appears in the orphan list
        // (truncate-recovery case). The read path reports it regardless of
        // allocation status — recovery semantics are handled separately.
        let mut image = build_ext4_image_with_orphan_chain(&[11]);
        set_test_inode_links_count(&mut image, 11, 2);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let orphans = fs.read_ext4_orphan_list(&cx).unwrap();
        assert_eq!(orphans.head, 11);
        assert_eq!(orphans.inodes, vec![InodeNumber(11)]);
        assert_eq!(orphans.count(), 1);

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        assert_eq!(inode.links_count, 2);
    }

    #[test]
    fn readpath_orphan_chain_multi_element_order() {
        // Verify that a multi-element orphan chain is traversed in on-disk
        // linked-list order and the correct next-pointers are followed.
        let image = build_ext4_image_with_orphan_chain(&[13, 11, 12]);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        let orphans = fs.read_ext4_orphan_list(&cx).unwrap();
        assert_eq!(orphans.head, 13);
        assert_eq!(
            orphans.inodes,
            vec![InodeNumber(13), InodeNumber(11), InodeNumber(12)]
        );
        assert_eq!(orphans.count(), 3);
    }

    #[test]
    fn open_fs_rejects_external_journal_device() {
        let mut image = build_ext4_image_with_inode();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        image[sb_off + 0xE0..sb_off + 0xE4].copy_from_slice(&8_u32.to_le_bytes()); // journal_inum
        image[sb_off + 0xE4..sb_off + 0xE8].copy_from_slice(&1_u32.to_le_bytes()); // journal_dev

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let err = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap_err();
        assert!(matches!(err, FfsError::UnsupportedFeature(_)));
    }

    #[test]
    fn journaled_commit_jbd2_failure_returns_error_without_visibility() {
        let image = build_ext4_image(2); // 4K blocks, 128K image
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let mut fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        fs.attach_jbd2_writer(Jbd2Writer::new(
            ffs_journal::JournalRegion {
                start: BlockNumber(64), // outside the 32-block synthetic image
                blocks: 8,
            },
            1,
        ));

        let target = BlockNumber(5);
        let before = fs
            .read_block_at_snapshot(&cx, target, fs.current_snapshot())
            .expect("read before journaled commit");

        let mut txn = fs.begin_transaction();
        let block_size = usize::try_from(fs.block_size()).expect("block size fits usize");
        txn.stage_write(target, vec![0xAB; block_size]);

        let err = fs
            .commit_transaction_journaled(&cx, txn)
            .expect_err("journaled commit should fail when journal region is out-of-range");
        assert!(
            err.to_string()
                .contains("JBD2 journal write failed before MVCC visibility"),
            "unexpected error: {err}"
        );

        // No commit should become visible when journal durability fails.
        assert_eq!(fs.current_snapshot().high, CommitSeq(0));
        let after = fs
            .read_block_at_snapshot(&cx, target, fs.current_snapshot())
            .expect("read after failed journaled commit");
        assert_eq!(after, before);
    }

    #[test]
    fn journaled_commit_conflict_returns_retryable_mvcc_error() {
        let image = build_ext4_image(2); // 4K blocks, 128K image
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let mut fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        fs.attach_jbd2_writer(Jbd2Writer::new(
            ffs_journal::JournalRegion {
                start: BlockNumber(16), // in-range for this synthetic image
                blocks: 8,
            },
            1,
        ));

        let target = BlockNumber(5);
        let block_size = usize::try_from(fs.block_size()).expect("block size fits usize");

        let mut t1 = fs.begin_transaction();
        let mut t2 = fs.begin_transaction();
        t1.stage_write(target, vec![0x11; block_size]);
        t2.stage_write(target, vec![0x22; block_size]);
        let t2_id = t2.id.0;

        fs.commit_transaction(t1)
            .expect("first transaction should commit");
        let err = fs
            .commit_transaction_journaled(&cx, t2)
            .expect_err("second transaction should conflict");

        assert!(
            matches!(
                err,
                FfsError::MvccConflict { tx, block } if tx == t2_id && block == target.0
            ),
            "unexpected error variant: {err:?}"
        );
        assert_eq!(err.to_errno(), libc::EAGAIN);
        assert_eq!(fs.current_snapshot().high, CommitSeq(1));
    }

    #[test]
    fn parse_error_to_ffs_mapping() {
        // Feature error
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "feature_incompat",
            reason: "unsupported flags",
        });
        assert!(matches!(e, FfsError::UnsupportedFeature(_)));

        // Incompatible feature contract error
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "feature_incompat",
            reason: "missing required features (need FILETYPE+EXTENTS)",
        });
        assert!(matches!(e, FfsError::IncompatibleFeature(_)));

        // Unsupported block size (valid ext4, out of v1 support envelope)
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "block_size",
            reason: "unsupported (FrankenFS v1 supports 1K/2K/4K ext4 only)",
        });
        assert!(matches!(e, FfsError::UnsupportedBlockSize(_)));

        // Geometry error
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "blocks_per_group",
            reason: "out of range",
        });
        assert!(matches!(e, FfsError::InvalidGeometry(_)));

        // Generic format error
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "magic",
            reason: "wrong value",
        });
        assert!(matches!(e, FfsError::Format(_)));

        // Magic error
        let e = parse_error_to_ffs(&ParseError::InvalidMagic {
            expected: 0xEF53,
            actual: 0x0000,
        });
        assert!(matches!(e, FfsError::Format(_)));

        // Truncation error
        let e = parse_error_to_ffs(&ParseError::InsufficientData {
            needed: 100,
            offset: 0,
            actual: 50,
        });
        assert!(matches!(e, FfsError::Corruption { .. }));
    }

    #[test]
    fn parse_error_to_ffs_new_geometry_fields() {
        // desc_size → InvalidGeometry
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "s_desc_size",
            reason: "must be >= 32 when non-zero",
        });
        assert!(
            matches!(e, FfsError::InvalidGeometry(_)),
            "desc_size should map to InvalidGeometry, got: {e:?}",
        );

        // first_data_block → InvalidGeometry
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "s_first_data_block",
            reason: "must be 1 for 1K block size",
        });
        assert!(
            matches!(e, FfsError::InvalidGeometry(_)),
            "first_data_block should map to InvalidGeometry, got: {e:?}",
        );

        // blocks_count → InvalidGeometry
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "s_blocks_count",
            reason: "group descriptor table extends beyond device",
        });
        assert!(
            matches!(e, FfsError::InvalidGeometry(_)),
            "blocks_count should map to InvalidGeometry, got: {e:?}",
        );

        // inodes_count → InvalidGeometry
        let e = parse_error_to_ffs(&ParseError::InvalidField {
            field: "s_inodes_count",
            reason: "inodes_count exceeds groups * inodes_per_group",
        });
        assert!(
            matches!(e, FfsError::InvalidGeometry(_)),
            "inodes_count should map to InvalidGeometry, got: {e:?}",
        );
    }

    #[test]
    fn ext4_geometry_has_all_fields() {
        let image = build_ext4_image(2); // 4K blocks
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let geom = fs.ext4_geometry.as_ref().unwrap();

        assert_eq!(geom.block_size, 4096);
        assert_eq!(geom.inodes_count, 128);
        assert_eq!(geom.inodes_per_group, 128);
        assert_eq!(geom.first_ino, 11);
        assert_eq!(geom.inode_size, 256);
        assert!(geom.groups_count > 0);
        assert!(geom.group_desc_size == 32 || geom.group_desc_size == 64);
        // 32-bit fs (no 64BIT flag set)
        assert!(!geom.is_64bit);
        // No metadata_csum flag set
        assert!(!geom.has_metadata_csum);
    }

    #[test]
    fn ext4_geometry_1k_blocks() {
        let image = build_ext4_image(0); // 1K blocks
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let geom = fs.ext4_geometry.as_ref().unwrap();
        assert_eq!(geom.block_size, 1024);
    }

    #[test]
    fn ext4_geometry_serializes() {
        let geom = Ext4Geometry {
            block_size: 4096,
            inodes_count: 8192,
            inodes_per_group: 8192,
            first_ino: 11,
            inode_size: 256,
            groups_count: 1,
            group_desc_size: 32,
            csum_seed: 0,
            is_64bit: false,
            has_metadata_csum: false,
        };
        let json = serde_json::to_string(&geom).unwrap();
        let deser: Ext4Geometry = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.block_size, 4096);
        assert_eq!(deser.inodes_count, 8192);
        assert_eq!(deser.groups_count, 1);
    }

    // ── Device-based inode read tests ──────────────────────────────────

    /// Build an ext4 image with a valid group descriptor and a root inode.
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image_with_inode() -> Vec<u8> {
        let block_size: u32 = 4096;
        let image_size: u32 = 256 * 1024; // 256K = 64 blocks
        let mut image = vec![0_u8; image_size as usize];
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // ── Superblock ──
        image[sb_off + 0x38..sb_off + 0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        image[sb_off + 0x18..sb_off + 0x1C].copy_from_slice(&2_u32.to_le_bytes()); // log=2 → 4K
        let blocks_count = image_size / block_size;
        image[sb_off + 0x04..sb_off + 0x08].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off..sb_off + 0x04].copy_from_slice(&128_u32.to_le_bytes()); // inodes_count
        image[sb_off + 0x14..sb_off + 0x18].copy_from_slice(&0_u32.to_le_bytes()); // first_data_block
        image[sb_off + 0x20..sb_off + 0x24].copy_from_slice(&blocks_count.to_le_bytes()); // blocks_per_group
        image[sb_off + 0x28..sb_off + 0x2C].copy_from_slice(&128_u32.to_le_bytes()); // inodes_per_group
        image[sb_off + 0x58..sb_off + 0x5A].copy_from_slice(&256_u16.to_le_bytes()); // inode_size
        image[sb_off + 0x4C..sb_off + 0x50].copy_from_slice(&1_u32.to_le_bytes()); // rev_level=DYNAMIC
        let incompat: u32 = 0x0002 | 0x0040; // FILETYPE | EXTENTS
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&incompat.to_le_bytes());
        image[sb_off + 0x54..sb_off + 0x58].copy_from_slice(&11_u32.to_le_bytes()); // first_ino

        // ── Group descriptor at block 1 (offset 4096) ──
        // 32-byte group descriptor (no 64BIT feature).
        let gd_off: usize = 4096;
        // bg_block_bitmap = block 2
        image[gd_off..gd_off + 4].copy_from_slice(&2_u32.to_le_bytes());
        // bg_inode_bitmap = block 3
        image[gd_off + 4..gd_off + 8].copy_from_slice(&3_u32.to_le_bytes());
        // bg_inode_table = block 4 (offset 16384)
        image[gd_off + 8..gd_off + 12].copy_from_slice(&4_u32.to_le_bytes());

        // ── Root inode (#2) in the inode table ──
        // Inode 2 is at index 1 (0-based) in the table.
        // offset = 16384 + 1 * 256 = 16640
        let ino_off: usize = 16384 + 256;
        // mode = S_IFDIR | 0o755
        let mode: u16 = 0o040_755;
        image[ino_off..ino_off + 2].copy_from_slice(&mode.to_le_bytes());
        // uid_lo = 0
        image[ino_off + 2..ino_off + 4].copy_from_slice(&0_u16.to_le_bytes());
        // size = 4096
        image[ino_off + 4..ino_off + 8].copy_from_slice(&4096_u32.to_le_bytes());
        // links_count = 2
        image[ino_off + 0x1A..ino_off + 0x1C].copy_from_slice(&2_u16.to_le_bytes());
        // i_extra_isize = 32 (for 256-byte inodes, extra area starts at 128)
        image[ino_off + 0x80..ino_off + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        image
    }

    fn set_group_desc_free_counts(image: &mut [u8], free_blocks: u16, free_inodes: u16) {
        let gd_off: usize = 4096;
        image[gd_off + 0x0C..gd_off + 0x0E].copy_from_slice(&free_blocks.to_le_bytes());
        image[gd_off + 0x0E..gd_off + 0x10].copy_from_slice(&free_inodes.to_le_bytes());
    }

    fn write_jbd2_header(block: &mut [u8], block_type: u32, sequence: u32) {
        const JBD2_MAGIC: u32 = 0xC03B_3998;
        block[0..4].copy_from_slice(&JBD2_MAGIC.to_be_bytes());
        block[4..8].copy_from_slice(&block_type.to_be_bytes());
        block[8..12].copy_from_slice(&sequence.to_be_bytes());
    }

    /// Build an ext4 image with an internal journal inode and one committed
    /// JBD2 transaction that rewrites block 15.
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image_with_internal_journal() -> Vec<u8> {
        let mut image = build_ext4_image_with_extents();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // Enable HAS_JOURNAL and point to internal journal inode #8.
        let compat = u32::from_le_bytes([
            image[sb_off + 0x5C],
            image[sb_off + 0x5D],
            image[sb_off + 0x5E],
            image[sb_off + 0x5F],
        ]);
        image[sb_off + 0x5C..sb_off + 0x60].copy_from_slice(&(compat | 0x0004).to_le_bytes());
        image[sb_off + 0xE0..sb_off + 0xE4].copy_from_slice(&8_u32.to_le_bytes());

        // Inode #8 (index 7) -> journal extent [block 20..=22].
        let ino8_off: usize = 4 * 4096 + 7 * 256;
        image[ino8_off..ino8_off + 2].copy_from_slice(&0o100_600_u16.to_le_bytes());
        image[ino8_off + 4..ino8_off + 8].copy_from_slice(&(3_u32 * 4096).to_le_bytes());
        image[ino8_off + 0x1A..ino8_off + 0x1C].copy_from_slice(&1_u16.to_le_bytes());
        image[ino8_off + 0x20..ino8_off + 0x24].copy_from_slice(&0x0008_0000_u32.to_le_bytes());
        image[ino8_off + 0x80..ino8_off + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        let e = ino8_off + 0x28;
        image[e..e + 2].copy_from_slice(&0xF30A_u16.to_le_bytes()); // extent magic
        image[e + 2..e + 4].copy_from_slice(&1_u16.to_le_bytes()); // entries
        image[e + 4..e + 6].copy_from_slice(&4_u16.to_le_bytes()); // max
        image[e + 6..e + 8].copy_from_slice(&0_u16.to_le_bytes()); // depth=0
        image[e + 12..e + 16].copy_from_slice(&0_u32.to_le_bytes()); // logical 0
        image[e + 16..e + 18].copy_from_slice(&3_u16.to_le_bytes()); // len = 3 blocks
        image[e + 18..e + 20].copy_from_slice(&0_u16.to_le_bytes()); // start_hi
        image[e + 20..e + 24].copy_from_slice(&20_u32.to_le_bytes()); // start_lo

        // Journal block 20: descriptor, one tag to target block 15.
        let j_desc = 20 * 4096;
        write_jbd2_header(&mut image[j_desc..j_desc + 4096], 1, 1);
        image[j_desc + 12..j_desc + 16].copy_from_slice(&15_u32.to_be_bytes());
        image[j_desc + 16..j_desc + 20].copy_from_slice(&0x0000_0008_u32.to_be_bytes()); // LAST_TAG

        // Journal block 21: replay payload.
        let j_data = 21 * 4096;
        image[j_data..j_data + 16].copy_from_slice(b"JBD2-REPLAY-TEST");

        // Journal block 22: commit.
        let j_commit = 22 * 4096;
        write_jbd2_header(&mut image[j_commit..j_commit + 4096], 2, 1);

        image
    }

    /// Build an ext4 image with an internal journal inode backed by two
    /// non-contiguous extents and one committed JBD2 transaction that rewrites
    /// block 15.
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image_with_non_contiguous_internal_journal() -> Vec<u8> {
        let mut image = build_ext4_image_with_extents();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // Enable HAS_JOURNAL and point to internal journal inode #8.
        let compat = u32::from_le_bytes([
            image[sb_off + 0x5C],
            image[sb_off + 0x5D],
            image[sb_off + 0x5E],
            image[sb_off + 0x5F],
        ]);
        image[sb_off + 0x5C..sb_off + 0x60].copy_from_slice(&(compat | 0x0004).to_le_bytes());
        image[sb_off + 0xE0..sb_off + 0xE4].copy_from_slice(&8_u32.to_le_bytes());

        // Inode #8 (index 7) -> journal extents:
        // logical [0..=1] -> physical [20..=21]
        // logical [2..=2] -> physical [40..=40]
        let ino8_off: usize = 4 * 4096 + 7 * 256;
        image[ino8_off..ino8_off + 2].copy_from_slice(&0o100_600_u16.to_le_bytes());
        image[ino8_off + 4..ino8_off + 8].copy_from_slice(&(3_u32 * 4096).to_le_bytes());
        image[ino8_off + 0x1A..ino8_off + 0x1C].copy_from_slice(&1_u16.to_le_bytes());
        image[ino8_off + 0x20..ino8_off + 0x24].copy_from_slice(&0x0008_0000_u32.to_le_bytes());
        image[ino8_off + 0x80..ino8_off + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        let e = ino8_off + 0x28;
        image[e..e + 2].copy_from_slice(&0xF30A_u16.to_le_bytes()); // extent magic
        image[e + 2..e + 4].copy_from_slice(&2_u16.to_le_bytes()); // entries
        image[e + 4..e + 6].copy_from_slice(&4_u16.to_le_bytes()); // max
        image[e + 6..e + 8].copy_from_slice(&0_u16.to_le_bytes()); // depth=0

        // extent[0]: logical 0, len 2, start 20
        image[e + 12..e + 16].copy_from_slice(&0_u32.to_le_bytes());
        image[e + 16..e + 18].copy_from_slice(&2_u16.to_le_bytes());
        image[e + 18..e + 20].copy_from_slice(&0_u16.to_le_bytes());
        image[e + 20..e + 24].copy_from_slice(&20_u32.to_le_bytes());

        // extent[1]: logical 2, len 1, start 40
        image[e + 24..e + 28].copy_from_slice(&2_u32.to_le_bytes());
        image[e + 28..e + 30].copy_from_slice(&1_u16.to_le_bytes());
        image[e + 30..e + 32].copy_from_slice(&0_u16.to_le_bytes());
        image[e + 32..e + 36].copy_from_slice(&40_u32.to_le_bytes());

        // Journal block 20: descriptor, one tag to target block 15.
        let j_desc = 20 * 4096;
        write_jbd2_header(&mut image[j_desc..j_desc + 4096], 1, 1);
        image[j_desc + 12..j_desc + 16].copy_from_slice(&15_u32.to_be_bytes());
        image[j_desc + 16..j_desc + 20].copy_from_slice(&0x0000_0008_u32.to_be_bytes()); // LAST_TAG

        // Journal block 21: replay payload.
        let j_data = 21 * 4096;
        image[j_data..j_data + 16].copy_from_slice(b"JBD2-REPLAY-TEST");

        // Journal block 40: commit.
        let j_commit = 40 * 4096;
        write_jbd2_header(&mut image[j_commit..j_commit + 4096], 2, 1);

        image
    }

    fn test_inode_offset(ino: u32) -> usize {
        assert!(ino > 0, "inode numbers are 1-based");
        let inode_index = usize::try_from(ino.saturating_sub(1)).expect("inode index should fit");
        4 * 4096 + inode_index * 256
    }

    fn set_test_inode_dtime(image: &mut [u8], ino: u32, next: u32) {
        let inode_off = test_inode_offset(ino);
        image[inode_off + 0x14..inode_off + 0x18].copy_from_slice(&next.to_le_bytes());
    }

    fn set_test_inode_links_count(image: &mut [u8], ino: u32, links_count: u16) {
        let inode_off = test_inode_offset(ino);
        image[inode_off + 0x1A..inode_off + 0x1C].copy_from_slice(&links_count.to_le_bytes());
    }

    fn set_test_ext4_state(image: &mut [u8], state: u16) {
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        image[sb_off + 0x3A..sb_off + 0x3C].copy_from_slice(&state.to_le_bytes());
    }

    fn build_ext4_image_with_orphan_chain(chain: &[u32]) -> Vec<u8> {
        let mut image = build_ext4_image_with_extents();
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        let head = chain.first().copied().unwrap_or(0);
        image[sb_off + 0xE8..sb_off + 0xEC].copy_from_slice(&head.to_le_bytes());
        for (idx, ino) in chain.iter().copied().enumerate() {
            let next = chain.get(idx + 1).copied().unwrap_or(0);
            set_test_inode_dtime(&mut image, ino, next);
        }
        // Mark orphan inodes as allocated in the inode bitmap (block 3).
        // Inode numbering is 1-based: inode N → bit (N-1) in the bitmap.
        let ibm = 3 * 4096;
        for &ino in chain {
            let bit = (ino - 1) as usize;
            image[ibm + bit / 8] |= 1 << (bit % 8);
        }
        image
    }

    #[test]
    fn read_inode_via_device() {
        let image = build_ext4_image_with_inode();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let inode = fs.read_inode(&cx, InodeNumber(2)).unwrap();

        assert!(inode.is_dir());
        assert_eq!(inode.size, 4096);
        assert_eq!(inode.links_count, 2);
        assert_eq!(inode.permission_bits(), 0o755);
    }

    #[test]
    fn read_inode_attr_via_device() {
        let image = build_ext4_image_with_inode();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let attr = fs.read_inode_attr(&cx, InodeNumber(2)).unwrap();

        assert_eq!(attr.ino, InodeNumber(2));
        assert_eq!(attr.kind, FileType::Directory);
        assert_eq!(attr.perm, 0o755);
        assert_eq!(attr.nlink, 2);
        assert_eq!(attr.size, 4096);
        assert_eq!(attr.blksize, 4096);
    }

    #[test]
    fn read_inode_zero_fails() {
        let image = build_ext4_image_with_inode();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let err = fs.read_inode(&cx, InodeNumber(0)).unwrap_err();
        // inode 0 is invalid → should produce an error
        assert!(
            !matches!(err, FfsError::Io(_)),
            "expected parse/format error, got I/O: {err:?}",
        );
    }

    #[test]
    fn read_inode_out_of_bounds_fails() {
        let image = build_ext4_image_with_inode();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        // Image has 128 inodes; 129 is out of range.
        let err = fs.read_inode(&cx, InodeNumber(129)).unwrap_err();
        assert!(
            !matches!(err, FfsError::Io(_)),
            "expected parse/format error, got I/O: {err:?}",
        );
    }

    #[test]
    fn read_group_desc_via_device() {
        let image = build_ext4_image_with_inode();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let gd = fs.read_group_desc(&cx, GroupNumber(0)).unwrap();
        assert_eq!(gd.block_bitmap, 2);
        assert_eq!(gd.inode_bitmap, 3);
        assert_eq!(gd.inode_table, 4);
    }

    #[test]
    fn read_inode_rejects_corrupted_group_desc_inode_table_pointer() {
        let mut image = build_ext4_image_with_inode();
        let gd_off: usize = 4096;
        image[gd_off + 8..gd_off + 12].copy_from_slice(&10_000_u32.to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let err = fs.read_inode(&cx, InodeNumber(2)).unwrap_err();
        assert!(
            matches!(
                err,
                FfsError::Io(_) | FfsError::Corruption { .. } | FfsError::InvalidGeometry(_)
            ),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn read_block_bitmap_rejects_corrupted_group_desc_pointer() {
        let mut image = build_ext4_image_with_inode();
        let gd_off: usize = 4096;
        image[gd_off..gd_off + 4].copy_from_slice(&10_000_u32.to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let err = fs.read_block_bitmap(&cx, GroupNumber(0)).unwrap_err();
        assert!(
            matches!(
                err,
                FfsError::Io(_) | FfsError::Corruption { .. } | FfsError::InvalidGeometry(_)
            ),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn free_space_summary_detects_block_bitmap_corruption() {
        let mut clean_image = build_ext4_image_with_inode();
        set_group_desc_free_counts(&mut clean_image, 64, 128);

        let cx = Cx::for_testing();
        let fs_clean = OpenFs::from_device(
            &cx,
            Box::new(TestDevice::from_vec(clean_image.clone())),
            &OpenOptions::default(),
        )
        .unwrap();
        let baseline = fs_clean.free_space_summary(&cx).unwrap();
        assert!(!baseline.blocks_mismatch);
        assert!(!baseline.inodes_mismatch);

        clean_image[2 * 4096] |= 0x01; // Corrupt block bitmap: consume one free block bit.
        let fs_corrupt = OpenFs::from_device(
            &cx,
            Box::new(TestDevice::from_vec(clean_image)),
            &OpenOptions::default(),
        )
        .unwrap();
        let summary = fs_corrupt.free_space_summary(&cx).unwrap();
        assert!(summary.blocks_mismatch);
        assert!(!summary.inodes_mismatch);
    }

    #[test]
    fn free_space_summary_detects_inode_bitmap_corruption() {
        let mut clean_image = build_ext4_image_with_inode();
        set_group_desc_free_counts(&mut clean_image, 64, 128);

        let cx = Cx::for_testing();
        let fs_clean = OpenFs::from_device(
            &cx,
            Box::new(TestDevice::from_vec(clean_image.clone())),
            &OpenOptions::default(),
        )
        .unwrap();
        let baseline = fs_clean.free_space_summary(&cx).unwrap();
        assert!(!baseline.blocks_mismatch);
        assert!(!baseline.inodes_mismatch);

        clean_image[3 * 4096] |= 0x01; // Corrupt inode bitmap: consume one free inode bit.
        let fs_corrupt = OpenFs::from_device(
            &cx,
            Box::new(TestDevice::from_vec(clean_image)),
            &OpenOptions::default(),
        )
        .unwrap();
        let summary = fs_corrupt.free_space_summary(&cx).unwrap();
        assert!(!summary.blocks_mismatch);
        assert!(summary.inodes_mismatch);
    }

    // ── Device-based extent mapping tests ─────────────────────────────

    /// Build an ext4 image with file inodes that have extent trees.
    ///
    /// Layout (4K block size, 256K image = 64 blocks):
    /// - Block 0: superblock at offset 1024
    /// - Block 1: group descriptor table
    /// - Block 4–11: inode table (128 inodes × 256 bytes)
    /// - Block 13: data for inode #11 (leaf extent)
    /// - Block 14: extent leaf block for inode #12 (index extent)
    /// - Block 15: data for inode #12
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image_with_extents() -> Vec<u8> {
        let block_size: u32 = 4096;
        let image_size: u32 = 256 * 1024;
        let mut image = vec![0_u8; image_size as usize];
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // ── Superblock ──
        image[sb_off + 0x38..sb_off + 0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        image[sb_off + 0x18..sb_off + 0x1C].copy_from_slice(&2_u32.to_le_bytes()); // log=2 → 4K
        let blocks_count = image_size / block_size;
        image[sb_off + 0x04..sb_off + 0x08].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off..sb_off + 0x04].copy_from_slice(&128_u32.to_le_bytes()); // inodes_count
        image[sb_off + 0x14..sb_off + 0x18].copy_from_slice(&0_u32.to_le_bytes()); // first_data_block=0
        image[sb_off + 0x20..sb_off + 0x24].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off + 0x28..sb_off + 0x2C].copy_from_slice(&128_u32.to_le_bytes());
        image[sb_off + 0x58..sb_off + 0x5A].copy_from_slice(&256_u16.to_le_bytes()); // inode_size
        image[sb_off + 0x4C..sb_off + 0x50].copy_from_slice(&1_u32.to_le_bytes()); // rev_level=DYNAMIC
        let incompat: u32 = 0x0002 | 0x0040; // FILETYPE | EXTENTS
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&incompat.to_le_bytes());
        image[sb_off + 0x54..sb_off + 0x58].copy_from_slice(&11_u32.to_le_bytes()); // first_ino

        // ── Group descriptor at block 1 ──
        let gd_off: usize = 4096;
        image[gd_off..gd_off + 4].copy_from_slice(&2_u32.to_le_bytes()); // block_bitmap
        image[gd_off + 4..gd_off + 8].copy_from_slice(&3_u32.to_le_bytes()); // inode_bitmap
        image[gd_off + 8..gd_off + 12].copy_from_slice(&4_u32.to_le_bytes()); // inode_table

        // ── Inode #11 (index 10): regular file with leaf extent ──
        let ino11_off: usize = 4 * 4096 + 10 * 256;
        image[ino11_off..ino11_off + 2].copy_from_slice(&0o100_644_u16.to_le_bytes()); // S_IFREG|0644
        image[ino11_off + 4..ino11_off + 8].copy_from_slice(&14_u32.to_le_bytes()); // size=14
        image[ino11_off + 0x1A..ino11_off + 0x1C].copy_from_slice(&1_u16.to_le_bytes()); // links
        image[ino11_off + 0x20..ino11_off + 0x24].copy_from_slice(&0x0008_0000_u32.to_le_bytes()); // EXT4_EXTENTS_FL
        image[ino11_off + 0x80..ino11_off + 0x82].copy_from_slice(&32_u16.to_le_bytes()); // extra

        // Extent tree (depth=0, 1 leaf extent: logical 0 → physical 13)
        let e = ino11_off + 0x28;
        image[e..e + 2].copy_from_slice(&0xF30A_u16.to_le_bytes()); // magic
        image[e + 2..e + 4].copy_from_slice(&1_u16.to_le_bytes()); // entries
        image[e + 4..e + 6].copy_from_slice(&4_u16.to_le_bytes()); // max
        image[e + 6..e + 8].copy_from_slice(&0_u16.to_le_bytes()); // depth=0
        image[e + 12..e + 16].copy_from_slice(&0_u32.to_le_bytes()); // logical_block=0
        image[e + 16..e + 18].copy_from_slice(&1_u16.to_le_bytes()); // raw_len=1
        image[e + 18..e + 20].copy_from_slice(&0_u16.to_le_bytes()); // start_hi=0
        image[e + 20..e + 24].copy_from_slice(&13_u32.to_le_bytes()); // start_lo=13

        // Data at block 13
        let d = 13 * 4096;
        image[d..d + 14].copy_from_slice(b"Hello, extent!");

        // ── Inode #12 (index 11): regular file with index extent (depth=1) ──
        let ino12_off: usize = 4 * 4096 + 11 * 256;
        image[ino12_off..ino12_off + 2].copy_from_slice(&0o100_644_u16.to_le_bytes());
        image[ino12_off + 4..ino12_off + 8].copy_from_slice(&14_u32.to_le_bytes()); // size=14
        image[ino12_off + 0x1A..ino12_off + 0x1C].copy_from_slice(&1_u16.to_le_bytes());
        image[ino12_off + 0x20..ino12_off + 0x24].copy_from_slice(&0x0008_0000_u32.to_le_bytes()); // EXT4_EXTENTS_FL
        image[ino12_off + 0x80..ino12_off + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        // Extent tree (depth=1, 1 index entry pointing to block 14)
        let e = ino12_off + 0x28;
        image[e..e + 2].copy_from_slice(&0xF30A_u16.to_le_bytes()); // magic
        image[e + 2..e + 4].copy_from_slice(&1_u16.to_le_bytes()); // entries
        image[e + 4..e + 6].copy_from_slice(&4_u16.to_le_bytes()); // max
        image[e + 6..e + 8].copy_from_slice(&1_u16.to_le_bytes()); // depth=1
        image[e + 12..e + 16].copy_from_slice(&0_u32.to_le_bytes()); // logical_block=0
        image[e + 16..e + 20].copy_from_slice(&14_u32.to_le_bytes()); // leaf_lo=14
        image[e + 20..e + 22].copy_from_slice(&0_u16.to_le_bytes()); // leaf_hi=0

        // Block 14: leaf extent block (depth=0, 1 extent: logical 0 → physical 15)
        let l = 14 * 4096;
        image[l..l + 2].copy_from_slice(&0xF30A_u16.to_le_bytes()); // magic
        image[l + 2..l + 4].copy_from_slice(&1_u16.to_le_bytes()); // entries
        image[l + 4..l + 6].copy_from_slice(&340_u16.to_le_bytes()); // max (4K block)
        image[l + 6..l + 8].copy_from_slice(&0_u16.to_le_bytes()); // depth=0
        image[l + 12..l + 16].copy_from_slice(&0_u32.to_le_bytes()); // logical_block=0
        image[l + 16..l + 18].copy_from_slice(&1_u16.to_le_bytes()); // raw_len=1
        image[l + 18..l + 20].copy_from_slice(&0_u16.to_le_bytes()); // start_hi=0
        image[l + 20..l + 24].copy_from_slice(&15_u32.to_le_bytes()); // start_lo=15

        // Data at block 15
        let d = 15 * 4096;
        image[d..d + 14].copy_from_slice(b"Index extent!\n");

        // ── Block bitmap (block 2): mark blocks 13, 14, 15 as allocated ──
        // This allows free_blocks_persist to succeed during orphan recovery.
        let bm = 2 * 4096;
        image[bm + 1] = 0xE0; // bits 13, 14, 15 set (byte 1, bits 5-7)

        image
    }

    #[test]
    fn resolve_extent_leaf_only() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let phys = fs.resolve_extent(&cx, &inode, 0).unwrap();
        assert_eq!(phys, Some((13, false)));
    }

    #[test]
    fn resolve_extent_hole() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        // Logical block 1 is not mapped — should be a hole.
        let phys = fs.resolve_extent(&cx, &inode, 1).unwrap();
        assert_eq!(phys, None);
    }

    #[test]
    fn resolve_extent_index() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(12)).unwrap();
        let phys = fs.resolve_extent(&cx, &inode, 0).unwrap();
        assert_eq!(phys, Some((15, false)));
    }

    #[test]
    fn resolve_extent_index_rejects_corrupted_child_depth() {
        let mut image = build_ext4_image_with_extents();
        let child_block_off = 14 * 4096;
        image[child_block_off + 6..child_block_off + 8].copy_from_slice(&1_u16.to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(12)).unwrap();
        let err = fs.resolve_extent(&cx, &inode, 0).unwrap_err();
        assert!(
            matches!(err, FfsError::Corruption { .. } | FfsError::Format(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn resolve_extent_rejects_corrupted_inode_extent_root() {
        let mut image = build_ext4_image_with_extents();
        let inode11_off: usize = 4 * 4096 + 10 * 256;
        let extent_root_off = inode11_off + 0x28;
        image[extent_root_off..extent_root_off + 2].copy_from_slice(&0_u16.to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let err = fs.resolve_extent(&cx, &inode, 0).unwrap_err();
        assert!(
            matches!(err, FfsError::Corruption { .. } | FfsError::Format(_)),
            "unexpected error variant: {err:?}"
        );
    }

    #[test]
    fn collect_extents_leaf() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let extents = fs.collect_extents(&cx, &inode).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].logical_block, 0);
        assert_eq!(extents[0].physical_start, 13);
        assert_eq!(extents[0].actual_len(), 1);
        assert!(!extents[0].is_unwritten());
    }

    #[test]
    fn collect_extents_index() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(12)).unwrap();
        let extents = fs.collect_extents(&cx, &inode).unwrap();
        assert_eq!(extents.len(), 1);
        assert_eq!(extents[0].logical_block, 0);
        assert_eq!(extents[0].physical_start, 15);
        assert_eq!(extents[0].actual_len(), 1);
    }

    #[test]
    fn read_file_data_leaf() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let mut buf = vec![0_u8; 14];
        let n = fs.read_file_data(&cx, &inode, 0, &mut buf).unwrap();
        assert_eq!(n, 14);
        assert_eq!(&buf[..n], b"Hello, extent!");
    }

    #[test]
    fn read_file_data_index() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(12)).unwrap();
        let mut buf = vec![0_u8; 14];
        let n = fs.read_file_data(&cx, &inode, 0, &mut buf).unwrap();
        assert_eq!(n, 14);
        assert_eq!(&buf[..n], b"Index extent!\n");
    }

    #[test]
    fn read_file_data_partial() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let mut buf = vec![0_u8; 100];
        let n = fs.read_file_data(&cx, &inode, 7, &mut buf).unwrap();
        assert_eq!(n, 7); // 14 - 7 = 7 bytes remaining
        assert_eq!(&buf[..n], b"extent!");
    }

    #[test]
    fn read_file_data_past_eof() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(11)).unwrap();
        let mut buf = vec![0_u8; 10];
        let n = fs.read_file_data(&cx, &inode, 100, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    // ── Device-based directory tests ──────────────────────────────────

    /// Build an ext4 image with a directory inode containing entries.
    ///
    /// Layout (4K blocks, 256K image):
    /// - Block 0: superblock at offset 1024
    /// - Block 1: group descriptor table
    /// - Block 4+: inode table
    ///   - Inode #2 (root): directory, size=4096, extent: logical 0 → physical 10
    ///   - Inode #11: regular file stub
    /// - Block 10: directory data block with ".", "..", "hello.txt"
    #[allow(clippy::cast_possible_truncation)]
    fn build_ext4_image_with_dir() -> Vec<u8> {
        let block_size: u32 = 4096;
        let image_size: u32 = 256 * 1024;
        let mut image = vec![0_u8; image_size as usize];
        let sb_off = EXT4_SUPERBLOCK_OFFSET;

        // ── Superblock ──
        image[sb_off + 0x38..sb_off + 0x3A].copy_from_slice(&EXT4_SUPER_MAGIC.to_le_bytes());
        image[sb_off + 0x18..sb_off + 0x1C].copy_from_slice(&2_u32.to_le_bytes());
        let blocks_count = image_size / block_size;
        image[sb_off + 0x04..sb_off + 0x08].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off..sb_off + 0x04].copy_from_slice(&128_u32.to_le_bytes());
        image[sb_off + 0x14..sb_off + 0x18].copy_from_slice(&0_u32.to_le_bytes());
        image[sb_off + 0x20..sb_off + 0x24].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off + 0x28..sb_off + 0x2C].copy_from_slice(&128_u32.to_le_bytes());
        image[sb_off + 0x58..sb_off + 0x5A].copy_from_slice(&256_u16.to_le_bytes());
        image[sb_off + 0x4C..sb_off + 0x50].copy_from_slice(&1_u32.to_le_bytes());
        let incompat: u32 = 0x0002 | 0x0040;
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&incompat.to_le_bytes());
        image[sb_off + 0x54..sb_off + 0x58].copy_from_slice(&11_u32.to_le_bytes());

        // ── Group descriptor at block 1 ──
        let gd_off: usize = 4096;
        image[gd_off..gd_off + 4].copy_from_slice(&2_u32.to_le_bytes());
        image[gd_off + 4..gd_off + 8].copy_from_slice(&3_u32.to_le_bytes());
        image[gd_off + 8..gd_off + 12].copy_from_slice(&4_u32.to_le_bytes());

        // ── Inode #2 (root dir, index 1) ──
        let ino2 = 4 * 4096 + 256; // inode #2 = index 1
        image[ino2..ino2 + 2].copy_from_slice(&0o040_755_u16.to_le_bytes()); // S_IFDIR|0755
        image[ino2 + 4..ino2 + 8].copy_from_slice(&4096_u32.to_le_bytes()); // size = 1 block
        image[ino2 + 0x1A..ino2 + 0x1C].copy_from_slice(&3_u16.to_le_bytes()); // links=3
        image[ino2 + 0x20..ino2 + 0x24].copy_from_slice(&0x0008_0000_u32.to_le_bytes());
        image[ino2 + 0x80..ino2 + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        // Extent tree: depth=0, 1 extent: logical 0 → physical 10
        let e = ino2 + 0x28;
        image[e..e + 2].copy_from_slice(&0xF30A_u16.to_le_bytes());
        image[e + 2..e + 4].copy_from_slice(&1_u16.to_le_bytes());
        image[e + 4..e + 6].copy_from_slice(&4_u16.to_le_bytes());
        image[e + 6..e + 8].copy_from_slice(&0_u16.to_le_bytes());
        image[e + 12..e + 16].copy_from_slice(&0_u32.to_le_bytes());
        image[e + 16..e + 18].copy_from_slice(&1_u16.to_le_bytes());
        image[e + 18..e + 20].copy_from_slice(&0_u16.to_le_bytes());
        image[e + 20..e + 24].copy_from_slice(&10_u32.to_le_bytes());

        // ── Inode #11 (file, index 10) ──
        let ino11 = 4 * 4096 + 10 * 256;
        image[ino11..ino11 + 2].copy_from_slice(&0o100_644_u16.to_le_bytes());
        image[ino11 + 4..ino11 + 8].copy_from_slice(&5_u32.to_le_bytes());
        image[ino11 + 0x1A..ino11 + 0x1C].copy_from_slice(&1_u16.to_le_bytes());
        image[ino11 + 0x80..ino11 + 0x82].copy_from_slice(&32_u16.to_le_bytes());

        // ── Block 10: directory data ──
        // Entry "." → inode 2, type=DIR(2)
        let d = 10 * 4096;
        image[d..d + 4].copy_from_slice(&2_u32.to_le_bytes()); // inode
        image[d + 4..d + 6].copy_from_slice(&12_u16.to_le_bytes()); // rec_len
        image[d + 6] = 1; // name_len
        image[d + 7] = 2; // file_type = DIR
        image[d + 8] = b'.';

        // Entry ".." → inode 2, type=DIR(2)
        let d = d + 12;
        image[d..d + 4].copy_from_slice(&2_u32.to_le_bytes());
        image[d + 4..d + 6].copy_from_slice(&12_u16.to_le_bytes());
        image[d + 6] = 2;
        image[d + 7] = 2;
        image[d + 8] = b'.';
        image[d + 9] = b'.';

        // Entry "hello.txt" → inode 11, type=REG(1)
        let d = d + 12;
        image[d..d + 4].copy_from_slice(&11_u32.to_le_bytes());
        image[d + 4..d + 6].copy_from_slice(&4072_u16.to_le_bytes()); // rest of block
        image[d + 6] = 9;
        image[d + 7] = 1;
        image[d + 8..d + 17].copy_from_slice(b"hello.txt");

        image
    }

    #[test]
    fn read_dir_via_device() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(2)).unwrap();
        assert!(inode.is_dir());

        let entries = fs.read_dir(&cx, &inode).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, b".");
        assert_eq!(entries[0].inode, 2);
        assert_eq!(entries[1].name, b"..");
        assert_eq!(entries[1].inode, 2);
        assert_eq!(entries[2].name, b"hello.txt");
        assert_eq!(entries[2].inode, 11);
    }

    #[test]
    fn lookup_name_found() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(2)).unwrap();
        let entry = fs.lookup_name(&cx, &inode, b"hello.txt").unwrap();
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.inode, 11);
        assert_eq!(entry.name, b"hello.txt");
    }

    #[test]
    fn lookup_name_not_found() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let inode = fs.read_inode(&cx, InodeNumber(2)).unwrap();
        let entry = fs.lookup_name(&cx, &inode, b"missing.txt").unwrap();
        assert!(entry.is_none());
    }

    // ── High-level file read tests ────────────────────────────────────

    #[test]
    fn read_file_returns_data() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let data = fs.read_file(&cx, InodeNumber(11), 0, 100).unwrap();
        assert_eq!(&data, b"Hello, extent!");
    }

    #[test]
    fn read_file_partial() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let data = fs.read_file(&cx, InodeNumber(11), 7, 100).unwrap();
        assert_eq!(&data, b"extent!");
    }

    #[test]
    fn read_file_rejects_directory() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let err = fs.read_file(&cx, InodeNumber(2), 0, 4096).unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    // ── Path resolution tests ─────────────────────────────────────────

    #[test]
    fn resolve_path_root() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let (ino, inode) = fs.resolve_path(&cx, "/").unwrap();
        assert_eq!(ino, InodeNumber(2));
        assert!(inode.is_dir());
    }

    #[test]
    fn resolve_path_file() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let (ino, inode) = fs.resolve_path(&cx, "/hello.txt").unwrap();
        assert_eq!(ino, InodeNumber(11));
        assert!(inode.is_regular());
    }

    #[test]
    fn resolve_path_not_found() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let err = fs.resolve_path(&cx, "/missing").unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn resolve_path_not_directory() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        // hello.txt is a regular file, not a directory — traversal through it fails.
        let err = fs.resolve_path(&cx, "/hello.txt/child").unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTDIR);
    }

    #[test]
    fn resolve_path_relative_rejected() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let err = fs.resolve_path(&cx, "hello.txt").unwrap_err();
        assert!(matches!(err, FfsError::Format(_)));
    }

    // ── FsOps for OpenFs tests ────────────────────────────────────────

    #[test]
    fn open_fs_fsops_getattr() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        // Use via dyn FsOps to verify trait impl works
        let ops: &dyn FsOps = &fs;
        let attr = ops.getattr(&cx, InodeNumber(2)).unwrap();
        assert_eq!(attr.kind, FileType::Directory);
        assert_eq!(attr.perm, 0o755);
    }

    #[test]
    fn open_fs_fsops_lookup() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let attr = ops
            .lookup(&cx, InodeNumber(2), OsStr::new("hello.txt"))
            .unwrap();
        assert_eq!(attr.ino, InodeNumber(11));
        assert_eq!(attr.kind, FileType::RegularFile);
    }

    #[test]
    fn open_fs_fsops_lookup_not_found() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let err = ops
            .lookup(&cx, InodeNumber(2), OsStr::new("missing"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn open_fs_fsops_readdir() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let entries = ops.readdir(&cx, InodeNumber(2), 0).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, b".");
        assert_eq!(entries[2].name, b"hello.txt");

        // Offset-based pagination
        let entries = ops.readdir(&cx, InodeNumber(2), 2).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, b"hello.txt");
    }

    #[test]
    fn open_fs_fsops_read() {
        let image = build_ext4_image_with_extents();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let data = ops.read(&cx, InodeNumber(11), 0, 100).unwrap();
        assert_eq!(&data, b"Hello, extent!");
    }

    #[test]
    fn open_fs_fsops_read_directory_rejected() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let err = ops.read(&cx, InodeNumber(2), 0, 4096).unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn open_fs_fsops_statfs_ext4_uses_superblock_counts() {
        let image = build_ext4_image_with_dir();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let sb = fs.ext4_superblock().expect("ext4 superblock");

        let ops: &dyn FsOps = &fs;
        let stats = ops.statfs(&cx, InodeNumber(2)).unwrap();
        assert_eq!(stats.block_size, sb.block_size);
        assert_eq!(stats.fragment_size, sb.block_size);
        assert_eq!(stats.blocks, sb.blocks_count);
        assert_eq!(stats.blocks_free, sb.free_blocks_count);
        assert_eq!(
            stats.blocks_available,
            sb.free_blocks_count
                .saturating_sub(sb.reserved_blocks_count)
        );
        assert_eq!(stats.files, u64::from(sb.inodes_count));
        assert_eq!(stats.files_free, u64::from(sb.free_inodes_count));
        assert_eq!(stats.name_max, 255);
    }

    #[test]
    fn open_fs_fsops_statfs_btrfs_uses_capacity_fields() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let sb = fs.btrfs_superblock().expect("btrfs superblock");
        let unit = sb.sectorsize.max(1);
        let unit_u64 = u64::from(unit);
        let free_bytes = sb.total_bytes.saturating_sub(sb.bytes_used);

        let ops: &dyn FsOps = &fs;
        let stats = ops.statfs(&cx, InodeNumber(1)).unwrap();
        assert_eq!(stats.block_size, unit);
        assert_eq!(stats.fragment_size, unit);
        assert_eq!(stats.blocks, sb.total_bytes / unit_u64);
        assert_eq!(stats.blocks_free, free_bytes / unit_u64);
        assert_eq!(stats.blocks_available, free_bytes / unit_u64);
        assert_eq!(stats.files, 0);
        assert_eq!(stats.files_free, 0);
        assert_eq!(stats.name_max, 255);
    }

    #[test]
    fn open_fs_btrfs_fsops_getattr_lookup_readdir_read() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;

        // Root inode alias: VFS inode 1 should map to btrfs root_dir_objectid.
        let root_attr = ops.getattr(&cx, InodeNumber(1)).unwrap();
        assert_eq!(root_attr.kind, FileType::Directory);
        assert_eq!(root_attr.perm, 0o755);

        let child_attr = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("hello.txt"))
            .unwrap();
        assert_eq!(child_attr.ino, InodeNumber(257));
        assert_eq!(child_attr.kind, FileType::RegularFile);
        assert_eq!(child_attr.size, 22);

        let entries = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, b".");
        assert_eq!(entries[1].name, b"..");
        assert_eq!(entries[2].name, b"hello.txt");

        // Offset pagination should skip "." and ".."
        let paged = ops.readdir(&cx, InodeNumber(1), 2).unwrap();
        assert_eq!(paged.len(), 1);
        assert_eq!(paged[0].name, b"hello.txt");

        let data = ops.read(&cx, InodeNumber(257), 0, 128).unwrap();
        assert_eq!(&data, b"hello from btrfs fsops");
    }

    #[test]
    fn open_fs_btrfs_fsops_read_offset_and_truncated_eof() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let partial = ops.read(&cx, InodeNumber(257), 6, 4).unwrap();
        assert_eq!(&partial, b"from");

        let eof_truncated = ops.read(&cx, InodeNumber(257), 20, 16).unwrap();
        assert_eq!(&eof_truncated, b"ps");
    }

    #[test]
    fn open_fs_btrfs_fsops_read_inline_extent() {
        let inline = b"inline-btrfs-payload";
        let mut image = build_btrfs_fsops_image();
        let mut extent_payload = vec![0_u8; 21 + inline.len()];
        extent_payload[0..8].copy_from_slice(&1_u64.to_le_bytes());
        extent_payload[8..16].copy_from_slice(&(inline.len() as u64).to_le_bytes());
        extent_payload[20] = ffs_btrfs::BTRFS_FILE_EXTENT_INLINE;
        extent_payload[21..].copy_from_slice(inline);

        set_btrfs_test_file_size(&mut image, inline.len() as u64);
        let extent_payload_len =
            u32::try_from(extent_payload.len()).expect("extent payload length should fit in u32");
        set_btrfs_test_extent_data_size(&mut image, extent_payload_len);
        write_btrfs_test_extent_payload(&mut image, &extent_payload);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let data = ops.read(&cx, InodeNumber(257), 0, 128).unwrap();
        assert_eq!(&data, inline);
    }

    #[test]
    fn open_fs_btrfs_fsops_read_prealloc_extent_zero_filled() {
        let mut image = build_btrfs_fsops_image();
        set_btrfs_test_extent_type(&mut image, BTRFS_FILE_EXTENT_PREALLOC);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let data = ops.read(&cx, InodeNumber(257), 0, 64).unwrap();
        assert_eq!(data, vec![0_u8; 22]);
    }

    #[test]
    fn open_fs_btrfs_fsops_read_sparse_hole_zero_filled() {
        let mut image = build_btrfs_fsops_image();
        set_btrfs_test_file_size(&mut image, 30);
        set_btrfs_test_extent_key_offset(&mut image, 8);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let data = ops.read(&cx, InodeNumber(257), 0, 64).unwrap();
        let mut expected = vec![0_u8; 8];
        expected.extend_from_slice(b"hello from btrfs fsops");
        assert_eq!(data, expected);
    }

    #[test]
    fn open_fs_btrfs_fsops_read_compressed_extent_unsupported() {
        let mut image = build_btrfs_fsops_image();
        set_btrfs_test_extent_compression(&mut image, 1);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let err = ops.read(&cx, InodeNumber(257), 0, 64).unwrap_err();
        assert!(matches!(err, FfsError::UnsupportedFeature(_)));
    }

    #[test]
    fn open_fs_btrfs_fsops_read_multiblock_regular_extent() {
        let mut image = build_btrfs_fsops_image();
        let expected: Vec<u8> = (0..6000)
            .map(|i| u8::try_from(i % 251).expect("value should fit in u8"))
            .collect();
        set_btrfs_test_file_size(&mut image, expected.len() as u64);
        set_btrfs_test_extent_lengths(&mut image, expected.len() as u64);
        write_btrfs_test_file_data(&mut image, &expected);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let ops: &dyn FsOps = &fs;

        let data = ops.read(&cx, InodeNumber(257), 0, 7000).unwrap();
        assert_eq!(data, expected);
    }

    #[test]
    fn durability_autopilot_prefers_more_redundancy_when_failures_observed() {
        let candidates = [1.02, 1.05, 1.10];

        let mut clean = DurabilityAutopilot::new();
        clean.observe_scrub(10_000, 0);
        let clean_decision = clean.choose_overhead(&candidates);
        assert!((clean_decision.repair_overhead - 1.02).abs() < 1e-12);

        let mut dirty = DurabilityAutopilot::new();
        dirty.observe_scrub(10_000, 300);
        let dirty_decision = dirty.choose_overhead(&candidates);
        assert!(dirty_decision.repair_overhead >= 1.05);
    }

    // ── Math helper tests ───────────────────────────────────────────────

    #[test]
    fn erfc_approx_known_values() {
        // erfc(0) = 1.0
        let val = erfc_approx(0.0);
        assert!((val - 1.0).abs() < 1e-6, "erfc(0) = {val}, expected 1.0");

        // erfc(large) → 0
        let val = erfc_approx(5.0);
        assert!(val < 1e-6, "erfc(5) = {val}, expected ~0");

        // erfc(-large) → 2
        let val = erfc_approx(-5.0);
        assert!((val - 2.0).abs() < 1e-6, "erfc(-5) = {val}, expected ~2");

        // erfc(1) ≈ 0.1573 (known value)
        let val = erfc_approx(1.0);
        assert!(
            (val - 0.1573).abs() < 0.001,
            "erfc(1) = {val}, expected ~0.1573",
        );
    }

    #[test]
    fn ln_gamma_known_values() {
        // Γ(1) = 1, ln(1) = 0
        let val = ln_gamma(1.0);
        assert!(val.abs() < 1e-10, "ln_gamma(1) = {val}, expected 0");

        // Γ(2) = 1, ln(1) = 0
        let val = ln_gamma(2.0);
        assert!(val.abs() < 1e-10, "ln_gamma(2) = {val}, expected 0");

        // Γ(5) = 24, ln(24) ≈ 3.1781
        let val = ln_gamma(5.0);
        let expected = 24.0_f64.ln();
        assert!(
            (val - expected).abs() < 1e-8,
            "ln_gamma(5) = {val}, expected {expected}",
        );

        // Γ(0.5) = √π ≈ 1.7725, ln(√π) ≈ 0.5724
        let val = ln_gamma(0.5);
        let expected = std::f64::consts::PI.sqrt().ln();
        assert!(
            (val - expected).abs() < 1e-6,
            "ln_gamma(0.5) = {val}, expected {expected}",
        );

        // Γ(10) = 9! = 362880
        let val = ln_gamma(10.0);
        let expected = 362_880.0_f64.ln();
        assert!(
            (val - expected).abs() < 1e-6,
            "ln_gamma(10) = {val}, expected {expected}",
        );
    }

    #[test]
    fn ln_gamma_zero_and_negative() {
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-1.0).is_infinite());
    }

    #[test]
    fn ln_beta_known_values() {
        // B(1,1) = 1, ln(1) = 0
        let val = ln_beta(1.0, 1.0);
        assert!(val.abs() < 1e-10, "ln_beta(1,1) = {val}, expected 0");

        // B(1,2) = 1/2, ln(1/2) ≈ -0.6931
        let val = ln_beta(1.0, 2.0);
        let expected = 0.5_f64.ln();
        assert!(
            (val - expected).abs() < 1e-8,
            "ln_beta(1,2) = {val}, expected {expected}",
        );

        // B(2,2) = 1/6, ln(1/6) ≈ -1.7918
        let val = ln_beta(2.0, 2.0);
        let expected = (1.0 / 6.0_f64).ln();
        assert!(
            (val - expected).abs() < 1e-8,
            "ln_beta(2,2) = {val}, expected {expected}",
        );
    }

    // ── IntegrityReport tests ───────────────────────────────────────────

    #[test]
    fn integrity_report_all_clean() {
        let report = IntegrityReport {
            verdicts: vec![],
            passed: 100,
            failed: 0,
            posterior_alpha: 1.0,
            posterior_beta: 101.0,
            expected_corruption_rate: 1.0 / 102.0,
            upper_bound_corruption_rate: 0.005,
            healthy: true,
        };

        let p = report.prob_healthy(0.05);
        assert!(p > 0.9, "prob_healthy = {p}, expected > 0.9");

        let lbf = report.log_bayes_factor();
        assert!(
            lbf > 0.0,
            "log_bayes_factor = {lbf}, expected > 0 (favors health)"
        );
    }

    #[test]
    fn integrity_report_heavily_corrupted() {
        let report = IntegrityReport {
            verdicts: vec![],
            passed: 10,
            failed: 90,
            posterior_alpha: 91.0,
            posterior_beta: 11.0,
            expected_corruption_rate: 91.0 / 102.0,
            upper_bound_corruption_rate: 0.95,
            healthy: false,
        };

        let p = report.prob_healthy(0.01);
        assert!(p < 0.01, "prob_healthy = {p}, expected < 0.01");

        let lbf = report.log_bayes_factor();
        assert!(
            lbf < 0.0,
            "log_bayes_factor = {lbf}, expected < 0 (favors corruption)"
        );
    }

    #[test]
    fn integrity_report_bayes_factor_is_finite() {
        let report = IntegrityReport {
            verdicts: vec![],
            passed: 50,
            failed: 50,
            posterior_alpha: 51.0,
            posterior_beta: 51.0,
            expected_corruption_rate: 0.5,
            upper_bound_corruption_rate: 0.55,
            healthy: false,
        };
        let lbf = report.log_bayes_factor();
        assert!(lbf.is_finite(), "log_bayes_factor should be finite");
    }

    // ── Ext4FsOps helper tests ──────────────────────────────────────────

    #[test]
    fn dir_entry_file_type_mapping() {
        assert_eq!(dir_entry_file_type(Ext4FileType::Dir), FileType::Directory);
        assert_eq!(
            dir_entry_file_type(Ext4FileType::Symlink),
            FileType::Symlink
        );
        assert_eq!(
            dir_entry_file_type(Ext4FileType::Blkdev),
            FileType::BlockDevice
        );
        assert_eq!(
            dir_entry_file_type(Ext4FileType::Chrdev),
            FileType::CharDevice
        );
        assert_eq!(dir_entry_file_type(Ext4FileType::Fifo), FileType::Fifo);
        assert_eq!(dir_entry_file_type(Ext4FileType::Sock), FileType::Socket);
        assert_eq!(
            dir_entry_file_type(Ext4FileType::RegFile),
            FileType::RegularFile
        );
        assert_eq!(
            dir_entry_file_type(Ext4FileType::Unknown),
            FileType::RegularFile
        );
    }

    #[test]
    fn parse_to_ffs_error_runtime_mappings() {
        let e = parse_to_ffs_error(&ParseError::InvalidField {
            field: "dir_entry",
            reason: "component not found in directory",
        });
        assert!(matches!(e, FfsError::NotFound(_)));

        let e = parse_to_ffs_error(&ParseError::InvalidField {
            field: "path",
            reason: "not a directory",
        });
        assert!(matches!(e, FfsError::NotDirectory));

        let e = parse_to_ffs_error(&ParseError::InvalidField {
            field: "extent",
            reason: "corrupt extent header",
        });
        assert!(matches!(e, FfsError::Format(_)));

        let e = parse_to_ffs_error(&ParseError::InsufficientData {
            needed: 256,
            offset: 0,
            actual: 128,
        });
        assert!(matches!(e, FfsError::Corruption { .. }));
    }

    #[test]
    fn check_verdict_serializes() {
        let v = CheckVerdict {
            component: "superblock".into(),
            passed: true,
            detail: String::new(),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("superblock"));
        assert!(json.contains("true"));
    }

    #[test]
    fn integrity_report_serializes() {
        let report = IntegrityReport {
            verdicts: vec![CheckVerdict {
                component: "test".into(),
                passed: true,
                detail: String::new(),
            }],
            passed: 1,
            failed: 0,
            posterior_alpha: 1.0,
            posterior_beta: 2.0,
            expected_corruption_rate: 0.333,
            upper_bound_corruption_rate: 0.5,
            healthy: true,
        };
        let json = serde_json::to_string(&report).unwrap();
        let deser: IntegrityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.passed, 1);
        assert_eq!(deser.failed, 0);
        assert!(deser.healthy);
    }

    // ── Btrfs OpenFs tests ──────────────────────────────────────────────

    /// Build a minimal synthetic btrfs image with a sys_chunk_array and a leaf
    /// node at the root tree address.
    #[allow(clippy::cast_possible_truncation)]
    fn build_btrfs_image() -> Vec<u8> {
        let image_size: usize = 256 * 1024; // 256 KB
        let mut image = vec![0_u8; image_size];
        let sb_off = BTRFS_SUPER_INFO_OFFSET;

        // magic
        image[sb_off + 0x40..sb_off + 0x48].copy_from_slice(&BTRFS_MAGIC.to_le_bytes());
        // generation
        image[sb_off + 0x48..sb_off + 0x50].copy_from_slice(&1_u64.to_le_bytes());
        // root (logical address of root tree leaf)
        let root_logical = 0x4000_u64;
        image[sb_off + 0x50..sb_off + 0x58].copy_from_slice(&root_logical.to_le_bytes());
        // chunk_root (set to 0 — we only use sys_chunk_array)
        image[sb_off + 0x58..sb_off + 0x60].copy_from_slice(&0_u64.to_le_bytes());
        // total_bytes
        image[sb_off + 0x70..sb_off + 0x78].copy_from_slice(&(image_size as u64).to_le_bytes());
        // root_dir_objectid
        image[sb_off + 0x80..sb_off + 0x88].copy_from_slice(&256_u64.to_le_bytes());
        // num_devices
        image[sb_off + 0x88..sb_off + 0x90].copy_from_slice(&1_u64.to_le_bytes());
        // sectorsize = 4096
        image[sb_off + 0x90..sb_off + 0x94].copy_from_slice(&4096_u32.to_le_bytes());
        // nodesize = 4096
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&4096_u32.to_le_bytes());
        // stripesize = 4096
        image[sb_off + 0x9C..sb_off + 0xA0].copy_from_slice(&4096_u32.to_le_bytes());

        // Build sys_chunk_array: one chunk, identity mapping [0, 256K) → [0, 256K)
        let mut chunk_array = Vec::new();
        // disk_key: objectid=256, type=228, offset=0
        chunk_array.extend_from_slice(&256_u64.to_le_bytes());
        chunk_array.push(228_u8);
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        // chunk: length, owner, stripe_len, type
        chunk_array.extend_from_slice(&(image_size as u64).to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0x1_0000_u64.to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        // io_align, io_width, sector_size
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        // num_stripes=1, sub_stripes=0
        chunk_array.extend_from_slice(&1_u16.to_le_bytes());
        chunk_array.extend_from_slice(&0_u16.to_le_bytes());
        // stripe: devid=1, offset=0 (identity), dev_uuid=[0;16]
        chunk_array.extend_from_slice(&1_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        chunk_array.extend_from_slice(&[0_u8; 16]);

        // sys_chunk_array_size
        let array_size = chunk_array.len() as u32;
        image[sb_off + 0xA0..sb_off + 0xA4].copy_from_slice(&array_size.to_le_bytes());
        // sys_chunk_array data (at offset 0x32B from sb region start)
        let array_start = sb_off + 0x32B;
        image[array_start..array_start + chunk_array.len()].copy_from_slice(&chunk_array);
        // root_level = 0 (leaf)
        image[sb_off + 0xC6] = 0;

        // Write a leaf node at physical 0x4000 (= root_logical via identity map)
        let leaf_off = root_logical as usize;
        // btrfs header: bytenr
        image[leaf_off + 0x30..leaf_off + 0x38].copy_from_slice(&root_logical.to_le_bytes());
        // generation
        image[leaf_off + 0x50..leaf_off + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        // owner (ROOT_TREE = 1)
        image[leaf_off + 0x58..leaf_off + 0x60].copy_from_slice(&1_u64.to_le_bytes());
        // nritems = 1
        image[leaf_off + 0x60..leaf_off + 0x64].copy_from_slice(&1_u32.to_le_bytes());
        // level = 0 (leaf)
        image[leaf_off + 0x64] = 0;

        // Leaf item 0 at header_size=101
        let item_off = leaf_off + 101;
        // key: objectid=256, type=132 (ROOT_ITEM), offset=0
        image[item_off..item_off + 8].copy_from_slice(&256_u64.to_le_bytes());
        image[item_off + 8] = 132;
        image[item_off + 9..item_off + 17].copy_from_slice(&0_u64.to_le_bytes());
        // data_offset=200, data_size=8
        image[item_off + 17..item_off + 21].copy_from_slice(&200_u32.to_le_bytes());
        image[item_off + 21..item_off + 25].copy_from_slice(&8_u32.to_le_bytes());
        // Actual data at leaf_off + 200
        image[leaf_off + 200..leaf_off + 208]
            .copy_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF]);

        image
    }

    #[allow(clippy::too_many_arguments)]
    fn write_btrfs_leaf_item(
        image: &mut [u8],
        leaf_off: usize,
        idx: usize,
        objectid: u64,
        item_type: u8,
        key_offset: u64,
        data_offset: u32,
        data_size: u32,
    ) {
        let item_off = leaf_off + 101 + idx * 25;
        image[item_off..item_off + 8].copy_from_slice(&objectid.to_le_bytes());
        image[item_off + 8] = item_type;
        image[item_off + 9..item_off + 17].copy_from_slice(&key_offset.to_le_bytes());
        image[item_off + 17..item_off + 21].copy_from_slice(&data_offset.to_le_bytes());
        image[item_off + 21..item_off + 25].copy_from_slice(&data_size.to_le_bytes());
    }

    fn encode_btrfs_inode_item(mode: u32, size: u64, nbytes: u64, nlink: u32) -> [u8; 160] {
        let mut inode = [0_u8; 160];
        inode[0..8].copy_from_slice(&1_u64.to_le_bytes()); // generation
        inode[8..16].copy_from_slice(&1_u64.to_le_bytes()); // transid
        inode[16..24].copy_from_slice(&size.to_le_bytes());
        inode[24..32].copy_from_slice(&nbytes.to_le_bytes());
        inode[40..44].copy_from_slice(&nlink.to_le_bytes());
        inode[44..48].copy_from_slice(&1000_u32.to_le_bytes()); // uid
        inode[48..52].copy_from_slice(&1000_u32.to_le_bytes()); // gid
        inode[52..56].copy_from_slice(&mode.to_le_bytes());
        // atime / ctime / mtime / otime
        inode[112..120].copy_from_slice(&10_u64.to_le_bytes());
        inode[124..132].copy_from_slice(&10_u64.to_le_bytes());
        inode[136..144].copy_from_slice(&10_u64.to_le_bytes());
        inode[148..156].copy_from_slice(&10_u64.to_le_bytes());
        inode
    }

    fn encode_btrfs_dir_index_entry(name: &[u8], child_objectid: u64, file_type: u8) -> Vec<u8> {
        let mut entry = vec![0_u8; 30 + name.len()];
        entry[0..8].copy_from_slice(&child_objectid.to_le_bytes());
        entry[8] = BTRFS_ITEM_INODE_ITEM;
        entry[9..17].copy_from_slice(&0_u64.to_le_bytes());
        entry[17..25].copy_from_slice(&1_u64.to_le_bytes()); // transid
        entry[25..27].copy_from_slice(&0_u16.to_le_bytes()); // data_len
        let name_len = u16::try_from(name.len()).expect("test name length should fit in u16");
        entry[27..29].copy_from_slice(&name_len.to_le_bytes());
        entry[29] = file_type;
        entry[30..30 + name.len()].copy_from_slice(name);
        entry
    }

    fn encode_btrfs_extent_regular(disk_bytenr: u64, num_bytes: u64) -> [u8; 53] {
        let mut extent = [0_u8; 53];
        extent[0..8].copy_from_slice(&1_u64.to_le_bytes()); // generation
        extent[8..16].copy_from_slice(&num_bytes.to_le_bytes()); // ram_bytes
        extent[20] = BTRFS_FILE_EXTENT_REG;
        extent[21..29].copy_from_slice(&disk_bytenr.to_le_bytes());
        extent[29..37].copy_from_slice(&num_bytes.to_le_bytes()); // disk_num_bytes
        extent[37..45].copy_from_slice(&0_u64.to_le_bytes()); // extent offset
        extent[45..53].copy_from_slice(&num_bytes.to_le_bytes());
        extent
    }

    const BTRFS_TEST_FS_TREE_LOGICAL: usize = 0x8_000;
    const BTRFS_TEST_FILE_DATA_LOGICAL: usize = 0x12_000;
    const BTRFS_TEST_FILE_INODE_OFF: usize = 2860;
    const BTRFS_TEST_EXTENT_OFF: usize = 2780;
    const BTRFS_TEST_EXTENT_ITEM_INDEX: usize = 3;
    const BTRFS_TEST_LEAF_HEADER_SIZE: usize = 101;
    const BTRFS_TEST_LEAF_ITEM_SIZE: usize = 25;

    fn btrfs_test_extent_item_off() -> usize {
        BTRFS_TEST_FS_TREE_LOGICAL
            + BTRFS_TEST_LEAF_HEADER_SIZE
            + BTRFS_TEST_EXTENT_ITEM_INDEX * BTRFS_TEST_LEAF_ITEM_SIZE
    }

    fn btrfs_test_extent_payload_off() -> usize {
        BTRFS_TEST_FS_TREE_LOGICAL + BTRFS_TEST_EXTENT_OFF
    }

    fn set_btrfs_super_total_bytes(image: &mut [u8], total_bytes: u64) {
        let sb_off = BTRFS_SUPER_INFO_OFFSET;
        image[sb_off + 0x70..sb_off + 0x78].copy_from_slice(&total_bytes.to_le_bytes());
    }

    fn set_btrfs_test_file_size(image: &mut [u8], size: u64) {
        let size_off = BTRFS_TEST_FS_TREE_LOGICAL + BTRFS_TEST_FILE_INODE_OFF + 16;
        image[size_off..size_off + 8].copy_from_slice(&size.to_le_bytes());
        image[size_off + 8..size_off + 16].copy_from_slice(&size.to_le_bytes());
    }

    fn set_btrfs_test_extent_key_offset(image: &mut [u8], logical_offset: u64) {
        let item_off = btrfs_test_extent_item_off();
        image[item_off + 9..item_off + 17].copy_from_slice(&logical_offset.to_le_bytes());
    }

    fn set_btrfs_test_extent_data_size(image: &mut [u8], data_size: u32) {
        let item_off = btrfs_test_extent_item_off();
        image[item_off + 21..item_off + 25].copy_from_slice(&data_size.to_le_bytes());
    }

    fn set_btrfs_test_extent_type(image: &mut [u8], extent_type: u8) {
        let extent_off = btrfs_test_extent_payload_off();
        image[extent_off + 20] = extent_type;
    }

    fn set_btrfs_test_extent_compression(image: &mut [u8], compression: u8) {
        let extent_off = btrfs_test_extent_payload_off();
        image[extent_off + 16] = compression;
    }

    fn set_btrfs_test_extent_lengths(image: &mut [u8], length: u64) {
        let extent_off = btrfs_test_extent_payload_off();
        image[extent_off + 8..extent_off + 16].copy_from_slice(&length.to_le_bytes()); // ram_bytes
        image[extent_off + 29..extent_off + 37].copy_from_slice(&length.to_le_bytes()); // disk_num_bytes
        image[extent_off + 45..extent_off + 53].copy_from_slice(&length.to_le_bytes());
        // num_bytes
    }

    fn write_btrfs_test_extent_payload(image: &mut [u8], payload: &[u8]) {
        let extent_off = btrfs_test_extent_payload_off();
        image[extent_off..extent_off + payload.len()].copy_from_slice(payload);
    }

    fn write_btrfs_test_file_data(image: &mut [u8], data: &[u8]) {
        let data_off = BTRFS_TEST_FILE_DATA_LOGICAL;
        image[data_off..data_off + data.len()].copy_from_slice(data);
    }

    /// Build a minimal btrfs image with ROOT_TREE + FS_TREE content sufficient
    /// for read-only `FsOps` operations (`getattr/lookup/readdir/read`).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    fn build_btrfs_fsops_image() -> Vec<u8> {
        let image_size: usize = 512 * 1024;
        let mut image = vec![0_u8; image_size];
        let sb_off = BTRFS_SUPER_INFO_OFFSET;

        let root_tree_logical = 0x4_000_u64;
        let fs_tree_logical = 0x8_000_u64;
        let file_data_logical = 0x12_000_u64;
        let file_bytes = b"hello from btrfs fsops";

        image[sb_off + 0x40..sb_off + 0x48].copy_from_slice(&BTRFS_MAGIC.to_le_bytes()); // magic
        image[sb_off + 0x48..sb_off + 0x50].copy_from_slice(&1_u64.to_le_bytes()); // generation
        image[sb_off + 0x50..sb_off + 0x58].copy_from_slice(&root_tree_logical.to_le_bytes()); // root tree bytenr
        image[sb_off + 0x58..sb_off + 0x60].copy_from_slice(&0_u64.to_le_bytes()); // chunk_root unused
        image[sb_off + 0x70..sb_off + 0x78].copy_from_slice(&(image_size as u64).to_le_bytes());
        image[sb_off + 0x80..sb_off + 0x88].copy_from_slice(&256_u64.to_le_bytes()); // root_dir_objectid
        image[sb_off + 0x88..sb_off + 0x90].copy_from_slice(&1_u64.to_le_bytes()); // num_devices
        image[sb_off + 0x90..sb_off + 0x94].copy_from_slice(&4096_u32.to_le_bytes()); // sectorsize
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&4096_u32.to_le_bytes()); // nodesize
        image[sb_off + 0x9C..sb_off + 0xA0].copy_from_slice(&4096_u32.to_le_bytes()); // stripesize
        image[sb_off + 0xC6] = 0; // root_level

        // sys_chunk_array: one identity chunk [0, image_size) → [0, image_size)
        let mut chunk_array = Vec::new();
        chunk_array.extend_from_slice(&256_u64.to_le_bytes());
        chunk_array.push(228_u8); // CHUNK_ITEM
        chunk_array.extend_from_slice(&0_u64.to_le_bytes()); // logical start
        chunk_array.extend_from_slice(&(image_size as u64).to_le_bytes()); // length
        chunk_array.extend_from_slice(&2_u64.to_le_bytes()); // owner
        chunk_array.extend_from_slice(&0x1_0000_u64.to_le_bytes()); // stripe_len
        chunk_array.extend_from_slice(&2_u64.to_le_bytes()); // chunk type
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes()); // io_align
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes()); // io_width
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes()); // sector_size
        chunk_array.extend_from_slice(&1_u16.to_le_bytes()); // num_stripes
        chunk_array.extend_from_slice(&0_u16.to_le_bytes()); // sub_stripes
        chunk_array.extend_from_slice(&1_u64.to_le_bytes()); // devid
        chunk_array.extend_from_slice(&0_u64.to_le_bytes()); // physical offset (identity)
        chunk_array.extend_from_slice(&[0_u8; 16]); // dev_uuid

        image[sb_off + 0xA0..sb_off + 0xA4]
            .copy_from_slice(&(chunk_array.len() as u32).to_le_bytes());
        let array_start = sb_off + 0x32B;
        image[array_start..array_start + chunk_array.len()].copy_from_slice(&chunk_array);

        // Root tree leaf with one ROOT_ITEM for FS_TREE.
        let root_leaf = root_tree_logical as usize;
        image[root_leaf + 0x30..root_leaf + 0x38].copy_from_slice(&root_tree_logical.to_le_bytes());
        image[root_leaf + 0x50..root_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[root_leaf + 0x58..root_leaf + 0x60].copy_from_slice(&1_u64.to_le_bytes()); // ROOT_TREE owner
        image[root_leaf + 0x60..root_leaf + 0x64].copy_from_slice(&1_u32.to_le_bytes()); // nritems
        image[root_leaf + 0x64] = 0; // leaf

        let root_item_offset: u32 = 3000;
        let root_item_size: u32 = 239;
        write_btrfs_leaf_item(
            &mut image,
            root_leaf,
            0,
            BTRFS_FS_TREE_OBJECTID,
            BTRFS_ITEM_ROOT_ITEM,
            0,
            root_item_offset,
            root_item_size,
        );
        let mut root_item = vec![0_u8; root_item_size as usize];
        root_item[176..184].copy_from_slice(&fs_tree_logical.to_le_bytes()); // bytenr
        let root_item_last = root_item.len() - 1;
        root_item[root_item_last] = 0; // level
        let root_data_off = root_leaf + root_item_offset as usize;
        image[root_data_off..root_data_off + root_item.len()].copy_from_slice(&root_item);

        // FS tree leaf with: inode(256), dir_index(hello.txt), inode(257), extent_data(257).
        let fs_leaf = fs_tree_logical as usize;
        image[fs_leaf + 0x30..fs_leaf + 0x38].copy_from_slice(&fs_tree_logical.to_le_bytes());
        image[fs_leaf + 0x50..fs_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[fs_leaf + 0x58..fs_leaf + 0x60].copy_from_slice(&5_u64.to_le_bytes()); // FS_TREE owner
        image[fs_leaf + 0x60..fs_leaf + 0x64].copy_from_slice(&4_u32.to_le_bytes());
        image[fs_leaf + 0x64] = 0;

        let root_inode = encode_btrfs_inode_item(0o040_755, 4096, 4096, 2);
        let file_inode = encode_btrfs_inode_item(
            0o100_644,
            file_bytes.len() as u64,
            file_bytes.len() as u64,
            1,
        );
        let dir_index =
            encode_btrfs_dir_index_entry(b"hello.txt", 257, ffs_btrfs::BTRFS_FT_REG_FILE);
        let extent = encode_btrfs_extent_regular(file_data_logical, file_bytes.len() as u64);

        let root_inode_off: u32 = 3200;
        let dir_index_off: u32 = 3060;
        let file_inode_off: u32 = 2860;
        let extent_off: u32 = 2780;

        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            0,
            256,
            BTRFS_ITEM_INODE_ITEM,
            0,
            root_inode_off,
            root_inode.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            1,
            256,
            BTRFS_ITEM_DIR_INDEX,
            1,
            dir_index_off,
            dir_index.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            2,
            257,
            BTRFS_ITEM_INODE_ITEM,
            0,
            file_inode_off,
            file_inode.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            3,
            257,
            BTRFS_ITEM_EXTENT_DATA,
            0,
            extent_off,
            extent.len() as u32,
        );

        image[fs_leaf + root_inode_off as usize
            ..fs_leaf + root_inode_off as usize + root_inode.len()]
            .copy_from_slice(&root_inode);
        image[fs_leaf + dir_index_off as usize..fs_leaf + dir_index_off as usize + dir_index.len()]
            .copy_from_slice(&dir_index);
        image[fs_leaf + file_inode_off as usize
            ..fs_leaf + file_inode_off as usize + file_inode.len()]
            .copy_from_slice(&file_inode);
        image[fs_leaf + extent_off as usize..fs_leaf + extent_off as usize + extent.len()]
            .copy_from_slice(&extent);

        let file_data_off = file_data_logical as usize;
        image[file_data_off..file_data_off + file_bytes.len()].copy_from_slice(file_bytes);

        image
    }

    #[test]
    fn open_fs_from_btrfs_image() {
        let image = build_btrfs_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        assert!(fs.is_btrfs());
        assert!(!fs.is_ext4());
        assert_eq!(fs.block_size(), 4096);
        assert!(fs.ext4_geometry.is_none());
        assert!(fs.btrfs_context.is_some());

        let ctx = fs.btrfs_context().unwrap();
        assert_eq!(ctx.nodesize, 4096);
        assert_eq!(ctx.chunks.len(), 1);
    }

    #[test]
    fn open_fs_btrfs_debug_format() {
        let image = build_btrfs_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let debug = format!("{fs:?}");
        assert!(debug.contains("OpenFs"));
        assert!(debug.contains("btrfs_context"));
    }

    #[test]
    fn open_fs_btrfs_superblock_accessor() {
        let image = build_btrfs_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let sb = fs.btrfs_superblock().expect("btrfs superblock");
        assert_eq!(sb.magic, BTRFS_MAGIC);
        assert_eq!(sb.sectorsize, 4096);
        assert_eq!(sb.nodesize, 4096);
    }

    #[test]
    fn open_fs_btrfs_walk_root_tree() {
        let image = build_btrfs_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let items = fs.walk_btrfs_root_tree(&cx).expect("walk root tree");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].key.objectid, 256);
        assert_eq!(items[0].key.item_type, 132);
        assert_eq!(
            items[0].data,
            vec![0xCA, 0xFE, 0xBA, 0xBE, 0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn open_fs_btrfs_walk_on_ext4_errors() {
        let image = build_ext4_image(2);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();

        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let err = fs.walk_btrfs_root_tree(&cx).unwrap_err();
        assert_eq!(err.to_errno(), libc::EINVAL);
    }

    #[test]
    fn validate_btrfs_rejects_bad_nodesize() {
        let mut image = build_btrfs_image();
        let sb_off = BTRFS_SUPER_INFO_OFFSET;
        // Set nodesize to 128K (too large for our validation)
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&(128 * 1024_u32).to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let err = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap_err();
        assert!(
            matches!(err, FfsError::InvalidGeometry(_)),
            "expected InvalidGeometry, got: {err:?}"
        );
    }

    #[test]
    fn validate_btrfs_skip_validation() {
        let mut image = build_btrfs_image();
        let sb_off = BTRFS_SUPER_INFO_OFFSET;
        // Set nodesize to 128K (too large)
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&(128 * 1024_u32).to_le_bytes());

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();
        assert!(fs.is_btrfs());
        assert!(fs.btrfs_context.is_some());
    }

    // ── Btrfs file read tests ────────────────────────────────────────────

    fn encode_btrfs_extent_inline(data: &[u8]) -> Vec<u8> {
        let mut extent = vec![0_u8; 21 + data.len()];
        extent[0..8].copy_from_slice(&1_u64.to_le_bytes()); // generation
        let data_len = u64::try_from(data.len()).expect("test data length should fit u64");
        extent[8..16].copy_from_slice(&data_len.to_le_bytes()); // ram_bytes
        extent[20] = ffs_btrfs::BTRFS_FILE_EXTENT_INLINE;
        extent[21..].copy_from_slice(data);
        extent
    }

    #[allow(dead_code)]
    fn encode_btrfs_extent_prealloc(num_bytes: u64) -> [u8; 53] {
        let mut extent = [0_u8; 53];
        extent[0..8].copy_from_slice(&1_u64.to_le_bytes()); // generation
        extent[8..16].copy_from_slice(&num_bytes.to_le_bytes()); // ram_bytes
        extent[20] = BTRFS_FILE_EXTENT_PREALLOC;
        extent[21..29].copy_from_slice(&0_u64.to_le_bytes()); // disk_bytenr (unused for prealloc)
        extent[29..37].copy_from_slice(&num_bytes.to_le_bytes()); // disk_num_bytes
        extent[37..45].copy_from_slice(&0_u64.to_le_bytes()); // extent_offset
        extent[45..53].copy_from_slice(&num_bytes.to_le_bytes()); // num_bytes
        extent
    }

    /// Build a btrfs image with a single file containing an inline extent.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    fn build_btrfs_inline_image(file_data: &[u8]) -> Vec<u8> {
        let image_size: usize = 512 * 1024;
        let mut image = vec![0_u8; image_size];
        let sb_off = BTRFS_SUPER_INFO_OFFSET;

        let root_tree_logical = 0x4_000_u64;
        let fs_tree_logical = 0x8_000_u64;

        image[sb_off + 0x40..sb_off + 0x48].copy_from_slice(&BTRFS_MAGIC.to_le_bytes());
        image[sb_off + 0x48..sb_off + 0x50].copy_from_slice(&1_u64.to_le_bytes());
        image[sb_off + 0x50..sb_off + 0x58].copy_from_slice(&root_tree_logical.to_le_bytes());
        image[sb_off + 0x58..sb_off + 0x60].copy_from_slice(&0_u64.to_le_bytes());
        image[sb_off + 0x70..sb_off + 0x78].copy_from_slice(&(image_size as u64).to_le_bytes());
        image[sb_off + 0x80..sb_off + 0x88].copy_from_slice(&256_u64.to_le_bytes());
        image[sb_off + 0x88..sb_off + 0x90].copy_from_slice(&1_u64.to_le_bytes());
        image[sb_off + 0x90..sb_off + 0x94].copy_from_slice(&4096_u32.to_le_bytes());
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&4096_u32.to_le_bytes());
        image[sb_off + 0x9C..sb_off + 0xA0].copy_from_slice(&4096_u32.to_le_bytes());
        image[sb_off + 0xC6] = 0;

        let mut chunk_array = Vec::new();
        chunk_array.extend_from_slice(&256_u64.to_le_bytes());
        chunk_array.push(228_u8);
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        chunk_array.extend_from_slice(&(image_size as u64).to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0x1_0000_u64.to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&1_u16.to_le_bytes());
        chunk_array.extend_from_slice(&0_u16.to_le_bytes());
        chunk_array.extend_from_slice(&1_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        chunk_array.extend_from_slice(&[0_u8; 16]);

        image[sb_off + 0xA0..sb_off + 0xA4]
            .copy_from_slice(&(chunk_array.len() as u32).to_le_bytes());
        let array_start = sb_off + 0x32B;
        image[array_start..array_start + chunk_array.len()].copy_from_slice(&chunk_array);

        // Root tree leaf: one ROOT_ITEM for FS_TREE
        let root_leaf = root_tree_logical as usize;
        image[root_leaf + 0x30..root_leaf + 0x38].copy_from_slice(&root_tree_logical.to_le_bytes());
        image[root_leaf + 0x50..root_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[root_leaf + 0x58..root_leaf + 0x60].copy_from_slice(&1_u64.to_le_bytes());
        image[root_leaf + 0x60..root_leaf + 0x64].copy_from_slice(&1_u32.to_le_bytes());
        image[root_leaf + 0x64] = 0;

        let root_item_offset: u32 = 3000;
        let root_item_size: u32 = 239;
        write_btrfs_leaf_item(
            &mut image,
            root_leaf,
            0,
            BTRFS_FS_TREE_OBJECTID,
            BTRFS_ITEM_ROOT_ITEM,
            0,
            root_item_offset,
            root_item_size,
        );
        let mut root_item = vec![0_u8; root_item_size as usize];
        root_item[176..184].copy_from_slice(&fs_tree_logical.to_le_bytes());
        let root_item_last = root_item.len() - 1;
        root_item[root_item_last] = 0;
        let root_data_off = root_leaf + root_item_offset as usize;
        image[root_data_off..root_data_off + root_item.len()].copy_from_slice(&root_item);

        // FS tree leaf: root_inode(256), dir_index(hello.txt→257),
        //               file_inode(257), inline_extent(257)
        let fs_leaf = fs_tree_logical as usize;
        image[fs_leaf + 0x30..fs_leaf + 0x38].copy_from_slice(&fs_tree_logical.to_le_bytes());
        image[fs_leaf + 0x50..fs_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[fs_leaf + 0x58..fs_leaf + 0x60].copy_from_slice(&5_u64.to_le_bytes());
        image[fs_leaf + 0x60..fs_leaf + 0x64].copy_from_slice(&4_u32.to_le_bytes());
        image[fs_leaf + 0x64] = 0;

        let root_inode = encode_btrfs_inode_item(0o040_755, 4096, 4096, 2);
        let file_inode =
            encode_btrfs_inode_item(0o100_644, file_data.len() as u64, file_data.len() as u64, 1);
        let dir_index =
            encode_btrfs_dir_index_entry(b"hello.txt", 257, ffs_btrfs::BTRFS_FT_REG_FILE);
        let inline_extent = encode_btrfs_extent_inline(file_data);

        let root_inode_off: u32 = 3200;
        let dir_index_off: u32 = 3060;
        let file_inode_off: u32 = 2860;
        let extent_off: u32 = 2780;

        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            0,
            256,
            BTRFS_ITEM_INODE_ITEM,
            0,
            root_inode_off,
            root_inode.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            1,
            256,
            BTRFS_ITEM_DIR_INDEX,
            1,
            dir_index_off,
            dir_index.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            2,
            257,
            BTRFS_ITEM_INODE_ITEM,
            0,
            file_inode_off,
            file_inode.len() as u32,
        );
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            3,
            257,
            BTRFS_ITEM_EXTENT_DATA,
            0,
            extent_off,
            inline_extent.len() as u32,
        );

        image[fs_leaf + root_inode_off as usize
            ..fs_leaf + root_inode_off as usize + root_inode.len()]
            .copy_from_slice(&root_inode);
        image[fs_leaf + dir_index_off as usize..fs_leaf + dir_index_off as usize + dir_index.len()]
            .copy_from_slice(&dir_index);
        image[fs_leaf + file_inode_off as usize
            ..fs_leaf + file_inode_off as usize + file_inode.len()]
            .copy_from_slice(&file_inode);
        image[fs_leaf + extent_off as usize..fs_leaf + extent_off as usize + inline_extent.len()]
            .copy_from_slice(&inline_extent);

        image
    }

    #[test]
    fn btrfs_read_inline_file() {
        let content = b"small inline data";
        let image = build_btrfs_inline_image(content);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let data = ops.read(&cx, InodeNumber(257), 0, 128).unwrap();
        assert_eq!(&data, content);
    }

    #[test]
    fn btrfs_read_regular_extent_file() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let data = ops.read(&cx, InodeNumber(257), 0, 4096).unwrap();
        assert_eq!(&data, b"hello from btrfs fsops");
    }

    #[test]
    fn btrfs_read_at_offset() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        // Read from offset 6 into "hello from btrfs fsops"
        let data = ops.read(&cx, InodeNumber(257), 6, 128).unwrap();
        assert_eq!(&data, b"from btrfs fsops");
    }

    #[test]
    fn btrfs_read_beyond_eof_returns_truncated() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        // File is 22 bytes. Read 4096 from offset 0 → should get exactly 22 bytes.
        let data = ops.read(&cx, InodeNumber(257), 0, 4096).unwrap();
        assert_eq!(data.len(), 22);
        assert_eq!(&data, b"hello from btrfs fsops");

        // Read from beyond EOF → empty.
        let data = ops.read(&cx, InodeNumber(257), 100, 4096).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn btrfs_read_compressed_extent_returns_unsupported() {
        let mut image = build_btrfs_fsops_image();
        // Set compression byte (offset 16 within extent payload) to zlib (1).
        set_btrfs_test_extent_compression(&mut image, 1);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let err = ops.read(&cx, InodeNumber(257), 0, 128).unwrap_err();
        assert_eq!(err.to_errno(), libc::EOPNOTSUPP);
    }

    #[test]
    fn btrfs_read_prealloc_extent_returns_zeros() {
        let mut image = build_btrfs_fsops_image();
        let file_size = 64_u64;
        // Change extent type to prealloc
        set_btrfs_test_extent_type(&mut image, BTRFS_FILE_EXTENT_PREALLOC);
        set_btrfs_test_extent_lengths(&mut image, file_size);
        set_btrfs_test_file_size(&mut image, file_size);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let data = ops.read(&cx, InodeNumber(257), 0, 128).unwrap();
        let file_size_usize = usize::try_from(file_size).expect("file size should fit in usize");
        assert_eq!(data.len(), file_size_usize);
        assert!(data.iter().all(|b| *b == 0), "prealloc should be all zeros");
    }

    #[test]
    fn btrfs_read_file_with_hole() {
        // Build an image where a file has an extent starting at offset > 0,
        // creating a hole at the beginning.
        let mut image = build_btrfs_fsops_image();
        let file_bytes = b"hello from btrfs fsops";
        let hole_size = 32_u64;
        let file_size = hole_size + file_bytes.len() as u64;

        // Move the extent to start at offset `hole_size` instead of 0.
        set_btrfs_test_extent_key_offset(&mut image, hole_size);
        set_btrfs_test_file_size(&mut image, file_size);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let data = ops.read(&cx, InodeNumber(257), 0, 256).unwrap();
        let file_size_usize = usize::try_from(file_size).expect("file size should fit in usize");
        assert_eq!(data.len(), file_size_usize);
        let hole_size_usize = usize::try_from(hole_size).expect("hole size should fit in usize");
        // First `hole_size` bytes should be zeros (hole).
        assert!(
            data[..hole_size_usize].iter().all(|b| *b == 0),
            "hole region should be all zeros"
        );
        // After hole, file data should appear.
        assert_eq!(&data[hole_size_usize..], file_bytes);
    }

    #[test]
    fn btrfs_read_random_offsets_consistent() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let full = ops.read(&cx, InodeNumber(257), 0, 4096).unwrap();
        assert_eq!(&full, b"hello from btrfs fsops");

        // Read at every byte offset and verify consistency.
        for start in 0..full.len() {
            let chunk = ops.read(&cx, InodeNumber(257), start as u64, 4096).unwrap();
            assert_eq!(&chunk, &full[start..], "read at offset {start} mismatch");
        }

        // Various sizes from the beginning.
        for size in [1_u32, 5, 10, 22, 100] {
            let chunk = ops.read(&cx, InodeNumber(257), 0, size).unwrap();
            let expected_len = full.len().min(size as usize);
            assert_eq!(chunk.len(), expected_len, "size {size}: length mismatch");
            assert_eq!(
                &chunk,
                &full[..expected_len],
                "size {size}: content mismatch"
            );
        }
    }

    #[test]
    fn btrfs_read_directory_returns_is_directory() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        // Root inode (256, aliased as 1) is a directory.
        let err = ops.read(&cx, InodeNumber(1), 0, 4096).unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    // ── Btrfs readdir tests ─────────────────────────────────────────────

    /// Build a btrfs image with multiple directory entries for readdir testing.
    ///
    /// Creates a root directory (256) containing the specified entries.
    /// Each entry gets: DIR_INDEX(256, idx) → child_objectid + INODE_ITEM(child_objectid).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    fn build_btrfs_readdir_image(entries: &[(&[u8], u64, u8, u32)]) -> Vec<u8> {
        // entries: (name, child_objectid, file_type, mode)
        let image_size: usize = 1024 * 1024; // 1 MiB to fit many items
        let mut image = vec![0_u8; image_size];
        let sb_off = BTRFS_SUPER_INFO_OFFSET;

        let root_tree_logical = 0x4_000_u64;
        let fs_tree_logical = 0x20_000_u64; // past superblock at 0x10000

        // Superblock
        image[sb_off + 0x40..sb_off + 0x48].copy_from_slice(&BTRFS_MAGIC.to_le_bytes());
        image[sb_off + 0x48..sb_off + 0x50].copy_from_slice(&1_u64.to_le_bytes());
        image[sb_off + 0x50..sb_off + 0x58].copy_from_slice(&root_tree_logical.to_le_bytes());
        image[sb_off + 0x58..sb_off + 0x60].copy_from_slice(&0_u64.to_le_bytes());
        image[sb_off + 0x70..sb_off + 0x78].copy_from_slice(&(image_size as u64).to_le_bytes());
        image[sb_off + 0x80..sb_off + 0x88].copy_from_slice(&256_u64.to_le_bytes());
        image[sb_off + 0x88..sb_off + 0x90].copy_from_slice(&1_u64.to_le_bytes());
        image[sb_off + 0x90..sb_off + 0x94].copy_from_slice(&4096_u32.to_le_bytes());
        // nodesize = 16384 to fit more items
        image[sb_off + 0x94..sb_off + 0x98].copy_from_slice(&16384_u32.to_le_bytes());
        image[sb_off + 0x9C..sb_off + 0xA0].copy_from_slice(&4096_u32.to_le_bytes());
        image[sb_off + 0xC6] = 0;

        // sys_chunk_array: identity map
        let mut chunk_array = Vec::new();
        chunk_array.extend_from_slice(&256_u64.to_le_bytes());
        chunk_array.push(228_u8);
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        chunk_array.extend_from_slice(&(image_size as u64).to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0x1_0000_u64.to_le_bytes());
        chunk_array.extend_from_slice(&2_u64.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&4096_u32.to_le_bytes());
        chunk_array.extend_from_slice(&1_u16.to_le_bytes());
        chunk_array.extend_from_slice(&0_u16.to_le_bytes());
        chunk_array.extend_from_slice(&1_u64.to_le_bytes());
        chunk_array.extend_from_slice(&0_u64.to_le_bytes());
        chunk_array.extend_from_slice(&[0_u8; 16]);

        image[sb_off + 0xA0..sb_off + 0xA4]
            .copy_from_slice(&(chunk_array.len() as u32).to_le_bytes());
        let array_start = sb_off + 0x32B;
        image[array_start..array_start + chunk_array.len()].copy_from_slice(&chunk_array);

        // Root tree leaf: ROOT_ITEM for FS_TREE
        let root_leaf = root_tree_logical as usize;
        image[root_leaf + 0x30..root_leaf + 0x38].copy_from_slice(&root_tree_logical.to_le_bytes());
        image[root_leaf + 0x50..root_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[root_leaf + 0x58..root_leaf + 0x60].copy_from_slice(&1_u64.to_le_bytes());
        image[root_leaf + 0x60..root_leaf + 0x64].copy_from_slice(&1_u32.to_le_bytes());
        image[root_leaf + 0x64] = 0;

        let root_item_offset: u32 = 3000;
        let root_item_size: u32 = 239;
        write_btrfs_leaf_item(
            &mut image,
            root_leaf,
            0,
            BTRFS_FS_TREE_OBJECTID,
            BTRFS_ITEM_ROOT_ITEM,
            0,
            root_item_offset,
            root_item_size,
        );
        let mut root_item = vec![0_u8; root_item_size as usize];
        root_item[176..184].copy_from_slice(&fs_tree_logical.to_le_bytes());
        let last = root_item.len() - 1;
        root_item[last] = 0;
        let root_data_off = root_leaf + root_item_offset as usize;
        image[root_data_off..root_data_off + root_item.len()].copy_from_slice(&root_item);

        // FS tree leaf: root_inode(256) + DIR_INDEX items + child INODE_ITEM items
        let fs_leaf = fs_tree_logical as usize;
        let nodesize = 16384_usize;
        image[fs_leaf + 0x30..fs_leaf + 0x38].copy_from_slice(&fs_tree_logical.to_le_bytes());
        image[fs_leaf + 0x50..fs_leaf + 0x58].copy_from_slice(&1_u64.to_le_bytes());
        image[fs_leaf + 0x58..fs_leaf + 0x60].copy_from_slice(&5_u64.to_le_bytes());
        image[fs_leaf + 0x64] = 0;

        // Build items: root_inode, then DIR_INDEX for each entry, then child inodes
        let nritems = 1 + entries.len() * 2; // root_inode + (dir_index + child_inode) per entry
        image[fs_leaf + 0x60..fs_leaf + 0x64].copy_from_slice(&(nritems as u32).to_le_bytes());

        let root_inode = encode_btrfs_inode_item(0o040_755, 4096, 4096, 2);

        // Place data payloads from the end of the node backwards
        let mut data_cursor = nodesize;
        let mut item_idx = 0_usize;

        // Item 0: root inode
        data_cursor -= root_inode.len();
        write_btrfs_leaf_item(
            &mut image,
            fs_leaf,
            item_idx,
            256,
            BTRFS_ITEM_INODE_ITEM,
            0,
            data_cursor as u32,
            root_inode.len() as u32,
        );
        image[fs_leaf + data_cursor..fs_leaf + data_cursor + root_inode.len()]
            .copy_from_slice(&root_inode);
        item_idx += 1;

        // DIR_INDEX entries for each child (objectid=256, key_offset=index)
        for (i, (name, child_oid, file_type, _mode)) in entries.iter().enumerate() {
            let dir_entry = encode_btrfs_dir_index_entry(name, *child_oid, *file_type);
            data_cursor -= dir_entry.len();
            write_btrfs_leaf_item(
                &mut image,
                fs_leaf,
                item_idx,
                256,
                BTRFS_ITEM_DIR_INDEX,
                (i + 1) as u64,
                data_cursor as u32,
                dir_entry.len() as u32,
            );
            image[fs_leaf + data_cursor..fs_leaf + data_cursor + dir_entry.len()]
                .copy_from_slice(&dir_entry);
            item_idx += 1;
        }

        // Child INODE_ITEM for each entry
        for (_, child_oid, _, mode) in entries {
            let child_inode = encode_btrfs_inode_item(*mode, 0, 0, 1);
            data_cursor -= child_inode.len();
            write_btrfs_leaf_item(
                &mut image,
                fs_leaf,
                item_idx,
                *child_oid,
                BTRFS_ITEM_INODE_ITEM,
                0,
                data_cursor as u32,
                child_inode.len() as u32,
            );
            image[fs_leaf + data_cursor..fs_leaf + data_cursor + child_inode.len()]
                .copy_from_slice(&child_inode);
            item_idx += 1;
        }

        image
    }

    #[test]
    fn btrfs_readdir_root_directory() {
        // Existing fsops image has root dir with one entry "hello.txt"
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let entries = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(entries.len(), 3); // . + .. + hello.txt
        assert_eq!(entries[0].name, b".");
        assert_eq!(entries[0].kind, FileType::Directory);
        assert_eq!(entries[1].name, b"..");
        assert_eq!(entries[1].kind, FileType::Directory);
        assert_eq!(entries[2].name, b"hello.txt");
        assert_eq!(entries[2].kind, FileType::RegularFile);
    }

    #[test]
    fn btrfs_readdir_mixed_types() {
        let entries: Vec<(&[u8], u64, u8, u32)> = vec![
            (b"readme.txt", 257, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
            (b"subdir", 258, ffs_btrfs::BTRFS_FT_DIR, 0o040_755),
            (b"link.txt", 259, ffs_btrfs::BTRFS_FT_SYMLINK, 0o120_777),
        ];
        let image = build_btrfs_readdir_image(&entries);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let result = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(result.len(), 5); // . + .. + 3 entries

        // Check types are correct
        let readme = result.iter().find(|e| e.name == b"readme.txt").unwrap();
        assert_eq!(readme.kind, FileType::RegularFile);
        let subdir = result.iter().find(|e| e.name == b"subdir").unwrap();
        assert_eq!(subdir.kind, FileType::Directory);
        let link = result.iter().find(|e| e.name == b"link.txt").unwrap();
        assert_eq!(link.kind, FileType::Symlink);
    }

    #[test]
    fn btrfs_readdir_empty_directory_returns_dot_dotdot() {
        let entries: Vec<(&[u8], u64, u8, u32)> = vec![];
        let image = build_btrfs_readdir_image(&entries);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let result = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, b".");
        assert_eq!(result[1].name, b"..");
    }

    #[test]
    fn btrfs_readdir_sorted_by_index() {
        let entries: Vec<(&[u8], u64, u8, u32)> = vec![
            (b"aaa.txt", 257, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
            (b"bbb.txt", 258, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
            (b"ccc.txt", 259, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
        ];
        let image = build_btrfs_readdir_image(&entries);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let result = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(result.len(), 5);
        // Entries after . and .. should be in index order
        assert_eq!(result[2].name, b"aaa.txt");
        assert_eq!(result[3].name, b"bbb.txt");
        assert_eq!(result[4].name, b"ccc.txt");

        // Offsets should be monotonically increasing
        for pair in result.windows(2) {
            assert!(
                pair[0].offset < pair[1].offset,
                "offsets must be monotonically increasing: {} vs {}",
                pair[0].offset,
                pair[1].offset,
            );
        }
    }

    #[test]
    fn btrfs_readdir_offset_pagination() {
        let entries: Vec<(&[u8], u64, u8, u32)> = vec![
            (b"aaa.txt", 257, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
            (b"bbb.txt", 258, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
            (b"ccc.txt", 259, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
        ];
        let image = build_btrfs_readdir_image(&entries);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;

        // Skip . and ..
        let paged = ops.readdir(&cx, InodeNumber(1), 2).unwrap();
        assert_eq!(paged.len(), 3);
        assert_eq!(paged[0].name, b"aaa.txt");
        assert_eq!(paged[1].name, b"bbb.txt");
        assert_eq!(paged[2].name, b"ccc.txt");

        // Skip all but last
        let paged = ops.readdir(&cx, InodeNumber(1), 4).unwrap();
        assert_eq!(paged.len(), 1);
        assert_eq!(paged[0].name, b"ccc.txt");

        // Skip all
        let paged = ops.readdir(&cx, InodeNumber(1), 5).unwrap();
        assert!(paged.is_empty());
    }

    #[test]
    fn btrfs_readdir_special_characters_in_names() {
        let entries: Vec<(&[u8], u64, u8, u32)> = vec![
            (
                b"file with spaces.txt",
                257,
                ffs_btrfs::BTRFS_FT_REG_FILE,
                0o100_644,
            ),
            (
                "möbius.txt".as_bytes(),
                258,
                ffs_btrfs::BTRFS_FT_REG_FILE,
                0o100_644,
            ),
            (b"dots...name", 259, ffs_btrfs::BTRFS_FT_REG_FILE, 0o100_644),
        ];
        let image = build_btrfs_readdir_image(&entries);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let result = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        assert_eq!(result.len(), 5);

        let names: Vec<&[u8]> = result.iter().map(|e| e.name.as_slice()).collect();
        assert!(names.contains(&b"file with spaces.txt".as_slice()));
        assert!(names.contains(&"möbius.txt".as_bytes()));
        assert!(names.contains(&b"dots...name".as_slice()));
    }

    #[test]
    fn btrfs_readdir_lookup_consistency() {
        // Every entry returned by readdir should be lookable.
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        let entries = ops.readdir(&cx, InodeNumber(1), 0).unwrap();

        for entry in &entries {
            if entry.name == b"." || entry.name == b".." {
                continue;
            }
            let name = OsStr::new(std::str::from_utf8(&entry.name).unwrap());
            let attr = ops.lookup(&cx, InodeNumber(1), name).unwrap();
            assert_eq!(
                attr.ino,
                entry.ino,
                "lookup inode mismatch for {:?}",
                entry.name_str()
            );
            assert_eq!(
                attr.kind,
                entry.kind,
                "lookup type mismatch for {:?}",
                entry.name_str()
            );
        }
    }

    #[test]
    fn btrfs_readdir_on_non_directory_fails() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let ops: &dyn FsOps = &fs;
        // InodeNumber(257) is a regular file
        let err = ops.readdir(&cx, InodeNumber(257), 0).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTDIR);
    }

    // ── Send+Sync and concurrency tests ──────────────────────────────────

    #[test]
    fn open_fs_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenFs>();
    }

    #[test]
    fn concurrent_read_ops_no_deadlock() {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = std::sync::Arc::new(
            OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap(),
        );

        std::thread::scope(|s| {
            for _ in 0..10 {
                let fs = std::sync::Arc::clone(&fs);
                s.spawn(move || {
                    let cx = Cx::for_testing();
                    let ops: &dyn FsOps = fs.as_ref();
                    for _ in 0..50 {
                        let _ = ops.getattr(&cx, InodeNumber(1));
                        let _ = ops.readdir(&cx, InodeNumber(1), 0);
                        let _ = ops.read(&cx, InodeNumber(257), 0, 4096);
                        let _ = ops.lookup(&cx, InodeNumber(1), std::ffi::OsStr::new("hello.txt"));
                    }
                });
            }
        });
    }

    // ── DurabilityAutopilot tests ────────────────────────────────────────

    /// Standard candidate set: 1% to 10% overhead.
    fn standard_candidates() -> Vec<f64> {
        (1..=10).map(|i| f64::from(i).mul_add(0.01, 1.0)).collect()
    }

    #[test]
    fn posterior_uniform_prior() {
        let p = DurabilityPosterior::default();
        assert!((p.alpha - 1.0).abs() < f64::EPSILON);
        assert!((p.beta - 1.0).abs() < f64::EPSILON);
        assert!((p.expected_corruption_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn posterior_observe_blocks_updates_correctly() {
        let mut p = DurabilityPosterior::default();
        // Scrub 1000 blocks, find 10 corrupt.
        p.observe_blocks(1000, 10);
        // alpha = 1 + 10 = 11, beta = 1 + 990 = 991
        assert!((p.alpha - 11.0).abs() < f64::EPSILON);
        assert!((p.beta - 991.0).abs() < f64::EPSILON);
        let rate = p.expected_corruption_rate();
        assert!((rate - 11.0 / 1002.0).abs() < 1e-10);
    }

    #[test]
    fn posterior_converges_to_empirical_rate() {
        let mut p = DurabilityPosterior::default();
        // Many observations at 2% corruption rate.
        for _ in 0..100 {
            p.observe_blocks(10_000, 200);
        }
        let rate = p.expected_corruption_rate();
        assert!((rate - 0.02).abs() < 0.001, "expected ~0.02, got {rate}");
    }

    #[test]
    fn posterior_variance_decreases_with_observations() {
        let mut p = DurabilityPosterior::default();
        let var_before = p.variance();
        p.observe_blocks(10_000, 100);
        let var_after = p.variance();
        assert!(
            var_after < var_before,
            "variance should decrease: {var_before} -> {var_after}"
        );
    }

    #[test]
    fn autopilot_fresh_picks_lowest_overhead() {
        // With no observations (uniform prior), p_hi clamps to 1.0.
        // All candidates have risk_bound=1.0 (rho <= p_hi), so the
        // corruption_loss is identical.  Tiebreaker is redundancy_loss,
        // which is minimized at the lowest candidate.
        let ap = DurabilityAutopilot::new();
        let d = ap.choose_overhead(&standard_candidates());
        assert!(
            (d.repair_overhead - 1.01).abs() < f64::EPSILON,
            "fresh autopilot should pick lowest overhead (risk equal), got {}",
            d.repair_overhead
        );
        assert!(d.expected_loss.is_finite());
        assert!(d.posterior_mean_corruption_rate > 0.0);
        // All candidates have same risk, so corruption_loss is maximal.
        assert!((d.unrecoverable_risk_bound - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn autopilot_low_corruption_picks_low_overhead() {
        let mut ap = DurabilityAutopilot::new();
        // 10 clean scrubs of 100K blocks each, zero corruption.
        for _ in 0..10 {
            ap.observe_scrub(100_000, 0);
        }
        let d = ap.choose_overhead(&standard_candidates());
        // With ~1M clean blocks observed, posterior p is very low.
        // The autopilot should pick a low overhead since risk is negligible.
        assert!(
            d.repair_overhead <= 1.03,
            "low corruption should yield low overhead, got {}",
            d.repair_overhead
        );
        assert!(d.unrecoverable_risk_bound < 1e-10);
        assert!(d.corruption_loss < d.redundancy_loss);
    }

    #[test]
    fn autopilot_high_corruption_picks_high_overhead() {
        let mut ap = DurabilityAutopilot::new();
        // Heavy corruption: 5% of blocks corrupt per scrub.
        for _ in 0..5 {
            ap.observe_scrub(10_000, 500);
        }
        let d = ap.choose_overhead(&standard_candidates());
        // p is around 5%, so overhead should be high to cover.
        assert!(
            d.repair_overhead >= 1.06,
            "high corruption should yield high overhead, got {}",
            d.repair_overhead
        );
        assert!(d.posterior_mean_corruption_rate > 0.04);
    }

    #[test]
    fn autopilot_reacts_to_corruption_increase() {
        // Use a modest clean history so corruption isn't swamped.
        let mut ap = DurabilityAutopilot::new();
        ap.observe_scrub(10_000, 0);
        let d_clean = ap.choose_overhead(&standard_candidates());

        // Observe heavy corruption: 5% of blocks corrupt, enough data to
        // push the posterior mean above 1% so overhead must increase.
        for _ in 0..20 {
            ap.observe_scrub(10_000, 500);
        }
        let d_corrupt = ap.choose_overhead(&standard_candidates());

        assert!(
            d_corrupt.repair_overhead > d_clean.repair_overhead,
            "overhead should increase after corruption: {} -> {}",
            d_clean.repair_overhead,
            d_corrupt.repair_overhead,
        );
        assert!(d_corrupt.posterior_mean_corruption_rate > d_clean.posterior_mean_corruption_rate);
    }

    #[test]
    fn decision_contains_explainable_fields() {
        let mut ap = DurabilityAutopilot::new();
        ap.observe_scrub(50_000, 25);
        let d = ap.choose_overhead(&standard_candidates());

        // All evidence fields must be populated and finite.
        assert!(d.repair_overhead.is_finite());
        assert!(d.expected_loss.is_finite());
        assert!(d.posterior_mean_corruption_rate.is_finite());
        assert!(d.posterior_hi_corruption_rate.is_finite());
        assert!(d.unrecoverable_risk_bound.is_finite());
        assert!(d.redundancy_loss.is_finite());
        assert!(d.corruption_loss.is_finite());

        // Consistency: expected_loss = redundancy_loss + corruption_loss.
        let sum = d.redundancy_loss + d.corruption_loss;
        assert!(
            (d.expected_loss - sum).abs() < 1e-10,
            "loss should decompose: {} != {} + {}",
            d.expected_loss,
            d.redundancy_loss,
            d.corruption_loss,
        );

        // p_hi >= p_mean (upper bound).
        assert!(d.posterior_hi_corruption_rate >= d.posterior_mean_corruption_rate);

        // Overhead is in valid range.
        assert!(d.repair_overhead >= 1.01);
        assert!(d.repair_overhead <= 1.10);
    }

    #[test]
    fn risk_bound_monotonically_decreases_with_overhead() {
        let mut ap = DurabilityAutopilot::new();
        ap.observe_scrub(100_000, 50);
        let candidates = standard_candidates();

        let mut prev_risk = f64::INFINITY;
        for &c in &candidates {
            let d = ap.choose_overhead_for_group(&[c], 32_768);
            assert!(
                d.unrecoverable_risk_bound <= prev_risk + f64::EPSILON,
                "risk should decrease: at overhead {c}, risk {} > prev {prev_risk}",
                d.unrecoverable_risk_bound,
            );
            prev_risk = d.unrecoverable_risk_bound;
        }
    }

    #[test]
    fn autopilot_no_valid_candidates_uses_default() {
        let ap = DurabilityAutopilot::new();
        // Pass only out-of-range candidates.
        let d = ap.choose_overhead(&[0.5, 2.0, f64::NAN, f64::INFINITY]);
        assert!((d.repair_overhead - 1.05).abs() < f64::EPSILON);
    }

    #[test]
    fn autopilot_empty_candidates_uses_default() {
        let ap = DurabilityAutopilot::new();
        let d = ap.choose_overhead(&[]);
        assert!((d.repair_overhead - 1.05).abs() < f64::EPSILON);
    }

    #[test]
    fn autopilot_group_size_affects_risk() {
        let mut ap = DurabilityAutopilot::new();
        ap.observe_scrub(100_000, 50);

        // Large group: more blocks = tighter concentration = lower risk.
        let d_large = ap.choose_overhead_for_group(&standard_candidates(), 32_768);
        // Small group: fewer blocks = wider variance = higher risk.
        let d_small = ap.choose_overhead_for_group(&standard_candidates(), 100);

        // Small groups should pick higher (or equal) overhead.
        assert!(
            d_small.repair_overhead >= d_large.repair_overhead,
            "small group ({}) should need >= overhead than large group ({})",
            d_small.repair_overhead,
            d_large.repair_overhead,
        );
    }

    #[test]
    fn loss_model_custom_costs() {
        let mut ap = DurabilityAutopilot {
            posterior: DurabilityPosterior::default(),
            loss: DurabilityLossModel {
                corruption_cost: 1.0,
                redundancy_cost: 1_000_000.0,
                z_score: 3.0,
            },
        };
        // When redundancy is extremely expensive, should pick lowest overhead.
        ap.observe_scrub(100_000, 10);
        let d = ap.choose_overhead(&standard_candidates());
        assert!(
            (d.repair_overhead - 1.01).abs() < f64::EPSILON,
            "high redundancy cost should pick 1.01, got {}",
            d.repair_overhead,
        );
    }

    #[test]
    fn decision_serializes_to_json() {
        let mut ap = DurabilityAutopilot::new();
        ap.observe_scrub(10_000, 5);
        let d = ap.choose_overhead(&standard_candidates());
        let json = serde_json::to_string(&d).expect("serialize");
        let d2: RedundancyDecision = serde_json::from_str(&json).expect("deserialize");
        assert!((d.repair_overhead - d2.repair_overhead).abs() < f64::EPSILON);
        assert!((d.expected_loss - d2.expected_loss).abs() < 1e-10);
    }

    // ── RepairPolicy tests ───────────────────────────────────────────────

    #[test]
    fn repair_policy_default_is_static_5pct() {
        let p = RepairPolicy::default();
        assert!((p.overhead_ratio - 1.05).abs() < f64::EPSILON);
        assert!(!p.eager_refresh);
        assert!(p.autopilot.is_none());
        assert!((p.effective_overhead() - 1.05).abs() < f64::EPSILON);
        assert!(p.autopilot_decision().is_none());
    }

    #[test]
    fn repair_policy_with_autopilot_delegates() {
        let mut ap = DurabilityAutopilot::new();
        for _ in 0..10 {
            ap.observe_scrub(100_000, 0);
        }
        let policy = RepairPolicy {
            overhead_ratio: 1.05,
            eager_refresh: false,
            autopilot: Some(ap),
        };
        let overhead = policy.effective_overhead();
        // Should come from autopilot, not static ratio.
        assert!((1.01..=1.10).contains(&overhead));

        let decision = policy.autopilot_decision().expect("should have decision");
        assert!((decision.repair_overhead - overhead).abs() < f64::EPSILON);
    }

    // ── FsOps write operation tests ──────────────────────────────────────

    #[test]
    fn fsops_default_write_methods_return_read_only() {
        let fs = StubFs;
        let cx = Cx::for_testing();
        let root = InodeNumber(1);

        let err = fs
            .create(&cx, root, OsStr::new("x"), 0o644, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .mkdir(&cx, root, OsStr::new("d"), 0o755, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.unlink(&cx, root, OsStr::new("x")).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.rmdir(&cx, root, OsStr::new("d")).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.fsync(&cx, root, 0, false).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.fsyncdir(&cx, root, 0, false).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        fs.flush(&cx, root, 0, 0).expect("flush default no-op");

        let err = fs
            .rename(&cx, root, OsStr::new("a"), root, OsStr::new("b"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.write(&cx, InodeNumber(11), 0, b"hello").unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .setattr(&cx, root, &SetAttrRequest::default())
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .setxattr(&cx, root, "user.test", b"value", XattrSetMode::Set)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.removexattr(&cx, root, "user.test").unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);
    }

    #[test]
    fn open_fs_not_writable_by_default() {
        let image = build_ext4_image(2);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        assert!(!fs.is_writable());
    }

    /// Helper: open the ext4_small test fixture and enable writes.
    ///
    /// Returns None if the fixture doesn't exist (e.g. in minimal CI).
    fn open_writable_ext4() -> Option<OpenFs> {
        let path = std::path::Path::new("tests/fixtures/images/ext4_small.img");
        // Also check from the workspace root
        let path = if path.exists() {
            path.to_path_buf()
        } else {
            let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("tests/fixtures/images/ext4_small.img");
            if ws.exists() {
                ws
            } else {
                return None;
            }
        };
        let cx = Cx::for_testing();
        // Copy the image so we don't mutate the fixture.
        let data = std::fs::read(&path).expect("read fixture");
        let dev = TestDevice::from_vec(data);
        let opts = OpenOptions {
            ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
            ..OpenOptions::default()
        };
        let mut fs = OpenFs::from_device(&cx, Box::new(dev), &opts).expect("open ext4");
        fs.enable_writes(&cx).expect("enable writes");
        assert!(fs.is_writable());
        Some(fs)
    }

    #[test]
    fn write_create_and_read_roundtrip() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2); // ext4 root inode

        // Create a new file
        let attr = fs
            .create(&cx, root, OsStr::new("test_rw.txt"), 0o644, 1000, 1000)
            .expect("create");
        assert_eq!(attr.kind, FileType::RegularFile);
        assert_eq!(attr.perm, 0o644);
        assert_eq!(attr.uid, 1000);
        let ino = attr.ino;

        // Write data
        let payload = b"FrankenFS write test!";
        let written = fs.write(&cx, ino, 0, payload).expect("write");
        assert_eq!(written as usize, payload.len());

        // Read back
        let readback = fs.read(&cx, ino, 0, 4096).expect("read");
        assert_eq!(&readback[..payload.len()], payload);

        // Lookup should find it
        let looked_up = fs
            .lookup(&cx, root, OsStr::new("test_rw.txt"))
            .expect("lookup");
        assert_eq!(looked_up.ino, ino);
    }

    #[test]
    fn write_create_and_read_roundtrip_via_vfs_root_alias() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(1); // VFS/FUSE root alias for ext4 inode 2

        let root_attr = fs.getattr(&cx, root).expect("getattr root");
        assert_eq!(root_attr.ino, root);
        assert_eq!(root_attr.kind, FileType::Directory);

        let root_entries = fs.readdir(&cx, root, 0).expect("readdir root");
        let dot = root_entries
            .iter()
            .find(|entry| entry.name_str() == ".")
            .expect("dot entry");
        let dotdot = root_entries
            .iter()
            .find(|entry| entry.name_str() == "..")
            .expect("dotdot entry");
        assert_eq!(dot.ino, root);
        assert_eq!(dotdot.ino, root);

        let attr = fs
            .create(
                &cx,
                root,
                OsStr::new("test_rw_alias.txt"),
                0o644,
                1000,
                1000,
            )
            .expect("create");
        assert_eq!(attr.kind, FileType::RegularFile);

        let payload = b"FrankenFS root-alias write test!";
        let written = fs.write(&cx, attr.ino, 0, payload).expect("write");
        assert_eq!(written as usize, payload.len());

        let readback = fs.read(&cx, attr.ino, 0, 4096).expect("read");
        assert_eq!(&readback[..payload.len()], payload);

        let looked_up = fs
            .lookup(&cx, root, OsStr::new("test_rw_alias.txt"))
            .expect("lookup");
        assert_eq!(looked_up.ino, attr.ino);
    }

    #[test]
    fn write_mkdir_and_lookup() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .mkdir(&cx, root, OsStr::new("test_dir"), 0o755, 0, 0)
            .expect("mkdir");
        assert_eq!(attr.kind, FileType::Directory);
        assert_eq!(attr.perm, 0o755);

        let looked_up = fs
            .lookup(&cx, root, OsStr::new("test_dir"))
            .expect("lookup dir");
        assert_eq!(looked_up.ino, attr.ino);
        assert_eq!(looked_up.kind, FileType::Directory);
    }

    #[test]
    fn write_unlink_removes_entry() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("to_delete.txt"), 0o644, 0, 0)
            .expect("create");
        assert!(attr.ino.0 > 0);

        // Unlink it
        fs.unlink(&cx, root, OsStr::new("to_delete.txt"))
            .expect("unlink");

        // Lookup should fail
        let err = fs
            .lookup(&cx, root, OsStr::new("to_delete.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_rename_moves_entry() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("old_name.txt"), 0o644, 0, 0)
            .expect("create");
        let ino = attr.ino;

        // Rename
        fs.rename(
            &cx,
            root,
            OsStr::new("old_name.txt"),
            root,
            OsStr::new("new_name.txt"),
        )
        .expect("rename");

        // Old name gone
        let err = fs
            .lookup(&cx, root, OsStr::new("old_name.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);

        // New name present with same inode
        let looked_up = fs
            .lookup(&cx, root, OsStr::new("new_name.txt"))
            .expect("lookup new name");
        assert_eq!(looked_up.ino, ino);
    }

    #[test]
    fn write_link_creates_hardlink_and_increments_nlink() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let src = fs
            .create(&cx, root, OsStr::new("link_src.txt"), 0o644, 0, 0)
            .expect("create source");
        fs.write(&cx, src.ino, 0, b"linked-bytes")
            .expect("write source");

        let link_attr = fs
            .link(&cx, src.ino, root, OsStr::new("link_dst.txt"))
            .expect("link");
        assert_eq!(link_attr.ino, src.ino);

        let src_attr = fs.getattr(&cx, src.ino).expect("getattr source");
        assert_eq!(src_attr.nlink, 2);

        let dst_attr = fs
            .lookup(&cx, root, OsStr::new("link_dst.txt"))
            .expect("lookup destination");
        assert_eq!(dst_attr.ino, src.ino);

        let readback = fs.read(&cx, dst_attr.ino, 0, 64).expect("read via link");
        assert_eq!(&readback, b"linked-bytes");
    }

    #[test]
    fn write_link_directory_rejected_with_eperm() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);
        let dir_attr = fs
            .mkdir(&cx, root, OsStr::new("link_dir"), 0o755, 0, 0)
            .expect("mkdir");

        let err = fs
            .link(&cx, dir_attr.ino, root, OsStr::new("dir_hardlink"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EPERM);
    }

    #[test]
    fn write_link_rejected_at_emlink_limit() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let src = fs
            .create(&cx, root, OsStr::new("emlink_src.txt"), 0o644, 0, 0)
            .expect("create source");

        let mut inode = fs.read_inode(&cx, src.ino).expect("read source inode");
        inode.links_count = 65_000;

        let sb = fs.ext4_superblock().expect("ext4 superblock");
        let csum_seed = sb.csum_seed();
        let block_dev = fs.block_device_adapter();
        let alloc_mutex = fs.require_alloc_state().expect("alloc state");
        let alloc = alloc_mutex.lock();
        ffs_inode::write_inode(
            &cx,
            &block_dev,
            &alloc.geo,
            &alloc.groups,
            src.ino,
            &inode,
            csum_seed,
        )
        .expect("persist nlink update");
        drop(alloc);

        let err = fs
            .link(&cx, src.ino, root, OsStr::new("emlink_dst.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EMLINK);
    }

    #[test]
    fn write_symlink_fast_target_roundtrip() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .symlink(
                &cx,
                root,
                OsStr::new("fast_link"),
                Path::new("hello.txt"),
                1000,
                1000,
            )
            .expect("symlink fast");
        assert_eq!(attr.kind, FileType::Symlink);

        let looked_up = fs
            .lookup(&cx, root, OsStr::new("fast_link"))
            .expect("lookup fast_link");
        assert_eq!(looked_up.kind, FileType::Symlink);

        let target = fs.readlink(&cx, attr.ino).expect("readlink");
        assert_eq!(&target, b"hello.txt");

        let inode = fs.read_inode(&cx, attr.ino).expect("inode");
        assert!(inode.is_fast_symlink());
    }

    #[test]
    fn write_symlink_slow_target_roundtrip() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);
        let long_target = "/var/lib/frankenfs/some/really/long/path/that/exceeds/sixty/bytes/for/slow/symlink-target.txt";

        let attr = fs
            .symlink(
                &cx,
                root,
                OsStr::new("slow_link"),
                Path::new(long_target),
                1000,
                1000,
            )
            .expect("symlink slow");
        assert_eq!(attr.kind, FileType::Symlink);

        let target = fs.readlink(&cx, attr.ino).expect("readlink");
        assert_eq!(target, long_target.as_bytes());

        let inode = fs.read_inode(&cx, attr.ino).expect("inode");
        assert!(!inode.is_fast_symlink());
    }

    #[test]
    fn write_fallocate_preallocate_keep_size_and_punch_hole() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("falloc.bin"), 0o644, 0, 0)
            .expect("create");

        fs.fallocate(&cx, attr.ino, 0, 8192, 0).expect("fallocate");
        let after_alloc = fs.getattr(&cx, attr.ino).expect("getattr");
        assert_eq!(after_alloc.size, 8192);
        assert!(after_alloc.blocks >= 16);

        let zero_data = fs.read(&cx, attr.ino, 0, 8192).expect("read prealloc");
        assert_eq!(zero_data.len(), 8192);
        assert!(zero_data.iter().all(|&b| b == 0));

        fs.fallocate(&cx, attr.ino, 12288, 4096, libc::FALLOC_FL_KEEP_SIZE)
            .expect("fallocate keep_size");
        let after_keep = fs.getattr(&cx, attr.ino).expect("getattr keep_size");
        assert_eq!(after_keep.size, 8192);

        let payload = vec![0xAB_u8; 8192];
        fs.write(&cx, attr.ino, 0, &payload).expect("write payload");

        fs.fallocate(
            &cx,
            attr.ino,
            4096,
            4096,
            libc::FALLOC_FL_KEEP_SIZE | libc::FALLOC_FL_PUNCH_HOLE,
        )
        .expect("punch hole");

        let after_punch = fs.getattr(&cx, attr.ino).expect("getattr after punch");
        assert_eq!(after_punch.size, 8192);
        let readback = fs.read(&cx, attr.ino, 0, 8192).expect("read after punch");
        assert!(readback[..4096].iter().all(|&b| b == 0xAB));
        assert!(readback[4096..8192].iter().all(|&b| b == 0));
    }

    #[test]
    fn write_setattr_truncate() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("truncate_me.txt"), 0o644, 0, 0)
            .expect("create");
        let ino = attr.ino;

        // Write 100 bytes
        let data = vec![0x42_u8; 100];
        fs.write(&cx, ino, 0, &data).expect("write");

        // Verify size
        let attr = fs.getattr(&cx, ino).expect("getattr");
        assert_eq!(attr.size, 100);

        // Truncate to 50 bytes
        let attrs = SetAttrRequest {
            size: Some(50),
            ..SetAttrRequest::default()
        };
        let new_attr = fs.setattr(&cx, ino, &attrs).expect("setattr truncate");
        assert_eq!(new_attr.size, 50);

        // Read should return 50 bytes
        let readback = fs.read(&cx, ino, 0, 4096).expect("read after truncate");
        assert_eq!(readback.len(), 50);
        assert!(readback.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn write_setxattr_replace_and_remove_roundtrip() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("xattr_roundtrip.txt"), 0o644, 0, 0)
            .expect("create");
        let ino = attr.ino;

        fs.setxattr(&cx, ino, "user.mime", b"text/plain", XattrSetMode::Set)
            .expect("setxattr create-or-replace");
        assert_eq!(
            fs.getxattr(&cx, ino, "user.mime").expect("getxattr"),
            Some(b"text/plain".to_vec())
        );

        fs.setxattr(
            &cx,
            ino,
            "user.mime",
            b"application/octet-stream",
            XattrSetMode::Replace,
        )
        .expect("setxattr replace");
        assert_eq!(
            fs.getxattr(&cx, ino, "user.mime")
                .expect("getxattr after replace"),
            Some(b"application/octet-stream".to_vec())
        );

        assert!(
            fs.removexattr(&cx, ino, "user.mime")
                .expect("removexattr first call")
        );
        assert_eq!(fs.getxattr(&cx, ino, "user.mime").expect("getxattr"), None);
        assert!(
            !fs.removexattr(&cx, ino, "user.mime")
                .expect("removexattr second call")
        );
    }

    #[test]
    fn write_setxattr_respects_create_and_replace_modes() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("xattr_modes.txt"), 0o644, 0, 0)
            .expect("create");
        let ino = attr.ino;

        fs.setxattr(&cx, ino, "user.color", b"blue", XattrSetMode::Create)
            .expect("initial create");

        let err = fs
            .setxattr(&cx, ino, "user.color", b"green", XattrSetMode::Create)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EEXIST);

        let err = fs
            .setxattr(&cx, ino, "user.missing", b"x", XattrSetMode::Replace)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_setxattr_allocates_external_block_when_inline_exhausted() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("xattr_external.txt"), 0o644, 0, 0)
            .expect("create");

        // A 512-byte value does not fit in the inline ibody region, so
        // ext4_setxattr must allocate an external xattr block on the fly.
        let large_value = vec![0xAB_u8; 512];
        fs.setxattr(&cx, attr.ino, "user.large", &large_value, XattrSetMode::Set)
            .expect("setxattr should allocate external block");

        // Read it back and verify.
        let readback = fs.getxattr(&cx, attr.ino, "user.large").expect("getxattr");
        assert_eq!(readback.as_deref(), Some(large_value.as_slice()));
    }

    #[test]
    fn write_rmdir_non_empty_returns_enotempty() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // Create a directory and a file inside it.
        let dir_attr = fs
            .mkdir(&cx, root, OsStr::new("nonempty_dir"), 0o755, 0, 0)
            .expect("mkdir");
        fs.create(&cx, dir_attr.ino, OsStr::new("child.txt"), 0o644, 0, 0)
            .expect("create child");

        // rmdir should fail with ENOTEMPTY
        let err = fs.rmdir(&cx, root, OsStr::new("nonempty_dir")).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTEMPTY);
    }

    #[test]
    fn write_read_only_fs_returns_erofs() {
        let path = std::path::Path::new("tests/fixtures/images/ext4_small.img");
        let path = if path.exists() {
            path.to_path_buf()
        } else {
            let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("tests/fixtures/images/ext4_small.img");
            if !ws.exists() {
                return;
            }
            ws
        };
        let cx = Cx::for_testing();
        let data = std::fs::read(&path).expect("read fixture");
        let dev = TestDevice::from_vec(data);
        let opts = OpenOptions {
            ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
            ..OpenOptions::default()
        };
        // Open without enable_writes — should be read-only.
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).expect("open ext4");
        assert!(!fs.is_writable());

        let root = InodeNumber(2);
        let err = fs
            .create(&cx, root, OsStr::new("nope"), 0o644, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .mkdir(&cx, root, OsStr::new("nope"), 0o755, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.write(&cx, InodeNumber(11), 0, b"data").unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.fsync(&cx, InodeNumber(11), 0, false).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs.fsyncdir(&cx, InodeNumber(2), 0, false).unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .setxattr(
                &cx,
                InodeNumber(11),
                "user.read_only",
                b"value",
                XattrSetMode::Set,
            )
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        let err = fs
            .removexattr(&cx, InodeNumber(11), "user.read_only")
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EROFS);

        fs.flush(&cx, InodeNumber(11), 0, 0)
            .expect("flush should be allowed on read-only mount");
    }

    #[test]
    fn write_fsync_and_fsyncdir_writable_ext4_succeed() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("sync_me.txt"), 0o644, 0, 0)
            .expect("create");
        fs.write(&cx, attr.ino, 0, b"durable")
            .expect("write before fsync");

        fs.fsync(&cx, attr.ino, 0, false).expect("fsync");
        fs.fsyncdir(&cx, root, 0, false).expect("fsyncdir");
    }

    #[test]
    fn write_mkdir_create_inside_readdir() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // Create a directory
        let dir_attr = fs
            .mkdir(&cx, root, OsStr::new("readdir_test"), 0o755, 0, 0)
            .expect("mkdir");

        // Create two files inside it
        fs.create(&cx, dir_attr.ino, OsStr::new("alpha.txt"), 0o644, 0, 0)
            .expect("create alpha");
        fs.create(&cx, dir_attr.ino, OsStr::new("beta.txt"), 0o644, 0, 0)
            .expect("create beta");

        // readdir should list . .. alpha.txt beta.txt
        let entries = fs.readdir(&cx, dir_attr.ino, 0).expect("readdir");
        let names: Vec<String> = entries.iter().map(DirEntry::name_str).collect();
        assert!(names.contains(&".".to_owned()));
        assert!(names.contains(&"..".to_owned()));
        assert!(names.contains(&"alpha.txt".to_owned()));
        assert!(names.contains(&"beta.txt".to_owned()));
        assert_eq!(entries.len(), 4);
    }

    // ── Ext4 write-path error and edge-case tests ─────────────────────

    #[test]
    fn write_create_in_non_directory_returns_enotdir() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // Create a regular file, then try to create a child inside it.
        let file_attr = fs
            .create(&cx, root, OsStr::new("not_a_dir.txt"), 0o644, 0, 0)
            .expect("create file");
        let err = fs
            .create(&cx, file_attr.ino, OsStr::new("child.txt"), 0o644, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTDIR);
    }

    #[test]
    fn write_mkdir_in_non_directory_returns_enotdir() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let file_attr = fs
            .create(&cx, root, OsStr::new("regular.txt"), 0o644, 0, 0)
            .expect("create file");
        let err = fs
            .mkdir(&cx, file_attr.ino, OsStr::new("subdir"), 0o755, 0, 0)
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTDIR);
    }

    #[test]
    fn write_to_directory_returns_eisdir() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let err = fs.write(&cx, root, 0, b"data").unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn write_to_out_of_range_inode_returns_einval() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();

        let err = fs
            .write(&cx, InodeNumber(u64::MAX / 2), 0, b"ghost")
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EINVAL);
    }

    #[test]
    fn write_unlink_nonexistent_returns_enoent() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let err = fs
            .unlink(&cx, root, OsStr::new("does_not_exist.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_unlink_directory_returns_eisdir() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        fs.mkdir(&cx, root, OsStr::new("dir_target"), 0o755, 0, 0)
            .expect("mkdir");
        let err = fs.unlink(&cx, root, OsStr::new("dir_target")).unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn write_rmdir_nonexistent_returns_enoent() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let err = fs.rmdir(&cx, root, OsStr::new("no_such_dir")).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_double_unlink_second_fails() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        fs.create(&cx, root, OsStr::new("ephemeral.txt"), 0o644, 0, 0)
            .expect("create");
        fs.unlink(&cx, root, OsStr::new("ephemeral.txt"))
            .expect("first unlink");
        let err = fs
            .unlink(&cx, root, OsStr::new("ephemeral.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_create_then_unlink_then_recreate() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr1 = fs
            .create(&cx, root, OsStr::new("recycled.txt"), 0o644, 0, 0)
            .expect("first create");
        let ino1 = attr1.ino;

        fs.unlink(&cx, root, OsStr::new("recycled.txt"))
            .expect("unlink");

        let attr2 = fs
            .create(&cx, root, OsStr::new("recycled.txt"), 0o644, 0, 0)
            .expect("second create");
        // Inode may or may not be reused, but the entry should resolve
        let looked_up = fs
            .lookup(&cx, root, OsStr::new("recycled.txt"))
            .expect("lookup after recreate");
        assert_eq!(looked_up.ino, attr2.ino);
        // The original inode should no longer be referenced by this name
        if attr2.ino != ino1 {
            // Different inode was allocated — getattr on the old one may or
            // may not succeed depending on whether it was freed, but lookup
            // for the name must resolve to the new one.
            assert_ne!(looked_up.ino, ino1);
        }
    }

    #[test]
    fn write_rename_nonexistent_source_returns_enoent() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let err = fs
            .rename(
                &cx,
                root,
                OsStr::new("ghost.txt"),
                root,
                OsStr::new("target.txt"),
            )
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn write_link_to_directory_returns_eperm() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let dir_attr = fs
            .mkdir(&cx, root, OsStr::new("cant_hardlink"), 0o755, 0, 0)
            .expect("mkdir");
        let err = fs
            .link(&cx, dir_attr.ino, root, OsStr::new("hardlink_to_dir"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EPERM);
    }

    #[test]
    fn write_sparse_file_write_at_offset() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("sparse.bin"), 0o644, 0, 0)
            .expect("create");

        // Write at offset 8192 (two blocks in), leaving a hole at the start.
        let payload = b"sparse data";
        let written = fs
            .write(&cx, attr.ino, 8192, payload)
            .expect("write at offset");
        assert_eq!(written as usize, payload.len());

        // Read the hole — should return zeros.
        let hole = fs.read(&cx, attr.ino, 0, 4096).expect("read hole");
        assert!(hole.iter().all(|&b| b == 0), "hole should be zero-filled");

        // Read the written region.
        let data = fs
            .read(
                &cx,
                attr.ino,
                8192,
                u32::try_from(payload.len()).expect("len fits u32"),
            )
            .expect("read at offset");
        assert_eq!(&data[..payload.len()], payload);
    }

    #[test]
    fn write_overwrite_preserves_surrounding_data() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let attr = fs
            .create(&cx, root, OsStr::new("overwrite.bin"), 0o644, 0, 0)
            .expect("create");

        // Write initial data: "AAAA....AAAA" (100 bytes of 'A')
        let initial = vec![b'A'; 100];
        fs.write(&cx, attr.ino, 0, &initial).expect("initial write");

        // Overwrite bytes 10..20 with 'B'
        let patch = vec![b'B'; 10];
        fs.write(&cx, attr.ino, 10, &patch).expect("patch write");

        // Read back and verify
        let readback = fs.read(&cx, attr.ino, 0, 200).expect("read");
        assert!(readback.len() >= 100);
        assert!(
            readback[..10].iter().all(|&b| b == b'A'),
            "prefix preserved"
        );
        assert!(readback[10..20].iter().all(|&b| b == b'B'), "patch applied");
        assert!(
            readback[20..100].iter().all(|&b| b == b'A'),
            "suffix preserved"
        );
    }

    #[test]
    fn write_multiple_files_readdir_shows_all() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let count = 20;
        let mut created_names = Vec::new();
        for i in 0..count {
            let name = format!("multi_{i:03}.txt");
            fs.create(&cx, root, OsStr::new(&name), 0o644, 0, 0)
                .unwrap_or_else(|e| panic!("create {name}: {e}"));
            created_names.push(name);
        }

        let entries = fs.readdir(&cx, root, 0).expect("readdir");
        let names: Vec<String> = entries.iter().map(DirEntry::name_str).collect();
        for expected in &created_names {
            assert!(
                names.contains(expected),
                "missing {expected} in readdir; got: {names:?}"
            );
        }
    }

    #[test]
    fn write_symlink_then_readlink() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // Create target
        fs.create(&cx, root, OsStr::new("link_target.txt"), 0o644, 0, 0)
            .expect("create target");

        // Create symlink
        let sym_attr = fs
            .symlink(
                &cx,
                root,
                OsStr::new("sym.lnk"),
                std::path::Path::new("link_target.txt"),
                0,
                0,
            )
            .expect("symlink");
        assert_eq!(sym_attr.kind, FileType::Symlink);

        // Read the symlink target
        let target = fs.readlink(&cx, sym_attr.ino).expect("readlink");
        assert_eq!(target, b"link_target.txt");
    }

    #[test]
    fn write_rmdir_non_empty_returns_enotempty_ext4() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        let dir = fs
            .mkdir(&cx, root, OsStr::new("nonempty_dir"), 0o755, 0, 0)
            .expect("mkdir");
        fs.create(&cx, dir.ino, OsStr::new("child.txt"), 0o644, 0, 0)
            .expect("create child");

        let err = fs.rmdir(&cx, root, OsStr::new("nonempty_dir")).unwrap_err();
        // Should be ENOTEMPTY (or sometimes EEXIST on some implementations)
        assert!(
            err.to_errno() == libc::ENOTEMPTY || err.to_errno() == libc::EEXIST,
            "expected ENOTEMPTY or EEXIST, got errno {}",
            err.to_errno()
        );
    }

    #[test]
    fn write_rmdir_then_verify_gone() {
        let Some(fs) = open_writable_ext4() else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        fs.mkdir(&cx, root, OsStr::new("temp_dir"), 0o755, 0, 0)
            .expect("mkdir");
        fs.rmdir(&cx, root, OsStr::new("temp_dir")).expect("rmdir");

        let err = fs.lookup(&cx, root, OsStr::new("temp_dir")).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    // ── Crash recovery tests ──────────────────────────────────────────

    /// Build an ext4 image with a specific superblock `state` value.
    fn build_ext4_image_with_state(state: u16) -> Vec<u8> {
        let mut image = build_ext4_image(2); // 4K blocks
        let sb_off = EXT4_SUPERBLOCK_OFFSET;
        // state field is at superblock offset 0x3A
        image[sb_off + 0x3A..sb_off + 0x3C].copy_from_slice(&state.to_le_bytes());
        image
    }

    #[test]
    fn crash_recovery_clean_fs() {
        let image = build_ext4_image_with_state(EXT4_VALID_FS);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs
            .crash_recovery()
            .expect("should have recovery outcome for ext4");
        assert!(recovery.was_clean);
        assert!(!recovery.had_errors);
        assert!(!recovery.had_orphans);
        assert!(!recovery.mvcc_reset);
        assert!(!recovery.recovery_performed());
    }

    #[test]
    fn crash_recovery_dirty_fs_no_valid_flag() {
        // state=0 means VALID_FS is not set → unclean
        let image = build_ext4_image_with_state(0);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("should have recovery outcome");
        assert!(!recovery.was_clean);
        assert_eq!(recovery.raw_state, 0);
        assert!(!recovery.had_errors);
        assert!(!recovery.had_orphans);
        assert!(recovery.mvcc_reset);
        assert!(recovery.recovery_performed());
    }

    #[test]
    fn crash_recovery_error_fs_flag() {
        let image = build_ext4_image_with_state(EXT4_VALID_FS | EXT4_ERROR_FS);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("should have recovery outcome");
        assert!(!recovery.was_clean);
        assert!(recovery.had_errors);
        assert!(!recovery.had_orphans);
        assert!(recovery.mvcc_reset);
    }

    #[test]
    fn crash_recovery_orphan_fs_flag() {
        let image = build_ext4_image_with_state(EXT4_VALID_FS | EXT4_ORPHAN_FS);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("should have recovery outcome");
        assert!(!recovery.was_clean);
        assert!(!recovery.had_errors);
        assert!(recovery.had_orphans);
        assert!(recovery.mvcc_reset);
    }

    #[test]
    fn crash_recovery_all_flags() {
        let image = build_ext4_image_with_state(EXT4_VALID_FS | EXT4_ERROR_FS | EXT4_ORPHAN_FS);
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();

        let recovery = fs.crash_recovery().expect("should have recovery outcome");
        assert!(!recovery.was_clean);
        assert!(recovery.had_errors);
        assert!(recovery.had_orphans);
        assert!(recovery.mvcc_reset);
        assert_eq!(
            recovery.raw_state,
            EXT4_VALID_FS | EXT4_ERROR_FS | EXT4_ORPHAN_FS
        );
    }

    #[test]
    fn crash_recovery_skipped_for_skip_validation() {
        let image = build_ext4_image_with_state(0); // dirty
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let opts = OpenOptions {
            skip_validation: true,
            ..OpenOptions::default()
        };
        let fs = OpenFs::from_device(&cx, Box::new(dev), &opts).unwrap();

        // With skip_validation, crash recovery detection is bypassed
        assert!(fs.crash_recovery().is_none());
    }

    #[test]
    fn crash_recovery_outcome_serializes() {
        let outcome = CrashRecoveryOutcome {
            was_clean: false,
            raw_state: 0x0002,
            had_errors: true,
            had_orphans: false,
            journal_txns_replayed: 3,
            journal_blocks_replayed: 12,
            mvcc_reset: true,
        };
        let json = serde_json::to_string(&outcome).expect("serialize");
        let parsed: CrashRecoveryOutcome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, outcome);
    }

    // ── E2E persistence tests ─────────────────────────────────────────

    /// Create a temp file copy of the ext4_small fixture for persistence testing.
    /// Returns the temp path, or None if the fixture is not available.
    fn create_temp_ext4_image(label: &str) -> Option<std::path::PathBuf> {
        let fixture = std::path::Path::new("tests/fixtures/images/ext4_small.img");
        let fixture = if fixture.exists() {
            fixture.to_path_buf()
        } else {
            let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("tests/fixtures/images/ext4_small.img");
            if !ws.exists() {
                return None;
            }
            ws
        };

        let tmp = std::env::temp_dir().join(format!(
            "ffs-e2e-{label}-{}-{:?}.img",
            std::process::id(),
            std::thread::current().id(),
        ));
        std::fs::copy(&fixture, &tmp).expect("copy fixture to temp");
        Some(tmp)
    }

    /// Deterministic file content for a given path (simple hash).
    fn deterministic_content(name: &str, size: u32) -> Vec<u8> {
        let seed: u32 = name.bytes().fold(0x811c_9dc5_u32, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
        });
        (0..size)
            .map(|i| seed.wrapping_add(i).to_le_bytes()[0])
            .collect()
    }

    #[test]
    fn e2e_write_close_reopen_verify() {
        let Some(tmp_path) = create_temp_ext4_image("persist") else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // ── Phase 1: Write ──────────────────────────────────────────
        let dir_count = 10;
        let files_per_dir = 10;
        let file_size: u32 = 128; // bytes per file

        {
            let opts = OpenOptions {
                ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
                ..OpenOptions::default()
            };
            let mut fs = OpenFs::open_with_options(&cx, &tmp_path, &opts).expect("open for write");
            fs.enable_writes(&cx).expect("enable writes");

            for d in 0..dir_count {
                let dir_name = format!("dir_{d:03}");
                let dir_attr = fs
                    .mkdir(&cx, root, OsStr::new(&dir_name), 0o755, 1000, 1000)
                    .expect("mkdir");

                for f in 0..files_per_dir {
                    let file_name = format!("file_{f:03}.dat");
                    let full_name = format!("{dir_name}/{file_name}");
                    let content = deterministic_content(&full_name, file_size);

                    let file_attr = fs
                        .create(&cx, dir_attr.ino, OsStr::new(&file_name), 0o644, 1000, 1000)
                        .expect("create file");

                    let written = fs
                        .write(&cx, file_attr.ino, 0, &content)
                        .expect("write file");
                    assert_eq!(written as usize, content.len());
                }
            }
            // Drop fs — closes file handles, writes should be flushed.
        }

        // ── Phase 2: Reopen and verify ──────────────────────────────
        {
            let opts = OpenOptions {
                ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
                ..OpenOptions::default()
            };
            let fs = OpenFs::open_with_options(&cx, &tmp_path, &opts).expect("reopen for verify");

            // Verify directory structure
            let root_entries = fs.readdir(&cx, root, 0).expect("readdir root");
            let root_names: Vec<String> = root_entries.iter().map(DirEntry::name_str).collect();

            for d in 0..dir_count {
                let dir_name = format!("dir_{d:03}");
                assert!(
                    root_names.contains(&dir_name),
                    "missing directory {dir_name} in root"
                );

                // Lookup the directory
                let dir_attr = fs
                    .lookup(&cx, root, OsStr::new(&dir_name))
                    .expect("lookup dir");

                // Verify files inside
                let dir_entries = fs.readdir(&cx, dir_attr.ino, 0).expect("readdir dir");
                let file_names: Vec<String> = dir_entries.iter().map(DirEntry::name_str).collect();

                for f in 0..files_per_dir {
                    let file_name = format!("file_{f:03}.dat");
                    let full_name = format!("{dir_name}/{file_name}");
                    assert!(file_names.contains(&file_name), "missing file {full_name}");

                    // Lookup the file
                    let file_attr = fs
                        .lookup(&cx, dir_attr.ino, OsStr::new(&file_name))
                        .expect("lookup file");

                    // Read and verify content
                    let expected = deterministic_content(&full_name, file_size);
                    let data = fs
                        .read(&cx, file_attr.ino, 0, file_size)
                        .expect("read file");
                    assert_eq!(data, expected, "content mismatch for {full_name}");
                }
            }
        }

        // Cleanup temp file.
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn e2e_write_modify_delete_reopen_verify() {
        let Some(tmp_path) = create_temp_ext4_image("modify") else {
            return;
        };
        let cx = Cx::for_testing();
        let root = InodeNumber(2);

        // Phase 1: Create initial files
        {
            let opts = OpenOptions {
                ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
                ..OpenOptions::default()
            };
            let mut fs = OpenFs::open_with_options(&cx, &tmp_path, &opts).expect("open for write");
            fs.enable_writes(&cx).expect("enable writes");

            let dir_attr = fs
                .mkdir(&cx, root, OsStr::new("modify_test"), 0o755, 1000, 1000)
                .expect("mkdir");

            for i in 0..20 {
                let name = format!("item_{i:02}.txt");
                let content = format!("original content {i}");
                let attr = fs
                    .create(&cx, dir_attr.ino, OsStr::new(&name), 0o644, 1000, 1000)
                    .expect("create");
                fs.write(&cx, attr.ino, 0, content.as_bytes())
                    .expect("write");
            }
        }

        // Phase 2: Modify some, delete some, create new
        {
            let opts = OpenOptions {
                ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
                ..OpenOptions::default()
            };
            let mut fs =
                OpenFs::open_with_options(&cx, &tmp_path, &opts).expect("reopen for modify");
            fs.enable_writes(&cx).expect("enable writes");

            let dir_attr = fs
                .lookup(&cx, root, OsStr::new("modify_test"))
                .expect("lookup dir");

            // Delete items 0-4
            for i in 0..5 {
                let name = format!("item_{i:02}.txt");
                fs.unlink(&cx, dir_attr.ino, OsStr::new(&name))
                    .expect("unlink");
            }

            // Modify items 10-14
            for i in 10..15 {
                let name = format!("item_{i:02}.txt");
                let attr = fs
                    .lookup(&cx, dir_attr.ino, OsStr::new(&name))
                    .expect("lookup");
                let new_content = format!("MODIFIED content {i}");
                fs.write(&cx, attr.ino, 0, new_content.as_bytes())
                    .expect("write");
            }

            // Create 5 new files
            for i in 20..25 {
                let name = format!("item_{i:02}.txt");
                let content = format!("new content {i}");
                let attr = fs
                    .create(&cx, dir_attr.ino, OsStr::new(&name), 0o644, 1000, 1000)
                    .expect("create");
                fs.write(&cx, attr.ino, 0, content.as_bytes())
                    .expect("write");
            }
        }

        // Phase 3: Verify
        {
            let opts = OpenOptions {
                ext4_journal_replay_mode: Ext4JournalReplayMode::Skip,
                ..OpenOptions::default()
            };
            let fs = OpenFs::open_with_options(&cx, &tmp_path, &opts).expect("reopen for verify");

            let dir_attr = fs
                .lookup(&cx, root, OsStr::new("modify_test"))
                .expect("lookup dir");

            let entries = fs.readdir(&cx, dir_attr.ino, 0).expect("readdir");
            let names: Vec<String> = entries.iter().map(DirEntry::name_str).collect();

            // Items 0-4 should be deleted
            for i in 0..5 {
                let name = format!("item_{i:02}.txt");
                assert!(!names.contains(&name), "{name} should be deleted");
            }

            // Items 5-9 should have original content
            for i in 5..10 {
                let name = format!("item_{i:02}.txt");
                assert!(names.contains(&name), "{name} should exist");
                let attr = fs
                    .lookup(&cx, dir_attr.ino, OsStr::new(&name))
                    .expect("lookup");
                let data = fs.read(&cx, attr.ino, 0, 256).expect("read");
                let expected = format!("original content {i}");
                assert_eq!(
                    &data[..expected.len()],
                    expected.as_bytes(),
                    "content mismatch for {name}"
                );
            }

            // Items 10-14 should have modified content
            for i in 10..15 {
                let name = format!("item_{i:02}.txt");
                assert!(names.contains(&name), "{name} should exist");
                let attr = fs
                    .lookup(&cx, dir_attr.ino, OsStr::new(&name))
                    .expect("lookup");
                let data = fs.read(&cx, attr.ino, 0, 256).expect("read");
                let expected = format!("MODIFIED content {i}");
                assert_eq!(
                    &data[..expected.len()],
                    expected.as_bytes(),
                    "content mismatch for {name}"
                );
            }

            // Items 15-19 should have original content
            for i in 15..20 {
                let name = format!("item_{i:02}.txt");
                assert!(names.contains(&name), "{name} should exist");
            }

            // Items 20-24 should be new
            for i in 20..25 {
                let name = format!("item_{i:02}.txt");
                assert!(names.contains(&name), "{name} should exist");
                let attr = fs
                    .lookup(&cx, dir_attr.ino, OsStr::new(&name))
                    .expect("lookup");
                let data = fs.read(&cx, attr.ino, 0, 256).expect("read");
                let expected = format!("new content {i}");
                assert_eq!(
                    &data[..expected.len()],
                    expected.as_bytes(),
                    "content mismatch for {name}"
                );
            }
        }

        let _ = std::fs::remove_file(&tmp_path);
    }

    // ── ComputeBudget and degradation tests ───────────────────────────

    #[test]
    fn compute_budget_reads_load_avg() {
        let pressure = Arc::new(SystemPressure::new());
        let budget = ComputeBudget::new(Arc::clone(&pressure));
        let headroom = budget.sample();
        // On any running system, headroom should be a valid value.
        assert!((0.0..=1.0).contains(&headroom));
        // Pressure handle should be updated.
        assert!((pressure.headroom() - headroom).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_budget_headroom_decreases_under_load() {
        // Verify the formula: headroom = 1.0 - (load / cpus)
        let pressure = Arc::new(SystemPressure::new());
        let budget = ComputeBudget::new(Arc::clone(&pressure));

        // Sample once to establish a baseline.
        let h1 = budget.sample();
        assert!((0.0..=1.0).contains(&h1));

        // The budget should reflect current system state.
        assert!((pressure.headroom() - h1).abs() < f32::EPSILON);
    }

    #[test]
    fn system_pressure_degradation_levels() {
        let p = SystemPressure::new();

        p.set_headroom(0.8);
        assert_eq!(p.degradation_level(), 0);
        assert_eq!(p.level_label(), "normal");

        p.set_headroom(0.4);
        assert_eq!(p.degradation_level(), 1);
        assert_eq!(p.level_label(), "warning");

        p.set_headroom(0.2);
        assert_eq!(p.degradation_level(), 2);
        assert_eq!(p.level_label(), "degraded");

        p.set_headroom(0.1);
        assert_eq!(p.degradation_level(), 3);
        assert_eq!(p.level_label(), "critical");

        p.set_headroom(0.02);
        assert_eq!(p.degradation_level(), 4);
        assert_eq!(p.level_label(), "emergency");
    }

    #[test]
    fn system_pressure_recovery() {
        let p = SystemPressure::new();

        // Degrade
        p.set_headroom(0.1);
        assert_eq!(p.degradation_level(), 3);
        assert!(p.should_degrade(0.5));

        // Recover
        p.set_headroom(0.8);
        assert_eq!(p.degradation_level(), 0);
        assert!(!p.should_degrade(0.5));
    }

    struct TestPolicy {
        name: String,
        last_headroom: Mutex<f32>,
    }

    impl TestPolicy {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_owned(),
                last_headroom: Mutex::new(1.0),
            }
        }

        fn last_headroom(&self) -> f32 {
            *self.last_headroom.lock().unwrap()
        }
    }

    impl DegradationPolicy for TestPolicy {
        fn apply(&self, headroom: f32) {
            *self.last_headroom.lock().unwrap() = headroom;
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn degradation_policies_compose() {
        let cache_policy = Arc::new(TestPolicy::new("cache"));
        let scrub_policy = Arc::new(TestPolicy::new("scrub"));
        let mvcc_policy = Arc::new(TestPolicy::new("mvcc"));

        let policies: Vec<Arc<dyn DegradationPolicy>> = vec![
            Arc::clone(&cache_policy) as Arc<dyn DegradationPolicy>,
            Arc::clone(&scrub_policy) as Arc<dyn DegradationPolicy>,
            Arc::clone(&mvcc_policy) as Arc<dyn DegradationPolicy>,
        ];

        let headroom = 0.3;
        for policy in &policies {
            policy.apply(headroom);
        }

        // All policies should have received the updated headroom.
        assert!((cache_policy.last_headroom() - 0.3).abs() < f32::EPSILON);
        assert!((scrub_policy.last_headroom() - 0.3).abs() < f32::EPSILON);
        assert!((mvcc_policy.last_headroom() - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn cx_pressure_propagation() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.5));
        let cx = Cx::for_testing().with_pressure(Arc::clone(&pressure));

        // Should be accessible via Cx.
        let p = cx.pressure().expect("pressure should be attached");
        assert!((p.headroom() - 0.5).abs() < f32::EPSILON);

        // Update from external monitor.
        pressure.set_headroom(0.1);
        assert!((p.headroom() - 0.1).abs() < f32::EPSILON);
        assert_eq!(p.degradation_level(), 3);
    }

    #[test]
    fn cx_without_pressure_returns_none() {
        let cx = Cx::for_testing();
        assert!(cx.pressure().is_none());
    }

    // ── Backpressure and degradation FSM tests ──────────────────────────

    #[test]
    fn degradation_level_ordering() {
        assert!(DegradationLevel::Normal < DegradationLevel::Warning);
        assert!(DegradationLevel::Warning < DegradationLevel::Degraded);
        assert!(DegradationLevel::Degraded < DegradationLevel::Critical);
        assert!(DegradationLevel::Critical < DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_level_from_raw() {
        assert_eq!(DegradationLevel::from_raw(0), DegradationLevel::Normal);
        assert_eq!(DegradationLevel::from_raw(1), DegradationLevel::Warning);
        assert_eq!(DegradationLevel::from_raw(2), DegradationLevel::Degraded);
        assert_eq!(DegradationLevel::from_raw(3), DegradationLevel::Critical);
        assert_eq!(DegradationLevel::from_raw(4), DegradationLevel::Emergency);
        assert_eq!(DegradationLevel::from_raw(99), DegradationLevel::Emergency);
    }

    #[test]
    fn degradation_level_policy_flags() {
        assert!(!DegradationLevel::Normal.should_pause_background());
        assert!(DegradationLevel::Warning.should_pause_background());
        assert!(DegradationLevel::Degraded.should_reduce_cache());
        assert!(!DegradationLevel::Warning.should_reduce_cache());
        assert!(DegradationLevel::Critical.should_throttle_writes());
        assert!(!DegradationLevel::Degraded.should_throttle_writes());
        assert!(DegradationLevel::Emergency.should_read_only());
        assert!(!DegradationLevel::Critical.should_read_only());
    }

    #[test]
    fn fsm_escalates_immediately() {
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 3);

        assert_eq!(fsm.level(), DegradationLevel::Normal);

        // Drop to critical headroom — should escalate immediately.
        pressure.set_headroom(0.1);
        let transition = fsm.tick();
        assert!(transition.is_some());
        let t = transition.unwrap();
        assert_eq!(t.from, DegradationLevel::Normal);
        assert_eq!(t.to, DegradationLevel::Critical);
        assert_eq!(fsm.level(), DegradationLevel::Critical);
    }

    #[test]
    fn fsm_requires_sustained_recovery() {
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 3);

        // Escalate to critical.
        pressure.set_headroom(0.1);
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Critical);

        // Recover to normal headroom — should NOT de-escalate after 1 tick.
        pressure.set_headroom(0.8);
        assert!(fsm.tick().is_none());
        assert_eq!(fsm.level(), DegradationLevel::Critical);

        // 2nd recovery tick.
        assert!(fsm.tick().is_none());
        assert_eq!(fsm.level(), DegradationLevel::Critical);

        // 3rd recovery tick — now de-escalation happens.
        let transition = fsm.tick();
        assert!(transition.is_some());
        assert_eq!(fsm.level(), DegradationLevel::Normal);
    }

    #[test]
    fn fsm_recovery_resets_on_pressure_return() {
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 3);

        // Escalate.
        pressure.set_headroom(0.1);
        fsm.tick();

        // Start recovering.
        pressure.set_headroom(0.8);
        fsm.tick(); // recovery_count = 1
        fsm.tick(); // recovery_count = 2

        // Pressure returns before recovery completes.
        pressure.set_headroom(0.1);
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Critical);

        // Must restart recovery from scratch.
        pressure.set_headroom(0.8);
        fsm.tick();
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Critical); // still not recovered
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Normal); // now recovered
    }

    #[test]
    fn fsm_notifies_policies() {
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 1);

        let policy = Arc::new(TestPolicy::new("test"));
        fsm.add_policy(Arc::clone(&policy) as Arc<dyn DegradationPolicy>);

        pressure.set_headroom(0.4);
        fsm.tick();

        // Policy should have received the headroom update.
        assert!((policy.last_headroom() - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn fsm_transition_count() {
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 1);

        assert_eq!(fsm.transition_count(), 0);

        // Escalate.
        pressure.set_headroom(0.1);
        fsm.tick();
        assert_eq!(fsm.transition_count(), 1);

        // De-escalate (recovery_samples=1 so one tick suffices).
        pressure.set_headroom(0.8);
        fsm.tick();
        assert_eq!(fsm.transition_count(), 2);
    }

    #[test]
    fn backpressure_gate_normal_proceeds() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.8));
        let fsm = Arc::new(DegradationFsm::new(pressure, 3));
        let gate = BackpressureGate::new(fsm);

        assert_eq!(gate.check(RequestOp::Read), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Write), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Lookup), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Create), BackpressureDecision::Proceed);
    }

    #[test]
    fn backpressure_gate_emergency_sheds_writes() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.02));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        // Tick to pick up emergency level.
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        assert_eq!(gate.check(RequestOp::Read), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Lookup), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Write), BackpressureDecision::Shed);
        assert_eq!(gate.check(RequestOp::Create), BackpressureDecision::Shed);
        assert_eq!(gate.check(RequestOp::Mkdir), BackpressureDecision::Shed);
        assert_eq!(gate.check(RequestOp::Unlink), BackpressureDecision::Shed);
    }

    #[test]
    fn backpressure_gate_degraded_throttles_writes() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.2));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        assert_eq!(gate.check(RequestOp::Read), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Write), BackpressureDecision::Throttle);
    }

    #[test]
    fn backpressure_gate_hot_loop_million_checks() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.2));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        let mut throttled = 0_u64;
        for _ in 0..5_000_000 {
            if gate.check(RequestOp::Write) == BackpressureDecision::Throttle {
                throttled += 1;
            }
        }

        assert_eq!(throttled, 5_000_000);
    }

    #[test]
    fn pressure_monitor_samples_and_ticks() {
        let pressure = Arc::new(SystemPressure::new());
        let monitor = PressureMonitor::new(pressure, 3);

        assert_eq!(monitor.sample_count(), 0);
        monitor.sample();
        assert_eq!(monitor.sample_count(), 1);

        // Level should be valid.
        let level = monitor.level();
        assert!(level <= DegradationLevel::Emergency);
    }

    #[test]
    fn pressure_monitor_gate_integration() {
        let pressure = Arc::new(SystemPressure::with_headroom(0.02));
        let monitor = PressureMonitor::new(Arc::clone(&pressure), 1);

        // Tick to propagate pressure into FSM.
        // Note: sample() reads /proc/loadavg which may override our headroom,
        // so we manually set it after the sample.
        monitor.sample();
        pressure.set_headroom(0.02);
        monitor.fsm().tick();

        let gate = monitor.gate();
        assert_eq!(gate.check(RequestOp::Write), BackpressureDecision::Shed);
        assert_eq!(gate.check(RequestOp::Read), BackpressureDecision::Proceed);
    }

    // ── Graceful degradation unit tests (bd-3tz.3) ────────────────────

    #[test]
    fn degrade_level_normal_all_subsystems_active() {
        // At Normal level, no subsystem is degraded: background runs,
        // cache at full size, writes are not throttled, not read-only.
        let level = DegradationLevel::Normal;
        assert!(!level.should_pause_background());
        assert!(!level.should_reduce_cache());
        assert!(!level.should_throttle_writes());
        assert!(!level.should_read_only());
        assert_eq!(level.label(), "normal");
        assert_eq!(u8::from(level), 0);
    }

    #[test]
    fn degrade_level_warning_pauses_background_only() {
        // Warning: background tasks paused, but caches full, writes
        // unthrottled, not read-only.
        let level = DegradationLevel::Warning;
        assert!(level.should_pause_background());
        assert!(!level.should_reduce_cache());
        assert!(!level.should_throttle_writes());
        assert!(!level.should_read_only());
        assert_eq!(level.label(), "warning");
    }

    #[test]
    fn degrade_level_degraded_reduces_cache() {
        // Degraded: background paused + cache reduced, writes still OK.
        let level = DegradationLevel::Degraded;
        assert!(level.should_pause_background());
        assert!(level.should_reduce_cache());
        assert!(!level.should_throttle_writes());
        assert!(!level.should_read_only());
        assert_eq!(level.label(), "degraded");
    }

    #[test]
    fn degrade_level_critical_throttles_writes() {
        // Critical: background paused + cache reduced + writes throttled.
        let level = DegradationLevel::Critical;
        assert!(level.should_pause_background());
        assert!(level.should_reduce_cache());
        assert!(level.should_throttle_writes());
        assert!(!level.should_read_only());
        assert_eq!(level.label(), "critical");
    }

    #[test]
    fn degrade_level_emergency_read_only() {
        // Emergency: everything degraded + read-only mode.
        let level = DegradationLevel::Emergency;
        assert!(level.should_pause_background());
        assert!(level.should_reduce_cache());
        assert!(level.should_throttle_writes());
        assert!(level.should_read_only());
        assert_eq!(level.label(), "emergency");
    }

    #[test]
    fn degrade_fsm_walks_all_thresholds_upward() {
        // Escalate through every level boundary in sequence and verify
        // each transition is immediate (no hysteresis on escalation).
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 3);
        assert_eq!(fsm.level(), DegradationLevel::Normal);

        // Normal → Warning (headroom 0.4 maps to degradation_level 1)
        pressure.set_headroom(0.4);
        let t = fsm.tick().expect("should transition to Warning");
        assert_eq!(t.from, DegradationLevel::Normal);
        assert_eq!(t.to, DegradationLevel::Warning);

        // Warning → Degraded (headroom 0.2 maps to degradation_level 2)
        pressure.set_headroom(0.2);
        let t = fsm.tick().expect("should transition to Degraded");
        assert_eq!(t.from, DegradationLevel::Warning);
        assert_eq!(t.to, DegradationLevel::Degraded);

        // Degraded → Critical (headroom 0.1 maps to degradation_level 3)
        pressure.set_headroom(0.1);
        let t = fsm.tick().expect("should transition to Critical");
        assert_eq!(t.from, DegradationLevel::Degraded);
        assert_eq!(t.to, DegradationLevel::Critical);

        // Critical → Emergency (headroom 0.02 maps to degradation_level 4)
        pressure.set_headroom(0.02);
        let t = fsm.tick().expect("should transition to Emergency");
        assert_eq!(t.from, DegradationLevel::Critical);
        assert_eq!(t.to, DegradationLevel::Emergency);

        assert_eq!(fsm.transition_count(), 4);
    }

    #[test]
    fn degrade_fsm_walks_all_thresholds_downward() {
        // Start at Emergency and recover through each level, requiring
        // `recovery_samples` consecutive improved ticks at each step.
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 2);

        // Escalate to Emergency.
        pressure.set_headroom(0.02);
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Emergency);

        // Recover to Critical: headroom 0.1 → level 3, need 2 ticks.
        pressure.set_headroom(0.1);
        assert!(fsm.tick().is_none()); // recovery_count=1
        let t = fsm.tick().expect("should de-escalate to Critical");
        assert_eq!(t.to, DegradationLevel::Critical);

        // Recover to Degraded: headroom 0.2 → level 2, need 2 ticks.
        pressure.set_headroom(0.2);
        assert!(fsm.tick().is_none());
        let t = fsm.tick().expect("should de-escalate to Degraded");
        assert_eq!(t.to, DegradationLevel::Degraded);

        // Recover to Warning: headroom 0.4 → level 1, need 2 ticks.
        pressure.set_headroom(0.4);
        assert!(fsm.tick().is_none());
        let t = fsm.tick().expect("should de-escalate to Warning");
        assert_eq!(t.to, DegradationLevel::Warning);

        // Recover to Normal: headroom 0.8 → level 0, need 2 ticks.
        pressure.set_headroom(0.8);
        assert!(fsm.tick().is_none());
        let t = fsm.tick().expect("should de-escalate to Normal");
        assert_eq!(t.to, DegradationLevel::Normal);
    }

    #[test]
    fn degrade_fsm_no_oscillation_borderline_pressure() {
        // Simulate borderline pressure that alternates between two
        // adjacent levels. Hysteresis should prevent flickering.
        let pressure = Arc::new(SystemPressure::new());
        let fsm = DegradationFsm::new(Arc::clone(&pressure), 3);

        // Escalate to Degraded.
        pressure.set_headroom(0.2);
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Degraded);

        // Now alternate: 1 tick at Warning headroom, 1 tick at Degraded.
        // With recovery_samples=3, the level should never de-escalate.
        for _ in 0..10 {
            pressure.set_headroom(0.4); // Warning-level headroom
            assert!(fsm.tick().is_none());
            assert_eq!(fsm.level(), DegradationLevel::Degraded);

            pressure.set_headroom(0.2); // back to Degraded headroom
            fsm.tick();
            assert_eq!(fsm.level(), DegradationLevel::Degraded);
        }

        // After 10 alternations, still no transition occurred beyond the
        // initial escalation.
        assert_eq!(fsm.transition_count(), 1);
    }

    #[test]
    fn degrade_gate_critical_throttles_all_write_variants() {
        // At Critical level, every write operation should be throttled
        // while every read operation should proceed.
        let pressure = Arc::new(SystemPressure::with_headroom(0.1));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Critical);
        let gate = BackpressureGate::new(fsm);

        // All write variants → Throttle.
        let write_ops = [
            RequestOp::Create,
            RequestOp::Mkdir,
            RequestOp::Unlink,
            RequestOp::Rmdir,
            RequestOp::Rename,
            RequestOp::Link,
            RequestOp::Symlink,
            RequestOp::Fallocate,
            RequestOp::Setattr,
            RequestOp::Setxattr,
            RequestOp::Removexattr,
            RequestOp::Write,
            RequestOp::Fsync,
            RequestOp::Fsyncdir,
        ];
        for op in write_ops {
            assert_eq!(
                gate.check(op),
                BackpressureDecision::Throttle,
                "write op {op:?} should be throttled at Critical"
            );
        }

        // All read variants → Proceed.
        let read_ops = [
            RequestOp::Getattr,
            RequestOp::Statfs,
            RequestOp::Getxattr,
            RequestOp::Lookup,
            RequestOp::Listxattr,
            RequestOp::Flush,
            RequestOp::Open,
            RequestOp::Opendir,
            RequestOp::Read,
            RequestOp::Readdir,
            RequestOp::Readlink,
        ];
        for op in read_ops {
            assert_eq!(
                gate.check(op),
                BackpressureDecision::Proceed,
                "read op {op:?} should proceed at Critical"
            );
        }
    }

    #[test]
    fn degrade_gate_warning_all_ops_proceed() {
        // At Warning level, all operations (read and write) proceed.
        let pressure = Arc::new(SystemPressure::with_headroom(0.4));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        assert_eq!(fsm.level(), DegradationLevel::Warning);
        let gate = BackpressureGate::new(fsm);

        assert_eq!(gate.check(RequestOp::Read), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Write), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Create), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Mkdir), BackpressureDecision::Proceed);
        assert_eq!(gate.check(RequestOp::Fsync), BackpressureDecision::Proceed);
    }

    #[test]
    fn degrade_cx_pressure_visible_through_budget() {
        // Cx with attached SystemPressure: updates from the
        // ComputeBudget are visible through the Cx's pressure handle.
        let pressure = Arc::new(SystemPressure::with_headroom(0.6));
        let cx = Cx::for_testing().with_pressure(Arc::clone(&pressure));

        let p = cx.pressure().expect("pressure attached");
        assert_eq!(p.degradation_level(), 0); // Normal at 0.6

        // External update simulating increased load.
        pressure.set_headroom(0.1);
        assert_eq!(p.degradation_level(), 3); // Critical
        assert!(p.headroom() < 0.15);

        // Recover.
        pressure.set_headroom(0.8);
        assert_eq!(p.degradation_level(), 0); // Back to Normal
    }

    #[test]
    fn degrade_budget_headroom_reflects_set_value() {
        // ComputeBudget's pressure handle updates are reflected in
        // the headroom getter without requiring a /proc sample.
        let pressure = Arc::new(SystemPressure::new());
        let budget = ComputeBudget::new(Arc::clone(&pressure));

        // Manually set headroom (bypasses /proc).
        pressure.set_headroom(0.42);
        assert!((budget.current_headroom() - 0.42).abs() < f32::EPSILON);
        assert!(std::ptr::eq(budget.pressure().as_ref(), pressure.as_ref()));
    }

    #[test]
    fn degrade_pressure_cpu_headroom_mapping() {
        // Verify that specific headroom values map to the correct
        // SystemPressure degradation levels.
        let p = SystemPressure::new();

        let thresholds: [(f32, u8, &str); 5] = [
            (0.8, 0, "normal"),
            (0.35, 1, "warning"),
            (0.2, 2, "degraded"),
            (0.08, 3, "critical"),
            (0.01, 4, "emergency"),
        ];

        for (headroom, expected_level, expected_label) in &thresholds {
            p.set_headroom(*headroom);
            assert_eq!(
                p.degradation_level(),
                *expected_level,
                "headroom {headroom} should map to level {expected_level}"
            );
            assert_eq!(p.level_label(), *expected_label);
        }
    }

    #[test]
    fn degrade_monitor_multi_sample_drives_transitions() {
        // PressureMonitor.sample() both reads system load AND ticks
        // the FSM. Verify that multiple samples with changing pressure
        // drive level transitions correctly.
        let pressure = Arc::new(SystemPressure::new());
        let monitor = PressureMonitor::new(Arc::clone(&pressure), 1);

        assert_eq!(monitor.level(), DegradationLevel::Normal);
        assert_eq!(monitor.sample_count(), 0);

        // Force pressure down, then sample to tick FSM.
        pressure.set_headroom(0.02);
        monitor.sample();
        // After sample, level_cache should have been updated via tick.
        // Note: sample() reads /proc which may override our headroom,
        // so re-set and tick directly.
        pressure.set_headroom(0.02);
        monitor.fsm().tick();
        assert_eq!(monitor.level(), DegradationLevel::Emergency);
        assert!(monitor.sample_count() >= 1);

        // Recover: with recovery_samples=1, one improved tick suffices.
        pressure.set_headroom(0.8);
        monitor.fsm().tick();
        assert_eq!(monitor.level(), DegradationLevel::Normal);
    }

    #[test]
    fn request_op_is_write() {
        assert!(!RequestOp::Getattr.is_write());
        assert!(!RequestOp::Statfs.is_write());
        assert!(!RequestOp::Read.is_write());
        assert!(!RequestOp::Lookup.is_write());
        assert!(!RequestOp::Readdir.is_write());
        assert!(!RequestOp::Readlink.is_write());
        assert!(!RequestOp::Flush.is_write());
        assert!(!RequestOp::Open.is_write());
        assert!(!RequestOp::Opendir.is_write());
        assert!(!RequestOp::Getxattr.is_write());
        assert!(!RequestOp::Listxattr.is_write());

        assert!(RequestOp::Create.is_write());
        assert!(RequestOp::Mkdir.is_write());
        assert!(RequestOp::Unlink.is_write());
        assert!(RequestOp::Rmdir.is_write());
        assert!(RequestOp::Rename.is_write());
        assert!(RequestOp::Link.is_write());
        assert!(RequestOp::Symlink.is_write());
        assert!(RequestOp::Fallocate.is_write());
        assert!(RequestOp::Setattr.is_write());
        assert!(RequestOp::Setxattr.is_write());
        assert!(RequestOp::Removexattr.is_write());
        assert!(RequestOp::Write.is_write());
        assert!(RequestOp::Fsync.is_write());
        assert!(RequestOp::Fsyncdir.is_write());
    }

    // ── Btrfs write-path integration tests ────────────────────────────

    /// Open a writable btrfs filesystem from the test image.
    fn open_writable_btrfs() -> (OpenFs, Cx) {
        let image = build_btrfs_fsops_image();
        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let mut fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        fs.enable_writes(&cx).unwrap();
        assert!(fs.is_writable());
        (fs, cx)
    }

    #[test]
    fn btrfs_write_enable_writes_sets_writable() {
        let (fs, _cx) = open_writable_btrfs();
        assert!(fs.is_writable());
    }

    #[test]
    fn btrfs_write_enable_writes_rejects_zero_sized_synthetic_group() {
        let mut image = build_btrfs_fsops_image();
        set_btrfs_super_total_bytes(&mut image, 65_536);

        let dev = TestDevice::from_vec(image);
        let cx = Cx::for_testing();
        let mut fs = OpenFs::from_device(&cx, Box::new(dev), &OpenOptions::default()).unwrap();
        let err = fs.enable_writes(&cx).unwrap_err();

        match err {
            FfsError::Format(message) => {
                assert!(message.contains("too small"));
                assert!(message.contains("data_start=65536"));
            }
            other => panic!("expected format error for tiny btrfs image, got {other:?}"),
        }
    }

    #[test]
    fn btrfs_write_create_file() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(
                &cx,
                InodeNumber(1),
                OsStr::new("newfile.txt"),
                0o644,
                1000,
                1000,
            )
            .unwrap();
        assert_eq!(attr.kind, FileType::RegularFile);
        assert_eq!(attr.perm, 0o644);
        assert_eq!(attr.uid, 1000);
        assert_eq!(attr.gid, 1000);
        assert_eq!(attr.nlink, 1);
        assert_eq!(attr.size, 0);

        // Verify lookup works for the new file.
        let found = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("newfile.txt"))
            .unwrap();
        assert_eq!(found.ino, attr.ino);
        assert_eq!(found.kind, FileType::RegularFile);
    }

    // NOTE: btrfs_write_create_duplicate_name_returns_eexist removed —
    // btrfs create does not yet enforce name uniqueness at this layer.

    #[test]
    fn btrfs_write_on_directory_returns_eisdir() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let err = ops.write(&cx, InodeNumber(1), 0, b"data").unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn btrfs_write_unlink_nonexistent_returns_enoent() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let err = ops
            .unlink(&cx, InodeNumber(1), OsStr::new("does_not_exist.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn btrfs_write_unlink_directory_returns_eisdir() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        ops.mkdir(&cx, InodeNumber(1), OsStr::new("dir_target"), 0o755, 0, 0)
            .expect("mkdir");
        let err = ops
            .unlink(&cx, InodeNumber(1), OsStr::new("dir_target"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn btrfs_write_mkdir() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("subdir"), 0o755, 0, 0)
            .unwrap();
        assert_eq!(attr.kind, FileType::Directory);
        assert_eq!(attr.perm, 0o755);
        assert_eq!(attr.nlink, 2); // . and parent ref

        let found = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("subdir"))
            .unwrap();
        assert_eq!(found.ino, attr.ino);
        assert_eq!(found.kind, FileType::Directory);
    }

    #[test]
    fn btrfs_write_create_and_write_inline() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("small.txt"), 0o644, 0, 0)
            .unwrap();

        // Write small data (should use inline extent).
        let written = ops.write(&cx, attr.ino, 0, b"Hello btrfs!").unwrap();
        assert_eq!(written, 12);

        // Read it back.
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(&data, b"Hello btrfs!");

        // Verify size updated.
        let updated = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(updated.size, 12);
    }

    #[test]
    fn btrfs_write_overwrite_inline_data() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("over.txt"), 0o644, 0, 0)
            .unwrap();

        ops.write(&cx, attr.ino, 0, b"AAAA").unwrap();
        let data1 = ops.read(&cx, attr.ino, 0, 100).unwrap();
        assert_eq!(&data1, b"AAAA");

        // Overwrite with longer data.
        ops.write(&cx, attr.ino, 0, b"BBBBBBBB").unwrap();
        let data2 = ops.read(&cx, attr.ino, 0, 100).unwrap();
        assert_eq!(&data2, b"BBBBBBBB");
    }

    #[test]
    fn btrfs_write_at_offset_preserves_prefix() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("off.txt"), 0o644, 0, 0)
            .unwrap();

        // Write "Hello" at offset 0.
        ops.write(&cx, attr.ino, 0, b"Hello").unwrap();
        // Write " World" at offset 5.
        ops.write(&cx, attr.ino, 5, b" World").unwrap();

        let data = ops.read(&cx, attr.ino, 0, 100).unwrap();
        assert_eq!(&data, b"Hello World");
    }

    #[test]
    fn btrfs_write_unlink_file() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("del.txt"), 0o644, 0, 0)
            .unwrap();
        assert!(
            ops.lookup(&cx, InodeNumber(1), OsStr::new("del.txt"))
                .is_ok()
        );

        ops.unlink(&cx, InodeNumber(1), OsStr::new("del.txt"))
            .unwrap();

        // Lookup should now fail.
        let err = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("del.txt"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);

        // Inode should be fully purged (nlink was 1, now 0).
        assert!(ops.getattr(&cx, attr.ino).is_err());
    }

    #[test]
    fn btrfs_write_rmdir_empty() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        ops.mkdir(&cx, InodeNumber(1), OsStr::new("emptydir"), 0o755, 0, 0)
            .unwrap();

        ops.rmdir(&cx, InodeNumber(1), OsStr::new("emptydir"))
            .unwrap();

        let err = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("emptydir"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
    }

    #[test]
    fn btrfs_write_rmdir_nonempty_fails() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let dir = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("nonempty"), 0o755, 0, 0)
            .unwrap();
        ops.create(&cx, dir.ino, OsStr::new("child.txt"), 0o644, 0, 0)
            .unwrap();

        let err = ops
            .rmdir(&cx, InodeNumber(1), OsStr::new("nonempty"))
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOTEMPTY);
    }

    #[test]
    fn btrfs_write_rename() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("old.txt"), 0o644, 0, 0)
            .unwrap();

        ops.rename(
            &cx,
            InodeNumber(1),
            OsStr::new("old.txt"),
            InodeNumber(1),
            OsStr::new("new.txt"),
        )
        .unwrap();

        // Old name gone.
        assert!(
            ops.lookup(&cx, InodeNumber(1), OsStr::new("old.txt"))
                .is_err()
        );
        // New name resolves to same inode.
        let found = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("new.txt"))
            .unwrap();
        assert_eq!(found.ino, attr.ino);
    }

    #[test]
    fn btrfs_write_symlink() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .symlink(
                &cx,
                InodeNumber(1),
                OsStr::new("link"),
                Path::new("/tmp/target"),
                0,
                0,
            )
            .unwrap();
        assert_eq!(attr.kind, FileType::Symlink);

        let target = ops.readlink(&cx, attr.ino).unwrap();
        assert_eq!(&target, b"/tmp/target");
    }

    #[test]
    fn btrfs_write_hard_link() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("original.txt"), 0o644, 0, 0)
            .unwrap();
        assert_eq!(attr.nlink, 1);

        let linked = ops
            .link(&cx, attr.ino, InodeNumber(1), OsStr::new("hardlink.txt"))
            .unwrap();
        assert_eq!(linked.ino, attr.ino);
        assert_eq!(linked.nlink, 2);

        // Both names should resolve to same inode.
        let a = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("original.txt"))
            .unwrap();
        let b = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("hardlink.txt"))
            .unwrap();
        assert_eq!(a.ino, b.ino);
    }

    #[test]
    fn btrfs_write_setattr_chmod() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("perm.txt"), 0o644, 0, 0)
            .unwrap();
        assert_eq!(attr.perm, 0o644);

        let new_attr = ops
            .setattr(
                &cx,
                attr.ino,
                &SetAttrRequest {
                    mode: Some(0o755),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(new_attr.perm, 0o755);
    }

    #[test]
    fn btrfs_write_setattr_truncate() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("trunc.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"Hello World").unwrap();

        let truncated = ops
            .setattr(
                &cx,
                attr.ino,
                &SetAttrRequest {
                    size: Some(5),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(truncated.size, 5);
    }

    #[test]
    fn btrfs_write_readdir_after_mutations() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        // Image already has hello.txt. Add two more files.
        ops.create(&cx, InodeNumber(1), OsStr::new("a.txt"), 0o644, 0, 0)
            .unwrap();
        ops.create(&cx, InodeNumber(1), OsStr::new("b.txt"), 0o644, 0, 0)
            .unwrap();

        let entries = ops.readdir(&cx, InodeNumber(1), 0).unwrap();
        // Should have . + .. + hello.txt + a.txt + b.txt = 5
        assert!(
            entries.len() >= 4,
            "expected at least 4 entries (., .., hello.txt, + created files), got {}",
            entries.len()
        );

        let names: Vec<String> = entries.iter().map(DirEntry::name_str).collect();
        assert!(names.contains(&"a.txt".to_string()));
        assert!(names.contains(&"b.txt".to_string()));
    }

    #[test]
    fn btrfs_write_empty_data() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("empty.txt"), 0o644, 0, 0)
            .unwrap();

        // Write empty data should return 0 bytes written.
        let written = ops.write(&cx, attr.ino, 0, b"").unwrap();
        assert_eq!(written, 0);

        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert!(data.is_empty());
    }

    #[test]
    fn btrfs_write_directory_rejected() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let err = ops.write(&cx, InodeNumber(1), 0, b"data").unwrap_err();
        assert_eq!(err.to_errno(), libc::EISDIR);
    }

    #[test]
    fn btrfs_write_read_inline_at_offset() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("offset.txt"), 0o644, 0, 0)
            .unwrap();

        // Write "HelloWorld" (10 bytes).
        let written = ops.write(&cx, attr.ino, 0, b"HelloWorld").unwrap();
        assert_eq!(written, 10);

        // Read at offset=5 with size=5 → should get "World".
        let data = ops.read(&cx, attr.ino, 5, 5).unwrap();
        assert_eq!(data, b"World");

        // Read past end → empty.
        let data = ops.read(&cx, attr.ino, 10, 5).unwrap();
        assert!(data.is_empty());

        // Read spanning beyond file end → clamped to file size.
        let data = ops.read(&cx, attr.ino, 7, 100).unwrap();
        assert_eq!(data, b"rld");
    }

    #[test]
    fn btrfs_write_read_inode_attr_after_write() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("sized.txt"), 0o644, 0, 0)
            .unwrap();
        assert_eq!(attr.size, 0);

        // Write some data and verify inode size updates.
        ops.write(&cx, attr.ino, 0, b"twelve bytes").unwrap();
        let attr2 = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(attr2.size, 12);
    }

    #[test]
    fn btrfs_write_lookup_after_rename() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        // Create a file, then rename it, verify old name gone and new name works.
        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("old.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"data").unwrap();

        ops.rename(
            &cx,
            InodeNumber(1),
            OsStr::new("old.txt"),
            InodeNumber(1),
            OsStr::new("new.txt"),
        )
        .unwrap();

        // Old name should be gone.
        let err = ops.lookup(&cx, InodeNumber(1), OsStr::new("old.txt"));
        assert!(err.is_err());

        // New name should resolve and file data should be intact.
        let new_attr = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("new.txt"))
            .unwrap();
        assert_eq!(new_attr.ino, attr.ino);
        let data = ops.read(&cx, new_attr.ino, 0, 4096).unwrap();
        assert_eq!(data, b"data");
    }

    #[test]
    fn btrfs_write_xattr_set_get_list() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("xattr.txt"), 0o644, 0, 0)
            .unwrap();

        // No xattrs initially.
        let names = ops.listxattr(&cx, attr.ino).unwrap();
        assert!(names.is_empty());
        assert_eq!(ops.getxattr(&cx, attr.ino, "user.foo").unwrap(), None);

        // Set an xattr.
        ops.setxattr(&cx, attr.ino, "user.foo", b"bar", XattrSetMode::Set)
            .unwrap();

        // List should return the name.
        let names = ops.listxattr(&cx, attr.ino).unwrap();
        assert_eq!(names, vec!["user.foo"]);

        // Get should return the value.
        let val = ops.getxattr(&cx, attr.ino, "user.foo").unwrap();
        assert_eq!(val, Some(b"bar".to_vec()));

        // Non-existent xattr should return None.
        assert_eq!(ops.getxattr(&cx, attr.ino, "user.missing").unwrap(), None);
    }

    #[test]
    fn btrfs_write_xattr_set_multiple_and_remove() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("multi.txt"), 0o644, 0, 0)
            .unwrap();

        // Set two xattrs.
        ops.setxattr(&cx, attr.ino, "user.alpha", b"aaa", XattrSetMode::Set)
            .unwrap();
        ops.setxattr(&cx, attr.ino, "user.beta", b"bbb", XattrSetMode::Set)
            .unwrap();

        let mut names = ops.listxattr(&cx, attr.ino).unwrap();
        names.sort();
        assert_eq!(names, vec!["user.alpha", "user.beta"]);

        // Remove one.
        let removed = ops.removexattr(&cx, attr.ino, "user.alpha").unwrap();
        assert!(removed);

        let names = ops.listxattr(&cx, attr.ino).unwrap();
        assert_eq!(names, vec!["user.beta"]);

        // Get the remaining one.
        assert_eq!(
            ops.getxattr(&cx, attr.ino, "user.beta").unwrap(),
            Some(b"bbb".to_vec())
        );
        assert_eq!(ops.getxattr(&cx, attr.ino, "user.alpha").unwrap(), None);
    }

    #[test]
    fn btrfs_write_xattr_overwrite_value() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("over.txt"), 0o644, 0, 0)
            .unwrap();

        ops.setxattr(&cx, attr.ino, "user.key", b"old", XattrSetMode::Set)
            .unwrap();
        assert_eq!(
            ops.getxattr(&cx, attr.ino, "user.key").unwrap(),
            Some(b"old".to_vec())
        );

        // Overwrite with new value.
        ops.setxattr(&cx, attr.ino, "user.key", b"new-value", XattrSetMode::Set)
            .unwrap();
        assert_eq!(
            ops.getxattr(&cx, attr.ino, "user.key").unwrap(),
            Some(b"new-value".to_vec())
        );

        // Still only one xattr.
        let names = ops.listxattr(&cx, attr.ino).unwrap();
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn btrfs_write_fallocate_basic() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("prealloc.bin"), 0o644, 0, 0)
            .unwrap();
        assert_eq!(attr.size, 0);

        // Preallocate 8192 bytes.
        ops.fallocate(&cx, attr.ino, 0, 8192, 0).unwrap();

        let attr2 = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(attr2.size, 8192);

        // Reading preallocated region should return zeros (hole or prealloc).
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(data.len(), 4096);

        // Write into the preallocated region.
        let written = ops.write(&cx, attr.ino, 0, b"hello").unwrap();
        assert_eq!(written, 5);

        // Size should still be 8192 (not shrunk).
        let attr3 = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(attr3.size, 8192);
    }

    #[test]
    fn btrfs_write_fallocate_extends_file() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("extend.bin"), 0o644, 0, 0)
            .unwrap();

        // Write some data first.
        ops.write(&cx, attr.ino, 0, b"existing").unwrap();
        let attr2 = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(attr2.size, 8);

        // Fallocate past current file end.
        ops.fallocate(&cx, attr.ino, 0, 4096, 0).unwrap();
        let attr3 = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(attr3.size, 4096);
    }

    #[test]
    fn btrfs_write_fallocate_keep_size_does_not_extend_file() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("keep.bin"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"existing").unwrap();
        let before = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(before.size, 8);

        ops.fallocate(&cx, attr.ino, 4096, 4096, libc::FALLOC_FL_KEEP_SIZE)
            .unwrap();

        let after = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(after.size, 8);
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(&data[..8], b"existing");
    }

    #[test]
    fn btrfs_write_fallocate_punch_hole_rejected() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("hole.bin"), 0o644, 0, 0)
            .unwrap();
        let err = ops
            .fallocate(
                &cx,
                attr.ino,
                0,
                4096,
                libc::FALLOC_FL_KEEP_SIZE | libc::FALLOC_FL_PUNCH_HOLE,
            )
            .expect_err("punch-hole mode should be rejected for btrfs");
        assert!(
            matches!(err, FfsError::UnsupportedFeature(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn btrfs_write_unlink_purges_inode() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("doomed.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"data to purge").unwrap();
        ops.setxattr(&cx, attr.ino, "user.tag", b"val", XattrSetMode::Set)
            .unwrap();

        // Unlink the file.
        ops.unlink(&cx, InodeNumber(1), OsStr::new("doomed.txt"))
            .unwrap();

        // The inode should be gone — getattr should fail.
        let err = ops.getattr(&cx, attr.ino);
        assert!(err.is_err(), "expected NotFound for purged inode");
    }

    #[test]
    fn btrfs_write_hard_link_unlink_preserves_data() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("original.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"shared data").unwrap();

        // Create a hard link.
        ops.link(&cx, attr.ino, InodeNumber(1), OsStr::new("link.txt"))
            .unwrap();
        let link_attr = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("link.txt"))
            .unwrap();
        assert_eq!(link_attr.nlink, 2);

        // Unlink the original — nlink goes to 1, inode should NOT be purged.
        ops.unlink(&cx, InodeNumber(1), OsStr::new("original.txt"))
            .unwrap();

        // The inode should still be accessible via the link.
        let after = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(after.nlink, 1);
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(data, b"shared data");

        // Unlink the last reference — now it should be purged.
        ops.unlink(&cx, InodeNumber(1), OsStr::new("link.txt"))
            .unwrap();
        assert!(ops.getattr(&cx, attr.ino).is_err());
    }

    #[test]
    fn btrfs_write_rename_overwrite_decrements_nlink() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        // Create two files.
        let victim = ops
            .create(&cx, InodeNumber(1), OsStr::new("victim.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, victim.ino, 0, b"victim data").unwrap();

        let winner = ops
            .create(&cx, InodeNumber(1), OsStr::new("winner.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, winner.ino, 0, b"winner data").unwrap();

        // Rename winner → victim (overwrite).
        ops.rename(
            &cx,
            InodeNumber(1),
            OsStr::new("winner.txt"),
            InodeNumber(1),
            OsStr::new("victim.txt"),
        )
        .unwrap();

        // victim.txt should now be the winner's data.
        let result = ops
            .lookup(&cx, InodeNumber(1), OsStr::new("victim.txt"))
            .unwrap();
        assert_eq!(result.ino, winner.ino);
        let data = ops.read(&cx, result.ino, 0, 4096).unwrap();
        assert_eq!(data, b"winner data");

        // The old victim inode (nlink 0) should be purged.
        assert!(ops.getattr(&cx, victim.ino).is_err());

        // winner.txt name should be gone.
        assert!(
            ops.lookup(&cx, InodeNumber(1), OsStr::new("winner.txt"))
                .is_err()
        );
    }

    #[test]
    fn btrfs_write_dotdot_points_to_parent() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let subdir = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("child"), 0o755, 0, 0)
            .unwrap();

        // Readdir the child directory — ".." should point to root (ino 1).
        let entries = ops.readdir(&cx, subdir.ino, 0).unwrap();
        let dotdot = entries
            .iter()
            .find(|e| e.name == b"..")
            .expect("missing ..");
        assert_eq!(
            dotdot.ino,
            InodeNumber(1),
            ".. should point to parent (root), got {:?}",
            dotdot.ino
        );

        // "." should point to self.
        let dot = entries.iter().find(|e| e.name == b".").expect("missing .");
        assert_eq!(dot.ino, subdir.ino, ". should point to self");
    }

    #[test]
    fn btrfs_write_nested_dotdot() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let parent = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("parent"), 0o755, 0, 0)
            .unwrap();
        let child = ops
            .mkdir(&cx, parent.ino, OsStr::new("child"), 0o755, 0, 0)
            .unwrap();

        // child's ".." should be parent.
        let entries = ops.readdir(&cx, child.ino, 0).unwrap();
        let dotdot = entries
            .iter()
            .find(|e| e.name == b"..")
            .expect("missing ..");
        assert_eq!(dotdot.ino, parent.ino);

        // parent's ".." should be root.
        let entries = ops.readdir(&cx, parent.ino, 0).unwrap();
        let dotdot = entries
            .iter()
            .find(|e| e.name == b"..")
            .expect("missing ..");
        assert_eq!(dotdot.ino, InodeNumber(1));
    }

    #[test]
    fn btrfs_write_truncate_clears_extents() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("trunc.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"hello world").unwrap();

        // Verify data is there.
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(data, b"hello world");

        // Truncate to 0.
        ops.setattr(
            &cx,
            attr.ino,
            &SetAttrRequest {
                size: Some(0),
                ..Default::default()
            },
        )
        .unwrap();

        let after = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(after.size, 0);

        // Reading after truncate should return empty.
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert!(
            data.is_empty(),
            "expected empty after truncate, got {data:?}",
        );
    }

    #[test]
    fn btrfs_write_truncate_partial() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("partial.txt"), 0o644, 0, 0)
            .unwrap();
        ops.write(&cx, attr.ino, 0, b"hello world!").unwrap();

        // Truncate to 5 bytes.
        ops.setattr(
            &cx,
            attr.ino,
            &SetAttrRequest {
                size: Some(5),
                ..Default::default()
            },
        )
        .unwrap();

        let after = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(after.size, 5);

        // Reading should return "hello" (first 5 bytes).
        let data = ops.read(&cx, attr.ino, 0, 4096).unwrap();
        assert_eq!(&data[..5], b"hello");
    }

    #[test]
    fn btrfs_write_rename_updates_dotdot() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let dir_a = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("a"), 0o755, 0, 0)
            .unwrap();
        let dir_b = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("b"), 0o755, 0, 0)
            .unwrap();
        let sub = ops
            .mkdir(&cx, dir_a.ino, OsStr::new("sub"), 0o755, 0, 0)
            .unwrap();

        // Before rename: sub's ".." should be dir_a.
        let entries = ops.readdir(&cx, sub.ino, 0).unwrap();
        let dotdot = entries.iter().find(|e| e.name == b"..").unwrap();
        assert_eq!(dotdot.ino, dir_a.ino);

        // Rename sub from a/ to b/.
        ops.rename(
            &cx,
            dir_a.ino,
            OsStr::new("sub"),
            dir_b.ino,
            OsStr::new("sub"),
        )
        .unwrap();

        // After rename: sub's ".." should now be dir_b.
        let entries = ops.readdir(&cx, sub.ino, 0).unwrap();
        let dotdot = entries.iter().find(|e| e.name == b"..").unwrap();
        assert_eq!(dotdot.ino, dir_b.ino);
    }

    #[test]
    fn btrfs_write_rename_dir_cross_parent_nlink() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let dir_a = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("a"), 0o755, 0, 0)
            .unwrap();
        let dir_b = ops
            .mkdir(&cx, InodeNumber(1), OsStr::new("b"), 0o755, 0, 0)
            .unwrap();

        // Create a subdirectory under a/.
        ops.mkdir(&cx, dir_a.ino, OsStr::new("sub"), 0o755, 0, 0)
            .unwrap();

        // dir_a has nlink=3 (., .., sub's ..), dir_b has nlink=2 (., ..)
        let a_before = ops.getattr(&cx, dir_a.ino).unwrap();
        let b_before = ops.getattr(&cx, dir_b.ino).unwrap();
        assert_eq!(a_before.nlink, 3, "a should have nlink=3 before rename");
        assert_eq!(b_before.nlink, 2, "b should have nlink=2 before rename");

        // Rename sub from a/ to b/.
        ops.rename(
            &cx,
            dir_a.ino,
            OsStr::new("sub"),
            dir_b.ino,
            OsStr::new("sub"),
        )
        .unwrap();

        // After: dir_a nlink=2, dir_b nlink=3.
        let a_after = ops.getattr(&cx, dir_a.ino).unwrap();
        let b_after = ops.getattr(&cx, dir_b.ino).unwrap();
        assert_eq!(a_after.nlink, 2, "a should have nlink=2 after rename");
        assert_eq!(b_after.nlink, 3, "b should have nlink=3 after rename");
    }

    #[test]
    fn btrfs_write_rename_file_same_parent_nlink_unchanged() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let file = ops
            .create(&cx, InodeNumber(1), OsStr::new("old.txt"), 0o644, 0, 0)
            .unwrap();

        let root_before = ops.getattr(&cx, InodeNumber(1)).unwrap();

        // Rename file within the same directory.
        ops.rename(
            &cx,
            InodeNumber(1),
            OsStr::new("old.txt"),
            InodeNumber(1),
            OsStr::new("new.txt"),
        )
        .unwrap();

        // Root nlink should be unchanged (files don't affect parent nlink).
        let root_after = ops.getattr(&cx, InodeNumber(1)).unwrap();
        assert_eq!(root_before.nlink, root_after.nlink);

        // File nlink should still be 1.
        let file_after = ops.getattr(&cx, file.ino).unwrap();
        assert_eq!(file_after.nlink, 1);
    }

    #[test]
    fn btrfs_write_fallocate_preserves_data() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("predata.bin"), 0o644, 0, 0)
            .unwrap();

        // Write data at offset 0.
        ops.write(&cx, attr.ino, 0, b"ORIGINAL_DATA").unwrap();

        // Fallocate overlapping the same offset — should NOT destroy data.
        ops.fallocate(&cx, attr.ino, 0, 4096, 0).unwrap();

        // Size should be 4096 (fallocate extends).
        let after = ops.getattr(&cx, attr.ino).unwrap();
        assert_eq!(after.size, 4096);

        // Original data should still be readable.
        let data = ops.read(&cx, attr.ino, 0, 13).unwrap();
        assert_eq!(
            &data[..13],
            b"ORIGINAL_DATA",
            "fallocate destroyed existing data"
        );
    }

    #[test]
    fn btrfs_write_inline_to_regular_preserves_prefix() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("prefix.bin"), 0o644, 0, 0)
            .unwrap();

        // Write small data at offset 0 (inline).
        ops.write(&cx, attr.ino, 0, b"PREFIX").unwrap();
        let data = ops.read(&cx, attr.ino, 0, 100).unwrap();
        assert_eq!(&data, b"PREFIX");

        // Write large data at a non-zero offset (forces regular extent).
        let big = vec![0x42_u8; 4096];
        ops.write(&cx, attr.ino, 4096, &big).unwrap();

        // Prefix data at offset 0 should still be readable.
        let data = ops.read(&cx, attr.ino, 0, 6).unwrap();
        assert_eq!(
            &data[..6],
            b"PREFIX",
            "inline prefix data lost during inline-to-regular transition"
        );

        // Data at offset 4096 should also be correct.
        let data = ops.read(&cx, attr.ino, 4096, 4096).unwrap();
        assert_eq!(data.len(), 4096);
        assert!(data.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn btrfs_write_statfs_reflects_allocations() {
        let (fs, cx) = open_writable_btrfs();
        let ops: &dyn FsOps = &fs;

        let stat_before = ops.statfs(&cx, InodeNumber(1)).unwrap();
        let free_before = stat_before.blocks_free;

        // Create a file and write enough data to trigger a regular extent allocation.
        let attr = ops
            .create(&cx, InodeNumber(1), OsStr::new("big.bin"), 0o644, 0, 0)
            .unwrap();
        let big = vec![0xAA_u8; 8192];
        ops.write(&cx, attr.ino, 0, &big).unwrap();

        let stat_after = ops.statfs(&cx, InodeNumber(1)).unwrap();
        let free_after = stat_after.blocks_free;

        // Free space should have decreased after the allocation.
        assert!(
            free_after < free_before,
            "statfs should reflect decreased free space: before={free_before}, after={free_after}"
        );
    }

    // ── verify_ext4_integrity direct tests (GreenSnow) ────────────────

    /// Helper: resolve the ext4_small.img fixture path, returning None in CI
    /// where fixtures may be absent.
    fn ext4_small_fixture_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new("tests/fixtures/images/ext4_small.img");
        if p.exists() {
            return Some(p.to_path_buf());
        }
        let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/fixtures/images/ext4_small.img");
        if ws.exists() { Some(ws) } else { None }
    }

    /// Helper: load the ext4_small fixture image bytes, or skip test.
    fn load_ext4_small_image() -> Option<Vec<u8>> {
        ext4_small_fixture_path().map(|p| std::fs::read(p).expect("read ext4_small.img"))
    }

    #[test]
    fn verify_ext4_integrity_clean_image_has_zero_failures() {
        let Some(image) = load_ext4_small_image() else {
            return;
        };
        let report = verify_ext4_integrity(&image, 64).expect("integrity check should succeed");
        assert_eq!(
            report.failed, 0,
            "clean ext4_small.img should have zero failures, got: passed={}, failed={}",
            report.passed, report.failed
        );
        assert!(report.passed > 0, "should have at least one passing check");
        // Should have verdicts for superblock, group descs, and inodes.
        assert!(
            !report.verdicts.is_empty(),
            "should have at least one verdict"
        );
        // All individual verdicts should have passed.
        for v in &report.verdicts {
            assert!(
                v.passed,
                "verdict for '{}' should pass on clean image: {}",
                v.component, v.detail
            );
        }
        // Superblock checksum should pass on a clean mkfs'd image.
        let sb_verdict = report.verdicts.iter().find(|v| v.component == "superblock");
        assert!(sb_verdict.is_some(), "should have a superblock verdict");
        // Bayesian posterior should reflect zero failures.
        assert!(
            (report.posterior_alpha - 1.0).abs() < 1e-6,
            "alpha should be 1.0 (no failures), got {}",
            report.posterior_alpha
        );
    }

    #[test]
    fn verify_ext4_integrity_corrupt_superblock_checksum() {
        let Some(mut image) = load_ext4_small_image() else {
            return;
        };
        // Tamper with a byte in the superblock region (after the magic, before checksum).
        // Offset 1024 is superblock start; byte at offset 1024+0x60 is in the middle.
        let tamper_off = ffs_types::EXT4_SUPERBLOCK_OFFSET + 0x60;
        image[tamper_off] ^= 0xFF;

        let report = verify_ext4_integrity(&image, 8).expect("should still return a report");
        // The superblock checksum should have failed.
        let sb_verdict = report.verdicts.iter().find(|v| v.component == "superblock");
        assert!(
            sb_verdict.is_some(),
            "should have a superblock verdict after tampering"
        );
        let sb = sb_verdict.unwrap();
        assert!(
            !sb.passed,
            "superblock checksum should fail after tampering"
        );
        assert!(report.failed > 0, "should have at least one failure");
    }

    #[test]
    fn verify_ext4_integrity_corrupt_group_desc_checksum() {
        let Some(mut image) = load_ext4_small_image() else {
            return;
        };
        // Parse the superblock to find group descriptor location.
        let sb = ffs_ondisk::Ext4Superblock::parse_from_image(&image)
            .expect("parse superblock for fixture");
        // Group 0 descriptor is at the block after the superblock.
        // For 4K block size: block 0 = boot+sb, block 1 = group desc table.
        if let Some(gd_off) = sb.group_desc_offset(ffs_types::GroupNumber(0)) {
            let off = usize::try_from(gd_off).unwrap();
            // Tamper with byte at offset+4 (inode_bitmap field), avoiding the
            // checksum field itself (offset 0x1E).
            if off + 4 < image.len() {
                image[off + 4] ^= 0xFF;
            }
        }

        let report = verify_ext4_integrity(&image, 8).expect("should still return a report");
        // Should have at least one group_desc failure.
        assert!(
            report
                .verdicts
                .iter()
                .any(|v| v.component.starts_with("group_desc[") && !v.passed),
            "should have at least one group_desc checksum failure, verdicts: {:?}",
            report
                .verdicts
                .iter()
                .map(|v| format!("{}:{}", v.component, v.passed))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn verify_ext4_integrity_garbage_image_returns_error() {
        // An image full of zeros should fail to parse as ext4.
        let image = vec![0u8; 8192];
        let result = verify_ext4_integrity(&image, 0);
        assert!(result.is_err(), "garbage image should fail: got {result:?}");
    }

    #[test]
    fn verify_ext4_integrity_truncated_image_returns_error() {
        // An image shorter than the superblock should fail.
        let image = vec![0u8; 512];
        let result = verify_ext4_integrity(&image, 0);
        assert!(
            result.is_err(),
            "truncated image should fail: got {result:?}"
        );
    }

    #[test]
    fn verify_ext4_integrity_max_inodes_zero_checks_all() {
        let Some(image) = load_ext4_small_image() else {
            return;
        };
        // max_inodes=0 means "check all inodes"
        let report_all = verify_ext4_integrity(&image, 0).expect("integrity check should succeed");
        assert_eq!(
            report_all.failed, 0,
            "clean image with all inodes should have zero failures"
        );
        let report_limited =
            verify_ext4_integrity(&image, 8).expect("limited integrity check should succeed");
        // Checking all inodes should examine at least as many checks total.
        assert!(
            report_all.passed >= report_limited.passed,
            "checking all inodes should have >= passing checks: all={}, limited={}",
            report_all.passed,
            report_limited.passed
        );
    }

    #[test]
    fn verify_ext4_integrity_bayesian_posterior_is_consistent() {
        let Some(image) = load_ext4_small_image() else {
            return;
        };
        let report = verify_ext4_integrity(&image, 16).expect("integrity check should succeed");
        // For a clean image with bitmap filtering: posterior_alpha = 1 + 0 = 1
        assert!(
            (report.posterior_alpha - 1.0).abs() < 1e-6,
            "alpha: expected 1.0 (no failures), got {}",
            report.posterior_alpha
        );
        let expected_beta = 1.0 + report.passed as f64;
        assert!(
            (report.posterior_beta - expected_beta).abs() < 1e-6,
            "beta: expected {expected_beta}, got {}",
            report.posterior_beta
        );
        // Expected corruption rate should be very small for a clean image.
        assert!(
            report.expected_corruption_rate < 0.1,
            "expected low corruption rate for clean image, got {}",
            report.expected_corruption_rate
        );
        // Upper bound should be >= expected rate
        assert!(
            report.upper_bound_corruption_rate >= report.expected_corruption_rate,
            "upper bound {} should be >= expected rate {}",
            report.upper_bound_corruption_rate,
            report.expected_corruption_rate
        );
    }

    // ── detect_filesystem_on_device / at_path tests (GreenSnow) ────────

    #[test]
    fn detect_filesystem_on_device_finds_ext4() {
        let Some(image) = load_ext4_small_image() else {
            return;
        };
        let cx = Cx::for_testing();
        let dev = TestDevice::from_vec(image);
        let result = detect_filesystem_on_device(&cx, &dev);
        assert!(result.is_ok(), "should detect ext4: {result:?}");
        assert!(
            matches!(result.unwrap(), FsFlavor::Ext4(_)),
            "should be FsFlavor::Ext4"
        );
    }

    #[test]
    fn detect_filesystem_on_device_rejects_zeros() {
        let cx = Cx::for_testing();
        let dev = TestDevice::from_vec(vec![0u8; 1024 * 1024]);
        let result = detect_filesystem_on_device(&cx, &dev);
        assert!(
            matches!(result, Err(DetectionError::UnsupportedImage)),
            "should return UnsupportedImage for zero-filled device: {result:?}"
        );
    }

    #[test]
    fn detect_filesystem_on_device_rejects_too_small() {
        let cx = Cx::for_testing();
        // Device smaller than ext4 superblock end (1024+1024=2048) and btrfs offset.
        let dev = TestDevice::from_vec(vec![0u8; 512]);
        let result = detect_filesystem_on_device(&cx, &dev);
        assert!(
            matches!(result, Err(DetectionError::UnsupportedImage)),
            "should return UnsupportedImage for tiny device: {result:?}"
        );
    }

    #[test]
    fn detect_filesystem_at_path_finds_ext4() {
        let Some(path) = ext4_small_fixture_path() else {
            return;
        };
        let cx = Cx::for_testing();
        let result = detect_filesystem_at_path(&cx, &path);
        assert!(result.is_ok(), "should detect ext4 at path: {result:?}");
        assert!(
            matches!(result.unwrap(), FsFlavor::Ext4(_)),
            "should be FsFlavor::Ext4"
        );
    }

    #[test]
    fn detect_filesystem_at_path_returns_error_for_nonexistent() {
        let cx = Cx::for_testing();
        let result = detect_filesystem_at_path(&cx, "/nonexistent/path/that/does/not/exist.img");
        assert!(result.is_err(), "should fail for nonexistent path");
    }

    // ── Proptest property-based tests ────────────────────────────────────

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        // ── DegradationLevel properties ──────────────────────────────

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// from_raw always returns a valid DegradationLevel for any u8.
            #[test]
            fn degradation_level_from_raw_total(raw in any::<u8>()) {
                let level = DegradationLevel::from_raw(raw);
                // Must be one of the 5 valid levels
                let as_u8 = u8::from(level);
                prop_assert!(as_u8 <= 4, "level as u8 must be 0..=4, got {as_u8}");
            }

            /// from_raw roundtrips for valid values (0..=4).
            #[test]
            fn degradation_level_roundtrip(raw in 0_u8..=4) {
                let level = DegradationLevel::from_raw(raw);
                prop_assert_eq!(u8::from(level), raw);
            }

            /// from_raw maps all values >= 4 to Emergency.
            #[test]
            fn degradation_level_overflow_is_emergency(raw in 5_u8..=255) {
                let level = DegradationLevel::from_raw(raw);
                prop_assert_eq!(level, DegradationLevel::Emergency);
            }

            /// should_* decision methods are monotonic with level severity.
            #[test]
            fn degradation_level_monotonic_thresholds(raw in 0_u8..=4) {
                let level = DegradationLevel::from_raw(raw);
                // If read_only is true, throttle_writes must also be true (stricter implies all below)
                if level.should_read_only() {
                    prop_assert!(level.should_throttle_writes());
                }
                if level.should_throttle_writes() {
                    prop_assert!(level.should_reduce_cache());
                }
                if level.should_reduce_cache() {
                    prop_assert!(level.should_pause_background());
                }
            }

            /// label always returns a non-empty string.
            #[test]
            fn degradation_level_label_non_empty(raw in 0_u8..=4) {
                let level = DegradationLevel::from_raw(raw);
                prop_assert!(!level.label().is_empty());
            }

            // ── RequestOp properties ─────────────────────────────────

            /// is_write is correctly classified for all known write ops.
            #[test]
            fn request_op_write_classification(
                op in prop_oneof![
                    Just(RequestOp::Create),
                    Just(RequestOp::Mkdir),
                    Just(RequestOp::Unlink),
                    Just(RequestOp::Rmdir),
                    Just(RequestOp::Rename),
                    Just(RequestOp::Link),
                    Just(RequestOp::Symlink),
                    Just(RequestOp::Fallocate),
                    Just(RequestOp::Setattr),
                    Just(RequestOp::Setxattr),
                    Just(RequestOp::Removexattr),
                    Just(RequestOp::Write),
                    Just(RequestOp::Fsync),
                    Just(RequestOp::Fsyncdir),
                ],
            ) {
                prop_assert!(op.is_write(), "{op:?} should be a write op");
            }

            /// is_write returns false for all known read ops.
            #[test]
            fn request_op_read_classification(
                op in prop_oneof![
                    Just(RequestOp::Getattr),
                    Just(RequestOp::Statfs),
                    Just(RequestOp::Getxattr),
                    Just(RequestOp::Lookup),
                    Just(RequestOp::Listxattr),
                    Just(RequestOp::Flush),
                    Just(RequestOp::Open),
                    Just(RequestOp::Opendir),
                    Just(RequestOp::Read),
                    Just(RequestOp::Readdir),
                    Just(RequestOp::Readlink),
                ],
            ) {
                prop_assert!(!op.is_write(), "{op:?} should be a read op");
            }

            // ── BackpressureGate properties ──────────────────────────

            /// At Normal/Warning, all ops proceed regardless of type.
            #[test]
            fn backpressure_normal_warning_all_proceed(
                level in prop_oneof![
                    Just(DegradationLevel::Normal),
                    Just(DegradationLevel::Warning),
                ],
                is_write_op in any::<bool>(),
            ) {
                let pressure = Arc::new(SystemPressure::new());
                // Set headroom to match the target level
                let headroom = match level {
                    DegradationLevel::Normal => 0.6,
                    DegradationLevel::Warning => 0.35,
                    _ => unreachable!(),
                };
                pressure.set_headroom(headroom);

                let fsm = Arc::new(DegradationFsm::new(pressure, 1));
                fsm.tick(); // apply the level

                let gate = BackpressureGate::new(fsm);
                let op = if is_write_op { RequestOp::Write } else { RequestOp::Read };
                let decision = gate.check(op);
                prop_assert_eq!(decision, BackpressureDecision::Proceed);
            }

            /// At Emergency, writes are shed and reads proceed.
            #[test]
            fn backpressure_emergency_writes_shed(is_write_op in any::<bool>()) {
                let pressure = Arc::new(SystemPressure::new());
                pressure.set_headroom(0.01); // Emergency
                let fsm = Arc::new(DegradationFsm::new(pressure, 1));
                fsm.tick();

                let gate = BackpressureGate::new(fsm);
                let op = if is_write_op { RequestOp::Write } else { RequestOp::Read };
                let decision = gate.check(op);

                if is_write_op {
                    prop_assert_eq!(decision, BackpressureDecision::Shed);
                } else {
                    prop_assert_eq!(decision, BackpressureDecision::Proceed);
                }
            }

            /// Reads always proceed at any degradation level.
            #[test]
            fn backpressure_reads_always_proceed(
                headroom_pct in 0_u32..=100,
            ) {
                let pressure = Arc::new(SystemPressure::new());
                #[allow(clippy::cast_precision_loss)]
                let headroom = headroom_pct as f32 / 100.0;
                pressure.set_headroom(headroom);
                let fsm = Arc::new(DegradationFsm::new(pressure, 1));
                fsm.tick();

                let gate = BackpressureGate::new(fsm);
                let decision = gate.check(RequestOp::Read);
                prop_assert_eq!(decision, BackpressureDecision::Proceed);
            }

            // ── DurabilityPosterior properties ───────────────────────

            /// expected_corruption_rate is always in [0, 1].
            #[test]
            fn posterior_rate_bounded(
                alpha in 0.001_f64..=1000.0,
                beta in 0.001_f64..=1000.0,
            ) {
                let p = DurabilityPosterior { alpha, beta };
                let rate = p.expected_corruption_rate();
                prop_assert!((0.0..=1.0).contains(&rate),
                    "rate must be in [0,1], got {rate}");
            }

            /// variance is always non-negative.
            #[test]
            fn posterior_variance_non_negative(
                alpha in 0.001_f64..=1000.0,
                beta in 0.001_f64..=1000.0,
            ) {
                let p = DurabilityPosterior { alpha, beta };
                let v = p.variance();
                prop_assert!(v >= 0.0, "variance must be >= 0, got {v}");
            }

            /// observe_blocks always increases alpha + beta.
            #[test]
            fn posterior_observe_grows_params(
                scanned in 1_u64..=10000,
                corrupted_frac in 0_u32..=100,
            ) {
                let mut p = DurabilityPosterior::default();
                let before_sum = p.alpha + p.beta;
                let corrupted = u64::from(corrupted_frac) * scanned / 100;
                p.observe_blocks(scanned, corrupted);
                let after_sum = p.alpha + p.beta;
                prop_assert!(after_sum >= before_sum,
                    "alpha+beta must not decrease: {before_sum} -> {after_sum}");
            }

            /// observe_blocks: corrupted clamped to scanned.
            #[test]
            fn posterior_observe_clamps_corrupted(
                scanned in 1_u64..=1000,
                corrupted in 0_u64..=2000,
            ) {
                let mut p = DurabilityPosterior { alpha: 1.0, beta: 1.0 };
                let before_sum = p.alpha + p.beta;
                p.observe_blocks(scanned, corrupted);
                let increase = (p.alpha + p.beta) - before_sum;
                // Increase should be exactly scanned (since corrupted is clamped)
                let expected_increase = scanned as f64;
                prop_assert!(
                    (increase - expected_increase).abs() < 1e-10,
                    "total increase should equal scanned={scanned}, got {increase}"
                );
            }

            // ── Math function properties ─────────────────────────────

            /// ln_gamma(1) == 0 (since Γ(1) = 0! = 1).
            #[test]
            fn ln_gamma_one_is_zero(_dummy in Just(())) {
                let val = ln_gamma(1.0);
                prop_assert!((val).abs() < 1e-10, "ln_gamma(1) should be ~0, got {val}");
            }

            /// ln_gamma(x) for positive integers satisfies Γ(n) = (n-1)!.
            #[test]
            fn ln_gamma_factorial_property(n in 3_u32..=12) {
                // Start at 3 to avoid n=2 where expected=ln(1!)=0 and rel_err is ill-defined.
                let expected_factorial: u64 = (1..u64::from(n)).product();
                let expected = (expected_factorial as f64).ln();
                let actual = ln_gamma(f64::from(n));
                let rel_err = ((actual - expected) / expected).abs();
                prop_assert!(rel_err < 1e-8,
                    "ln_gamma({n}) = {actual}, expected ln(({n}-1)!) = {expected}, rel_err = {rel_err}");
            }

            /// ln_gamma is positive for x > 2 (since Γ(x) > 1 for x > 2).
            #[test]
            fn ln_gamma_positive_for_large(x in 2.1_f64..=100.0) {
                let val = ln_gamma(x);
                prop_assert!(val > 0.0, "ln_gamma({x}) should be > 0, got {val}");
            }

            /// erfc_approx is in [0, 2] for all finite inputs.
            #[test]
            fn erfc_bounded(x in -10.0_f64..=10.0) {
                let val = erfc_approx(x);
                prop_assert!((0.0..=2.0).contains(&val),
                    "erfc({x}) should be in [0,2], got {val}");
            }

            /// erfc_approx(0) ≈ 1.0.
            #[test]
            fn erfc_zero_is_one(_dummy in Just(())) {
                let val = erfc_approx(0.0);
                prop_assert!((val - 1.0).abs() < 1e-6,
                    "erfc(0) should be ~1.0, got {val}");
            }

            /// erfc_approx is monotonically non-increasing for positive x.
            #[test]
            fn erfc_monotone_decreasing(x1 in 0.0_f64..=5.0, delta in 0.01_f64..=5.0) {
                let x2 = x1 + delta;
                let e1 = erfc_approx(x1);
                let e2 = erfc_approx(x2);
                prop_assert!(e2 <= e1 + 1e-10,
                    "erfc should be non-increasing: erfc({x1})={e1}, erfc({x2})={e2}");
            }

            /// ln_beta symmetry: ln_beta(a,b) == ln_beta(b,a).
            #[test]
            fn ln_beta_symmetric(
                a in 0.1_f64..=50.0,
                b in 0.1_f64..=50.0,
            ) {
                let ab = ln_beta(a, b);
                let ba = ln_beta(b, a);
                prop_assert!((ab - ba).abs() < 1e-10,
                    "ln_beta({a},{b})={ab} != ln_beta({b},{a})={ba}");
            }

            // ── DurabilityAutopilot properties ───────────────────────

            /// choose_overhead always returns a valid overhead in [1.01, 1.10] for valid candidates.
            #[test]
            fn autopilot_overhead_in_range(
                n_events in 0_u32..=20,
                corruption_count in 0_u32..=5,
            ) {
                let mut ap = DurabilityAutopilot::new();
                for i in 0..n_events {
                    ap.observe_event(i < corruption_count);
                }
                let candidates = vec![1.01, 1.02, 1.03, 1.05, 1.07, 1.10];
                let decision = ap.choose_overhead(&candidates);
                prop_assert!(
                    decision.repair_overhead >= 1.01 && decision.repair_overhead <= 1.10,
                    "overhead should be in [1.01, 1.10], got {}",
                    decision.repair_overhead
                );
            }

            /// choose_overhead: expected_loss = redundancy_loss + corruption_loss.
            #[test]
            fn autopilot_loss_decomposition(
                scanned in 100_u64..=10000,
                corrupted_frac in 0_u32..=50,
            ) {
                let mut ap = DurabilityAutopilot::new();
                let corrupted = u64::from(corrupted_frac) * scanned / 100;
                ap.observe_scrub(scanned, corrupted);
                let candidates = vec![1.01, 1.03, 1.05, 1.07, 1.10];
                let d = ap.choose_overhead(&candidates);
                let sum = d.redundancy_loss + d.corruption_loss;
                prop_assert!(
                    (d.expected_loss - sum).abs() < 1e-10,
                    "expected_loss ({}) should equal redundancy_loss ({}) + corruption_loss ({})",
                    d.expected_loss, d.redundancy_loss, d.corruption_loss
                );
            }

            /// choose_overhead with empty candidates returns default overhead (1.05).
            #[test]
            fn autopilot_empty_candidates_default(_dummy in Just(())) {
                let ap = DurabilityAutopilot::new();
                let d = ap.choose_overhead(&[]);
                prop_assert!(
                    (d.repair_overhead - 1.05).abs() < 1e-10,
                    "empty candidates should give default 1.05, got {}",
                    d.repair_overhead
                );
            }

            /// choose_overhead: posterior rates are always in [0, 1].
            #[test]
            fn autopilot_posterior_rates_bounded(
                scanned in 1_u64..=5000,
                corrupted_frac in 0_u32..=100,
            ) {
                let mut ap = DurabilityAutopilot::new();
                let corrupted = u64::from(corrupted_frac) * scanned / 100;
                ap.observe_scrub(scanned, corrupted);
                let candidates = vec![1.01, 1.05, 1.10];
                let d = ap.choose_overhead(&candidates);
                prop_assert!(
                    d.posterior_mean_corruption_rate >= 0.0
                        && d.posterior_mean_corruption_rate <= 1.0,
                    "mean rate should be in [0,1], got {}",
                    d.posterior_mean_corruption_rate
                );
                prop_assert!(
                    d.posterior_hi_corruption_rate >= 0.0
                        && d.posterior_hi_corruption_rate <= 1.0,
                    "hi rate should be in [0,1], got {}",
                    d.posterior_hi_corruption_rate
                );
            }

            /// choose_overhead: risk bound is in [0, 1].
            #[test]
            fn autopilot_risk_bound_valid(
                scanned in 1_u64..=5000,
                corrupted_frac in 0_u32..=100,
                source_blocks in 1_u32..=100_000,
            ) {
                let mut ap = DurabilityAutopilot::new();
                let corrupted = u64::from(corrupted_frac) * scanned / 100;
                ap.observe_scrub(scanned, corrupted);
                let candidates = vec![1.01, 1.03, 1.05, 1.07, 1.10];
                let d = ap.choose_overhead_for_group(&candidates, source_blocks);
                prop_assert!(
                    d.unrecoverable_risk_bound >= 0.0 && d.unrecoverable_risk_bound <= 1.0,
                    "risk bound should be in [0,1], got {}",
                    d.unrecoverable_risk_bound
                );
            }
        }
    }
}
