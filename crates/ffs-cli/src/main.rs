#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use asupersync::{Budget, Cx};
use clap::{Parser, Subcommand, ValueEnum};
use ffs_block::{BlockDevice, ByteBlockDevice, ByteDevice, FileByteDevice};
use ffs_core::{
    CrashRecoveryOutcome, Ext4JournalReplayMode, FsFlavor, FsOps, OpenFs, OpenOptions,
    detect_filesystem_at_path,
};
use ffs_fuse::MountOptions;
use ffs_harness::ParityReport;
use ffs_ondisk::{
    BtrfsChunkEntry, BtrfsSuperblock, Ext4DirEntry, Ext4Extent, Ext4ExtentHeader, Ext4ExtentIndex,
    Ext4GroupDesc, Ext4ImageReader, Ext4Inode, Ext4Superblock, ExtentTree, map_logical_to_physical,
    parse_dx_root, parse_extent_tree, parse_inode_extent_tree, verify_btrfs_superblock_checksum,
};
use ffs_repair::codec::encode_group;
use ffs_repair::evidence::{EvidenceEventType, EvidenceRecord};
use ffs_repair::recovery::{GroupRecoveryOrchestrator, RecoveryOutcome};
use ffs_repair::scrub::{
    BlockValidator, BtrfsSuperblockValidator, BtrfsTreeBlockValidator, CompositeValidator,
    Ext4SuperblockValidator, ScrubReport, Scrubber, Severity, ZeroCheckValidator,
};
use ffs_repair::storage::{REPAIR_DESC_SLOT_COUNT, RepairGroupLayout, RepairGroupStorage};
use ffs_repair::symbol::RepairGroupDescExt;
use ffs_types::{
    BTRFS_SUPER_INFO_OFFSET, BTRFS_SUPER_INFO_SIZE, BlockNumber, ByteOffset,
    EXT4_SUPERBLOCK_OFFSET, EXT4_SUPERBLOCK_SIZE, GroupNumber, InodeNumber,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env::VarError;
use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, info_span};
use tracing_subscriber::EnvFilter;

// ── Production Cx acquisition ───────────────────────────────────────────────

fn cli_cx() -> Cx {
    Cx::for_request()
}

#[allow(dead_code)]
fn cli_cx_with_timeout_secs(secs: u64) -> Cx {
    Cx::for_request_with_budget(Budget::with_deadline_secs(secs))
}

// ── CLI definition ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LogFormat {
    Human,
    Json,
}

impl LogFormat {
    const ENV_KEY: &'static str = "FFS_LOG_FORMAT";

    fn parse(raw: &str) -> Result<Self> {
        <Self as ValueEnum>::from_str(raw.trim(), true).map_err(|_| {
            anyhow::anyhow!(
                "invalid {key}={raw:?}; expected one of: human, json",
                key = Self::ENV_KEY
            )
        })
    }

    fn from_env() -> Result<Option<Self>> {
        match std::env::var(Self::ENV_KEY) {
            Ok(value) => Ok(Some(Self::parse(&value)?)),
            Err(VarError::NotPresent) => Ok(None),
            Err(VarError::NotUnicode(_)) => {
                bail!("{key} contains non-UTF-8 bytes", key = Self::ENV_KEY)
            }
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Json => "json",
        }
    }
}

fn default_env_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"))
}

fn env_bool(key: &str, default: bool) -> Result<bool> {
    match std::env::var(key) {
        Ok(value) => {
            let value = value.trim();
            if value == "1"
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("yes")
                || value.eq_ignore_ascii_case("on")
            {
                Ok(true)
            } else if value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("no")
                || value.eq_ignore_ascii_case("off")
            {
                Ok(false)
            } else {
                bail!("invalid {key}={value:?}; expected one of: 1,0,true,false,yes,no,on,off")
            }
        }
        Err(VarError::NotPresent) => Ok(default),
        Err(VarError::NotUnicode(_)) => bail!("{key} contains non-UTF-8 bytes"),
    }
}

fn init_logging(log_format_override: Option<LogFormat>) -> Result<LogFormat> {
    let format = log_format_override
        .or(LogFormat::from_env()?)
        .unwrap_or(LogFormat::Human);

    match format {
        LogFormat::Human => tracing_subscriber::fmt()
            .with_env_filter(default_env_filter())
            .with_target(true)
            .with_level(true)
            .compact()
            .try_init()
            .map_err(|err| anyhow::anyhow!("failed to initialize human logger: {err}"))?,
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_env_filter(default_env_filter())
            .with_target(true)
            .with_level(true)
            .try_init()
            .map_err(|err| anyhow::anyhow!("failed to initialize JSON logger: {err}"))?,
    }

    Ok(format)
}

#[derive(Parser)]
#[command(name = "ffs", about = "FrankenFS — memory-safe filesystem toolkit")]
struct Cli {
    /// Log output format (`human` or `json`).
    ///
    /// Precedence: `--log-format` > `FFS_LOG_FORMAT` > `human`.
    #[arg(long, value_enum, global = true)]
    log_format: Option<LogFormat>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect a filesystem image (ext4 or btrfs).
    Inspect {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Show MVCC and EBR version statistics for a filesystem image.
    MvccStats {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Show filesystem information (superblock + optional detailed sections).
    Info {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Include ext4 block group table.
        #[arg(long)]
        groups: bool,
        /// Include MVCC engine status.
        #[arg(long)]
        mvcc: bool,
        /// Include repair subsystem status.
        #[arg(long)]
        repair: bool,
        /// Include journal status.
        #[arg(long)]
        journal: bool,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Dump low-level filesystem metadata.
    Dump {
        #[command(subcommand)]
        command: DumpCommand,
    },
    /// Run offline filesystem checks.
    Fsck {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Attempt repair actions when possible.
        #[arg(long, short = 'r')]
        repair: bool,
        /// Force a full check even if the filesystem appears clean.
        #[arg(long, short = 'f')]
        force: bool,
        /// Emit detailed phase progress.
        #[arg(long, short = 'v')]
        verbose: bool,
        /// Restrict checks to one ext4 group or btrfs block-group index.
        #[arg(long)]
        block_group: Option<u32>,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Trigger manual repair workflows.
    Repair {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Scrub all groups (default behavior is stale-only intent).
        #[arg(long)]
        full_scrub: bool,
        /// Restrict to one ext4 group or btrfs block-group index.
        #[arg(long)]
        block_group: Option<u32>,
        /// Force re-encoding of repair symbols.
        #[arg(long)]
        rebuild_symbols: bool,
        /// Verify only; do not attempt repair writes.
        #[arg(long)]
        verify_only: bool,
        /// Maximum worker threads for repair workflow.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        max_threads: Option<u32>,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Mount a filesystem image via FUSE.
    Mount {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Mountpoint directory.
        mountpoint: PathBuf,
        /// Allow other users to access the mount.
        #[arg(long)]
        allow_other: bool,
        /// Mount read-write (default is read-only).
        #[arg(long)]
        rw: bool,
    },
    /// Run a read-only integrity scan (scrub) on a filesystem image.
    Scrub {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Show feature parity coverage report.
    Parity {
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
    /// Display the repair evidence ledger (JSONL).
    Evidence {
        /// Path to the evidence ledger file.
        ledger: PathBuf,
        /// Output in JSON format (array of records).
        #[arg(long)]
        json: bool,
        /// Filter by event type (e.g., corruption_detected, repair_succeeded).
        #[arg(long)]
        event_type: Option<String>,
        /// Show only the last N records.
        #[arg(long)]
        tail: Option<usize>,
    },
    /// Create a new ext4 filesystem image.
    ///
    /// Wraps `mkfs.ext4` to create a properly formatted ext4 image,
    /// then verifies the result via FrankenFS parsing.
    Mkfs {
        /// Output path for the new image file.
        output: PathBuf,
        /// Image size in megabytes.
        #[arg(long, default_value = "64")]
        size_mb: u64,
        /// Block size in bytes (1024, 2048, or 4096).
        #[arg(long, default_value = "4096")]
        block_size: u32,
        /// Volume label.
        #[arg(long, default_value = "frankenfs")]
        label: String,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum DumpCommand {
    /// Dump superblock fields.
    Superblock {
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
        /// Include a raw hex dump of the on-disk structure.
        #[arg(long)]
        hex: bool,
    },
    /// Dump one ext4 group descriptor.
    Group {
        /// Block group index.
        group: u32,
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
        /// Include a raw hex dump of the on-disk structure.
        #[arg(long)]
        hex: bool,
    },
    /// Dump one ext4 inode.
    Inode {
        /// Inode number.
        inode: u64,
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
        /// Include a raw hex dump of the on-disk structure.
        #[arg(long)]
        hex: bool,
    },
    /// Dump one ext4 inode's full extent tree.
    Extents {
        /// Inode number.
        inode: u64,
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
        /// Include raw hex dumps of extent nodes.
        #[arg(long)]
        hex: bool,
    },
    /// Dump one ext4 directory inode's entries (and htree metadata if indexed).
    Dir {
        /// Inode number.
        inode: u64,
        /// Path to the filesystem image.
        image: PathBuf,
        /// Output in JSON format.
        #[arg(long)]
        json: bool,
        /// Include raw hex dumps of directory data blocks.
        #[arg(long)]
        hex: bool,
    },
}

impl Command {
    const fn name(&self) -> &'static str {
        match self {
            Self::Inspect { .. } => "inspect",
            Self::MvccStats { .. } => "mvcc-stats",
            Self::Info { .. } => "info",
            Self::Dump { .. } => "dump",
            Self::Fsck { .. } => "fsck",
            Self::Repair { .. } => "repair",
            Self::Mount { .. } => "mount",
            Self::Scrub { .. } => "scrub",
            Self::Parity { .. } => "parity",
            Self::Evidence { .. } => "evidence",
            Self::Mkfs { .. } => "mkfs",
        }
    }
}

// ── Serializable outputs ────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(tag = "filesystem", rename_all = "lowercase")]
enum InspectOutput {
    Ext4 {
        block_size: u32,
        inodes_count: u32,
        blocks_count: u64,
        volume_name: String,
        free_blocks_total: u64,
        free_inodes_total: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        free_space_mismatch: Option<FreeSpaceMismatch>,
        #[serde(skip_serializing_if = "Option::is_none")]
        orphan_diagnostics: Option<Ext4OrphanDiagnosticsOutput>,
    },
    Btrfs {
        sectorsize: u32,
        nodesize: u32,
        generation: u64,
        label: String,
    },
}

/// Optional field indicating a mismatch between bitmap and group descriptor counts.
#[derive(Debug, Serialize)]
struct FreeSpaceMismatch {
    gd_free_blocks: u64,
    gd_free_inodes: u64,
}

#[derive(Debug, Serialize)]
struct Ext4OrphanDiagnosticsOutput {
    count: u32,
    sample_inodes: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct MvccStatsOutput {
    block_versions: BlockVersionStatsOutput,
    ebr_versions: EbrVersionStatsOutput,
}

#[derive(Debug, Serialize)]
struct BlockVersionStatsOutput {
    tracked_blocks: usize,
    max_chain_length: usize,
    chains_over_cap: usize,
    chains_over_critical: usize,
    chain_cap: Option<usize>,
    critical_chain_length: Option<usize>,
}

#[derive(Debug, Serialize)]
struct EbrVersionStatsOutput {
    #[serde(rename = "retired_versions")]
    retired: u64,
    #[serde(rename = "reclaimed_versions")]
    reclaimed: u64,
    #[serde(rename = "pending_versions")]
    pending: u64,
}

#[derive(Debug, Serialize)]
struct InfoOutput {
    filesystem: String,
    superblock: SuperblockInfoOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    groups: Option<Vec<Ext4GroupInfoOutput>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mvcc: Option<MvccInfoOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repair: Option<RepairInfoOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    journal: Option<JournalInfoOutput>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum SuperblockInfoOutput {
    Ext4 {
        uuid: String,
        label: String,
        block_size: u32,
        blocks_total: u64,
        blocks_free: u64,
        blocks_reserved: u64,
        inodes_total: u32,
        inodes_free: u32,
        blocks_per_group: u32,
        inodes_per_group: u32,
        groups_count: u32,
        mount_count: u16,
        max_mount_count: u16,
        state_flags: Vec<String>,
        feature_compat: String,
        feature_incompat: String,
        feature_ro_compat: String,
        checksum_type: String,
        checksum_seed: u32,
        mtime: u32,
        wtime: u32,
        lastcheck: u32,
        mkfs_time: u32,
    },
    Btrfs {
        fsid: String,
        label: String,
        sectorsize: u32,
        nodesize: u32,
        generation: u64,
        total_bytes: u64,
        bytes_used: u64,
        bytes_free: u64,
        num_devices: u64,
        csum_type: String,
        compat_flags_hex: String,
        compat_ro_flags_hex: String,
        incompat_flags_hex: String,
    },
}

#[derive(Debug, Serialize)]
struct Ext4GroupInfoOutput {
    group: u32,
    block_start: u64,
    block_end_inclusive: u64,
    free_blocks: u32,
    inode_start: u64,
    inode_end_inclusive: u64,
    free_inodes: u32,
    flags_raw: u16,
    flags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MvccInfoOutput {
    current_commit_seq: u64,
    active_snapshot_count: usize,
    oldest_active_snapshot: Option<u64>,
    total_versioned_blocks: usize,
    max_chain_depth: usize,
    average_chain_depth: String,
    blocks_pending_gc: u64,
    ssi_conflict_count: Option<u64>,
    abort_count: Option<u64>,
}

#[derive(Debug, Serialize)]
struct RepairInfoOutput {
    configured_overhead_ratio: f64,
    metrics_available: bool,
    note: String,
}

#[derive(Debug, Clone, Copy)]
struct InfoCommandOptions {
    sections: InfoSections,
    json: bool,
}

#[derive(Debug, Clone, Copy)]
struct InfoSections(u8);

impl InfoSections {
    const GROUPS: u8 = 1 << 0;
    const MVCC: u8 = 1 << 1;
    const REPAIR: u8 = 1 << 2;
    const JOURNAL: u8 = 1 << 3;

    const fn empty() -> Self {
        Self(0)
    }

    const fn with_groups(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::GROUPS;
        }
        self
    }

    const fn with_mvcc(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::MVCC;
        }
        self
    }

    const fn with_repair(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::REPAIR;
        }
        self
    }

    const fn with_journal(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::JOURNAL;
        }
        self
    }

    const fn groups(self) -> bool {
        (self.0 & Self::GROUPS) != 0
    }

    const fn mvcc(self) -> bool {
        (self.0 & Self::MVCC) != 0
    }

    const fn repair(self) -> bool {
        (self.0 & Self::REPAIR) != 0
    }

    const fn journal(self) -> bool {
        (self.0 & Self::JOURNAL) != 0
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum JournalInfoOutput {
    Ext4 {
        journal_inode: u32,
        external_journal_dev: u32,
        journal_uuid: String,
        journal_size_bytes: Option<u64>,
        replayed_transactions: u32,
        replayed_blocks: u64,
        scanned_blocks: u64,
        descriptor_blocks: u64,
        commit_blocks: u64,
        revoke_blocks: u64,
        skipped_revoked_blocks: u64,
        incomplete_transactions: u64,
    },
    Unsupported {
        reason: String,
    },
}

#[derive(Debug, Serialize)]
struct DumpSuperblockOutput {
    filesystem: String,
    superblock: SuperblockInfoOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_hex: Option<String>,
}

#[derive(Debug, Serialize)]
struct DumpGroupOutput {
    filesystem: String,
    group: u32,
    descriptor: Ext4GroupDesc,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_hex: Option<String>,
}

#[derive(Debug, Serialize)]
struct DumpInodeOutput {
    filesystem: String,
    inode: u64,
    parsed: Ext4Inode,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_hex: Option<String>,
}

#[derive(Debug, Serialize)]
struct DumpExtentOutput {
    filesystem: String,
    inode: u64,
    root_depth: u16,
    nodes: Vec<DumpExtentNodeOutput>,
    flattened_extents: Vec<DumpExtentEntryOutput>,
}

#[derive(Debug, Serialize)]
struct DumpExtentNodeOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    source_block: Option<u64>,
    header: Ext4ExtentHeader,
    #[serde(flatten)]
    node: DumpExtentNodeKindOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_hex: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "node_kind", rename_all = "lowercase")]
enum DumpExtentNodeKindOutput {
    Leaf { extents: Vec<DumpExtentEntryOutput> },
    Index { indexes: Vec<Ext4ExtentIndex> },
}

#[derive(Debug, Serialize)]
struct DumpExtentEntryOutput {
    logical_block: u32,
    physical_start: u64,
    physical_end_inclusive: u64,
    raw_len: u16,
    actual_len: u16,
    initialized: bool,
}

#[derive(Debug, Serialize)]
struct DumpDirOutput {
    filesystem: String,
    inode: u64,
    entries: Vec<DumpDirEntryOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    htree: Option<DumpDxRootOutput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_hex_blocks: Option<Vec<DumpHexBlockOutput>>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DumpDirEntryOutput {
    index: usize,
    inode: u32,
    rec_len: u32,
    file_type: String,
    name: String,
}

#[derive(Debug, Serialize)]
struct DumpDxRootOutput {
    hash_version: u8,
    indirect_levels: u8,
    entries: Vec<DumpDxEntryOutput>,
}

#[derive(Debug, Serialize)]
struct DumpDxEntryOutput {
    hash: u32,
    block: u32,
}

#[derive(Debug, Serialize)]
struct DumpHexBlockOutput {
    logical_block: u32,
    physical_block: u64,
    hex: String,
}

#[derive(Debug, Clone, Copy)]
struct FsckCommandOptions {
    flags: FsckFlags,
    block_group: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct FsckFlags(u8);

impl FsckFlags {
    const REPAIR: u8 = 1 << 0;
    const FORCE: u8 = 1 << 1;
    const VERBOSE: u8 = 1 << 2;
    const JSON: u8 = 1 << 3;

    const fn empty() -> Self {
        Self(0)
    }

    const fn with_repair(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::REPAIR;
        }
        self
    }

    const fn with_force(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::FORCE;
        }
        self
    }

    const fn with_verbose(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::VERBOSE;
        }
        self
    }

    const fn with_json(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::JSON;
        }
        self
    }

    const fn repair(self) -> bool {
        (self.0 & Self::REPAIR) != 0
    }

    const fn force(self) -> bool {
        (self.0 & Self::FORCE) != 0
    }

    const fn verbose(self) -> bool {
        (self.0 & Self::VERBOSE) != 0
    }

    const fn json(self) -> bool {
        (self.0 & Self::JSON) != 0
    }
}

#[derive(Debug, Serialize)]
struct FsckOutput {
    filesystem: String,
    scope: FsckScopeOutput,
    phases: Vec<FsckPhaseOutput>,
    scrub: FsckScrubOutput,
    repair_status: FsckRepairStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    ext4_recovery: Option<Ext4RecoveryOutput>,
    outcome: FsckOutcome,
    exit_code: i32,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FsckScopeOutput {
    Full,
    Ext4BlockGroup {
        group: u32,
        start_block: u64,
        block_count: u64,
    },
    BtrfsBlockGroup {
        group: u32,
        logical_start: u64,
        logical_bytes: u64,
        start_block: u64,
        block_count: u64,
    },
}

#[derive(Debug, Serialize)]
struct FsckPhaseOutput {
    phase: String,
    status: String,
    detail: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FsckRepairStatus {
    NotRequested,
    RequestedPerformed,
    RequestedNotPerformed,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum FsckOutcome {
    Clean,
    ErrorsFound,
}

#[derive(Debug, Serialize)]
struct FsckScrubOutput {
    scanned: u64,
    corrupt: u64,
    error_or_higher: u64,
    io_error: u64,
}

#[derive(Debug, Serialize)]
struct Ext4RecoveryOutput {
    recovery_performed: bool,
    #[serde(flatten)]
    crash_recovery: CrashRecoveryOutcome,
}

#[derive(Debug, Clone, Copy)]
struct RepairCommandOptions {
    flags: RepairFlags,
    block_group: Option<u32>,
    max_threads: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct RepairFlags(u8);

impl RepairFlags {
    const FULL_SCRUB: u8 = 1 << 0;
    const REBUILD_SYMBOLS: u8 = 1 << 1;
    const VERIFY_ONLY: u8 = 1 << 2;
    const JSON: u8 = 1 << 3;

    const fn empty() -> Self {
        Self(0)
    }

    const fn with_full_scrub(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::FULL_SCRUB;
        }
        self
    }

    const fn with_rebuild_symbols(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::REBUILD_SYMBOLS;
        }
        self
    }

    const fn with_verify_only(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::VERIFY_ONLY;
        }
        self
    }

    const fn with_json(mut self, enabled: bool) -> Self {
        if enabled {
            self.0 |= Self::JSON;
        }
        self
    }

    const fn full_scrub(self) -> bool {
        (self.0 & Self::FULL_SCRUB) != 0
    }

    const fn rebuild_symbols(self) -> bool {
        (self.0 & Self::REBUILD_SYMBOLS) != 0
    }

    const fn verify_only(self) -> bool {
        (self.0 & Self::VERIFY_ONLY) != 0
    }

    const fn json(self) -> bool {
        (self.0 & Self::JSON) != 0
    }
}

#[derive(Debug, Serialize)]
struct RepairOutput {
    filesystem: String,
    scope: RepairScopeOutput,
    action: RepairActionOutput,
    scrub: RepairScrubOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    ext4_recovery: Option<Ext4RecoveryOutput>,
    exit_code: i32,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RepairScopeOutput {
    Full,
    Ext4BlockGroup {
        group: u32,
        start_block: u64,
        block_count: u64,
    },
    BtrfsBlockGroup {
        group: u32,
        logical_start: u64,
        logical_bytes: u64,
        start_block: u64,
        block_count: u64,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum RepairActionOutput {
    VerifyOnly,
    RepairRequested,
    NoCorruptionDetected,
}

#[derive(Debug, Serialize)]
struct RepairScrubOutput {
    scanned: u64,
    corrupt: u64,
    error_or_higher: u64,
    io_error: u64,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<()> {
    let cli = Cli::parse();
    let log_format = init_logging(cli.log_format)?;
    let command_name = cli.command.name();
    let run_span = info_span!(
        target: "ffs::cli",
        "command",
        command = command_name,
        log_format = log_format.as_str()
    );
    let _run_guard = run_span.enter();
    let started = Instant::now();

    info!(
        target: "ffs::cli",
        command = command_name,
        log_format = log_format.as_str(),
        "command_start"
    );

    let result = match cli.command {
        Command::Inspect { image, json } => inspect(&image, json),
        Command::MvccStats { image, json } => mvcc_stats_cmd(&image, json),
        Command::Info {
            image,
            groups,
            mvcc,
            repair,
            journal,
            json,
        } => info_cmd(
            &image,
            InfoCommandOptions {
                sections: InfoSections::empty()
                    .with_groups(groups)
                    .with_mvcc(mvcc)
                    .with_repair(repair)
                    .with_journal(journal),
                json,
            },
        ),
        Command::Dump { command } => dump_cmd(&command),
        Command::Fsck {
            image,
            repair,
            force,
            verbose,
            block_group,
            json,
        } => fsck_cmd(
            &image,
            FsckCommandOptions {
                flags: FsckFlags::empty()
                    .with_repair(repair)
                    .with_force(force)
                    .with_verbose(verbose)
                    .with_json(json),
                block_group,
            },
        ),
        Command::Repair {
            image,
            full_scrub,
            block_group,
            rebuild_symbols,
            verify_only,
            max_threads,
            json,
        } => repair_cmd(
            &image,
            RepairCommandOptions {
                flags: RepairFlags::empty()
                    .with_full_scrub(full_scrub)
                    .with_rebuild_symbols(rebuild_symbols)
                    .with_verify_only(verify_only)
                    .with_json(json),
                block_group,
                max_threads,
            },
        ),
        Command::Mount {
            image,
            mountpoint,
            allow_other,
            rw,
        } => mount_cmd(&image, &mountpoint, allow_other, rw),
        Command::Scrub { image, json } => scrub_cmd(&image, json),
        Command::Parity { json } => parity(json),
        Command::Evidence {
            ledger,
            json,
            event_type,
            tail,
        } => evidence_cmd(&ledger, json, event_type.as_deref(), tail),
        Command::Mkfs {
            output,
            size_mb,
            block_size,
            label,
            json,
        } => mkfs_cmd(&output, size_mb, block_size, &label, json),
    };

    let duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
    if let Err(err) = &result {
        error!(
            target: "ffs::cli",
            command = command_name,
            duration_us,
            error = %err,
            "command_failed"
        );
    } else {
        info!(
            target: "ffs::cli",
            command = command_name,
            duration_us,
            "command_succeeded"
        );
    }

    result
}

fn inspect(path: &PathBuf, json: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::inspect",
        "inspect",
        image = %path.display(),
        output_json = json
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::inspect", "inspect_start");

    let cx = cli_cx();
    let open_opts = OpenOptions {
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
        ..OpenOptions::default()
    };
    let flavor = detect_filesystem_at_path(&cx, path)
        .with_context(|| format!("failed to detect ext4/btrfs metadata in {}", path.display()))?;

    let output = match &flavor {
        FsFlavor::Ext4(sb) => inspect_ext4_output(
            &cx,
            path,
            &open_opts,
            sb.block_size,
            sb.inodes_count,
            sb.blocks_count,
            &sb.volume_name,
        )?,
        FsFlavor::Btrfs(sb) => InspectOutput::Btrfs {
            sectorsize: sb.sectorsize,
            nodesize: sb.nodesize,
            generation: sb.generation,
            label: sb.label.clone(),
        },
    };

    info!(
        target: "ffs::cli::inspect",
        filesystem = match &flavor {
            FsFlavor::Ext4(_) => "ext4",
            FsFlavor::Btrfs(_) => "btrfs",
        },
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "inspect_detected_filesystem"
    );

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize output")?
        );
    } else {
        println!("FrankenFS Inspector");
        match output {
            InspectOutput::Ext4 {
                block_size,
                inodes_count,
                blocks_count,
                volume_name,
                free_blocks_total,
                free_inodes_total,
                free_space_mismatch,
                orphan_diagnostics,
            } => {
                println!("filesystem: ext4");
                println!("block_size: {block_size}");
                println!("inodes_count: {inodes_count}");
                println!("blocks_count: {blocks_count}");
                println!("volume_name: {volume_name}");
                println!("free_blocks: {free_blocks_total}");
                println!("free_inodes: {free_inodes_total}");
                if let Some(mismatch) = free_space_mismatch {
                    println!(
                        "WARNING: mismatch with group descriptors (gd_free_blocks={}, gd_free_inodes={})",
                        mismatch.gd_free_blocks, mismatch.gd_free_inodes
                    );
                }
                if let Some(orphan_diag) = orphan_diagnostics {
                    println!(
                        "orphans: count={} sample_inodes={:?}",
                        orphan_diag.count, orphan_diag.sample_inodes
                    );
                }
            }
            InspectOutput::Btrfs {
                sectorsize,
                nodesize,
                generation,
                label,
            } => {
                println!("filesystem: btrfs");
                println!("sectorsize: {sectorsize}");
                println!("nodesize: {nodesize}");
                println!("generation: {generation}");
                println!("label: {label}");
            }
        }
    }

    info!(
        target: "ffs::cli::inspect",
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "inspect_complete"
    );

    Ok(())
}

fn mvcc_stats_cmd(path: &PathBuf, json: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::mvcc_stats",
        "mvcc_stats",
        image = %path.display(),
        output_json = json
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::mvcc_stats", "mvcc_stats_start");

    let cx = cli_cx();
    let open_opts = OpenOptions {
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
        ..OpenOptions::default()
    };
    let open_fs = OpenFs::open_with_options(&cx, path, &open_opts)
        .with_context(|| format!("failed to open image: {}", path.display()))?;

    let mvcc_guard = open_fs.mvcc_store().read();
    let block_stats = mvcc_guard.block_version_stats();
    let ebr_stats = mvcc_guard.ebr_stats();
    drop(mvcc_guard);

    let output = MvccStatsOutput {
        block_versions: BlockVersionStatsOutput {
            tracked_blocks: block_stats.tracked_blocks,
            max_chain_length: block_stats.max_chain_length,
            chains_over_cap: block_stats.chains_over_cap,
            chains_over_critical: block_stats.chains_over_critical,
            chain_cap: block_stats.chain_cap,
            critical_chain_length: block_stats.critical_chain_length,
        },
        ebr_versions: EbrVersionStatsOutput {
            retired: ebr_stats.retired_versions,
            reclaimed: ebr_stats.reclaimed_versions,
            pending: ebr_stats.pending_versions(),
        },
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize mvcc stats output")?
        );
    } else {
        println!("FrankenFS MVCC/EBR Stats");
        println!("block_versions:");
        println!("  tracked_blocks: {}", output.block_versions.tracked_blocks);
        println!(
            "  max_chain_length: {}",
            output.block_versions.max_chain_length
        );
        println!(
            "  chains_over_cap: {}",
            output.block_versions.chains_over_cap
        );
        println!(
            "  chains_over_critical: {}",
            output.block_versions.chains_over_critical
        );
        println!("  chain_cap: {:?}", output.block_versions.chain_cap);
        println!(
            "  critical_chain_length: {:?}",
            output.block_versions.critical_chain_length
        );
        println!("ebr_versions:");
        println!("  retired_versions: {}", output.ebr_versions.retired);
        println!("  reclaimed_versions: {}", output.ebr_versions.reclaimed);
        println!("  pending_versions: {}", output.ebr_versions.pending);
    }

    info!(
        target: "ffs::cli::mvcc_stats",
        tracked_blocks = output.block_versions.tracked_blocks,
        max_chain_length = output.block_versions.max_chain_length,
        pending_versions = output.ebr_versions.pending,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "mvcc_stats_complete"
    );

    Ok(())
}

fn info_cmd(path: &PathBuf, options: InfoCommandOptions) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::info",
        "info",
        image = %path.display(),
        include_groups = options.sections.groups(),
        include_mvcc = options.sections.mvcc(),
        include_repair = options.sections.repair(),
        include_journal = options.sections.journal(),
        output_json = options.json
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::info", "info_start");

    let cx = cli_cx();
    let open_opts = OpenOptions {
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
        ..OpenOptions::default()
    };
    let open_fs = OpenFs::open_with_options(&cx, path, &open_opts)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let output = build_info_output(path, &cx, &open_fs, options)?;

    print_info_output(options.json, &output)?;

    info!(
        target: "ffs::cli::info",
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        limitations = output.limitations.len(),
        "info_complete"
    );

    Ok(())
}

fn build_info_output(
    path: &PathBuf,
    cx: &Cx,
    open_fs: &OpenFs,
    options: InfoCommandOptions,
) -> Result<InfoOutput> {
    let mut limitations = Vec::new();

    let groups_output = if options.sections.groups() {
        match &open_fs.flavor {
            FsFlavor::Ext4(sb) => Some(build_ext4_group_info(path, sb)?),
            FsFlavor::Btrfs(_) => {
                limitations
                    .push("--groups is currently implemented for ext4 images only".to_owned());
                None
            }
        }
    } else {
        None
    };

    let mvcc_output = if options.sections.mvcc() {
        limitations.push(
            "MVCC status currently exposes snapshot/version-chain metrics; transaction/SSI counters are not yet wired into this command"
                .to_owned(),
        );
        Some(build_mvcc_info(open_fs))
    } else {
        None
    };

    let repair_output = if options.sections.repair() {
        limitations.push(
            "Repair status currently reports static policy defaults; live symbol coverage/scrub counters are not yet exposed through CLI info"
                .to_owned(),
        );
        Some(RepairInfoOutput {
            configured_overhead_ratio: 1.05,
            metrics_available: false,
            note: "live repair metrics are not yet exposed through OpenFs".to_owned(),
        })
    } else {
        None
    };

    let journal_output = if options.sections.journal() {
        match &open_fs.flavor {
            FsFlavor::Ext4(sb) => Some(build_ext4_journal_info(cx, open_fs, sb)),
            FsFlavor::Btrfs(_) => Some(JournalInfoOutput::Unsupported {
                reason: "btrfs journal status is not applicable (btrfs is copy-on-write)"
                    .to_owned(),
            }),
        }
    } else {
        None
    };

    Ok(InfoOutput {
        filesystem: filesystem_name(&open_fs.flavor).to_owned(),
        superblock: superblock_info_for(&open_fs.flavor),
        groups: groups_output,
        mvcc: mvcc_output,
        repair: repair_output,
        journal: journal_output,
        limitations,
    })
}

fn filesystem_name(flavor: &FsFlavor) -> &'static str {
    match flavor {
        FsFlavor::Ext4(_) => "ext4",
        FsFlavor::Btrfs(_) => "btrfs",
    }
}

fn superblock_info_for(flavor: &FsFlavor) -> SuperblockInfoOutput {
    match flavor {
        FsFlavor::Ext4(sb) => {
            let checksum_type = if sb.checksum_type == 1 {
                "crc32c".to_owned()
            } else {
                format!("unknown({})", sb.checksum_type)
            };

            SuperblockInfoOutput::Ext4 {
                uuid: format_uuid(&sb.uuid),
                label: sb.volume_name.clone(),
                block_size: sb.block_size,
                blocks_total: sb.blocks_count,
                blocks_free: sb.free_blocks_count,
                blocks_reserved: sb.reserved_blocks_count,
                inodes_total: sb.inodes_count,
                inodes_free: sb.free_inodes_count,
                blocks_per_group: sb.blocks_per_group,
                inodes_per_group: sb.inodes_per_group,
                groups_count: sb.groups_count(),
                mount_count: sb.mnt_count,
                max_mount_count: sb.max_mnt_count,
                state_flags: ext4_state_flag_names(sb.state),
                feature_compat: format!("{}", sb.feature_compat),
                feature_incompat: format!("{}", sb.feature_incompat),
                feature_ro_compat: format!("{}", sb.feature_ro_compat),
                checksum_type,
                checksum_seed: sb.csum_seed(),
                mtime: sb.mtime,
                wtime: sb.wtime,
                lastcheck: sb.lastcheck,
                mkfs_time: sb.mkfs_time,
            }
        }
        FsFlavor::Btrfs(sb) => SuperblockInfoOutput::Btrfs {
            fsid: format_uuid(&sb.fsid),
            label: sb.label.clone(),
            sectorsize: sb.sectorsize,
            nodesize: sb.nodesize,
            generation: sb.generation,
            total_bytes: sb.total_bytes,
            bytes_used: sb.bytes_used,
            bytes_free: sb.total_bytes.saturating_sub(sb.bytes_used),
            num_devices: sb.num_devices,
            csum_type: btrfs_checksum_type_name(sb.csum_type),
            compat_flags_hex: format!("0x{:016x}", sb.compat_flags),
            compat_ro_flags_hex: format!("0x{:016x}", sb.compat_ro_flags),
            incompat_flags_hex: format!("0x{:016x}", sb.incompat_flags),
        },
    }
}

fn build_ext4_group_info(path: &PathBuf, sb: &Ext4Superblock) -> Result<Vec<Ext4GroupInfoOutput>> {
    let groups_count = sb.groups_count();
    let mut groups = Vec::with_capacity(usize::try_from(groups_count).unwrap_or(0));
    let inodes_total = u64::from(sb.inodes_count);

    for group in 0..groups_count {
        let (desc, _raw_desc) = read_ext4_group_desc_from_path(path, sb, group)?;

        let block_start = u64::from(sb.first_data_block)
            .saturating_add(u64::from(group).saturating_mul(u64::from(sb.blocks_per_group)));
        let block_end_exclusive = block_start
            .saturating_add(u64::from(sb.blocks_per_group))
            .min(sb.blocks_count);

        let inode_start = u64::from(group)
            .saturating_mul(u64::from(sb.inodes_per_group))
            .saturating_add(1);
        let inode_end_exclusive = inode_start
            .saturating_add(u64::from(sb.inodes_per_group))
            .min(inodes_total.saturating_add(1));

        groups.push(Ext4GroupInfoOutput {
            group,
            block_start,
            block_end_inclusive: block_end_exclusive.saturating_sub(1),
            free_blocks: desc.free_blocks_count,
            inode_start,
            inode_end_inclusive: inode_end_exclusive.saturating_sub(1),
            free_inodes: desc.free_inodes_count,
            flags_raw: desc.flags,
            flags: ext4_group_flag_names(desc.flags),
        });
    }

    Ok(groups)
}

fn build_mvcc_info(open_fs: &OpenFs) -> MvccInfoOutput {
    let mvcc_guard = open_fs.mvcc_store().read();
    let current_commit_seq = mvcc_guard.current_snapshot().high.0;
    let active_snapshot_count = mvcc_guard.active_snapshot_count();
    let oldest_active_snapshot = mvcc_guard.watermark().map(|seq| seq.0);
    let block_stats = mvcc_guard.block_version_stats();
    let total_versioned_entries = mvcc_guard.version_count();
    let ebr_stats = mvcc_guard.ebr_stats();
    drop(mvcc_guard);

    MvccInfoOutput {
        current_commit_seq,
        active_snapshot_count,
        oldest_active_snapshot,
        total_versioned_blocks: block_stats.tracked_blocks,
        max_chain_depth: block_stats.max_chain_length,
        average_chain_depth: format_ratio_thousandths(
            total_versioned_entries,
            block_stats.tracked_blocks,
        ),
        blocks_pending_gc: ebr_stats.pending_versions(),
        ssi_conflict_count: None,
        abort_count: None,
    }
}

fn build_ext4_journal_info(cx: &Cx, open_fs: &OpenFs, sb: &Ext4Superblock) -> JournalInfoOutput {
    let journal_size_bytes = if sb.journal_inum == 0 {
        None
    } else {
        open_fs
            .read_inode(cx, InodeNumber(u64::from(sb.journal_inum)))
            .ok()
            .map(|inode| inode.size)
    };

    let replayed_transactions = open_fs.ext4_journal_replay().map_or(0_u32, |replay| {
        u32::try_from(replay.committed_sequences.len()).unwrap_or(u32::MAX)
    });
    let replayed_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.replayed_blocks);
    let scanned_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.scanned_blocks);
    let descriptor_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.descriptor_blocks);
    let commit_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.commit_blocks);
    let revoke_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.revoke_blocks);
    let skipped_revoked_blocks = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.skipped_revoked_blocks);
    let incomplete_transactions = open_fs
        .ext4_journal_replay()
        .map_or(0_u64, |replay| replay.stats.incomplete_transactions);

    JournalInfoOutput::Ext4 {
        journal_inode: sb.journal_inum,
        external_journal_dev: sb.journal_dev,
        journal_uuid: format_uuid(&sb.journal_uuid),
        journal_size_bytes,
        replayed_transactions,
        replayed_blocks,
        scanned_blocks,
        descriptor_blocks,
        commit_blocks,
        revoke_blocks,
        skipped_revoked_blocks,
        incomplete_transactions,
    }
}

fn print_info_output(json: bool, output: &InfoOutput) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(output).context("serialize info output")?
        );
        return Ok(());
    }

    println!("FrankenFS Filesystem Info");
    println!("filesystem: {}", output.filesystem);
    print_superblock_info(&output.superblock);

    if let Some(groups) = &output.groups {
        println!();
        println!("groups: {}", groups.len());
        for group in groups {
            println!(
                "  group={} blocks={}..{} free_blocks={} inodes={}..{} free_inodes={} flags={}",
                group.group,
                group.block_start,
                group.block_end_inclusive,
                group.free_blocks,
                group.inode_start,
                group.inode_end_inclusive,
                group.free_inodes,
                group.flags.join("|")
            );
        }
    }

    if let Some(mvcc) = &output.mvcc {
        println!();
        println!("mvcc:");
        println!("  current_commit_seq: {}", mvcc.current_commit_seq);
        println!("  active_snapshot_count: {}", mvcc.active_snapshot_count);
        println!(
            "  oldest_active_snapshot: {}",
            mvcc.oldest_active_snapshot
                .map_or_else(|| "none".to_owned(), |value| value.to_string())
        );
        println!("  total_versioned_blocks: {}", mvcc.total_versioned_blocks);
        println!("  max_chain_depth: {}", mvcc.max_chain_depth);
        println!("  average_chain_depth: {}", mvcc.average_chain_depth);
        println!("  blocks_pending_gc: {}", mvcc.blocks_pending_gc);
    }

    if let Some(repair) = &output.repair {
        println!();
        println!("repair:");
        println!(
            "  configured_overhead_ratio: {:.3}",
            repair.configured_overhead_ratio
        );
        println!("  metrics_available: {}", repair.metrics_available);
        println!("  note: {}", repair.note);
    }

    if let Some(journal) = &output.journal {
        println!();
        match journal {
            JournalInfoOutput::Ext4 {
                journal_inode,
                external_journal_dev,
                journal_uuid,
                journal_size_bytes,
                replayed_transactions,
                replayed_blocks,
                scanned_blocks,
                descriptor_blocks,
                commit_blocks,
                revoke_blocks,
                skipped_revoked_blocks,
                incomplete_transactions,
            } => {
                println!("journal:");
                println!("  inode: {journal_inode}");
                println!("  external_dev: {external_journal_dev}");
                println!("  uuid: {journal_uuid}");
                println!(
                    "  size_bytes: {}",
                    journal_size_bytes
                        .map_or_else(|| "unknown".to_owned(), |value| value.to_string())
                );
                println!("  replayed_transactions: {replayed_transactions}");
                println!("  replayed_blocks: {replayed_blocks}");
                println!("  scanned_blocks: {scanned_blocks}");
                println!("  descriptor_blocks: {descriptor_blocks}");
                println!("  commit_blocks: {commit_blocks}");
                println!("  revoke_blocks: {revoke_blocks}");
                println!("  skipped_revoked_blocks: {skipped_revoked_blocks}");
                println!("  incomplete_transactions: {incomplete_transactions}");
            }
            JournalInfoOutput::Unsupported { reason } => {
                println!("journal: unsupported ({reason})");
            }
        }
    }

    if !output.limitations.is_empty() {
        println!();
        println!("limitations:");
        for limitation in &output.limitations {
            println!("  - {limitation}");
        }
    }

    Ok(())
}

fn print_superblock_info(superblock: &SuperblockInfoOutput) {
    match superblock {
        SuperblockInfoOutput::Ext4 {
            uuid,
            label,
            block_size,
            blocks_total,
            blocks_free,
            blocks_reserved,
            inodes_total,
            inodes_free,
            blocks_per_group,
            inodes_per_group,
            groups_count,
            mount_count,
            max_mount_count,
            state_flags,
            feature_compat,
            feature_incompat,
            feature_ro_compat,
            checksum_type,
            checksum_seed,
            mtime,
            wtime,
            lastcheck,
            mkfs_time,
        } => {
            println!("superblock (ext4):");
            println!("  uuid: {uuid}");
            println!("  label: {label}");
            println!("  block_size: {block_size}");
            println!("  blocks_total: {blocks_total}");
            println!("  blocks_free: {blocks_free}");
            println!("  blocks_reserved: {blocks_reserved}");
            println!("  inodes_total: {inodes_total}");
            println!("  inodes_free: {inodes_free}");
            println!("  blocks_per_group: {blocks_per_group}");
            println!("  inodes_per_group: {inodes_per_group}");
            println!("  groups_count: {groups_count}");
            println!("  mount_count: {mount_count}");
            println!("  max_mount_count: {max_mount_count}");
            println!("  state_flags: {}", state_flags.join("|"));
            println!("  feature_compat: {feature_compat}");
            println!("  feature_incompat: {feature_incompat}");
            println!("  feature_ro_compat: {feature_ro_compat}");
            println!("  checksum_type: {checksum_type}");
            println!("  checksum_seed: {checksum_seed}");
            println!("  mtime: {mtime}");
            println!("  wtime: {wtime}");
            println!("  lastcheck: {lastcheck}");
            println!("  mkfs_time: {mkfs_time}");
        }
        SuperblockInfoOutput::Btrfs {
            fsid,
            label,
            sectorsize,
            nodesize,
            generation,
            total_bytes,
            bytes_used,
            bytes_free,
            num_devices,
            csum_type,
            compat_flags_hex,
            compat_ro_flags_hex,
            incompat_flags_hex,
        } => {
            println!("superblock (btrfs):");
            println!("  fsid: {fsid}");
            println!("  label: {label}");
            println!("  sectorsize: {sectorsize}");
            println!("  nodesize: {nodesize}");
            println!("  generation: {generation}");
            println!("  total_bytes: {total_bytes}");
            println!("  bytes_used: {bytes_used}");
            println!("  bytes_free: {bytes_free}");
            println!("  num_devices: {num_devices}");
            println!("  checksum_type: {csum_type}");
            println!("  compat_flags: {compat_flags_hex}");
            println!("  compat_ro_flags: {compat_ro_flags_hex}");
            println!("  incompat_flags: {incompat_flags_hex}");
        }
    }
}

fn ext4_state_flag_names(state: u16) -> Vec<String> {
    const EXT4_VALID_FS: u16 = 0x0001;
    const EXT4_ERROR_FS: u16 = 0x0002;
    const EXT4_ORPHAN_FS: u16 = 0x0004;

    let mut names = Vec::new();
    if (state & EXT4_VALID_FS) != 0 {
        names.push("VALID_FS".to_owned());
    }
    if (state & EXT4_ERROR_FS) != 0 {
        names.push("ERROR_FS".to_owned());
    }
    if (state & EXT4_ORPHAN_FS) != 0 {
        names.push("ORPHAN_FS".to_owned());
    }

    let known = EXT4_VALID_FS | EXT4_ERROR_FS | EXT4_ORPHAN_FS;
    let unknown = state & !known;
    if unknown != 0 {
        names.push(format!("UNKNOWN(0x{unknown:04X})"));
    }
    if names.is_empty() {
        names.push("NONE".to_owned());
    }
    names
}

fn ext4_appears_clean_state(state: u16) -> bool {
    const EXT4_VALID_FS: u16 = 0x0001;
    const EXT4_ERROR_FS: u16 = 0x0002;
    const EXT4_ORPHAN_FS: u16 = 0x0004;

    (state & EXT4_VALID_FS) != 0 && (state & EXT4_ERROR_FS) == 0 && (state & EXT4_ORPHAN_FS) == 0
}

fn ext4_group_flag_names(flags: u16) -> Vec<String> {
    const EXT4_BG_INODE_UNINIT: u16 = 0x0001;
    const EXT4_BG_BLOCK_UNINIT: u16 = 0x0002;
    const EXT4_BG_INODE_ZEROED: u16 = 0x0004;

    let mut names = Vec::new();
    if (flags & EXT4_BG_INODE_UNINIT) != 0 {
        names.push("INODE_UNINIT".to_owned());
    }
    if (flags & EXT4_BG_BLOCK_UNINIT) != 0 {
        names.push("BLOCK_UNINIT".to_owned());
    }
    if (flags & EXT4_BG_INODE_ZEROED) != 0 {
        names.push("INODE_ZEROED".to_owned());
    }

    let known = EXT4_BG_INODE_UNINIT | EXT4_BG_BLOCK_UNINIT | EXT4_BG_INODE_ZEROED;
    let unknown = flags & !known;
    if unknown != 0 {
        names.push(format!("UNKNOWN(0x{unknown:04X})"));
    }
    if names.is_empty() {
        names.push("NONE".to_owned());
    }
    names
}

fn btrfs_checksum_type_name(csum_type: u16) -> String {
    if csum_type == ffs_types::BTRFS_CSUM_TYPE_CRC32C {
        "crc32c".to_owned()
    } else {
        format!("unknown({csum_type})")
    }
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

fn format_ratio_thousandths(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        return "0.000".to_owned();
    }

    let numerator_u128 = u128::try_from(numerator).unwrap_or(u128::MAX);
    let denominator_u128 = u128::try_from(denominator).unwrap_or(1);
    let milli = numerator_u128
        .saturating_mul(1000)
        .saturating_div(denominator_u128);
    let whole = milli / 1000;
    let fractional = milli % 1000;
    format!("{whole}.{fractional:03}")
}

fn dump_cmd(command: &DumpCommand) -> Result<()> {
    match command {
        DumpCommand::Superblock { image, json, hex } => dump_superblock_cmd(image, *json, *hex),
        DumpCommand::Group {
            group,
            image,
            json,
            hex,
        } => dump_group_cmd(*group, image, *json, *hex),
        DumpCommand::Inode {
            inode,
            image,
            json,
            hex,
        } => dump_inode_cmd(*inode, image, *json, *hex),
        DumpCommand::Extents {
            inode,
            image,
            json,
            hex,
        } => dump_extents_cmd(*inode, image, *json, *hex),
        DumpCommand::Dir {
            inode,
            image,
            json,
            hex,
        } => dump_dir_cmd(*inode, image, *json, *hex),
    }
}

fn dump_superblock_cmd(path: &PathBuf, json: bool, hex: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::dump::superblock",
        "dump_superblock",
        image = %path.display(),
        output_json = json,
        include_hex = hex
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::dump::superblock", "dump_superblock_start");

    let cx = cli_cx();
    let flavor = detect_filesystem_at_path(&cx, path)
        .with_context(|| format!("failed to detect ext4/btrfs metadata in {}", path.display()))?;

    let raw_hex = if hex {
        let (offset, len, label) = match &flavor {
            FsFlavor::Ext4(_) => (
                EXT4_SUPERBLOCK_OFFSET,
                EXT4_SUPERBLOCK_SIZE,
                "ext4 superblock",
            ),
            FsFlavor::Btrfs(_) => (
                BTRFS_SUPER_INFO_OFFSET,
                BTRFS_SUPER_INFO_SIZE,
                "btrfs superblock",
            ),
        };
        let bytes = read_file_region(path, offset, len, label)?;
        Some(bytes_to_hex_dump(&bytes))
    } else {
        None
    };

    let output = DumpSuperblockOutput {
        filesystem: filesystem_name(&flavor).to_owned(),
        superblock: superblock_info_for(&flavor),
        raw_hex,
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize dump superblock output")?
        );
    } else {
        println!("FrankenFS Dump: superblock");
        println!("filesystem: {}", output.filesystem);
        print_superblock_info(&output.superblock);
        if let Some(raw_hex) = &output.raw_hex {
            println!();
            println!("raw_hex:");
            println!("{raw_hex}");
        }
    }

    info!(
        target: "ffs::cli::dump::superblock",
        filesystem = output.filesystem,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "dump_superblock_complete"
    );

    Ok(())
}

fn dump_group_cmd(group: u32, path: &PathBuf, json: bool, hex: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::dump::group",
        "dump_group",
        image = %path.display(),
        group,
        output_json = json,
        include_hex = hex
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::dump::group", "dump_group_start");

    let sb = ext4_superblock_for_action(path, "dump group")?;
    let (desc, raw_desc) = read_ext4_group_desc_from_path(path, &sb, group)?;

    let raw_hex = if hex {
        Some(bytes_to_hex_dump(&raw_desc))
    } else {
        None
    };

    let output = DumpGroupOutput {
        filesystem: "ext4".to_owned(),
        group,
        descriptor: desc,
        raw_hex,
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize dump group output")?
        );
    } else {
        println!("FrankenFS Dump: group");
        println!("filesystem: {}", output.filesystem);
        println!("group: {}", output.group);
        println!("descriptor:");
        println!("  block_bitmap: {}", output.descriptor.block_bitmap);
        println!("  inode_bitmap: {}", output.descriptor.inode_bitmap);
        println!("  inode_table: {}", output.descriptor.inode_table);
        println!(
            "  free_blocks_count: {}",
            output.descriptor.free_blocks_count
        );
        println!(
            "  free_inodes_count: {}",
            output.descriptor.free_inodes_count
        );
        println!("  used_dirs_count: {}", output.descriptor.used_dirs_count);
        println!("  itable_unused: {}", output.descriptor.itable_unused);
        println!("  flags: 0x{:04X}", output.descriptor.flags);
        println!("  checksum: 0x{:04X}", output.descriptor.checksum);

        if let Some(raw_hex) = &output.raw_hex {
            println!();
            println!("raw_hex:");
            println!("{raw_hex}");
        }
    }

    info!(
        target: "ffs::cli::dump::group",
        group,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "dump_group_complete"
    );

    Ok(())
}

fn ext4_superblock_for_action(path: &PathBuf, action: &str) -> Result<Ext4Superblock> {
    let cx = cli_cx();
    let flavor = detect_filesystem_at_path(&cx, path)
        .with_context(|| format!("failed to detect ext4/btrfs metadata in {}", path.display()))?;
    match flavor {
        FsFlavor::Ext4(sb) => Ok(sb),
        FsFlavor::Btrfs(_) => bail!("{action} currently supports ext4 images only"),
    }
}

fn dump_inode_cmd(inode: u64, path: &PathBuf, json: bool, hex: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::dump::inode",
        "dump_inode",
        image = %path.display(),
        inode,
        output_json = json,
        include_hex = hex
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::dump::inode", "dump_inode_start");

    let sb = ext4_superblock_for_action(path, "dump inode")?;
    let inode_number = InodeNumber(inode);
    let (parsed, raw_inode) = read_ext4_inode_from_path(path, &sb, inode_number)
        .with_context(|| format!("failed to read inode {inode}"))?;

    let raw_hex = if hex {
        Some(bytes_to_hex_dump(&raw_inode))
    } else {
        None
    };

    let output = DumpInodeOutput {
        filesystem: "ext4".to_owned(),
        inode,
        parsed,
        raw_hex,
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize dump inode output")?
        );
    } else {
        println!("FrankenFS Dump: inode");
        println!("filesystem: {}", output.filesystem);
        println!("inode: {}", output.inode);
        println!("mode: 0x{:04X}", output.parsed.mode);
        println!("uid: {}", output.parsed.uid);
        println!("gid: {}", output.parsed.gid);
        println!("size: {}", output.parsed.size);
        println!("links_count: {}", output.parsed.links_count);
        println!("blocks: {}", output.parsed.blocks);
        println!("flags: 0x{:08X}", output.parsed.flags);
        println!("generation: {}", output.parsed.generation);
        println!("file_acl: {}", output.parsed.file_acl);
        println!("atime: {}", output.parsed.atime);
        println!("ctime: {}", output.parsed.ctime);
        println!("mtime: {}", output.parsed.mtime);
        println!("dtime: {}", output.parsed.dtime);
        println!("extra_isize: {}", output.parsed.extra_isize);
        println!("checksum: 0x{:08X}", output.parsed.checksum);
        println!("projid: {}", output.parsed.projid);

        if let Some(raw_hex) = &output.raw_hex {
            println!();
            println!("raw_hex:");
            println!("{raw_hex}");
        }
    }

    info!(
        target: "ffs::cli::dump::inode",
        inode,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "dump_inode_complete"
    );

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn dump_extents_cmd(inode: u64, path: &PathBuf, json: bool, hex: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::dump::extents",
        "dump_extents",
        image = %path.display(),
        inode,
        output_json = json,
        include_hex = hex
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::dump::extents", "dump_extents_start");

    let (image, reader) = load_ext4_reader(path, "dump extents")?;
    let inode_number = InodeNumber(inode);
    let parsed_inode = reader
        .read_inode(&image, inode_number)
        .with_context(|| format!("failed to read inode {inode}"))?;
    let (root_header, _) = parse_inode_extent_tree(&parsed_inode)
        .with_context(|| format!("inode {inode} is not extent-backed"))?;

    let mut nodes = Vec::new();
    collect_extent_nodes(
        &reader,
        &image,
        None,
        &parsed_inode.extent_bytes,
        root_header.depth,
        hex,
        &mut nodes,
    )?;

    let flattened_extents = reader
        .collect_extents(&image, &parsed_inode)
        .with_context(|| format!("failed to collect extents for inode {inode}"))?
        .into_iter()
        .map(dump_extent_entry)
        .collect();

    let output = DumpExtentOutput {
        filesystem: "ext4".to_owned(),
        inode,
        root_depth: root_header.depth,
        nodes,
        flattened_extents,
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize dump extents output")?
        );
    } else {
        println!("FrankenFS Dump: extents");
        println!("filesystem: {}", output.filesystem);
        println!("inode: {}", output.inode);
        println!("root_depth: {}", output.root_depth);
        println!("nodes: {}", output.nodes.len());

        for node in &output.nodes {
            let source = node
                .source_block
                .map_or_else(|| "inode_root".to_owned(), |block| block.to_string());
            println!(
                "  node source={} depth={} entries={} max_entries={} generation={}",
                source,
                node.header.depth,
                node.header.entries,
                node.header.max_entries,
                node.header.generation
            );
            match &node.node {
                DumpExtentNodeKindOutput::Leaf { extents } => {
                    for extent in extents {
                        println!(
                            "    leaf logical={} physical={}..{} len={} initialized={}",
                            extent.logical_block,
                            extent.physical_start,
                            extent.physical_end_inclusive,
                            extent.actual_len,
                            extent.initialized
                        );
                    }
                }
                DumpExtentNodeKindOutput::Index { indexes } => {
                    for index in indexes {
                        println!(
                            "    index logical={} child_block={}",
                            index.logical_block, index.leaf_block
                        );
                    }
                }
            }
            if let Some(raw_hex) = &node.raw_hex {
                println!("    raw_hex:");
                println!("{raw_hex}");
            }
        }
    }

    info!(
        target: "ffs::cli::dump::extents",
        inode,
        nodes = output.nodes.len(),
        flattened_extents = output.flattened_extents.len(),
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "dump_extents_complete"
    );

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn dump_dir_cmd(inode: u64, path: &PathBuf, json: bool, hex: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::dump::dir",
        "dump_dir",
        image = %path.display(),
        inode,
        output_json = json,
        include_hex = hex
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::dump::dir", "dump_dir_start");

    let (image, reader) = load_ext4_reader(path, "dump dir")?;
    let inode_number = InodeNumber(inode);
    let parsed_inode = reader
        .read_inode(&image, inode_number)
        .with_context(|| format!("failed to read inode {inode}"))?;
    let entries = reader
        .read_dir(&image, &parsed_inode)
        .with_context(|| format!("failed to read directory entries for inode {inode}"))?;

    let htree = match reader
        .resolve_extent(&image, &parsed_inode, 0)
        .with_context(|| format!("failed to resolve first directory block for inode {inode}"))?
    {
        Some(physical_block) => {
            let block = reader
                .read_block(&image, BlockNumber(physical_block))
                .with_context(|| format!("failed to read directory block {physical_block}"))?;
            parse_dx_root(block).ok().map(|root| DumpDxRootOutput {
                hash_version: root.hash_version,
                indirect_levels: root.indirect_levels,
                entries: root
                    .entries
                    .iter()
                    .map(|entry| DumpDxEntryOutput {
                        hash: entry.hash,
                        block: entry.block,
                    })
                    .collect(),
            })
        }
        None => None,
    };

    let raw_hex_blocks = if hex {
        Some(read_ext4_directory_hex_blocks(
            &image,
            &reader,
            &parsed_inode,
        )?)
    } else {
        None
    };

    let mut limitations = Vec::new();
    limitations.push(
        "directory entry byte offsets are not exposed by parser APIs; `index` preserves on-disk iteration order"
            .to_owned(),
    );
    if htree.is_none() {
        limitations.push(
            "htree metadata is only shown for indexed directories with a parseable dx root"
                .to_owned(),
        );
    }

    let output = DumpDirOutput {
        filesystem: "ext4".to_owned(),
        inode,
        entries: entries
            .iter()
            .enumerate()
            .map(|(index, entry)| dump_dir_entry(index, entry))
            .collect(),
        htree,
        raw_hex_blocks,
        limitations,
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("serialize dump dir output")?
        );
    } else {
        println!("FrankenFS Dump: dir");
        println!("filesystem: {}", output.filesystem);
        println!("inode: {}", output.inode);
        println!("entries: {}", output.entries.len());
        for entry in &output.entries {
            println!(
                "  index={} inode={} rec_len={} type={} name={}",
                entry.index, entry.inode, entry.rec_len, entry.file_type, entry.name
            );
        }

        if let Some(htree) = &output.htree {
            println!();
            println!("htree:");
            println!("  hash_version: {}", htree.hash_version);
            println!("  indirect_levels: {}", htree.indirect_levels);
            for entry in &htree.entries {
                println!("  entry hash=0x{:08X} block={}", entry.hash, entry.block);
            }
        }

        if let Some(raw_hex_blocks) = &output.raw_hex_blocks {
            println!();
            println!("raw_hex_blocks: {}", raw_hex_blocks.len());
            for block in raw_hex_blocks {
                println!(
                    "  logical_block={} physical_block={}",
                    block.logical_block, block.physical_block
                );
                println!("{}", block.hex);
            }
        }

        if !output.limitations.is_empty() {
            println!();
            println!("limitations:");
            for limitation in &output.limitations {
                println!("  - {limitation}");
            }
        }
    }

    info!(
        target: "ffs::cli::dump::dir",
        inode,
        entries = output.entries.len(),
        has_htree = output.htree.is_some(),
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "dump_dir_complete"
    );

    Ok(())
}

fn load_ext4_reader(path: &PathBuf, action: &str) -> Result<(Vec<u8>, Ext4ImageReader)> {
    let cx = cli_cx();
    let flavor = detect_filesystem_at_path(&cx, path)
        .with_context(|| format!("failed to detect ext4/btrfs metadata in {}", path.display()))?;
    if !matches!(flavor, FsFlavor::Ext4(_)) {
        bail!("{action} currently supports ext4 images only");
    }

    let image = std::fs::read(path)
        .with_context(|| format!("failed to read filesystem image: {}", path.display()))?;
    let reader = Ext4ImageReader::new(&image).context("failed to parse ext4 superblock")?;
    Ok((image, reader))
}

fn read_ext4_group_desc_from_path(
    path: &PathBuf,
    sb: &Ext4Superblock,
    group: u32,
) -> Result<(Ext4GroupDesc, Vec<u8>)> {
    let group_number = GroupNumber(group);
    let offset_u64 = sb
        .group_desc_offset(group_number)
        .ok_or_else(|| anyhow::anyhow!("group descriptor offset overflow for group {group}"))?;
    let offset = usize::try_from(offset_u64)
        .with_context(|| format!("group descriptor offset does not fit usize for group {group}"))?;
    let desc_size = sb.group_desc_size();
    let raw_desc = read_file_region(
        path,
        offset,
        usize::from(desc_size),
        "ext4 group descriptor",
    )
    .with_context(|| format!("failed to read ext4 group descriptor {group}"))?;
    let desc = Ext4GroupDesc::parse_from_bytes(&raw_desc, desc_size)
        .with_context(|| format!("failed to parse ext4 group descriptor {group}"))?;
    Ok((desc, raw_desc))
}

fn read_ext4_inode_from_path(
    path: &PathBuf,
    sb: &Ext4Superblock,
    inode: InodeNumber,
) -> Result<(Ext4Inode, Vec<u8>)> {
    let location = sb
        .locate_inode(inode)
        .with_context(|| format!("failed to locate inode {}", inode.0))?;
    let (group_desc, _raw_group_desc) = read_ext4_group_desc_from_path(path, sb, location.group.0)
        .with_context(|| format!("failed to read group descriptor {}", location.group.0))?;
    let inode_offset = sb
        .inode_device_offset(&location, group_desc.inode_table)
        .with_context(|| format!("failed to compute inode offset for inode {}", inode.0))?;
    let offset = usize::try_from(inode_offset)
        .with_context(|| format!("inode offset does not fit usize for inode {}", inode.0))?;
    let inode_size = usize::from(sb.inode_size);
    let raw_inode = read_file_region(path, offset, inode_size, "ext4 inode")
        .with_context(|| format!("failed to read raw inode bytes for inode {}", inode.0))?;
    let parsed = Ext4Inode::parse_from_bytes(&raw_inode)
        .with_context(|| format!("failed to parse inode {}", inode.0))?;
    Ok((parsed, raw_inode))
}

fn read_file_region(path: &PathBuf, offset: usize, len: usize, label: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open filesystem image: {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("failed to stat filesystem image: {}", path.display()))?
        .len();
    let offset_u64 = u64::try_from(offset)
        .with_context(|| format!("{label} offset does not fit in u64 (offset={offset})"))?;
    let len_u64 = u64::try_from(len)
        .with_context(|| format!("{label} length does not fit in u64 (len={len})"))?;
    let end = offset_u64
        .checked_add(len_u64)
        .ok_or_else(|| anyhow::anyhow!("{label} region overflow (offset={offset}, len={len})"))?;
    if end > file_len {
        bail!("{label} region out of bounds (offset={offset}, len={len}, image_len={file_len})");
    }

    file.seek(SeekFrom::Start(offset_u64))
        .with_context(|| format!("failed to seek to {label} offset {offset}"))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)
        .with_context(|| format!("failed to read {label} bytes"))?;
    Ok(bytes)
}

fn bytes_to_hex_dump(bytes: &[u8]) -> String {
    let mut out = String::new();
    for (line, chunk) in bytes.chunks(16).enumerate() {
        let offset = line.saturating_mul(16);
        write!(&mut out, "{offset:08x}:").expect("write to String cannot fail");
        for byte in chunk {
            write!(&mut out, " {byte:02x}").expect("write to String cannot fail");
        }
        out.push('\n');
    }
    out
}

fn dump_extent_entry(extent: Ext4Extent) -> DumpExtentEntryOutput {
    let actual_len = extent.actual_len();
    let initialized = !extent.is_unwritten();
    DumpExtentEntryOutput {
        logical_block: extent.logical_block,
        physical_start: extent.physical_start,
        physical_end_inclusive: extent
            .physical_start
            .saturating_add(u64::from(actual_len))
            .saturating_sub(1),
        raw_len: extent.raw_len,
        actual_len,
        initialized,
    }
}

fn collect_extent_nodes(
    reader: &Ext4ImageReader,
    image: &[u8],
    source_block: Option<u64>,
    raw_node: &[u8],
    expected_depth: u16,
    include_hex: bool,
    nodes: &mut Vec<DumpExtentNodeOutput>,
) -> Result<()> {
    let (header, tree) = parse_extent_tree(raw_node).context("failed to parse extent tree node")?;
    if header.depth != expected_depth {
        bail!(
            "extent tree depth mismatch: expected {expected_depth}, parsed {}",
            header.depth
        );
    }

    let raw_hex = include_hex.then(|| bytes_to_hex_dump(raw_node));

    match tree {
        ExtentTree::Leaf(extents) => {
            nodes.push(DumpExtentNodeOutput {
                source_block,
                header,
                node: DumpExtentNodeKindOutput::Leaf {
                    extents: extents.into_iter().map(dump_extent_entry).collect(),
                },
                raw_hex,
            });
        }
        ExtentTree::Index(indexes) => {
            nodes.push(DumpExtentNodeOutput {
                source_block,
                header,
                node: DumpExtentNodeKindOutput::Index {
                    indexes: indexes.clone(),
                },
                raw_hex,
            });

            let next_depth = expected_depth
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("invalid extent depth transition from 0"))?;
            for index in indexes {
                let child = reader
                    .read_block(image, BlockNumber(index.leaf_block))
                    .with_context(|| {
                        format!(
                            "failed to read extent child block {} (logical={})",
                            index.leaf_block, index.logical_block
                        )
                    })?;
                collect_extent_nodes(
                    reader,
                    image,
                    Some(index.leaf_block),
                    child,
                    next_depth,
                    include_hex,
                    nodes,
                )?;
            }
        }
    }

    Ok(())
}

fn dump_dir_entry(index: usize, entry: &Ext4DirEntry) -> DumpDirEntryOutput {
    DumpDirEntryOutput {
        index,
        inode: entry.inode,
        rec_len: entry.rec_len,
        file_type: format!("{:?}", entry.file_type).to_ascii_lowercase(),
        name: entry.name_str(),
    }
}

fn read_ext4_directory_hex_blocks(
    image: &[u8],
    reader: &Ext4ImageReader,
    inode: &Ext4Inode,
) -> Result<Vec<DumpHexBlockOutput>> {
    let block_size = u64::from(reader.sb.block_size);
    let block_count_u64 = inode.size.div_ceil(block_size);
    let block_count = u32::try_from(block_count_u64).with_context(|| {
        format!("directory block count exceeds supported range: {block_count_u64}")
    })?;

    let mut blocks = Vec::new();
    for logical_block in 0..block_count {
        if let Some(physical_block) = reader
            .resolve_extent(image, inode, logical_block)
            .with_context(|| format!("failed to resolve directory block {logical_block}"))?
        {
            let block = reader
                .read_block(image, BlockNumber(physical_block))
                .with_context(|| format!("failed to read directory block {physical_block}"))?;
            blocks.push(DumpHexBlockOutput {
                logical_block,
                physical_block,
                hex: bytes_to_hex_dump(block),
            });
        }
    }

    Ok(blocks)
}

fn inspect_ext4_output(
    cx: &Cx,
    path: &PathBuf,
    open_opts: &OpenOptions,
    block_size: u32,
    inodes_count: u32,
    blocks_count: u64,
    volume_name: &str,
) -> Result<InspectOutput> {
    // Open the filesystem to read bitmaps for free space and orphan diagnostics.
    let open_fs = OpenFs::open_with_options(cx, path, open_opts)
        .with_context(|| format!("failed to open ext4 image: {}", path.display()))?;
    let summary = open_fs
        .free_space_summary(cx)
        .context("failed to compute free space summary")?;
    let orphans = open_fs
        .read_ext4_orphan_list(cx)
        .context("failed to read ext4 orphan list")?;
    let orphan_diagnostics = if orphans.inodes.is_empty() {
        None
    } else {
        Some(Ext4OrphanDiagnosticsOutput {
            count: u32::try_from(orphans.count()).unwrap_or(u32::MAX),
            sample_inodes: orphans.inodes.iter().take(16).map(|ino| ino.0).collect(),
        })
    };

    let mismatch = if summary.blocks_mismatch || summary.inodes_mismatch {
        Some(FreeSpaceMismatch {
            gd_free_blocks: summary.gd_free_blocks_total,
            gd_free_inodes: summary.gd_free_inodes_total,
        })
    } else {
        None
    };

    Ok(InspectOutput::Ext4 {
        block_size,
        inodes_count,
        blocks_count,
        volume_name: volume_name.to_owned(),
        free_blocks_total: summary.free_blocks_total,
        free_inodes_total: summary.free_inodes_total,
        free_space_mismatch: mismatch,
        orphan_diagnostics,
    })
}

const fn ext4_mount_replay_mode(read_write: bool) -> Ext4JournalReplayMode {
    if read_write {
        Ext4JournalReplayMode::Apply
    } else {
        Ext4JournalReplayMode::SimulateOverlay
    }
}

fn mount_cmd(
    image_path: &PathBuf,
    mountpoint: &PathBuf,
    allow_other: bool,
    rw: bool,
) -> Result<()> {
    let auto_unmount = env_bool("FFS_AUTO_UNMOUNT", true)?;
    let command_span = info_span!(
        target: "ffs::cli::mount",
        "mount",
        image = %image_path.display(),
        mountpoint = %mountpoint.display(),
        allow_other,
        auto_unmount,
        read_write = rw
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::mount", "mount_start");

    let cx = cli_cx();
    let open_opts = OpenOptions {
        ext4_journal_replay_mode: ext4_mount_replay_mode(rw),
        ..OpenOptions::default()
    };
    let mut open_fs = OpenFs::open_with_options(&cx, image_path, &open_opts)
        .with_context(|| format!("failed to open filesystem image: {}", image_path.display()))?;

    let mode_str = if rw { "rw" } else { "ro" };
    match &open_fs.flavor {
        FsFlavor::Ext4(sb) => {
            eprintln!(
                "Mounting ext4 image (block_size={}, blocks={}, {mode_str}) at {}",
                sb.block_size,
                sb.blocks_count,
                mountpoint.display()
            );
        }
        FsFlavor::Btrfs(sb) => {
            eprintln!(
                "Mounting btrfs image (sectorsize={}, nodesize={}, label={:?}, {mode_str}) at {}",
                sb.sectorsize,
                sb.nodesize,
                sb.label,
                mountpoint.display()
            );
        }
    }

    if let Some(recovery) = open_fs.crash_recovery() {
        if recovery.recovery_performed() {
            eprintln!(
                "  crash recovery: unclean shutdown detected (state=0x{:04X}, errors={}, orphans={})",
                recovery.raw_state, recovery.had_errors, recovery.had_orphans
            );
            if recovery.journal_txns_replayed > 0 {
                eprintln!(
                    "  journal replay: {} transactions, {} blocks replayed",
                    recovery.journal_txns_replayed, recovery.journal_blocks_replayed
                );
            }
            if recovery.mvcc_reset {
                eprintln!("  mvcc: version store reset (in-flight transactions discarded)");
            }
        }
    }

    if rw {
        open_fs
            .enable_writes(&cx)
            .context("failed to enable write support")?;
    }

    let opts = MountOptions {
        read_only: !rw,
        allow_other,
        auto_unmount,
        worker_threads: 0,
    };

    let fs_ops: Box<dyn FsOps> = Box::new(open_fs);
    ffs_fuse::mount(fs_ops, mountpoint, &opts)
        .with_context(|| format!("FUSE mount failed at {}", mountpoint.display()))?;

    info!(
        target: "ffs::cli::mount",
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "mount_complete"
    );

    Ok(())
}

// ── Scrub command ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct ScrubOutput {
    blocks_scanned: u64,
    blocks_corrupt: u64,
    blocks_error_or_higher: u64,
    blocks_io_error: u64,
    findings: Vec<ScrubFindingOutput>,
}

#[derive(Debug, Serialize)]
struct ScrubFindingOutput {
    block: u64,
    kind: String,
    severity: String,
    detail: String,
}

fn choose_btrfs_scrub_block_size(image_len: u64, nodesize: u32, sectorsize: u32) -> Result<u32> {
    if nodesize == 0 || !nodesize.is_power_of_two() {
        bail!("invalid btrfs nodesize={nodesize}; expected non-zero power-of-two");
    }

    // Btrfs superblock region is 4 KiB; scrub block size must hold it.
    let min_block_size = if sectorsize.is_power_of_two() {
        sectorsize.max(4096)
    } else {
        4096
    };

    if min_block_size > nodesize {
        bail!(
            "invalid btrfs geometry: sectorsize={sectorsize} nodesize={nodesize} (expected sectorsize <= nodesize)"
        );
    }

    let mut candidate = nodesize;
    while candidate >= min_block_size {
        if image_len % u64::from(candidate) == 0 {
            return Ok(candidate);
        }
        candidate /= 2;
    }

    bail!(
        "image length is not aligned to any supported btrfs scrub block size: len_bytes={image_len}, nodesize={nodesize}, sectorsize={sectorsize}"
    )
}

fn count_blocks_at_severity_or_higher(report: &ScrubReport, min: Severity) -> u64 {
    report
        .findings
        .iter()
        .filter(|finding| finding.severity >= min)
        .map(|finding| finding.block.0)
        .collect::<BTreeSet<_>>()
        .len() as u64
}

fn scrub_validator(flavor: &FsFlavor, block_size: u32) -> Box<dyn BlockValidator> {
    match flavor {
        FsFlavor::Ext4(_) => Box::new(CompositeValidator::new(vec![
            Box::new(ZeroCheckValidator),
            Box::new(Ext4SuperblockValidator::new(block_size)),
        ])),
        FsFlavor::Btrfs(sb) => Box::new(CompositeValidator::new(vec![
            Box::new(ZeroCheckValidator),
            Box::new(BtrfsSuperblockValidator::new(block_size)),
            Box::new(BtrfsTreeBlockValidator::new(
                block_size,
                sb.fsid,
                sb.csum_type,
            )),
        ])),
    }
}

fn print_scrub_output(json: bool, output: &ScrubOutput, report: &ScrubReport) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(output).context("serialize scrub report")?
        );
    } else {
        println!("FrankenFS Scrub Report");
        println!(
            "scanned {} blocks: {} corrupt, {} error+, {} io_errors, {} findings",
            output.blocks_scanned,
            output.blocks_corrupt,
            output.blocks_error_or_higher,
            output.blocks_io_error,
            output.findings.len(),
        );
        if !report.findings.is_empty() {
            println!();
            for finding in &report.findings {
                println!("  {finding}");
            }
        }
    }

    Ok(())
}

fn scrub_cmd(path: &PathBuf, json: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::scrub",
        "scrub",
        image = %path.display(),
        output_json = json
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::scrub", "scrub_start");

    let cx = cli_cx();

    // Detect filesystem to get the block size.
    let flavor = detect_filesystem_at_path(&cx, path)
        .with_context(|| format!("failed to detect ext4/btrfs metadata in {}", path.display()))?;

    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let image_len = byte_dev.len_bytes();

    let block_size = match &flavor {
        FsFlavor::Ext4(sb) => sb.block_size,
        FsFlavor::Btrfs(sb) => choose_btrfs_scrub_block_size(image_len, sb.nodesize, sb.sectorsize)
            .with_context(|| {
                format!(
                    "failed to derive aligned btrfs scrub block size (len_bytes={image_len}, nodesize={}, sectorsize={})",
                    sb.nodesize, sb.sectorsize
                )
            })?,
    };

    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;

    let validator = scrub_validator(&flavor, block_size);

    if !json {
        let fs_name = match &flavor {
            FsFlavor::Ext4(_) => "ext4",
            FsFlavor::Btrfs(_) => "btrfs",
        };
        eprintln!(
            "Scrubbing {fs_name} image: {} ({} blocks, block_size={block_size})",
            path.display(),
            block_dev.block_count(),
        );
    }

    let report = Scrubber::new(&block_dev, &*validator)
        .scrub_all(&cx)
        .with_context(|| "scrub failed")?;

    let blocks_error_or_higher = count_blocks_at_severity_or_higher(&report, Severity::Error);

    let output = ScrubOutput {
        blocks_scanned: report.blocks_scanned,
        blocks_corrupt: report.blocks_corrupt,
        blocks_error_or_higher,
        blocks_io_error: report.blocks_io_error,
        findings: report
            .findings
            .iter()
            .map(|f| ScrubFindingOutput {
                block: f.block.0,
                kind: f.kind.to_string(),
                severity: f.severity.to_string(),
                detail: f.detail.clone(),
            })
            .collect(),
    };

    print_scrub_output(json, &output, &report)?;

    let has_error_findings = report.count_at_severity(Severity::Error) > 0;

    info!(
        target: "ffs::cli::scrub",
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        blocks_scanned = output.blocks_scanned,
        blocks_corrupt = output.blocks_corrupt,
        blocks_error_or_higher = output.blocks_error_or_higher,
        has_error_findings,
        "scrub_complete"
    );

    // Exit with non-zero status if corruption found at Error or above.
    if has_error_findings {
        std::process::exit(2);
    }

    Ok(())
}

fn fsck_cmd(path: &PathBuf, options: FsckCommandOptions) -> Result<()> {
    let flags = options.flags;
    let command_span = info_span!(
        target: "ffs::cli::fsck",
        "fsck",
        image = %path.display(),
        repair = flags.repair(),
        force = flags.force(),
        verbose = flags.verbose(),
        block_group = options.block_group.unwrap_or(u32::MAX),
        output_json = flags.json()
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::fsck", "fsck_start");

    let output = match build_fsck_output(path, options) {
        Ok(output) => output,
        Err(err) => {
            if flags.json() {
                let mut escaped = String::new();
                for ch in format!("{err:#}").chars() {
                    match ch {
                        '"' => escaped.push_str("\\\""),
                        '\\' => escaped.push_str("\\\\"),
                        '\n' => escaped.push_str("\\n"),
                        '\r' => escaped.push_str("\\r"),
                        '\t' => escaped.push_str("\\t"),
                        _ => escaped.push(ch),
                    }
                }
                println!(
                    "{{\"status\":\"operational_error\",\"exit_code\":4,\"error\":\"{escaped}\"}}"
                );
            } else {
                eprintln!("fsck operational error: {err:#}");
            }
            std::process::exit(4);
        }
    };

    print_fsck_output(flags.json(), &output)?;

    info!(
        target: "ffs::cli::fsck",
        filesystem = output.filesystem,
        outcome = ?output.outcome,
        repair_status = ?output.repair_status,
        exit_code = output.exit_code,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "fsck_complete"
    );

    if output.exit_code != 0 {
        std::process::exit(output.exit_code);
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn build_fsck_output(path: &PathBuf, options: FsckCommandOptions) -> Result<FsckOutput> {
    let flags = options.flags;
    let cx = cli_cx();
    let mut phases = Vec::new();
    let mut limitations = Vec::new();
    let (flavor, bootstrap_recovery_source) =
        detect_flavor_with_optional_btrfs_bootstrap(&cx, path, flags.repair(), &mut limitations)?;
    let mut ext4_recovery = None;
    let mut btrfs_repair_attempted = bootstrap_recovery_source.is_some();
    let mut btrfs_repair_performed = bootstrap_recovery_source.is_some();
    let mut btrfs_repair_detail = bootstrap_recovery_source.map(|source| {
        format!(
            "bootstrap restored primary btrfs superblock from backup mirror at byte offset {} \
             (generation={})",
            source.offset, source.generation
        )
    });
    if flags.repair() {
        if let FsFlavor::Ext4(_) = &flavor {
            ext4_recovery = Some(run_ext4_mount_recovery(path)?);
        }
    }

    phases.push(FsckPhaseOutput {
        phase: "superblock_validation".to_owned(),
        status: "ok".to_owned(),
        detail: match &flavor {
            FsFlavor::Ext4(sb) => format!(
                "ext4 superblock parsed (block_size={}, blocks={}, inodes={})",
                sb.block_size, sb.blocks_count, sb.inodes_count
            ),
            FsFlavor::Btrfs(sb) => format!(
                "btrfs superblock parsed (sectorsize={}, nodesize={}, generation={})",
                sb.sectorsize, sb.nodesize, sb.generation
            ),
        },
    });

    let image = std::fs::read(path)
        .with_context(|| format!("failed to read filesystem image: {}", path.display()))?;
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let image_len = byte_dev.len_bytes();

    let (scope, report, block_size, scrub_skipped_detail) = match &flavor {
        FsFlavor::Ext4(sb) => {
            let reader = Ext4ImageReader::new(&image).context("failed to parse ext4 superblock")?;
            let desc_status = validate_ext4_group_descriptors(&reader, &image, options.block_group);
            phases.push(desc_status);

            let block_size = sb.block_size;
            let block_dev = ByteBlockDevice::new(byte_dev, block_size).with_context(|| {
                format!("failed to create block device (block_size={block_size})")
            })?;
            let validator = scrub_validator(&flavor, block_size);

            let skip_full_scrub = options.block_group.is_none()
                && !flags.force()
                && ext4_appears_clean_state(sb.state);

            let (scope, report, skipped_detail) = if let Some(group) = options.block_group {
                let (start, count) = ext4_group_scrub_scope(sb, group)?;
                if flags.verbose() && !flags.json() {
                    eprintln!(
                        "fsck: ext4 group {} -> scrub blocks {}..{}",
                        group,
                        start.0,
                        start.0.saturating_add(count).saturating_sub(1)
                    );
                }
                let report = Scrubber::new(&block_dev, &*validator)
                    .scrub_range(&cx, start, count)
                    .with_context(|| format!("failed to scrub ext4 group {group}"))?;
                (
                    FsckScopeOutput::Ext4BlockGroup {
                        group,
                        start_block: start.0,
                        block_count: count,
                    },
                    report,
                    None,
                )
            } else if skip_full_scrub {
                let detail = format!(
                    "filesystem appears clean (state_flags={}); skipped block-level scrub; pass --force for full scrub",
                    ext4_state_flag_names(sb.state).join("|")
                );
                if flags.verbose() && !flags.json() {
                    eprintln!("fsck: {detail}");
                }
                limitations.push(
                    "filesystem appears clean; skipped block-level scrub (use --force for full scrub)"
                        .to_owned(),
                );
                (
                    FsckScopeOutput::Full,
                    ScrubReport {
                        findings: Vec::new(),
                        blocks_scanned: 0,
                        blocks_corrupt: 0,
                        blocks_io_error: 0,
                    },
                    Some(detail),
                )
            } else {
                if flags.verbose() && !flags.json() {
                    eprintln!(
                        "fsck: ext4 full scrub across {} blocks",
                        block_dev.block_count()
                    );
                }
                let report = Scrubber::new(&block_dev, &*validator)
                    .scrub_all(&cx)
                    .context("failed to scrub ext4 image")?;
                (FsckScopeOutput::Full, report, None)
            };
            (scope, report, block_size, skipped_detail)
        }
        FsFlavor::Btrfs(sb) => {
            phases.push(FsckPhaseOutput {
                phase: "group_descriptor_validation".to_owned(),
                status: "skipped".to_owned(),
                detail: "ext4-specific group descriptor checks do not apply to btrfs".to_owned(),
            });

            let block_size = choose_btrfs_scrub_block_size(image_len, sb.nodesize, sb.sectorsize)
                .with_context(|| {
                    format!(
                        "failed to derive aligned btrfs scrub block size (len_bytes={image_len}, nodesize={}, sectorsize={})",
                        sb.nodesize, sb.sectorsize
                    )
                })?;
            let (scope, scrub_start, scrub_count, mut report) = if let Some(group) =
                options.block_group
            {
                let specs = discover_btrfs_repair_group_specs(path, block_size)
                    .context("failed to discover btrfs block groups for scoped fsck")?;
                let spec = specs
                    .iter()
                    .find(|spec| spec.group == group)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "btrfs block group {group} is unavailable (valid range: 0..{})",
                            specs.len().saturating_sub(1)
                        )
                    })?;
                if flags.verbose() && !flags.json() {
                    eprintln!(
                        "fsck: btrfs group {} -> logical {}..{}, physical blocks {}..{}",
                        group,
                        spec.logical_start,
                        spec.logical_start
                            .saturating_add(spec.logical_bytes)
                            .saturating_sub(1),
                        spec.physical_start_block.0,
                        spec.physical_start_block
                            .0
                            .saturating_add(spec.physical_block_count)
                            .saturating_sub(1)
                    );
                }
                let report = scrub_range_for_repair(
                    path,
                    &flavor,
                    block_size,
                    spec.physical_start_block,
                    spec.physical_block_count,
                    None,
                    &mut limitations,
                )
                .with_context(|| format!("failed to scrub btrfs group {group}"))?;
                (
                    FsckScopeOutput::BtrfsBlockGroup {
                        group,
                        logical_start: spec.logical_start,
                        logical_bytes: spec.logical_bytes,
                        start_block: spec.physical_start_block.0,
                        block_count: spec.physical_block_count,
                    },
                    spec.physical_start_block,
                    spec.physical_block_count,
                    report,
                )
            } else {
                let block_dev = ByteBlockDevice::new(byte_dev, block_size).with_context(|| {
                    format!("failed to create block device (block_size={block_size})")
                })?;
                let validator = scrub_validator(&flavor, block_size);
                if flags.verbose() && !flags.json() {
                    eprintln!(
                        "fsck: btrfs full scrub across {} blocks (block_size={block_size})",
                        block_dev.block_count()
                    );
                }
                let report = Scrubber::new(&block_dev, &*validator)
                    .scrub_all(&cx)
                    .context("failed to scrub btrfs image")?;
                (FsckScopeOutput::Full, BlockNumber(0), u64::MAX, report)
            };

            if flags.repair() {
                let mut verify_after_repair_writes = false;
                let primary_superblock = primary_btrfs_superblock_block(block_size);
                let superblock_in_scope =
                    block_range_contains(scrub_start, scrub_count, primary_superblock);
                if superblock_in_scope
                    && report_has_error_or_higher_for_block(&report, primary_superblock)
                {
                    btrfs_repair_attempted = true;
                    match recover_primary_btrfs_superblock_from_backup(path, &mut limitations) {
                        Ok(Some(source)) => {
                            btrfs_repair_performed = true;
                            verify_after_repair_writes = true;
                            let detail = format!(
                                "restored primary btrfs superblock from backup mirror at byte offset {} (generation={})",
                                source.offset, source.generation
                            );
                            append_btrfs_repair_detail(&mut btrfs_repair_detail, detail.clone());
                            limitations.push(detail);
                        }
                        Ok(None) => {
                            let detail =
                                "btrfs superblock corruption detected, but no valid backup superblock mirror was available for restoration"
                                    .to_owned();
                            append_btrfs_repair_detail(&mut btrfs_repair_detail, detail.clone());
                            limitations.push(detail);
                        }
                        Err(error) => {
                            let detail =
                                format!("btrfs superblock recovery attempt failed: {error:#}");
                            append_btrfs_repair_detail(&mut btrfs_repair_detail, detail.clone());
                            limitations.push(detail);
                        }
                    }
                }

                match repair_corrupt_btrfs_superblock_mirrors_from_primary(
                    path,
                    block_size,
                    scrub_start,
                    scrub_count,
                    &mut limitations,
                ) {
                    Ok(mirror_outcome) => {
                        if mirror_outcome.attempted {
                            btrfs_repair_attempted = true;
                            if mirror_outcome.repaired > 0 {
                                btrfs_repair_performed = true;
                                verify_after_repair_writes = true;
                                append_btrfs_repair_detail(
                                    &mut btrfs_repair_detail,
                                    format!(
                                        "restored {} btrfs superblock mirror(s) from primary superblock",
                                        mirror_outcome.repaired
                                    ),
                                );
                            } else {
                                append_btrfs_repair_detail(
                                    &mut btrfs_repair_detail,
                                    "btrfs superblock mirror corruption detected but restoration from primary did not complete",
                                );
                            }
                        }
                    }
                    Err(error) => {
                        btrfs_repair_attempted = true;
                        let detail =
                            format!("btrfs superblock mirror recovery attempt failed: {error:#}");
                        append_btrfs_repair_detail(&mut btrfs_repair_detail, detail.clone());
                        limitations.push(detail);
                    }
                }

                // Attempt RaptorQ block-symbol recovery for non-superblock
                // corruption in the fsck --repair path.
                if count_blocks_at_severity_or_higher(&report, Severity::Error) > 0 {
                    match discover_btrfs_repair_group_specs(path, block_size) {
                        Ok(btrfs_specs) if !btrfs_specs.is_empty() => {
                            match recover_btrfs_corrupt_blocks(
                                path,
                                block_size,
                                sb.fsid,
                                &btrfs_specs,
                                &report,
                                &mut limitations,
                            ) {
                                Ok((recovered, _unrecovered, _repaired_groups)) => {
                                    if recovered > 0 {
                                        btrfs_repair_attempted = true;
                                        btrfs_repair_performed = true;
                                        verify_after_repair_writes = true;
                                        append_btrfs_repair_detail(
                                            &mut btrfs_repair_detail,
                                            format!(
                                                "recovered {recovered} btrfs block(s) via RaptorQ symbol reconstruction"
                                            ),
                                        );
                                    }
                                }
                                Err(error) => {
                                    btrfs_repair_attempted = true;
                                    let detail = format!(
                                        "btrfs block-symbol recovery attempt failed: {error:#}"
                                    );
                                    append_btrfs_repair_detail(
                                        &mut btrfs_repair_detail,
                                        detail.clone(),
                                    );
                                    limitations.push(detail);
                                }
                            }
                        }
                        Ok(_) => {
                            limitations.push(
                                "no btrfs block groups discovered; fsck block-symbol recovery skipped"
                                    .to_owned(),
                            );
                        }
                        Err(error) => {
                            limitations.push(format!(
                                "failed to discover btrfs repair group layout for fsck: {error:#}"
                            ));
                        }
                    }
                }

                if !btrfs_repair_performed && btrfs_repair_detail.is_none() {
                    if superblock_in_scope {
                        btrfs_repair_detail = Some(
                            "no primary btrfs superblock corruption detected in selected fsck scope"
                                .to_owned(),
                        );
                    } else {
                        btrfs_repair_detail = Some(
                            "repair scope does not include the primary btrfs superblock".to_owned(),
                        );
                    }
                }

                if verify_after_repair_writes {
                    report = scrub_range_for_repair(
                        path,
                        &flavor,
                        block_size,
                        scrub_start,
                        scrub_count,
                        None,
                        &mut limitations,
                    )
                    .context("failed to verify btrfs image after fsck repairs")?;
                }
            }

            (scope, report, block_size, None)
        }
    };
    let scrub = scrub_report_to_phase(&report);
    let scrub_phase_status = if scrub_skipped_detail.is_some() {
        "skipped"
    } else if scrub.error_or_higher == 0 {
        "ok"
    } else {
        "error"
    };
    let scrub_phase_detail = scrub_skipped_detail.unwrap_or_else(|| {
        format!(
            "scanned={} corrupt={} error_or_higher={} io_errors={}",
            scrub.scanned, scrub.corrupt, scrub.error_or_higher, scrub.io_error
        )
    });
    phases.push(FsckPhaseOutput {
        phase: "checksum_scrub".to_owned(),
        status: scrub_phase_status.to_owned(),
        detail: scrub_phase_detail,
    });

    let repair_status = if flags.repair() {
        if let Some(recovery) = &ext4_recovery {
            phases.push(FsckPhaseOutput {
                phase: "repair".to_owned(),
                status: "ok".to_owned(),
                detail: ext4_recovery_detail(recovery),
            });
            FsckRepairStatus::RequestedPerformed
        } else if matches!(flavor, FsFlavor::Btrfs(_)) {
            let (status, detail, performed) = if btrfs_repair_performed {
                (
                    "ok".to_owned(),
                    btrfs_repair_detail.unwrap_or_else(|| {
                        "restored primary btrfs superblock from backup mirror".to_owned()
                    }),
                    true,
                )
            } else if btrfs_repair_attempted {
                (
                    "error".to_owned(),
                    btrfs_repair_detail.unwrap_or_else(|| {
                        "btrfs superblock corruption detected but recovery did not complete"
                            .to_owned()
                    }),
                    false,
                )
            } else {
                (
                    "skipped".to_owned(),
                    btrfs_repair_detail.unwrap_or_else(|| {
                        "repair requested but no primary btrfs superblock corruption was detected in scope"
                            .to_owned()
                    }),
                    false,
                )
            };
            phases.push(FsckPhaseOutput {
                phase: "repair".to_owned(),
                status,
                detail,
            });
            if performed {
                FsckRepairStatus::RequestedPerformed
            } else {
                FsckRepairStatus::RequestedNotPerformed
            }
        } else {
            phases.push(FsckPhaseOutput {
                phase: "repair".to_owned(),
                status: "skipped".to_owned(),
                detail:
                    "repair requested but no write-side workflow is available for this filesystem flavor"
                        .to_owned(),
            });
            FsckRepairStatus::RequestedNotPerformed
        }
    } else {
        FsckRepairStatus::NotRequested
    };

    limitations.push(format!(
        "fsck currently covers superblock/group-descriptor validation plus block-level scrub checks (block_size={block_size})"
    ));

    let outcome = if scrub.error_or_higher > 0 {
        FsckOutcome::ErrorsFound
    } else {
        FsckOutcome::Clean
    };
    let exit_code = match outcome {
        FsckOutcome::Clean => 0,
        FsckOutcome::ErrorsFound => 1,
    };

    Ok(FsckOutput {
        filesystem: filesystem_name(&flavor).to_owned(),
        scope,
        phases,
        scrub,
        repair_status,
        ext4_recovery,
        outcome,
        exit_code,
        limitations,
    })
}

fn validate_ext4_group_descriptors(
    reader: &Ext4ImageReader,
    image: &[u8],
    only_group: Option<u32>,
) -> FsckPhaseOutput {
    if let Some(group) = only_group {
        return match reader.read_group_desc(image, GroupNumber(group)) {
            Ok(_) => FsckPhaseOutput {
                phase: "group_descriptor_validation".to_owned(),
                status: "ok".to_owned(),
                detail: format!("validated ext4 group descriptor {group}"),
            },
            Err(err) => FsckPhaseOutput {
                phase: "group_descriptor_validation".to_owned(),
                status: "error".to_owned(),
                detail: format!("group {group} failed validation: {err}"),
            },
        };
    }

    let groups = reader.sb.groups_count();
    for group in 0..groups {
        if let Err(err) = reader.read_group_desc(image, GroupNumber(group)) {
            return FsckPhaseOutput {
                phase: "group_descriptor_validation".to_owned(),
                status: "error".to_owned(),
                detail: format!("group {group} failed validation: {err}"),
            };
        }
    }

    FsckPhaseOutput {
        phase: "group_descriptor_validation".to_owned(),
        status: "ok".to_owned(),
        detail: format!("validated {groups} ext4 group descriptors"),
    }
}

fn ext4_group_scrub_scope(sb: &Ext4Superblock, group: u32) -> Result<(BlockNumber, u64)> {
    let groups = sb.groups_count();
    if group >= groups {
        bail!("block group {group} out of range (groups_count={groups})");
    }

    let start = u64::from(sb.first_data_block)
        .saturating_add(u64::from(group).saturating_mul(u64::from(sb.blocks_per_group)));
    let end_exclusive = start
        .saturating_add(u64::from(sb.blocks_per_group))
        .min(sb.blocks_count);
    let count = end_exclusive.saturating_sub(start);

    Ok((BlockNumber(start), count))
}

fn scrub_report_to_phase(report: &ScrubReport) -> FsckScrubOutput {
    FsckScrubOutput {
        scanned: report.blocks_scanned,
        corrupt: report.blocks_corrupt,
        error_or_higher: count_blocks_at_severity_or_higher(report, Severity::Error),
        io_error: report.blocks_io_error,
    }
}

fn run_ext4_mount_recovery(path: &PathBuf) -> Result<Ext4RecoveryOutput> {
    let cx = cli_cx();
    let open = OpenFs::open_with_options(
        &cx,
        path,
        &OpenOptions {
            skip_validation: false,
            ext4_journal_replay_mode: Ext4JournalReplayMode::Apply,
        },
    )
    .with_context(|| {
        format!(
            "failed to open ext4 image for repair workflow: {}",
            path.display()
        )
    })?;

    let outcome = open.crash_recovery().cloned().ok_or_else(|| {
        anyhow::anyhow!("ext4 repair workflow expected crash recovery outcome but found none")
    })?;
    Ok(Ext4RecoveryOutput {
        recovery_performed: outcome.recovery_performed(),
        crash_recovery: outcome,
    })
}

fn ext4_recovery_detail(recovery: &Ext4RecoveryOutput) -> String {
    format!(
        "recovery_performed={} clean={} had_errors={} had_orphans={} journal_txns_replayed={} journal_blocks_replayed={}",
        recovery.recovery_performed,
        recovery.crash_recovery.was_clean,
        recovery.crash_recovery.had_errors,
        recovery.crash_recovery.had_orphans,
        recovery.crash_recovery.journal_txns_replayed,
        recovery.crash_recovery.journal_blocks_replayed
    )
}

fn print_fsck_output(json: bool, output: &FsckOutput) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(output).context("serialize fsck output")?
        );
        return Ok(());
    }

    println!("FrankenFS FSCK");
    println!("filesystem: {}", output.filesystem);
    match &output.scope {
        FsckScopeOutput::Full => println!("scope: full"),
        FsckScopeOutput::Ext4BlockGroup {
            group,
            start_block,
            block_count,
        } => {
            println!(
                "scope: ext4 group {} (blocks {}..{})",
                group,
                start_block,
                start_block.saturating_add(*block_count).saturating_sub(1)
            );
        }
        FsckScopeOutput::BtrfsBlockGroup {
            group,
            logical_start,
            logical_bytes,
            start_block,
            block_count,
        } => {
            println!(
                "scope: btrfs group {} (logical {}..{}, physical blocks {}..{})",
                group,
                logical_start,
                logical_start
                    .saturating_add(*logical_bytes)
                    .saturating_sub(1),
                start_block,
                start_block.saturating_add(*block_count).saturating_sub(1)
            );
        }
    }
    println!("phases:");
    for phase in &output.phases {
        println!(
            "  - {}: {} ({})",
            phase.phase.replace('_', " "),
            phase.status,
            phase.detail
        );
    }
    println!(
        "scrub: scanned={} corrupt={} error_or_higher={} io_errors={}",
        output.scrub.scanned,
        output.scrub.corrupt,
        output.scrub.error_or_higher,
        output.scrub.io_error
    );
    if let Some(recovery) = &output.ext4_recovery {
        println!("ext4_recovery: {}", ext4_recovery_detail(recovery));
    }
    println!("repair_status: {:?}", output.repair_status);
    println!("outcome: {:?}", output.outcome);
    println!("exit_code: {}", output.exit_code);
    if !output.limitations.is_empty() {
        println!("limitations:");
        for limitation in &output.limitations {
            println!("  - {limitation}");
        }
    }

    Ok(())
}

fn repair_cmd(path: &PathBuf, options: RepairCommandOptions) -> Result<()> {
    let flags = options.flags;
    let command_span = info_span!(
        target: "ffs::cli::repair",
        "repair",
        image = %path.display(),
        full_scrub = flags.full_scrub(),
        verify_only = flags.verify_only(),
        rebuild_symbols = flags.rebuild_symbols(),
        block_group = options.block_group.unwrap_or(u32::MAX),
        max_threads = options.max_threads.unwrap_or(0),
        output_json = flags.json()
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::repair", "repair_start");

    let output = match build_repair_output(path, options) {
        Ok(output) => output,
        Err(err) => {
            if flags.json() {
                let mut escaped = String::new();
                for ch in format!("{err:#}").chars() {
                    match ch {
                        '"' => escaped.push_str("\\\""),
                        '\\' => escaped.push_str("\\\\"),
                        '\n' => escaped.push_str("\\n"),
                        '\r' => escaped.push_str("\\r"),
                        '\t' => escaped.push_str("\\t"),
                        _ => escaped.push(ch),
                    }
                }
                println!(
                    "{{\"status\":\"operational_error\",\"exit_code\":4,\"error\":\"{escaped}\"}}"
                );
            } else {
                eprintln!("repair operational error: {err:#}");
            }
            std::process::exit(4);
        }
    };

    print_repair_output(flags.json(), &output)?;

    info!(
        target: "ffs::cli::repair",
        filesystem = output.filesystem,
        action = ?output.action,
        exit_code = output.exit_code,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "repair_complete"
    );

    if output.exit_code != 0 {
        std::process::exit(output.exit_code);
    }

    Ok(())
}

fn repair_worker_limit(requested: Option<u32>) -> (usize, Option<u32>) {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let requested_raw = requested.unwrap_or(1).max(1);
    let requested_threads = usize::try_from(requested_raw).unwrap_or(usize::MAX);
    let effective = requested_threads.min(available).max(1);
    let capped = (requested_threads > effective).then_some(requested_raw);
    (effective, capped)
}

fn partition_scrub_range(
    start: BlockNumber,
    count: u64,
    workers: usize,
) -> Vec<(BlockNumber, u64)> {
    if count == 0 || workers == 0 {
        return Vec::new();
    }

    let workers_u64 = u64::try_from(workers).unwrap_or(u64::MAX).min(count);
    let base = count / workers_u64;
    let remainder = count % workers_u64;
    let mut cursor = start.0;
    let mut ranges = Vec::with_capacity(usize::try_from(workers_u64).unwrap_or(usize::MAX));
    for worker_idx in 0..workers_u64 {
        let width = base + u64::from(worker_idx < remainder);
        ranges.push((BlockNumber(cursor), width));
        cursor = cursor.saturating_add(width);
    }
    ranges
}

fn merge_scrub_reports(reports: Vec<ScrubReport>) -> ScrubReport {
    let mut merged = ScrubReport {
        findings: Vec::new(),
        blocks_scanned: 0,
        blocks_corrupt: 0,
        blocks_io_error: 0,
    };
    for report in reports {
        merged.blocks_scanned = merged.blocks_scanned.saturating_add(report.blocks_scanned);
        merged.blocks_corrupt = merged.blocks_corrupt.saturating_add(report.blocks_corrupt);
        merged.blocks_io_error = merged
            .blocks_io_error
            .saturating_add(report.blocks_io_error);
        merged.findings.extend(report.findings);
    }
    merged
}

const DEFAULT_REPAIR_OVERHEAD_RATIO: f64 = 1.05;
const DEFAULT_REPAIR_VALIDATION_BLOCK_COUNT: u32 = 0;
const BTRFS_EXTENT_TREE_OBJECTID: u64 = 2;
const BTRFS_ROOT_ITEM_TYPE: u8 = 132;
const BTRFS_BLOCK_GROUP_ITEM_TYPE: u8 = 192;
const BTRFS_SUPER_MIRROR_MAX: u32 = 3;
const BTRFS_SUPER_MIRROR_SHIFT: u32 = 12;
const BTRFS_SUPER_MIRROR_BASE: u64 = 16 * 1024;
const BTRFS_SUPER_BYTENR_OFFSET: usize = 0x30;
const BTRFS_SUPER_CSUM_OFFSET: usize = 0;
const BTRFS_SUPER_CSUM_LEN: usize = 4;
const BTRFS_SUPER_CSUM_DATA_OFFSET: usize = 0x20;

#[derive(Debug, Clone, Copy)]
struct BtrfsRepairGroupSpec {
    group: u32,
    logical_start: u64,
    logical_bytes: u64,
    physical_start_block: BlockNumber,
    physical_block_count: u64,
    /// Source payload blocks (total minus repair metadata).
    source_block_count: u32,
    /// Repair tail layout for RaptorQ symbol storage/recovery.
    layout: RepairGroupLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ext4RepairStaleness {
    Fresh,
    Stale,
    Untracked,
}

#[derive(Debug, Clone, Copy)]
struct Ext4RepairGroupSpec {
    group: u32,
    scrub_start_block: BlockNumber,
    scrub_block_count: u64,
    source_first_block: BlockNumber,
    source_block_count: u32,
    layout: RepairGroupLayout,
}

#[derive(Debug, Clone, Copy, Default)]
struct RepairExecutionStats {
    recovery_attempted: bool,
    recovery_unrecovered_blocks: u64,
    symbol_rebuild_attempted: bool,
    symbol_rebuild_groups: u64,
    symbol_rebuild_failed_groups: u64,
}

#[derive(Debug, Clone, Copy)]
struct BtrfsSuperblockRecoverySource {
    offset: u64,
    generation: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct BtrfsMirrorRepairOutcome {
    attempted: bool,
    repaired: u32,
}

fn btrfs_super_mirror_offsets(image_len: u64) -> Vec<u64> {
    let mut offsets = Vec::new();
    for mirror in 0..BTRFS_SUPER_MIRROR_MAX {
        let offset = if mirror == 0 {
            BTRFS_SUPER_INFO_OFFSET as u64
        } else {
            let shift = BTRFS_SUPER_MIRROR_SHIFT.saturating_mul(mirror);
            let Some(candidate) = BTRFS_SUPER_MIRROR_BASE.checked_shl(shift) else {
                continue;
            };
            candidate
        };
        let Some(end) =
            offset.checked_add(u64::try_from(BTRFS_SUPER_INFO_SIZE).unwrap_or(u64::MAX))
        else {
            continue;
        };
        if end <= image_len {
            offsets.push(offset);
        }
    }
    offsets
}

fn block_range_contains(start: BlockNumber, count: u64, target: BlockNumber) -> bool {
    if count == 0 || target.0 < start.0 {
        return false;
    }
    if count == u64::MAX {
        return true;
    }
    start.0.checked_add(count).is_none_or(|end| target.0 < end)
}

fn primary_btrfs_superblock_block(block_size: u32) -> BlockNumber {
    BlockNumber((BTRFS_SUPER_INFO_OFFSET as u64) / u64::from(block_size))
}

fn report_has_error_or_higher_for_block(report: &ScrubReport, block: BlockNumber) -> bool {
    report
        .findings
        .iter()
        .any(|finding| finding.block == block && finding.severity >= Severity::Error)
}

fn normalize_btrfs_superblock_for_offset(region: &mut [u8], bytenr: u64) -> Result<()> {
    if region.len() < BTRFS_SUPER_INFO_SIZE {
        bail!(
            "btrfs superblock region too short: expected {} bytes, got {}",
            BTRFS_SUPER_INFO_SIZE,
            region.len()
        );
    }
    region[BTRFS_SUPER_BYTENR_OFFSET..BTRFS_SUPER_BYTENR_OFFSET + 8]
        .copy_from_slice(&bytenr.to_le_bytes());
    let checksum = crc32c::crc32c(&region[BTRFS_SUPER_CSUM_DATA_OFFSET..BTRFS_SUPER_INFO_SIZE]);
    region[BTRFS_SUPER_CSUM_OFFSET..BTRFS_SUPER_CSUM_OFFSET + BTRFS_SUPER_CSUM_LEN]
        .copy_from_slice(&checksum.to_le_bytes());
    Ok(())
}

fn normalize_btrfs_superblock_as_primary(region: &mut [u8]) -> Result<()> {
    normalize_btrfs_superblock_for_offset(region, BTRFS_SUPER_INFO_OFFSET as u64)
}

fn append_btrfs_repair_detail(detail: &mut Option<String>, message: impl Into<String>) {
    let message = message.into();
    if let Some(existing) = detail.as_mut() {
        if !existing.contains(&message) {
            if !existing.is_empty() {
                existing.push_str("; ");
            }
            existing.push_str(&message);
        }
    } else {
        *detail = Some(message);
    }
}

#[allow(clippy::too_many_lines)]
fn repair_corrupt_btrfs_superblock_mirrors_from_primary(
    path: &PathBuf,
    block_size: u32,
    scrub_start: BlockNumber,
    scrub_count: u64,
    limitations: &mut Vec<String>,
) -> Result<BtrfsMirrorRepairOutcome> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let image_len = byte_dev.len_bytes();
    let primary_offset = BTRFS_SUPER_INFO_OFFSET as u64;
    let block_size_u64 = u64::from(block_size);

    let mut corrupt_mirror_offsets = Vec::new();
    for offset in btrfs_super_mirror_offsets(image_len)
        .into_iter()
        .filter(|offset| *offset != primary_offset)
    {
        if offset % block_size_u64 != 0 {
            limitations.push(format!(
                "cannot evaluate btrfs superblock mirror at byte offset {offset} with scrub block size {block_size}"
            ));
            continue;
        }
        let mirror_block = BlockNumber(offset / block_size_u64);
        if !block_range_contains(scrub_start, scrub_count, mirror_block) {
            continue;
        }

        let mut mirror_region = vec![0_u8; BTRFS_SUPER_INFO_SIZE];
        if let Err(error) = byte_dev.read_exact_at(&cx, ByteOffset(offset), &mut mirror_region) {
            limitations.push(format!(
                "failed to read btrfs superblock mirror at byte offset {offset} while checking repair scope: {error}"
            ));
            corrupt_mirror_offsets.push(offset);
            continue;
        }

        let mut mirror_valid = true;
        match BtrfsSuperblock::parse_superblock_region(&mirror_region) {
            Ok(parsed) => {
                if parsed.bytenr != offset {
                    mirror_valid = false;
                    limitations.push(format!(
                        "btrfs superblock mirror at byte offset {offset} reports bytenr={} and needs repair",
                        parsed.bytenr
                    ));
                }
            }
            Err(error) => {
                mirror_valid = false;
                limitations.push(format!(
                    "btrfs superblock mirror at byte offset {offset} failed structural validation and needs repair: {error}"
                ));
            }
        }
        if let Err(error) = verify_btrfs_superblock_checksum(&mirror_region) {
            mirror_valid = false;
            limitations.push(format!(
                "btrfs superblock mirror at byte offset {offset} failed checksum validation and needs repair: {error}"
            ));
        }

        if !mirror_valid {
            corrupt_mirror_offsets.push(offset);
        }
    }

    if corrupt_mirror_offsets.is_empty() {
        return Ok(BtrfsMirrorRepairOutcome::default());
    }

    let mut outcome = BtrfsMirrorRepairOutcome {
        attempted: true,
        repaired: 0,
    };
    let mut primary_region = vec![0_u8; BTRFS_SUPER_INFO_SIZE];
    if let Err(error) = byte_dev.read_exact_at(&cx, ByteOffset(primary_offset), &mut primary_region)
    {
        limitations.push(format!(
            "failed to read primary btrfs superblock while restoring mirror copies: {error}"
        ));
        return Ok(outcome);
    }

    let parsed_primary = match BtrfsSuperblock::parse_superblock_region(&primary_region) {
        Ok(parsed) => parsed,
        Err(error) => {
            limitations.push(format!(
                "failed to parse primary btrfs superblock while restoring mirror copies: {error}"
            ));
            return Ok(outcome);
        }
    };
    if parsed_primary.bytenr != primary_offset {
        limitations.push(format!(
            "primary btrfs superblock reports bytenr={} and cannot seed mirror restoration",
            parsed_primary.bytenr
        ));
        return Ok(outcome);
    }
    if let Err(error) = verify_btrfs_superblock_checksum(&primary_region) {
        limitations.push(format!(
            "primary btrfs superblock checksum is invalid; mirror restoration skipped: {error}"
        ));
        return Ok(outcome);
    }

    let mut repaired_offsets = Vec::new();
    for offset in corrupt_mirror_offsets {
        let mut mirror_region = primary_region.clone();
        if let Err(error) = normalize_btrfs_superblock_for_offset(&mut mirror_region, offset) {
            limitations.push(format!(
                "failed to retarget btrfs superblock mirror at byte offset {offset}: {error:#}"
            ));
            continue;
        }
        if let Err(error) = BtrfsSuperblock::parse_superblock_region(&mirror_region) {
            limitations.push(format!(
                "re-encoded btrfs superblock mirror at byte offset {offset} failed validation: {error}"
            ));
            continue;
        }
        if let Err(error) = verify_btrfs_superblock_checksum(&mirror_region) {
            limitations.push(format!(
                "re-encoded btrfs superblock mirror at byte offset {offset} failed checksum validation: {error}"
            ));
            continue;
        }
        match byte_dev.write_all_at(&cx, ByteOffset(offset), &mirror_region) {
            Ok(()) => repaired_offsets.push(offset),
            Err(error) => limitations.push(format!(
                "failed to write recovered btrfs superblock mirror at byte offset {offset}: {error}"
            )),
        }
    }

    if repaired_offsets.is_empty() {
        return Ok(outcome);
    }

    byte_dev
        .sync(&cx)
        .context("failed to sync image after btrfs superblock mirror recovery")?;
    outcome.repaired = u32::try_from(repaired_offsets.len()).unwrap_or(u32::MAX);
    let offsets = repaired_offsets
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    limitations.push(format!(
        "restored {} btrfs superblock mirror(s) from primary superblock at byte offset(s): {offsets}",
        outcome.repaired
    ));
    Ok(outcome)
}

fn recover_primary_btrfs_superblock_from_backup(
    path: &PathBuf,
    limitations: &mut Vec<String>,
) -> Result<Option<BtrfsSuperblockRecoverySource>> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let image_len = byte_dev.len_bytes();
    let primary_offset = BTRFS_SUPER_INFO_OFFSET as u64;

    let mut selected: Option<(Vec<u8>, BtrfsSuperblockRecoverySource)> = None;
    for offset in btrfs_super_mirror_offsets(image_len)
        .into_iter()
        .filter(|offset| *offset != primary_offset)
    {
        let mut region = vec![0_u8; BTRFS_SUPER_INFO_SIZE];
        if let Err(error) = byte_dev.read_exact_at(&cx, ByteOffset(offset), &mut region) {
            limitations.push(format!(
                "failed to read btrfs superblock mirror at byte offset {offset}: {error}"
            ));
            continue;
        }

        let parsed = match BtrfsSuperblock::parse_superblock_region(&region) {
            Ok(parsed) => parsed,
            Err(error) => {
                limitations.push(format!(
                    "btrfs superblock mirror at byte offset {offset} is structurally invalid: {error}"
                ));
                continue;
            }
        };
        if parsed.bytenr != offset {
            limitations.push(format!(
                "btrfs superblock mirror at byte offset {offset} reports bytenr={} and was skipped",
                parsed.bytenr
            ));
            continue;
        }
        if let Err(error) = verify_btrfs_superblock_checksum(&region) {
            limitations.push(format!(
                "btrfs superblock mirror at byte offset {offset} failed checksum validation: {error}"
            ));
            continue;
        }

        let source = BtrfsSuperblockRecoverySource {
            offset,
            generation: parsed.generation,
        };
        let keep_candidate = selected.as_ref().is_none_or(|(_, current)| {
            source.generation > current.generation
                || (source.generation == current.generation && source.offset < current.offset)
        });
        if keep_candidate {
            selected = Some((region, source));
        }
    }

    let Some((mut recovered_region, source)) = selected else {
        return Ok(None);
    };
    normalize_btrfs_superblock_as_primary(&mut recovered_region)?;
    BtrfsSuperblock::parse_superblock_region(&recovered_region)
        .context("recovered primary btrfs superblock failed structural validation")?;
    verify_btrfs_superblock_checksum(&recovered_region)
        .context("recovered primary btrfs superblock failed checksum validation")?;

    byte_dev
        .write_all_at(&cx, ByteOffset(primary_offset), &recovered_region)
        .context("failed to write recovered primary btrfs superblock")?;
    byte_dev
        .sync(&cx)
        .context("failed to sync image after primary btrfs superblock recovery")?;

    Ok(Some(source))
}

fn detect_flavor_with_optional_btrfs_bootstrap(
    cx: &Cx,
    path: &PathBuf,
    allow_bootstrap: bool,
    limitations: &mut Vec<String>,
) -> Result<(FsFlavor, Option<BtrfsSuperblockRecoverySource>)> {
    match detect_filesystem_at_path(cx, path) {
        Ok(flavor) => Ok((flavor, None)),
        Err(detect_error) => {
            if !allow_bootstrap {
                return Err(detect_error).with_context(|| {
                    format!("failed to detect ext4/btrfs metadata in {}", path.display())
                });
            }

            let source = match recover_primary_btrfs_superblock_from_backup(path, limitations) {
                Ok(Some(source)) => source,
                Ok(None) => {
                    return Err(detect_error).with_context(|| {
                        format!(
                            "failed to detect ext4/btrfs metadata in {}; bootstrap btrfs superblock \
                             recovery found no valid backup mirrors",
                            path.display()
                        )
                    });
                }
                Err(recover_error) => {
                    return Err(detect_error).with_context(|| {
                        format!(
                            "failed to detect ext4/btrfs metadata in {}; bootstrap btrfs superblock \
                             recovery failed: {recover_error:#}",
                            path.display()
                        )
                    });
                }
            };
            limitations.push(format!(
                "bootstrap restored primary btrfs superblock from backup mirror at byte offset {} \
                 (generation={})",
                source.offset, source.generation
            ));

            let flavor = detect_filesystem_at_path(cx, path).with_context(|| {
                format!(
                    "failed to detect ext4/btrfs metadata in {} after bootstrap btrfs superblock \
                     recovery",
                    path.display()
                )
            })?;
            Ok((flavor, Some(source)))
        }
    }
}

fn parse_btrfs_root_item_bytenr(data: &[u8]) -> Result<u64> {
    if data.len() < 184 {
        bail!(
            "btrfs root item payload too short: expected at least 184 bytes, got {}",
            data.len()
        );
    }
    let mut bytenr_raw = [0_u8; 8];
    bytenr_raw.copy_from_slice(&data[176..184]);
    let bytenr = u64::from_le_bytes(bytenr_raw);
    if bytenr == 0 {
        bail!("btrfs root item bytenr must be non-zero");
    }
    Ok(bytenr)
}

fn parse_btrfs_block_group_total_bytes(data: &[u8]) -> Result<u64> {
    if data.len() < 16 {
        bail!(
            "btrfs block-group payload too short: expected at least 16 bytes, got {}",
            data.len()
        );
    }
    let mut total_raw = [0_u8; 8];
    total_raw.copy_from_slice(&data[8..16]);
    let total_bytes = u64::from_le_bytes(total_raw);
    if total_bytes == 0 {
        bail!("btrfs block-group total_bytes must be non-zero");
    }
    Ok(total_bytes)
}

fn build_btrfs_repair_group_spec(
    group: u32,
    logical_start: u64,
    logical_bytes: u64,
    block_size: u32,
    chunks: &[BtrfsChunkEntry],
) -> Result<BtrfsRepairGroupSpec> {
    if logical_bytes == 0 {
        bail!("btrfs block group {group} has zero logical span");
    }
    let logical_end = logical_start
        .checked_add(logical_bytes.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("btrfs block group {group} logical range overflow"))?;
    let start_mapping = map_logical_to_physical(chunks, logical_start)
        .with_context(|| format!("failed to map btrfs group {group} logical start"))?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "btrfs block group {group} logical start {logical_start} is not covered by any chunk"
            )
        })?;
    let end_mapping = map_logical_to_physical(chunks, logical_end)
        .with_context(|| format!("failed to map btrfs group {group} logical end"))?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "btrfs block group {group} logical end {logical_end} is not covered by any chunk"
            )
        })?;
    if start_mapping.devid != end_mapping.devid {
        bail!(
            "btrfs block group {group} spans multiple devices (start devid={}, end devid={})",
            start_mapping.devid,
            end_mapping.devid
        );
    }
    let expected_end_physical = start_mapping
        .physical
        .checked_add(logical_bytes.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("btrfs block group {group} physical range overflow"))?;
    if end_mapping.physical != expected_end_physical {
        bail!(
            "btrfs block group {group} spans non-contiguous chunk mapping (start_physical={}, end_physical={}, expected_end={expected_end_physical})",
            start_mapping.physical,
            end_mapping.physical
        );
    }
    let block_size_u64 = u64::from(block_size);
    let start_block = start_mapping.physical / block_size_u64;
    let end_exclusive = expected_end_physical
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("btrfs block group {group} end overflow"))?;
    let end_block = end_exclusive.div_ceil(block_size_u64);
    let block_count = end_block.saturating_sub(start_block);
    if block_count == 0 {
        bail!("btrfs block group {group} maps to zero scrub blocks");
    }

    let blocks_per_group = u32::try_from(block_count).with_context(|| {
        format!("btrfs block group {group} block count does not fit u32 ({block_count})")
    })?;
    // Compute tail layout for RaptorQ repair symbols (same scheme as ext4).
    let source_and_repair_budget = if blocks_per_group > REPAIR_DESC_SLOT_COUNT + 1 {
        blocks_per_group.saturating_sub(REPAIR_DESC_SLOT_COUNT)
    } else {
        // Group too small for repair metadata — treat entire span as source.
        blocks_per_group
    };
    let desired_repair = RepairGroupDescExt::compute_repair_block_count(
        source_and_repair_budget,
        DEFAULT_REPAIR_OVERHEAD_RATIO,
    )
    .max(1);
    let repair_block_count = desired_repair
        .min(source_and_repair_budget.saturating_sub(1))
        .max(1);
    let source_block_count = source_and_repair_budget.saturating_sub(repair_block_count);
    let layout = RepairGroupLayout::new(
        GroupNumber(group),
        BlockNumber(start_block),
        blocks_per_group,
        DEFAULT_REPAIR_VALIDATION_BLOCK_COUNT,
        repair_block_count,
    )?;

    Ok(BtrfsRepairGroupSpec {
        group,
        logical_start,
        logical_bytes,
        physical_start_block: BlockNumber(start_block),
        physical_block_count: block_count,
        source_block_count,
        layout,
    })
}

fn discover_btrfs_repair_group_specs(
    path: &PathBuf,
    block_size: u32,
) -> Result<Vec<BtrfsRepairGroupSpec>> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let fs = OpenFs::from_device(&cx, Box::new(byte_dev), &OpenOptions::default())
        .with_context(|| format!("failed to open btrfs image at {}", path.display()))?;
    if !fs.is_btrfs() {
        bail!("image is not btrfs");
    }
    let chunks = fs
        .btrfs_context()
        .ok_or_else(|| anyhow::anyhow!("btrfs context is unavailable"))?
        .chunks
        .clone();

    let root_entries = fs
        .walk_btrfs_root_tree(&cx)
        .context("failed to walk btrfs root tree")?;
    let extent_root_entry = root_entries
        .iter()
        .find(|entry| {
            entry.key.objectid == BTRFS_EXTENT_TREE_OBJECTID
                && entry.key.item_type == BTRFS_ROOT_ITEM_TYPE
        })
        .ok_or_else(|| anyhow::anyhow!("failed to locate btrfs extent tree root item"))?;
    let extent_root_bytenr = parse_btrfs_root_item_bytenr(&extent_root_entry.data)
        .context("failed to parse btrfs extent tree root item")?;

    let extent_entries = fs
        .walk_btrfs_tree(&cx, extent_root_bytenr)
        .with_context(|| format!("failed to walk btrfs extent tree at {extent_root_bytenr}"))?;

    let mut specs = Vec::new();
    let mut group_index = 0_u32;
    for entry in extent_entries
        .iter()
        .filter(|entry| entry.key.item_type == BTRFS_BLOCK_GROUP_ITEM_TYPE)
    {
        let payload_total =
            parse_btrfs_block_group_total_bytes(&entry.data).with_context(|| {
                format!(
                    "failed to parse btrfs block-group payload for key objectid={} offset={}",
                    entry.key.objectid, entry.key.offset
                )
            })?;
        let logical_bytes = if entry.key.offset == 0 {
            payload_total
        } else {
            entry.key.offset
        };
        let spec = build_btrfs_repair_group_spec(
            group_index,
            entry.key.objectid,
            logical_bytes,
            block_size,
            &chunks,
        )?;
        specs.push(spec);
        group_index = group_index.saturating_add(1);
    }

    if specs.is_empty() {
        bail!("no btrfs block groups discovered from extent tree");
    }

    Ok(specs)
}

fn build_ext4_repair_group_spec(sb: &Ext4Superblock, group: u32) -> Result<Ext4RepairGroupSpec> {
    let (scrub_start, scrub_count) = ext4_group_scrub_scope(sb, group)?;
    let blocks_per_group = u32::try_from(scrub_count).with_context(|| {
        format!("ext4 group {group} scrub span does not fit u32 blocks ({scrub_count})")
    })?;
    if blocks_per_group <= REPAIR_DESC_SLOT_COUNT + 1 {
        bail!(
            "ext4 group {group} is too small for repair tail layout (blocks_per_group={blocks_per_group})"
        );
    }

    let source_and_repair_budget = blocks_per_group.saturating_sub(REPAIR_DESC_SLOT_COUNT);
    let desired_repair = RepairGroupDescExt::compute_repair_block_count(
        source_and_repair_budget,
        DEFAULT_REPAIR_OVERHEAD_RATIO,
    )
    .max(1);
    let repair_block_count = desired_repair
        .min(source_and_repair_budget.saturating_sub(1))
        .max(1);
    let source_block_count = source_and_repair_budget.saturating_sub(repair_block_count);
    if source_block_count == 0 {
        bail!(
            "ext4 group {group} has no source payload blocks after reserving repair metadata \
             (blocks_per_group={blocks_per_group}, repair_blocks={repair_block_count})"
        );
    }

    let layout = RepairGroupLayout::new(
        GroupNumber(group),
        scrub_start,
        blocks_per_group,
        DEFAULT_REPAIR_VALIDATION_BLOCK_COUNT,
        repair_block_count,
    )?;

    Ok(Ext4RepairGroupSpec {
        group,
        scrub_start_block: scrub_start,
        scrub_block_count: scrub_count,
        source_first_block: scrub_start,
        source_block_count,
        layout,
    })
}

fn build_ext4_repair_group_specs(sb: &Ext4Superblock) -> Result<Vec<Ext4RepairGroupSpec>> {
    let mut specs = Vec::with_capacity(usize::try_from(sb.groups_count()).unwrap_or(usize::MAX));
    for group in 0..sb.groups_count() {
        specs.push(build_ext4_repair_group_spec(sb, group)?);
    }
    Ok(specs)
}

fn probe_ext4_repair_staleness(
    path: &PathBuf,
    block_size: u32,
    specs: &[Ext4RepairGroupSpec],
) -> Result<Vec<(u32, Ext4RepairStaleness)>> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;

    let mut states = Vec::with_capacity(specs.len());
    for spec in specs {
        let storage = RepairGroupStorage::new(&block_dev, spec.layout);
        let state =
            storage
                .read_group_desc_ext(&cx)
                .map_or(Ext4RepairStaleness::Untracked, |desc| {
                    if desc.repair_generation == 0 {
                        Ext4RepairStaleness::Stale
                    } else {
                        match storage.read_repair_symbols(&cx) {
                            Ok(symbols) if symbols.is_empty() => Ext4RepairStaleness::Stale,
                            Ok(_) => Ext4RepairStaleness::Fresh,
                            Err(_) => Ext4RepairStaleness::Stale,
                        }
                    }
                });
        states.push((spec.group, state));
    }

    Ok(states)
}

fn select_ext4_repair_groups(
    flags: RepairFlags,
    ext4_clean: bool,
    all_groups: &[u32],
    staleness: &[(u32, Ext4RepairStaleness)],
) -> Vec<u32> {
    if flags.full_scrub() || flags.rebuild_symbols() {
        return all_groups.to_vec();
    }

    let stale: Vec<u32> = staleness
        .iter()
        .filter_map(|(group, state)| (*state == Ext4RepairStaleness::Stale).then_some(*group))
        .collect();
    if !stale.is_empty() {
        return stale;
    }

    let has_fresh = staleness
        .iter()
        .any(|(_, state)| *state == Ext4RepairStaleness::Fresh);
    if has_fresh || ext4_clean {
        return Vec::new();
    }

    all_groups.to_vec()
}

fn scrub_ext4_groups_for_repair(
    path: &PathBuf,
    flavor: &FsFlavor,
    specs: &[Ext4RepairGroupSpec],
    groups: &[u32],
    max_threads: Option<u32>,
    limitations: &mut Vec<String>,
) -> Result<ScrubReport> {
    if groups.is_empty() {
        return Ok(ScrubReport {
            findings: Vec::new(),
            blocks_scanned: 0,
            blocks_corrupt: 0,
            blocks_io_error: 0,
        });
    }

    let mut reports = Vec::with_capacity(groups.len());
    for group in groups {
        let spec = specs
            .iter()
            .find(|spec| spec.group == *group)
            .ok_or_else(|| {
                anyhow::anyhow!("ext4 repair group {group} is not available in the computed layout")
            })?;
        let report = scrub_range_for_repair(
            path,
            flavor,
            match flavor {
                FsFlavor::Ext4(sb) => sb.block_size,
                FsFlavor::Btrfs(_) => unreachable!("ext4 helper called for non-ext4 flavor"),
            },
            spec.scrub_start_block,
            spec.scrub_block_count,
            max_threads,
            limitations,
        )
        .with_context(|| format!("failed to scrub ext4 group {group}"))?;
        reports.push(report);
    }

    Ok(merge_scrub_reports(reports))
}

fn corrupt_blocks_at_error_or_higher(report: &ScrubReport) -> Vec<BlockNumber> {
    let mut blocks: Vec<BlockNumber> = report
        .findings
        .iter()
        .filter(|finding| finding.severity >= Severity::Error)
        .map(|finding| finding.block)
        .collect();
    blocks.sort_unstable_by_key(|block| block.0);
    blocks.dedup_by_key(|block| block.0);
    blocks
}

fn group_ext4_corrupt_blocks(
    report: &ScrubReport,
    specs: &[Ext4RepairGroupSpec],
) -> (BTreeMap<u32, Vec<BlockNumber>>, u64) {
    let mut grouped: BTreeMap<u32, Vec<BlockNumber>> = BTreeMap::new();
    let mut outside_source_ranges = 0_u64;

    for block in corrupt_blocks_at_error_or_higher(report) {
        if let Some(spec) = specs.iter().find(|spec| {
            let start = spec.source_first_block.0;
            let end = start.saturating_add(u64::from(spec.source_block_count));
            block.0 >= start && block.0 < end
        }) {
            grouped.entry(spec.group).or_default().push(block);
        } else {
            outside_source_ranges = outside_source_ranges.saturating_add(1);
        }
    }

    (grouped, outside_source_ranges)
}

#[allow(clippy::too_many_arguments)]
fn recover_ext4_corrupt_blocks(
    path: &PathBuf,
    block_size: u32,
    fs_uuid: [u8; 16],
    specs: &[Ext4RepairGroupSpec],
    report: &ScrubReport,
    limitations: &mut Vec<String>,
) -> Result<(u64, u64, Vec<u32>)> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;

    let (grouped, outside_source_ranges) = group_ext4_corrupt_blocks(report, specs);
    if outside_source_ranges > 0 {
        limitations.push(format!(
            "{outside_source_ranges} corrupt block(s) were outside ext4 source repair ranges and \
             could not be reconstructed with block symbols"
        ));
    }

    let mut recovered_blocks = 0_u64;
    let mut unrecovered_blocks = 0_u64;
    let mut repaired_groups = Vec::new();

    for (group, corrupt_blocks) in grouped {
        let spec = specs
            .iter()
            .find(|spec| spec.group == group)
            .ok_or_else(|| {
                anyhow::anyhow!("ext4 recovery group {group} missing from computed repair layout")
            })?;
        let orchestrator = GroupRecoveryOrchestrator::new(
            &block_dev,
            fs_uuid,
            spec.layout,
            spec.source_first_block,
            spec.source_block_count,
        )
        .with_context(|| format!("failed to create recovery orchestrator for group {group}"))?;

        let outcome = orchestrator.recover_from_corrupt_blocks(&cx, &corrupt_blocks);
        let repaired = u64::try_from(outcome.repaired_blocks.len()).unwrap_or(u64::MAX);
        let corrupt = u64::try_from(corrupt_blocks.len()).unwrap_or(u64::MAX);
        let unrecovered = corrupt.saturating_sub(repaired);
        recovered_blocks = recovered_blocks.saturating_add(repaired);
        unrecovered_blocks = unrecovered_blocks.saturating_add(unrecovered);
        if repaired > 0 {
            repaired_groups.push(group);
        }

        if outcome.evidence.outcome != RecoveryOutcome::Recovered || unrecovered > 0 {
            let reason = outcome.evidence.reason.as_deref().unwrap_or("unspecified");
            limitations.push(format!(
                "ext4 group {group} recovery outcome={:?} recovered={repaired} unrecovered={unrecovered} reason={reason}",
                outcome.evidence.outcome
            ));
        }
    }

    Ok((recovered_blocks, unrecovered_blocks, repaired_groups))
}

/// Group corrupt blocks from a scrub report into btrfs block-group buckets.
fn group_btrfs_corrupt_blocks(
    report: &ScrubReport,
    specs: &[BtrfsRepairGroupSpec],
) -> (BTreeMap<u32, Vec<BlockNumber>>, u64) {
    let mut grouped: BTreeMap<u32, Vec<BlockNumber>> = BTreeMap::new();
    let mut outside_source_ranges = 0_u64;

    for block in corrupt_blocks_at_error_or_higher(report) {
        if let Some(spec) = specs.iter().find(|spec| {
            let start = spec.physical_start_block.0;
            let end = start.saturating_add(u64::from(spec.source_block_count));
            block.0 >= start && block.0 < end
        }) {
            grouped.entry(spec.group).or_default().push(block);
        } else {
            outside_source_ranges = outside_source_ranges.saturating_add(1);
        }
    }

    (grouped, outside_source_ranges)
}

/// Attempt RaptorQ symbol reconstruction for corrupt btrfs blocks.
fn recover_btrfs_corrupt_blocks(
    path: &PathBuf,
    block_size: u32,
    fs_uuid: [u8; 16],
    specs: &[BtrfsRepairGroupSpec],
    report: &ScrubReport,
    limitations: &mut Vec<String>,
) -> Result<(u64, u64, Vec<u32>)> {
    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;

    let (grouped, outside_source_ranges) = group_btrfs_corrupt_blocks(report, specs);
    if outside_source_ranges > 0 {
        limitations.push(format!(
            "{outside_source_ranges} corrupt block(s) were outside btrfs source repair ranges and \
             could not be reconstructed with block symbols"
        ));
    }

    let mut recovered_blocks = 0_u64;
    let mut unrecovered_blocks = 0_u64;
    let mut repaired_groups = Vec::new();

    for (group, corrupt_blocks) in grouped {
        let spec = specs
            .iter()
            .find(|spec| spec.group == group)
            .ok_or_else(|| {
                anyhow::anyhow!("btrfs recovery group {group} missing from computed repair layout")
            })?;
        let orchestrator = GroupRecoveryOrchestrator::new(
            &block_dev,
            fs_uuid,
            spec.layout,
            spec.physical_start_block,
            spec.source_block_count,
        )
        .with_context(|| {
            format!("failed to create recovery orchestrator for btrfs group {group}")
        })?;

        let outcome = orchestrator.recover_from_corrupt_blocks(&cx, &corrupt_blocks);
        let repaired = u64::try_from(outcome.repaired_blocks.len()).unwrap_or(u64::MAX);
        let corrupt = u64::try_from(corrupt_blocks.len()).unwrap_or(u64::MAX);
        let unrecovered = corrupt.saturating_sub(repaired);
        recovered_blocks = recovered_blocks.saturating_add(repaired);
        unrecovered_blocks = unrecovered_blocks.saturating_add(unrecovered);
        if repaired > 0 {
            repaired_groups.push(group);
        }

        if outcome.evidence.outcome != RecoveryOutcome::Recovered || unrecovered > 0 {
            let reason = outcome.evidence.reason.as_deref().unwrap_or("unspecified");
            limitations.push(format!(
                "btrfs group {group} recovery outcome={:?} recovered={repaired} unrecovered={unrecovered} reason={reason}",
                outcome.evidence.outcome
            ));
        }
    }

    Ok((recovered_blocks, unrecovered_blocks, repaired_groups))
}

/// Re-encode RaptorQ repair symbols for the given btrfs block groups after
/// recovery, mirroring the ext4 symbol-rebuild workflow.
#[allow(clippy::too_many_lines)]
fn rebuild_btrfs_repair_symbols(
    path: &PathBuf,
    block_size: u32,
    fs_uuid: [u8; 16],
    specs: &[BtrfsRepairGroupSpec],
    groups: &[u32],
    limitations: &mut Vec<String>,
) -> Result<(u64, u64)> {
    if groups.is_empty() {
        return Ok((0, 0));
    }

    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;
    let spec_by_group: BTreeMap<u32, &BtrfsRepairGroupSpec> =
        specs.iter().map(|spec| (spec.group, spec)).collect();

    let mut unique_groups = BTreeSet::new();
    unique_groups.extend(groups.iter().copied());

    let mut rebuilt_groups = 0_u64;
    let mut failed_groups = 0_u64;
    for group in unique_groups {
        let Some(spec) = spec_by_group.get(&group).copied() else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for btrfs group {group}: group layout is unavailable"
            ));
            continue;
        };

        let encoded = match encode_group(
            &cx,
            &block_dev,
            &fs_uuid,
            GroupNumber(group),
            spec.physical_start_block,
            spec.source_block_count,
            spec.layout.repair_block_count,
        ) {
            Ok(encoded) => encoded,
            Err(error) => {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to encode repair symbols for btrfs group {group}: {error}"
                ));
                continue;
            }
        };

        let Ok(symbol_size) = u16::try_from(encoded.symbol_size) else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for btrfs group {group}: symbol_size {} exceeds u16",
                encoded.symbol_size
            ));
            continue;
        };
        let Ok(source_block_count_u16) = u16::try_from(encoded.source_block_count) else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for btrfs group {group}: source_block_count {} exceeds u16",
                encoded.source_block_count
            ));
            continue;
        };

        let storage = RepairGroupStorage::new(&block_dev, spec.layout);
        let current_generation = if let Ok(desc) = storage.read_group_desc_ext(&cx) {
            desc.repair_generation
        } else {
            let bootstrap = RepairGroupDescExt {
                transfer_length: u64::from(encoded.source_block_count)
                    .saturating_mul(u64::from(encoded.symbol_size)),
                symbol_size,
                source_block_count: source_block_count_u16,
                sub_blocks: 1,
                symbol_alignment: 4,
                repair_start_block: spec.layout.repair_start_block(),
                repair_block_count: spec.layout.repair_block_count,
                repair_generation: 0,
                checksum: 0,
            };
            if let Err(error) = storage.write_group_desc_ext(&cx, &bootstrap) {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to bootstrap repair descriptor for btrfs group {group}: {error}"
                ));
                continue;
            }
            0
        };

        let symbols: Vec<(u32, Vec<u8>)> = encoded
            .repair_symbols
            .into_iter()
            .map(|symbol| (symbol.esi, symbol.data))
            .collect();
        let new_generation = current_generation.saturating_add(1).max(1);
        match storage.write_repair_symbols(&cx, &symbols, new_generation) {
            Ok(()) => {
                rebuilt_groups = rebuilt_groups.saturating_add(1);
            }
            Err(error) => {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to publish rebuilt symbols for btrfs group {group}: {error}"
                ));
            }
        }
    }

    Ok((rebuilt_groups, failed_groups))
}

#[allow(clippy::too_many_lines)]
fn rebuild_ext4_repair_symbols(
    path: &PathBuf,
    block_size: u32,
    fs_uuid: [u8; 16],
    specs: &[Ext4RepairGroupSpec],
    groups: &[u32],
    limitations: &mut Vec<String>,
) -> Result<(u64, u64)> {
    if groups.is_empty() {
        return Ok((0, 0));
    }

    let cx = cli_cx();
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let block_dev = ByteBlockDevice::new(byte_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;
    let spec_by_group: BTreeMap<u32, Ext4RepairGroupSpec> = specs
        .iter()
        .copied()
        .map(|spec| (spec.group, spec))
        .collect();

    let mut unique_groups = BTreeSet::new();
    unique_groups.extend(groups.iter().copied());

    let mut rebuilt_groups = 0_u64;
    let mut failed_groups = 0_u64;
    for group in unique_groups {
        let Some(spec) = spec_by_group.get(&group).copied() else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for ext4 group {group}: group layout is unavailable"
            ));
            continue;
        };

        let encoded = match encode_group(
            &cx,
            &block_dev,
            &fs_uuid,
            GroupNumber(group),
            spec.source_first_block,
            spec.source_block_count,
            spec.layout.repair_block_count,
        ) {
            Ok(encoded) => encoded,
            Err(error) => {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to encode repair symbols for ext4 group {group}: {error}"
                ));
                continue;
            }
        };

        let Ok(symbol_size) = u16::try_from(encoded.symbol_size) else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for ext4 group {group}: symbol_size {} exceeds u16",
                encoded.symbol_size
            ));
            continue;
        };
        let Ok(source_block_count_u16) = u16::try_from(encoded.source_block_count) else {
            failed_groups = failed_groups.saturating_add(1);
            limitations.push(format!(
                "cannot rebuild symbols for ext4 group {group}: source_block_count {} exceeds u16",
                encoded.source_block_count
            ));
            continue;
        };

        let storage = RepairGroupStorage::new(&block_dev, spec.layout);
        let current_generation = if let Ok(desc) = storage.read_group_desc_ext(&cx) {
            desc.repair_generation
        } else {
            let bootstrap = RepairGroupDescExt {
                transfer_length: u64::from(encoded.source_block_count)
                    .saturating_mul(u64::from(encoded.symbol_size)),
                symbol_size,
                source_block_count: source_block_count_u16,
                sub_blocks: 1,
                symbol_alignment: 4,
                repair_start_block: spec.layout.repair_start_block(),
                repair_block_count: spec.layout.repair_block_count,
                repair_generation: 0,
                checksum: 0,
            };
            if let Err(error) = storage.write_group_desc_ext(&cx, &bootstrap) {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to bootstrap repair descriptor for ext4 group {group}: {error}"
                ));
                continue;
            }
            0
        };

        let symbols: Vec<(u32, Vec<u8>)> = encoded
            .repair_symbols
            .into_iter()
            .map(|symbol| (symbol.esi, symbol.data))
            .collect();
        let new_generation = current_generation.saturating_add(1).max(1);
        match storage.write_repair_symbols(&cx, &symbols, new_generation) {
            Ok(()) => {
                rebuilt_groups = rebuilt_groups.saturating_add(1);
            }
            Err(error) => {
                failed_groups = failed_groups.saturating_add(1);
                limitations.push(format!(
                    "failed to publish rebuilt symbols for ext4 group {group}: {error}"
                ));
            }
        }
    }

    Ok((rebuilt_groups, failed_groups))
}

fn scrub_range_for_repair(
    path: &PathBuf,
    flavor: &FsFlavor,
    block_size: u32,
    start: BlockNumber,
    count: u64,
    max_threads: Option<u32>,
    limitations: &mut Vec<String>,
) -> Result<ScrubReport> {
    let seed_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let seed_block_dev = ByteBlockDevice::new(seed_dev, block_size)
        .with_context(|| format!("failed to create block device (block_size={block_size})"))?;
    let available_blocks = seed_block_dev.block_count();
    let effective_count = if start.0 >= available_blocks {
        0
    } else {
        count.min(available_blocks.saturating_sub(start.0))
    };
    if effective_count == 0 {
        return Ok(ScrubReport {
            findings: Vec::new(),
            blocks_scanned: 0,
            blocks_corrupt: 0,
            blocks_io_error: 0,
        });
    }

    let (worker_limit, capped_from) = repair_worker_limit(max_threads);
    if let Some(requested) = capped_from {
        limitations.push(format!(
            "--max-threads={requested} exceeds host parallelism; capped at {worker_limit}"
        ));
    }

    if worker_limit <= 1 || effective_count <= 1 {
        let validator = scrub_validator(flavor, block_size);
        let cx = cli_cx();
        return Scrubber::new(&seed_block_dev, &*validator)
            .scrub_range(&cx, start, effective_count)
            .with_context(|| {
                format!(
                    "failed to scrub range starting at block {} for {} blocks",
                    start.0, effective_count
                )
            });
    }

    drop(seed_block_dev);
    let ranges = partition_scrub_range(start, effective_count, worker_limit);
    let mut reports = Vec::with_capacity(ranges.len());

    std::thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(ranges.len());
        for (worker_idx, (range_start, range_count)) in ranges.into_iter().enumerate() {
            let path = path.clone();
            let shard_flavor = flavor.clone();
            handles.push(scope.spawn(move || -> Result<ScrubReport> {
                let cx = cli_cx();
                let byte_dev = FileByteDevice::open(&path)
                    .with_context(|| format!("failed to open image: {}", path.display()))?;
                let block_dev = ByteBlockDevice::new(byte_dev, block_size).with_context(|| {
                    format!("failed to create block device (block_size={block_size})")
                })?;
                let validator = scrub_validator(&shard_flavor, block_size);
                Scrubber::new(&block_dev, &*validator)
                    .scrub_range(&cx, range_start, range_count)
                    .with_context(|| {
                        format!(
                            "repair scrub worker {worker_idx} failed at block {} for {} blocks",
                            range_start.0, range_count
                        )
                    })
            }));
        }

        for handle in handles {
            let shard = handle
                .join()
                .map_err(|_| anyhow::anyhow!("repair scrub worker panicked"))??;
            reports.push(shard);
        }
        Ok(())
    })?;

    Ok(merge_scrub_reports(reports))
}

#[allow(clippy::too_many_lines)]
fn build_repair_output(path: &PathBuf, options: RepairCommandOptions) -> Result<RepairOutput> {
    let flags = options.flags;
    let cx = cli_cx();
    let mut limitations = Vec::new();
    let (flavor, bootstrap_recovery_source) = detect_flavor_with_optional_btrfs_bootstrap(
        &cx,
        path,
        !flags.verify_only(),
        &mut limitations,
    )?;
    let rebuild_symbols_requested = flags.rebuild_symbols() && !flags.verify_only();
    if flags.rebuild_symbols() && flags.verify_only() {
        limitations.push("--rebuild-symbols is ignored when --verify-only is set".to_owned());
    }

    let ext4_recovery = if flags.verify_only() {
        None
    } else {
        match &flavor {
            FsFlavor::Ext4(_) => Some(run_ext4_mount_recovery(path)?),
            FsFlavor::Btrfs(_) => None,
        }
    };
    let byte_dev = FileByteDevice::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?;
    let image_len = byte_dev.len_bytes();

    let (scope, report, execution_stats) = match &flavor {
        FsFlavor::Ext4(sb) => {
            let specs = build_ext4_repair_group_specs(sb)
                .context("failed to compute ext4 repair group layout")?;
            let all_groups: Vec<u32> = specs.iter().map(|spec| spec.group).collect();
            let selected_groups = if let Some(group) = options.block_group {
                vec![group]
            } else {
                let staleness = probe_ext4_repair_staleness(path, sb.block_size, &specs)
                    .context("failed to inspect ext4 repair symbol staleness")?;
                let selection_flags = RepairFlags::empty()
                    .with_full_scrub(flags.full_scrub())
                    .with_rebuild_symbols(rebuild_symbols_requested);
                let selected = select_ext4_repair_groups(
                    selection_flags,
                    ext4_appears_clean_state(sb.state),
                    &all_groups,
                    &staleness,
                );

                if !selection_flags.full_scrub() && !selection_flags.rebuild_symbols() {
                    let stale_count = staleness
                        .iter()
                        .filter(|(_, state)| *state == Ext4RepairStaleness::Stale)
                        .count();
                    let fresh_count = staleness
                        .iter()
                        .filter(|(_, state)| *state == Ext4RepairStaleness::Fresh)
                        .count();
                    let untracked_count = staleness
                        .iter()
                        .filter(|(_, state)| *state == Ext4RepairStaleness::Untracked)
                        .count();

                    if selected.is_empty() {
                        if stale_count == 0 && fresh_count > 0 {
                            limitations.push(
                                "stale-only scope found no stale ext4 groups; scrub skipped (use --full-scrub to force)"
                                    .to_owned(),
                            );
                        } else if stale_count == 0
                            && fresh_count == 0
                            && ext4_appears_clean_state(sb.state)
                        {
                            limitations.push(
                                "stale-only scope found no tracked repair metadata and filesystem is clean; scrub skipped (use --full-scrub to force)"
                                    .to_owned(),
                            );
                        } else if stale_count == 0 && fresh_count == 0 && untracked_count > 0 {
                            limitations.push(
                                "stale-only scope could not infer group staleness from on-image repair metadata; running full scrub because filesystem is not clean"
                                    .to_owned(),
                            );
                        }
                    } else if selected.len() < all_groups.len() {
                        limitations.push(format!(
                            "stale-only scope selected {}/{} ext4 groups (stale={}, fresh={}, untracked={})",
                            selected.len(),
                            all_groups.len(),
                            stale_count,
                            fresh_count,
                            untracked_count
                        ));
                    }
                }

                if selected.is_empty() && !ext4_appears_clean_state(sb.state) {
                    limitations.push(
                        "stale-only scope yielded no explicit candidates on a non-clean filesystem; running full scrub"
                            .to_owned(),
                    );
                    all_groups
                } else {
                    selected
                }
            };

            let scope = if let Some(group) = options.block_group {
                let spec = specs
                    .iter()
                    .find(|spec| spec.group == group)
                    .ok_or_else(|| anyhow::anyhow!("ext4 block group {group} is unavailable"))?;
                RepairScopeOutput::Ext4BlockGroup {
                    group,
                    start_block: spec.scrub_start_block.0,
                    block_count: spec.scrub_block_count,
                }
            } else {
                RepairScopeOutput::Full
            };

            let selected_set: BTreeSet<u32> = selected_groups.iter().copied().collect();
            let selected_specs: Vec<Ext4RepairGroupSpec> = specs
                .iter()
                .copied()
                .filter(|spec| selected_set.contains(&spec.group))
                .collect();
            let mut report = scrub_ext4_groups_for_repair(
                path,
                &flavor,
                &specs,
                &selected_groups,
                options.max_threads,
                &mut limitations,
            )
            .context("failed to scrub ext4 image")?;

            let mut stats = RepairExecutionStats::default();
            if !flags.verify_only()
                && !selected_specs.is_empty()
                && count_blocks_at_severity_or_higher(&report, Severity::Error) > 0
            {
                let (_recovered, unrecovered, repaired_groups) = recover_ext4_corrupt_blocks(
                    path,
                    sb.block_size,
                    sb.uuid,
                    &selected_specs,
                    &report,
                    &mut limitations,
                )
                .context("failed to run ext4 block-symbol recovery")?;
                stats.recovery_attempted = true;
                stats.recovery_unrecovered_blocks = unrecovered;

                if !repaired_groups.is_empty() {
                    let (rebuilt, failed) = rebuild_ext4_repair_symbols(
                        path,
                        sb.block_size,
                        sb.uuid,
                        &selected_specs,
                        &repaired_groups,
                        &mut limitations,
                    )
                    .context("failed to refresh ext4 repair symbols after recovery")?;
                    stats.symbol_rebuild_attempted = true;
                    stats.symbol_rebuild_groups =
                        stats.symbol_rebuild_groups.saturating_add(rebuilt);
                    stats.symbol_rebuild_failed_groups =
                        stats.symbol_rebuild_failed_groups.saturating_add(failed);
                }
            }

            if rebuild_symbols_requested && !selected_specs.is_empty() {
                let (rebuilt, failed) = rebuild_ext4_repair_symbols(
                    path,
                    sb.block_size,
                    sb.uuid,
                    &selected_specs,
                    &selected_groups,
                    &mut limitations,
                )
                .context("failed to rebuild ext4 repair symbols")?;
                stats.symbol_rebuild_attempted = true;
                stats.symbol_rebuild_groups = stats.symbol_rebuild_groups.saturating_add(rebuilt);
                stats.symbol_rebuild_failed_groups =
                    stats.symbol_rebuild_failed_groups.saturating_add(failed);
            }

            if !flags.verify_only() && (stats.recovery_attempted || stats.symbol_rebuild_attempted)
            {
                report = scrub_ext4_groups_for_repair(
                    path,
                    &flavor,
                    &specs,
                    &selected_groups,
                    options.max_threads,
                    &mut limitations,
                )
                .context("failed to verify ext4 image after repair writes")?;
            }

            if stats.symbol_rebuild_failed_groups > 0 {
                limitations.push(format!(
                    "symbol re-encoding failed for {} ext4 group(s)",
                    stats.symbol_rebuild_failed_groups
                ));
            }

            (scope, report, stats)
        }
        FsFlavor::Btrfs(sb) => {
            if !flags.full_scrub() && !rebuild_symbols_requested && options.block_group.is_none() {
                limitations.push(
                    "stale-only mode currently maps to full btrfs scrub because btrfs group-level staleness tracking is not exposed"
                        .to_owned(),
                );
            }

            let block_size = choose_btrfs_scrub_block_size(image_len, sb.nodesize, sb.sectorsize)
                .with_context(|| {
                    format!(
                        "failed to derive aligned btrfs scrub block size (len_bytes={image_len}, nodesize={}, sectorsize={})",
                        sb.nodesize, sb.sectorsize
                    )
                })?;
            let (scope, scrub_start, scrub_count) = if let Some(group) = options.block_group {
                let specs = discover_btrfs_repair_group_specs(path, block_size)
                    .context("failed to discover btrfs block groups for scoped repair")?;
                let spec = specs
                    .iter()
                    .find(|spec| spec.group == group)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "btrfs block group {group} is unavailable (valid range: 0..{})",
                            specs.len().saturating_sub(1)
                        )
                    })?;
                (
                    RepairScopeOutput::BtrfsBlockGroup {
                        group,
                        logical_start: spec.logical_start,
                        logical_bytes: spec.logical_bytes,
                        start_block: spec.physical_start_block.0,
                        block_count: spec.physical_block_count,
                    },
                    spec.physical_start_block,
                    spec.physical_block_count,
                )
            } else {
                (RepairScopeOutput::Full, BlockNumber(0), u64::MAX)
            };

            let mut report = scrub_range_for_repair(
                path,
                &flavor,
                block_size,
                scrub_start,
                scrub_count,
                options.max_threads,
                &mut limitations,
            )
            .context("failed to scrub btrfs image")?;

            let mut stats = RepairExecutionStats {
                recovery_attempted: bootstrap_recovery_source.is_some(),
                ..RepairExecutionStats::default()
            };
            let mut verify_after_repair_writes = false;
            let primary_superblock = primary_btrfs_superblock_block(block_size);
            let superblock_in_scope =
                block_range_contains(scrub_start, scrub_count, primary_superblock);
            if !flags.verify_only()
                && superblock_in_scope
                && report_has_error_or_higher_for_block(&report, primary_superblock)
            {
                stats.recovery_attempted = true;
                match recover_primary_btrfs_superblock_from_backup(path, &mut limitations) {
                    Ok(Some(source)) => {
                        verify_after_repair_writes = true;
                        limitations.push(format!(
                            "restored primary btrfs superblock from backup mirror at byte offset {} (generation={})",
                            source.offset, source.generation
                        ));
                    }
                    Ok(None) => {
                        limitations.push(
                            "btrfs superblock corruption detected, but no valid backup superblock mirror was available for restoration"
                                .to_owned(),
                        );
                    }
                    Err(error) => {
                        limitations.push(format!(
                            "btrfs superblock recovery attempt failed: {error:#}"
                        ));
                    }
                }
            }

            if !flags.verify_only() {
                match repair_corrupt_btrfs_superblock_mirrors_from_primary(
                    path,
                    block_size,
                    scrub_start,
                    scrub_count,
                    &mut limitations,
                ) {
                    Ok(mirror_outcome) => {
                        if mirror_outcome.attempted {
                            stats.recovery_attempted = true;
                            if mirror_outcome.repaired > 0 {
                                verify_after_repair_writes = true;
                            }
                        }
                    }
                    Err(error) => {
                        stats.recovery_attempted = true;
                        limitations.push(format!(
                            "btrfs superblock mirror recovery attempt failed: {error:#}"
                        ));
                    }
                }
            }

            // Attempt RaptorQ block-symbol recovery for non-superblock corruption.
            let mut btrfs_repaired_groups: Vec<u32> = Vec::new();
            let mut btrfs_specs_for_rebuild: Vec<BtrfsRepairGroupSpec> = Vec::new();
            if !flags.verify_only()
                && count_blocks_at_severity_or_higher(&report, Severity::Error) > 0
            {
                match discover_btrfs_repair_group_specs(path, block_size) {
                    Ok(btrfs_specs) if !btrfs_specs.is_empty() => {
                        btrfs_specs_for_rebuild.clone_from(&btrfs_specs);
                        match recover_btrfs_corrupt_blocks(
                            path,
                            block_size,
                            sb.fsid,
                            &btrfs_specs,
                            &report,
                            &mut limitations,
                        ) {
                            Ok((recovered, unrecovered, repaired_groups)) => {
                                stats.recovery_attempted = true;
                                stats.recovery_unrecovered_blocks = unrecovered;
                                btrfs_repaired_groups = repaired_groups;
                                if recovered > 0 {
                                    verify_after_repair_writes = true;
                                }
                            }
                            Err(error) => {
                                stats.recovery_attempted = true;
                                limitations.push(format!(
                                    "btrfs block-symbol recovery attempt failed: {error:#}"
                                ));
                            }
                        }
                    }
                    Ok(_) => {
                        limitations.push(
                            "no btrfs block groups discovered; block-symbol recovery skipped"
                                .to_owned(),
                        );
                    }
                    Err(error) => {
                        limitations.push(format!(
                            "failed to discover btrfs repair group layout: {error:#}"
                        ));
                    }
                }
            }

            // Rebuild repair symbols for repaired groups or on explicit request.
            if !flags.verify_only() {
                let rebuild_groups = if rebuild_symbols_requested {
                    // --rebuild-symbols: discover all groups and rebuild.
                    if btrfs_specs_for_rebuild.is_empty() {
                        btrfs_specs_for_rebuild =
                            discover_btrfs_repair_group_specs(path, block_size).unwrap_or_default();
                    }
                    btrfs_specs_for_rebuild
                        .iter()
                        .map(|spec| spec.group)
                        .collect::<Vec<_>>()
                } else {
                    btrfs_repaired_groups
                };
                if !rebuild_groups.is_empty() {
                    stats.symbol_rebuild_attempted = true;
                    match rebuild_btrfs_repair_symbols(
                        path,
                        block_size,
                        sb.fsid,
                        &btrfs_specs_for_rebuild,
                        &rebuild_groups,
                        &mut limitations,
                    ) {
                        Ok((rebuilt, failed)) => {
                            stats.symbol_rebuild_groups = rebuilt;
                            stats.symbol_rebuild_failed_groups = failed;
                            if rebuilt > 0 {
                                verify_after_repair_writes = true;
                            }
                        }
                        Err(error) => {
                            limitations
                                .push(format!("btrfs symbol rebuild attempt failed: {error:#}"));
                        }
                    }
                }
            }

            if !flags.verify_only() && verify_after_repair_writes {
                report = scrub_range_for_repair(
                    path,
                    &flavor,
                    block_size,
                    scrub_start,
                    scrub_count,
                    options.max_threads,
                    &mut limitations,
                )
                .context("failed to verify btrfs image after repairs")?;
            }

            (scope, report, stats)
        }
    };
    let scrub = repair_scrub_from_report(&report);
    let action = if flags.verify_only() {
        RepairActionOutput::VerifyOnly
    } else if ext4_recovery.is_some()
        || execution_stats.recovery_attempted
        || execution_stats.symbol_rebuild_attempted
    {
        RepairActionOutput::RepairRequested
    } else if scrub.error_or_higher > 0 {
        limitations.push(
            "repair found actionable corruption, but no write-side workflow is available for the detected corruption class"
                .to_owned(),
        );
        RepairActionOutput::RepairRequested
    } else {
        RepairActionOutput::NoCorruptionDetected
    };

    if execution_stats.recovery_unrecovered_blocks > 0 {
        limitations.push(format!(
            "{} corrupt block(s) remained unrecovered after block-symbol reconstruction attempts",
            execution_stats.recovery_unrecovered_blocks
        ));
    }

    let exit_code = i32::from(
        scrub.error_or_higher > 0
            || execution_stats.recovery_unrecovered_blocks > 0
            || execution_stats.symbol_rebuild_failed_groups > 0,
    );

    Ok(RepairOutput {
        filesystem: filesystem_name(&flavor).to_owned(),
        scope,
        action,
        scrub,
        ext4_recovery,
        exit_code,
        limitations,
    })
}

fn repair_scrub_from_report(report: &ScrubReport) -> RepairScrubOutput {
    RepairScrubOutput {
        scanned: report.blocks_scanned,
        corrupt: report.blocks_corrupt,
        error_or_higher: count_blocks_at_severity_or_higher(report, Severity::Error),
        io_error: report.blocks_io_error,
    }
}

fn print_repair_output(json: bool, output: &RepairOutput) -> Result<()> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(output).context("serialize repair output")?
        );
        return Ok(());
    }

    println!("FrankenFS Repair");
    println!("filesystem: {}", output.filesystem);
    match &output.scope {
        RepairScopeOutput::Full => println!("scope: full"),
        RepairScopeOutput::Ext4BlockGroup {
            group,
            start_block,
            block_count,
        } => {
            println!(
                "scope: ext4 group {} (blocks {}..{})",
                group,
                start_block,
                start_block.saturating_add(*block_count).saturating_sub(1)
            );
        }
        RepairScopeOutput::BtrfsBlockGroup {
            group,
            logical_start,
            logical_bytes,
            start_block,
            block_count,
        } => {
            println!(
                "scope: btrfs group {} (logical {}..{}, physical blocks {}..{})",
                group,
                logical_start,
                logical_start
                    .saturating_add(*logical_bytes)
                    .saturating_sub(1),
                start_block,
                start_block.saturating_add(*block_count).saturating_sub(1)
            );
        }
    }
    println!("action: {:?}", output.action);
    println!(
        "scrub: scanned={} corrupt={} error_or_higher={} io_errors={}",
        output.scrub.scanned,
        output.scrub.corrupt,
        output.scrub.error_or_higher,
        output.scrub.io_error
    );
    if let Some(recovery) = &output.ext4_recovery {
        println!("ext4_recovery: {}", ext4_recovery_detail(recovery));
    }
    println!("exit_code: {}", output.exit_code);
    if !output.limitations.is_empty() {
        println!("limitations:");
        for limitation in &output.limitations {
            println!("  - {limitation}");
        }
    }

    Ok(())
}

fn evidence_cmd(
    path: &PathBuf,
    json: bool,
    event_type_filter: Option<&str>,
    tail: Option<usize>,
) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::evidence",
        "evidence",
        ledger = %path.display(),
        output_json = json,
        event_type_filter = event_type_filter.unwrap_or(""),
        tail = tail.unwrap_or(0)
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::evidence", "evidence_start");

    let records = load_evidence_records(path, event_type_filter, tail)?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&records).context("serialize evidence records")?
        );
    } else {
        if records.is_empty() {
            println!("No evidence records found.");
            return Ok(());
        }
        println!("FrankenFS Evidence Ledger ({} records)", records.len());
        println!();
        for record in &records {
            print_evidence_record(record);
        }
    }

    info!(
        target: "ffs::cli::evidence",
        record_count = records.len(),
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "evidence_complete"
    );

    Ok(())
}

fn load_evidence_records(
    path: &PathBuf,
    event_type_filter: Option<&str>,
    tail: Option<usize>,
) -> Result<Vec<EvidenceRecord>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open evidence ledger: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut records: Vec<EvidenceRecord> = Vec::new();
    let mut tail_records: VecDeque<EvidenceRecord> = VecDeque::new();
    let mut line_buf = Vec::new();

    loop {
        line_buf.clear();
        let bytes_read = reader
            .read_until(b'\n', &mut line_buf)
            .with_context(|| format!("failed reading evidence ledger line: {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        let Ok(line) = std::str::from_utf8(&line_buf) else {
            // Preserve torn-write behavior: skip malformed byte sequences.
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<EvidenceRecord>(line) else {
            // Preserve torn-write behavior: skip malformed lines.
            continue;
        };
        if let Some(filter) = event_type_filter {
            if evidence_event_type_name(record.event_type) != filter {
                continue;
            }
        }

        if let Some(limit) = tail {
            if limit == 0 {
                continue;
            }
            if tail_records.len() == limit {
                tail_records.pop_front();
            }
            tail_records.push_back(record);
        } else {
            records.push(record);
        }
    }

    if tail.is_some() {
        Ok(tail_records.into_iter().collect())
    } else {
        Ok(records)
    }
}

const fn evidence_event_type_name(event_type: EvidenceEventType) -> &'static str {
    match event_type {
        EvidenceEventType::CorruptionDetected => "corruption_detected",
        EvidenceEventType::RepairAttempted => "repair_attempted",
        EvidenceEventType::RepairSucceeded => "repair_succeeded",
        EvidenceEventType::RepairFailed => "repair_failed",
        EvidenceEventType::ScrubCycleComplete => "scrub_cycle_complete",
        EvidenceEventType::PolicyDecision => "policy_decision",
        EvidenceEventType::SymbolRefresh => "symbol_refresh",
        EvidenceEventType::WalRecovery => "wal_recovery",
        EvidenceEventType::TransactionCommit => "transaction_commit",
        EvidenceEventType::TxnAborted => "txn_aborted",
        EvidenceEventType::SerializationConflict => "serialization_conflict",
        EvidenceEventType::VersionGc => "version_gc",
        EvidenceEventType::SnapshotAdvanced => "snapshot_advanced",
        EvidenceEventType::FlushBatch => "flush_batch",
        EvidenceEventType::BackpressureActivated => "backpressure_activated",
        EvidenceEventType::DirtyBlockDiscarded => "dirty_block_discarded",
        EvidenceEventType::DurabilityPolicyChanged => "durability_policy_changed",
        EvidenceEventType::RefreshPolicyChanged => "refresh_policy_changed",
    }
}

fn print_evidence_record(record: &EvidenceRecord) {
    let ts_secs = record.timestamp_ns / 1_000_000_000;
    let ts_nanos = record.timestamp_ns % 1_000_000_000;
    let event = evidence_event_type_name(record.event_type);

    print!(
        "  [{ts_secs}.{ts_nanos:09}] {event:<24} group={}",
        record.block_group
    );

    if let Some((start, end)) = record.block_range {
        print!(" blocks={start}..{end}");
    }

    print_evidence_record_event_payload(record);
    println!();
}

fn print_evidence_record_event_payload(record: &EvidenceRecord) {
    match record.event_type {
        EvidenceEventType::CorruptionDetected => print_corruption_payload(record),
        EvidenceEventType::RepairAttempted
        | EvidenceEventType::RepairSucceeded
        | EvidenceEventType::RepairFailed => print_repair_payload(record),
        EvidenceEventType::ScrubCycleComplete => print_scrub_cycle_payload(record),
        EvidenceEventType::PolicyDecision => print_policy_payload(record),
        EvidenceEventType::SymbolRefresh => print_symbol_refresh_payload(record),
        EvidenceEventType::WalRecovery => print_wal_recovery_payload(record),
        EvidenceEventType::TransactionCommit => print_transaction_commit_payload(record),
        EvidenceEventType::TxnAborted => print_txn_aborted_payload(record),
        EvidenceEventType::SerializationConflict => print_serialization_conflict_payload(record),
        EvidenceEventType::VersionGc => print_version_gc_payload(record),
        EvidenceEventType::SnapshotAdvanced => print_snapshot_advanced_payload(record),
        EvidenceEventType::FlushBatch => print_flush_batch_payload(record),
        EvidenceEventType::BackpressureActivated => print_backpressure_payload(record),
        EvidenceEventType::DirtyBlockDiscarded => print_dirty_block_discarded_payload(record),
        EvidenceEventType::DurabilityPolicyChanged => {
            print_durability_policy_changed_payload(record);
        }
        EvidenceEventType::RefreshPolicyChanged => print_refresh_policy_changed_payload(record),
    }
}

fn print_corruption_payload(record: &EvidenceRecord) {
    if let Some(c) = record.corruption.as_ref() {
        print!(
            " blocks_affected={} kind={} severity={}",
            c.blocks_affected, c.corruption_kind, c.severity
        );
    }
}

fn print_repair_payload(record: &EvidenceRecord) {
    if let Some(r) = record.repair.as_ref() {
        print!(
            " corrupt={} symbols={}/{} verify={}",
            r.corrupt_count, r.symbols_used, r.symbols_available, r.verify_pass
        );
        if let Some(reason) = r.reason.as_ref() {
            print!(" reason=\"{reason}\"");
        }
    }
}

fn print_scrub_cycle_payload(record: &EvidenceRecord) {
    if let Some(s) = record.scrub_cycle.as_ref() {
        print!(
            " scanned={} corrupt={} io_errors={} findings={}",
            s.blocks_scanned, s.blocks_corrupt, s.blocks_io_error, s.findings_count
        );
    }
}

fn print_policy_payload(record: &EvidenceRecord) {
    if let Some(p) = record.policy.as_ref() {
        print!(
            " posterior={:.4} overhead={:.3} risk_bound={:.1e} decision=\"{}\"",
            p.corruption_posterior, p.overhead_ratio, p.risk_bound, p.decision
        );
    }
}

fn print_symbol_refresh_payload(record: &EvidenceRecord) {
    if let Some(s) = record.symbol_refresh.as_ref() {
        print!(
            " gen={}→{} symbols={}",
            s.previous_generation, s.new_generation, s.symbols_generated
        );
    }
}

fn print_wal_recovery_payload(record: &EvidenceRecord) {
    if let Some(w) = record.wal_recovery.as_ref() {
        print!(
            " commits={} versions={} discarded={} valid={}/{}",
            w.commits_replayed,
            w.versions_replayed,
            w.records_discarded,
            w.wal_valid_bytes,
            w.wal_total_bytes
        );
        if w.used_checkpoint {
            if let Some(seq) = w.checkpoint_commit_seq {
                print!(" checkpoint_seq={seq}");
            }
        }
    }
}

fn print_transaction_commit_payload(record: &EvidenceRecord) {
    if let Some(t) = record.transaction_commit.as_ref() {
        print!(
            " txn_id={} commit_seq={} write_set_size={} duration_us={}",
            t.txn_id, t.commit_seq, t.write_set_size, t.duration_us
        );
    }
}

fn print_txn_aborted_payload(record: &EvidenceRecord) {
    if let Some(t) = record.txn_aborted.as_ref() {
        let reason = serde_json::to_value(t.reason)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("{:?}", t.reason));
        print!(
            " txn_id={} reason={reason} read_set_size={} write_set_size={}",
            t.txn_id, t.read_set_size, t.write_set_size
        );
        if let Some(detail) = t.detail.as_ref() {
            print!(" detail=\"{detail}\"");
        }
    }
}

fn print_serialization_conflict_payload(record: &EvidenceRecord) {
    if let Some(c) = record.serialization_conflict.as_ref() {
        print!(" txn_id={} conflict_type={}", c.txn_id, c.conflict_type);
        if let Some(conflicting_txn) = c.conflicting_txn {
            print!(" conflicting_txn={conflicting_txn}");
        }
    }
}

fn print_version_gc_payload(record: &EvidenceRecord) {
    if let Some(gc) = record.version_gc.as_ref() {
        print!(
            " block_id={} versions_freed={} oldest_retained_commit_seq={}",
            gc.block_id, gc.versions_freed, gc.oldest_retained_commit_seq
        );
    }
}

fn print_snapshot_advanced_payload(record: &EvidenceRecord) {
    if let Some(s) = record.snapshot_advanced.as_ref() {
        print!(
            " old_commit_seq={} new_commit_seq={} versions_eligible={}",
            s.old_commit_seq, s.new_commit_seq, s.versions_eligible
        );
    }
}

fn print_flush_batch_payload(record: &EvidenceRecord) {
    if let Some(f) = record.flush_batch.as_ref() {
        print!(
            " blocks_flushed={} bytes_written={} flush_duration_us={}",
            f.blocks_flushed, f.bytes_written, f.flush_duration_us
        );
    }
}

fn print_backpressure_payload(record: &EvidenceRecord) {
    if let Some(b) = record.backpressure_activated.as_ref() {
        print!(
            " dirty_ratio={:.4} threshold={:.4}",
            b.dirty_ratio, b.threshold
        );
    }
}

fn print_dirty_block_discarded_payload(record: &EvidenceRecord) {
    if let Some(d) = record.dirty_block_discarded.as_ref() {
        let reason = serde_json::to_value(d.reason)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| format!("{:?}", d.reason));
        print!(
            " block_id={} txn_id={} reason={reason}",
            d.block_id, d.txn_id
        );
    }
}

fn print_durability_policy_changed_payload(record: &EvidenceRecord) {
    if let Some(d) = record.durability_policy_changed.as_ref() {
        print!(
            " old_overhead={:.4} new_overhead={:.4} posterior=({:.3},{:.3},{:.4})",
            d.old_overhead, d.new_overhead, d.posterior_alpha, d.posterior_beta, d.posterior_mean
        );
    }
}

fn print_refresh_policy_changed_payload(record: &EvidenceRecord) {
    if let Some(p) = record.refresh_policy_changed.as_ref() {
        print!(
            " policy=\"{}\"->\"{}\" policy_group={}",
            p.old_policy, p.new_policy, p.block_group
        );
    }
}

fn parity(json: bool) -> Result<()> {
    let command_span = info_span!(
        target: "ffs::cli::parity",
        "parity",
        output_json = json
    );
    let _command_guard = command_span.enter();
    let started = Instant::now();
    info!(target: "ffs::cli::parity", "parity_start");

    let report = ParityReport::current();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize parity report")?
        );
    } else {
        println!("FrankenFS Feature Parity Report");
        println!();
        for domain in &report.domains {
            println!(
                "  {:<35} {:>2}/{:<2}  ({:.1}%)",
                domain.domain, domain.implemented, domain.total, domain.coverage_percent
            );
        }
        println!();
        println!(
            "  {:<35} {:>2}/{:<2}  ({:.1}%)",
            "OVERALL",
            report.overall_implemented,
            report.overall_total,
            report.overall_coverage_percent
        );
    }

    info!(
        target: "ffs::cli::parity",
        overall_coverage_percent = report.overall_coverage_percent,
        duration_us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        "parity_complete"
    );

    Ok(())
}

// ── Mkfs command ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct MkfsOutput {
    path: String,
    size_bytes: u64,
    block_size: u32,
    label: String,
    block_count: u64,
    groups_count: u32,
    inodes_count: u32,
}

fn mkfs_cmd(
    output: &PathBuf,
    size_mb: u64,
    block_size: u32,
    label: &str,
    json: bool,
) -> Result<()> {
    if ![1024, 2048, 4096].contains(&block_size) {
        bail!("block_size must be 1024, 2048, or 4096 (got {block_size})");
    }
    if size_mb == 0 {
        bail!("size_mb must be > 0");
    }
    if output.exists() {
        bail!("output file already exists: {}", output.display());
    }

    let size_bytes = size_mb * 1024 * 1024;

    // Create sparse image file.
    let f = std::fs::File::create(output)
        .with_context(|| format!("create image file {}", output.display()))?;
    f.set_len(size_bytes)
        .with_context(|| format!("set image size to {size_bytes}"))?;
    drop(f);

    // Run mkfs.ext4.
    let mkfs_output = std::process::Command::new("mkfs.ext4")
        .args([
            "-F",
            "-b",
            &block_size.to_string(),
            "-L",
            label,
            &output.display().to_string(),
        ])
        .output()
        .context("failed to run mkfs.ext4 (is it installed?)")?;

    if !mkfs_output.status.success() {
        let stderr = String::from_utf8_lossy(&mkfs_output.stderr);
        // Clean up the partial image on failure.
        let _ = std::fs::remove_file(output);
        bail!("mkfs.ext4 failed: {stderr}");
    }

    // Verify the new image by opening it with FrankenFS.
    let cx = cli_cx();
    let fs = OpenFs::open(&cx, output)
        .with_context(|| format!("verify new image at {}", output.display()))?;

    let result = match &fs.flavor {
        FsFlavor::Ext4(sb) => MkfsOutput {
            path: output.display().to_string(),
            size_bytes,
            block_size: sb.block_size,
            label: label.to_owned(),
            block_count: sb.blocks_count,
            groups_count: sb.groups_count(),
            inodes_count: sb.inodes_count,
        },
        FsFlavor::Btrfs(_) => unreachable!("mkfs.ext4 created a btrfs image"),
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).context("serialize mkfs output")?
        );
    } else {
        println!("FrankenFS mkfs");
        println!("  Path:        {}", result.path);
        println!(
            "  Size:        {} MiB ({} bytes)",
            size_mb, result.size_bytes
        );
        println!("  Block size:  {}", result.block_size);
        println!("  Label:       {}", result.label);
        println!("  Blocks:      {}", result.block_count);
        println!("  Groups:      {}", result.groups_count);
        println!("  Inodes:      {}", result.inodes_count);
        println!("Image created successfully.");
    }

    info!(
        target: "ffs::cli",
        path = %output.display(),
        size_bytes,
        block_size,
        label,
        "mkfs_complete"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Cli, Command, DumpCommand, EvidenceEventType, Ext4JournalReplayMode, Ext4RepairStaleness,
        FsckCommandOptions, FsckFlags, LogFormat, RepairCommandOptions, RepairFlags,
        btrfs_super_mirror_offsets, build_btrfs_repair_group_spec, build_ext4_group_info,
        build_fsck_output, build_repair_output, ext4_appears_clean_state, ext4_mount_replay_mode,
        load_evidence_records, merge_scrub_reports, normalize_btrfs_superblock_as_primary,
        partition_scrub_range, read_ext4_group_desc_from_path, read_ext4_inode_from_path,
        read_file_region, repair_worker_limit, select_ext4_repair_groups,
    };
    use clap::Parser;
    use serde_json::Value;
    use std::io::{self, Seek, SeekFrom, Write};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tracing::{info, info_span};
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::MakeWriter;

    #[derive(Clone, Default)]
    struct SharedLogBuffer {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl SharedLogBuffer {
        fn as_string(&self) -> String {
            let bytes = self.bytes.lock().expect("log buffer lock poisoned").clone();
            String::from_utf8(bytes).expect("log buffer must be utf-8")
        }
    }

    struct SharedLogWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for SharedLogWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes
                .lock()
                .expect("log buffer lock poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> MakeWriter<'writer> for SharedLogBuffer {
        type Writer = SharedLogWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            SharedLogWriter {
                bytes: Arc::clone(&self.bytes),
            }
        }
    }

    fn parse_first_json_line(buffer: &SharedLogBuffer) -> Value {
        let logs = buffer.as_string();
        let line = logs
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("expected at least one log line");
        serde_json::from_str(line).expect("line should parse as JSON")
    }

    #[allow(clippy::cast_possible_truncation)]
    fn build_test_ext4_image_with_state(state: u16) -> Vec<u8> {
        const BLOCK_SIZE_LOG: u32 = 2; // 4K blocks
        let block_size = 1024_u32 << BLOCK_SIZE_LOG;
        let image_size: u32 = 128 * 1024;
        let mut image = vec![0_u8; image_size as usize];
        let sb_off = ffs_types::EXT4_SUPERBLOCK_OFFSET;

        // Core superblock fields needed by parser + geometry checks.
        image[sb_off + 0x38..sb_off + 0x3A]
            .copy_from_slice(&ffs_types::EXT4_SUPER_MAGIC.to_le_bytes());
        image[sb_off + 0x18..sb_off + 0x1C].copy_from_slice(&BLOCK_SIZE_LOG.to_le_bytes());
        let blocks_count = image_size / block_size;
        image[sb_off + 0x04..sb_off + 0x08].copy_from_slice(&blocks_count.to_le_bytes());
        image[sb_off..sb_off + 0x04].copy_from_slice(&128_u32.to_le_bytes()); // inodes_count
        image[sb_off + 0x14..sb_off + 0x18].copy_from_slice(&0_u32.to_le_bytes()); // first_data_block
        image[sb_off + 0x20..sb_off + 0x24].copy_from_slice(&blocks_count.to_le_bytes()); // blocks_per_group
        image[sb_off + 0x28..sb_off + 0x2C].copy_from_slice(&128_u32.to_le_bytes()); // inodes_per_group
        image[sb_off + 0x58..sb_off + 0x5A].copy_from_slice(&256_u16.to_le_bytes()); // inode_size
        image[sb_off + 0x4C..sb_off + 0x50].copy_from_slice(&1_u32.to_le_bytes()); // rev_level
        image[sb_off + 0x54..sb_off + 0x58].copy_from_slice(&11_u32.to_le_bytes()); // first_ino
        let filetype: u32 = 0x0002;
        let extents: u32 = 0x0040;
        image[sb_off + 0x60..sb_off + 0x64].copy_from_slice(&(filetype | extents).to_le_bytes());
        image[sb_off + 0x3A..sb_off + 0x3C].copy_from_slice(&state.to_le_bytes());

        // Minimal group descriptor so group-descriptor validation succeeds.
        let gdt_off = block_size as usize;
        image[gdt_off + 0x08..gdt_off + 0x0C].copy_from_slice(&5_u32.to_le_bytes()); // inode table block

        image
    }

    fn build_test_btrfs_superblock(bytenr: u64, generation: u64) -> Vec<u8> {
        let mut sb = vec![0_u8; super::BTRFS_SUPER_INFO_SIZE];
        sb[0x40..0x48].copy_from_slice(&ffs_types::BTRFS_MAGIC.to_le_bytes());
        sb[0x30..0x38].copy_from_slice(&bytenr.to_le_bytes());
        sb[0x48..0x50].copy_from_slice(&generation.to_le_bytes());
        sb[0x90..0x94].copy_from_slice(&4096_u32.to_le_bytes()); // sectorsize
        sb[0x94..0x98].copy_from_slice(&16384_u32.to_le_bytes()); // nodesize
        sb[0x9C..0xA0].copy_from_slice(&4096_u32.to_le_bytes()); // stripesize
        sb[0xC4..0xC6].copy_from_slice(&ffs_types::BTRFS_CSUM_TYPE_CRC32C.to_le_bytes());
        let checksum = crc32c::crc32c(&sb[0x20..super::BTRFS_SUPER_INFO_SIZE]);
        sb[0..4].copy_from_slice(&checksum.to_le_bytes());
        sb
    }

    fn test_btrfs_chunk_entry(
        logical_start: u64,
        length: u64,
        physical_start: u64,
    ) -> ffs_ondisk::BtrfsChunkEntry {
        ffs_ondisk::BtrfsChunkEntry {
            key: ffs_ondisk::BtrfsKey {
                objectid: 0,
                item_type: 0,
                offset: logical_start,
            },
            length,
            owner: 0,
            stripe_len: 0,
            chunk_type: 0,
            io_align: 0,
            io_width: 0,
            sector_size: 4096,
            num_stripes: 1,
            sub_stripes: 1,
            stripes: vec![ffs_ondisk::BtrfsStripe {
                devid: 1,
                offset: physical_start,
                dev_uuid: [0_u8; 16],
            }],
        }
    }

    fn write_sparse_test_image(path: &std::path::Path, image: &[u8]) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.set_len(u64::try_from(image.len()).expect("test image length should fit into u64"))?;

        let mut run_start: Option<usize> = None;
        for (idx, byte) in image.iter().enumerate() {
            if *byte == 0 {
                if let Some(start) = run_start.take() {
                    file.seek(SeekFrom::Start(
                        u64::try_from(start).expect("run start should fit into u64"),
                    ))?;
                    file.write_all(&image[start..idx])?;
                }
            } else if run_start.is_none() {
                run_start = Some(idx);
            }
        }

        if let Some(start) = run_start {
            file.seek(SeekFrom::Start(
                u64::try_from(start).expect("run start should fit into u64"),
            ))?;
            file.write_all(&image[start..])?;
        }

        file.sync_all()
    }

    fn with_temp_image_path<T>(image: &[u8], f: impl FnOnce(PathBuf) -> T) -> T {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        path.push(format!(
            "ffs-cli-fsck-force-{}-{ts}.img",
            std::process::id()
        ));
        write_sparse_test_image(&path, image).expect("write test filesystem image");
        let result = f(path.clone());
        let _ = std::fs::remove_file(path);
        result
    }

    #[test]
    fn read_file_region_reads_exact_window() {
        let image: Vec<u8> = (0_u8..64).collect();
        with_temp_image_path(&image, |path| {
            let region = read_file_region(&path, 16, 8, "test window")
                .expect("region read should succeed for in-bounds window");
            assert_eq!(region, image[16..24]);
        });
    }

    #[test]
    fn read_file_region_rejects_out_of_bounds_window() {
        let image = vec![0_u8; 32];
        with_temp_image_path(&image, |path| {
            let err = read_file_region(&path, 24, 16, "test window")
                .expect_err("region read should fail for out-of-bounds window");
            let message = format!("{err:#}");
            assert!(
                message.contains("out of bounds"),
                "expected out-of-bounds error, got: {message}"
            );
        });
    }

    #[test]
    fn load_evidence_records_skips_invalid_lines_and_filters_by_event_type() {
        let ledger = concat!(
            "{\"timestamp_ns\":1,\"event_type\":\"corruption_detected\",\"block_group\":1}\n",
            "{not-json\n",
            "\n",
            "{\"timestamp_ns\":2,\"event_type\":\"repair_failed\",\"block_group\":2}\n",
            "{\"timestamp_ns\":3,\"event_type\":\"repair_failed\",\"block_group\":3}\n",
        );
        with_temp_image_path(ledger.as_bytes(), |path| {
            let records = load_evidence_records(&path, Some("repair_failed"), None)
                .expect("filtered evidence read should succeed");
            assert_eq!(records.len(), 2);
            assert!(
                records
                    .iter()
                    .all(|record| record.event_type == EvidenceEventType::RepairFailed)
            );
            assert_eq!(records[0].block_group, 2);
            assert_eq!(records[1].block_group, 3);
        });
    }

    #[test]
    fn load_evidence_records_tail_keeps_last_filtered_records() {
        let ledger = concat!(
            "{\"timestamp_ns\":1,\"event_type\":\"repair_failed\",\"block_group\":1}\n",
            "{\"timestamp_ns\":2,\"event_type\":\"corruption_detected\",\"block_group\":2}\n",
            "{\"timestamp_ns\":3,\"event_type\":\"repair_failed\",\"block_group\":3}\n",
            "{\"timestamp_ns\":4,\"event_type\":\"repair_failed\",\"block_group\":4}\n",
        );
        with_temp_image_path(ledger.as_bytes(), |path| {
            let records = load_evidence_records(&path, Some("repair_failed"), Some(2))
                .expect("tailed evidence read should succeed");
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].timestamp_ns, 3);
            assert_eq!(records[1].timestamp_ns, 4);
            assert_eq!(records[0].block_group, 3);
            assert_eq!(records[1].block_group, 4);
        });
    }

    #[test]
    fn load_evidence_records_skips_non_utf8_lines() {
        let mut ledger = Vec::new();
        ledger.extend_from_slice(
            b"{\"timestamp_ns\":1,\"event_type\":\"repair_failed\",\"block_group\":1}\n",
        );
        ledger.extend_from_slice(&[0xFF, 0xFE, 0x00, b'\n']);
        ledger.extend_from_slice(
            b"{\"timestamp_ns\":2,\"event_type\":\"repair_failed\",\"block_group\":2}\n",
        );

        with_temp_image_path(&ledger, |path| {
            let records = load_evidence_records(&path, Some("repair_failed"), None)
                .expect("evidence read should tolerate non-utf8 torn lines");
            assert_eq!(records.len(), 2);
            assert_eq!(records[0].timestamp_ns, 1);
            assert_eq!(records[1].timestamp_ns, 2);
        });
    }

    #[test]
    fn build_ext4_group_info_reads_descriptors_from_disk_regions() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let sb =
            super::Ext4Superblock::parse_from_image(&image).expect("parse test ext4 superblock");
        let expected_groups = usize::try_from(sb.groups_count()).expect("groups_count fits usize");

        with_temp_image_path(&image, |path| {
            let groups = build_ext4_group_info(&path, &sb).expect("build ext4 group info");
            assert_eq!(groups.len(), expected_groups);
            assert_eq!(groups[0].group, 0);
        });
    }

    #[test]
    fn read_ext4_group_desc_from_path_reads_single_descriptor_window() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let sb =
            super::Ext4Superblock::parse_from_image(&image).expect("parse test ext4 superblock");

        with_temp_image_path(&image, |path| {
            let (desc, raw) = read_ext4_group_desc_from_path(&path, &sb, 0)
                .expect("read ext4 group descriptor from path");
            assert_eq!(desc.inode_table, 5);
            assert_eq!(raw.len(), usize::from(sb.group_desc_size()));
        });
    }

    #[test]
    fn read_ext4_inode_from_path_reads_inode_window() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let sb =
            super::Ext4Superblock::parse_from_image(&image).expect("parse test ext4 superblock");

        with_temp_image_path(&image, |path| {
            let (inode, raw) = read_ext4_inode_from_path(&path, &sb, ffs_types::InodeNumber(2))
                .expect("read ext4 inode from path");
            assert_eq!(raw.len(), usize::from(sb.inode_size));
            assert_eq!(inode.mode, 0);
        });
    }

    #[test]
    fn log_format_parser_supports_human_and_json() {
        assert_eq!(
            LogFormat::parse("human").expect("parse human"),
            LogFormat::Human
        );
        assert_eq!(
            LogFormat::parse("JSON").expect("parse json"),
            LogFormat::Json
        );
        assert!(LogFormat::parse("invalid").is_err());
    }

    #[test]
    fn json_log_serializes_domain_fields() {
        let buffer = SharedLogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_env_filter(EnvFilter::new("info"))
            .with_writer(buffer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            info!(
                target: "ffs::test",
                event_name = "transaction_commit",
                txn_id = 42_u64,
                write_set_size = 3_u64,
                duration_us = 900_u64,
                "transaction_commit"
            );
        });

        let json = parse_first_json_line(&buffer);
        assert_eq!(json["event_name"], "transaction_commit");
        assert_eq!(json["txn_id"], 42);
        assert_eq!(json["write_set_size"], 3);
        assert_eq!(json["duration_us"], 900);
        assert_eq!(json["target"], "ffs::test");
        assert_eq!(json["level"], "INFO");
    }

    #[test]
    fn json_log_preserves_span_context() {
        let buffer = SharedLogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_env_filter(EnvFilter::new("info"))
            .with_writer(buffer.clone())
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let span = info_span!("mount", image = "/tmp/ext4.img", mode = "ro");
            let _guard = span.enter();
            info!(target: "ffs::test", action = "mount_begin", "mount_begin");
        });

        let json = parse_first_json_line(&buffer);
        assert_eq!(json["action"], "mount_begin");
        assert_eq!(json["span"]["name"], "mount");
        assert_eq!(json["span"]["image"], "/tmp/ext4.img");
        assert_eq!(json["span"]["mode"], "ro");
    }

    #[test]
    fn ext4_mount_replay_mode_is_persistent_for_rw() {
        assert_eq!(ext4_mount_replay_mode(true), Ext4JournalReplayMode::Apply);
        assert_eq!(
            ext4_mount_replay_mode(false),
            Ext4JournalReplayMode::SimulateOverlay
        );
    }

    #[test]
    fn cli_parses_info_command_with_all_flags() {
        let cli = Cli::try_parse_from([
            "ffs",
            "info",
            "--groups",
            "--mvcc",
            "--repair",
            "--journal",
            "--json",
            "/tmp/fs.img",
        ])
        .expect("info command should parse");

        match cli.command {
            Command::Info {
                image,
                groups,
                mvcc,
                repair,
                journal,
                json,
            } => {
                assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                assert!(groups);
                assert!(mvcc);
                assert!(repair);
                assert!(journal);
                assert!(json);
            }
            _ => panic!("expected info command"),
        }
    }

    #[test]
    fn cli_parses_info_command_without_optional_flags() {
        let cli = Cli::try_parse_from(["ffs", "info", "/tmp/fs.img"])
            .expect("minimal info command should parse");

        match cli.command {
            Command::Info {
                groups,
                mvcc,
                repair,
                journal,
                json,
                ..
            } => {
                assert!(!groups);
                assert!(!mvcc);
                assert!(!repair);
                assert!(!journal);
                assert!(!json);
            }
            _ => panic!("expected info command"),
        }
    }

    #[test]
    fn cli_parses_dump_superblock_command_with_flags() {
        let cli = Cli::try_parse_from([
            "ffs",
            "dump",
            "superblock",
            "--json",
            "--hex",
            "/tmp/fs.img",
        ])
        .expect("dump superblock command should parse");

        match cli.command {
            Command::Dump { command } => match command {
                DumpCommand::Superblock { image, json, hex } => {
                    assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                    assert!(json);
                    assert!(hex);
                }
                _ => panic!("expected dump superblock command"),
            },
            _ => panic!("expected dump command"),
        }
    }

    #[test]
    fn cli_parses_dump_group_command() {
        let cli = Cli::try_parse_from(["ffs", "dump", "group", "7", "/tmp/fs.img"])
            .expect("dump group command should parse");

        match cli.command {
            Command::Dump { command } => match command {
                DumpCommand::Group {
                    group,
                    image,
                    json,
                    hex,
                } => {
                    assert_eq!(group, 7);
                    assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                    assert!(!json);
                    assert!(!hex);
                }
                _ => panic!("expected dump group command"),
            },
            _ => panic!("expected dump command"),
        }
    }

    #[test]
    fn cli_parses_fsck_with_all_flags() {
        let cli = Cli::try_parse_from([
            "ffs",
            "fsck",
            "-r",
            "-f",
            "-v",
            "--block-group",
            "3",
            "--json",
            "/tmp/fs.img",
        ])
        .expect("fsck command should parse");

        match cli.command {
            Command::Fsck {
                image,
                repair,
                force,
                verbose,
                block_group,
                json,
            } => {
                assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                assert!(repair);
                assert!(force);
                assert!(verbose);
                assert_eq!(block_group, Some(3));
                assert!(json);
            }
            _ => panic!("expected fsck command"),
        }
    }

    #[test]
    fn cli_parses_fsck_minimal() {
        let cli = Cli::try_parse_from(["ffs", "fsck", "/tmp/fs.img"]).expect("fsck should parse");

        match cli.command {
            Command::Fsck {
                image,
                repair,
                force,
                verbose,
                block_group,
                json,
            } => {
                assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                assert!(!repair);
                assert!(!force);
                assert!(!verbose);
                assert_eq!(block_group, None);
                assert!(!json);
            }
            _ => panic!("expected fsck command"),
        }
    }

    #[test]
    fn cli_parses_repair_with_all_flags() {
        let cli = Cli::try_parse_from([
            "ffs",
            "repair",
            "--full-scrub",
            "--block-group",
            "9",
            "--rebuild-symbols",
            "--verify-only",
            "--max-threads",
            "4",
            "--json",
            "/tmp/fs.img",
        ])
        .expect("repair command should parse");

        match cli.command {
            Command::Repair {
                image,
                full_scrub,
                block_group,
                rebuild_symbols,
                verify_only,
                max_threads,
                json,
            } => {
                assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                assert!(full_scrub);
                assert_eq!(block_group, Some(9));
                assert!(rebuild_symbols);
                assert!(verify_only);
                assert_eq!(max_threads, Some(4));
                assert!(json);
            }
            _ => panic!("expected repair command"),
        }
    }

    #[test]
    fn cli_parses_repair_minimal() {
        let cli =
            Cli::try_parse_from(["ffs", "repair", "/tmp/fs.img"]).expect("repair should parse");

        match cli.command {
            Command::Repair {
                image,
                full_scrub,
                block_group,
                rebuild_symbols,
                verify_only,
                max_threads,
                json,
            } => {
                assert_eq!(image, std::path::PathBuf::from("/tmp/fs.img"));
                assert!(!full_scrub);
                assert_eq!(block_group, None);
                assert!(!rebuild_symbols);
                assert!(!verify_only);
                assert_eq!(max_threads, None);
                assert!(!json);
            }
            _ => panic!("expected repair command"),
        }
    }

    #[test]
    fn repair_worker_limit_defaults_to_single() {
        let (workers, capped) = repair_worker_limit(None);
        assert_eq!(workers, 1);
        assert_eq!(capped, None);
    }

    #[test]
    fn partition_scrub_range_covers_exact_range() {
        let ranges = partition_scrub_range(super::BlockNumber(10), 11, 3);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], (super::BlockNumber(10), 4));
        assert_eq!(ranges[1], (super::BlockNumber(14), 4));
        assert_eq!(ranges[2], (super::BlockNumber(18), 3));
    }

    #[test]
    fn merge_scrub_reports_accumulates_counts() {
        let merged = merge_scrub_reports(vec![
            super::ScrubReport {
                findings: Vec::new(),
                blocks_scanned: 10,
                blocks_corrupt: 2,
                blocks_io_error: 1,
            },
            super::ScrubReport {
                findings: Vec::new(),
                blocks_scanned: 7,
                blocks_corrupt: 0,
                blocks_io_error: 3,
            },
        ]);
        assert_eq!(merged.blocks_scanned, 17);
        assert_eq!(merged.blocks_corrupt, 2);
        assert_eq!(merged.blocks_io_error, 4);
    }

    #[test]
    fn ext4_clean_state_detection_matches_expected_flags() {
        assert!(ext4_appears_clean_state(0x0001));
        assert!(!ext4_appears_clean_state(0x0000));
        assert!(!ext4_appears_clean_state(0x0001 | 0x0002));
        assert!(!ext4_appears_clean_state(0x0001 | 0x0004));
    }

    #[test]
    fn fsck_skips_clean_ext4_scrub_without_force() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let output = with_temp_image_path(&image, |path| {
            build_fsck_output(
                &path,
                FsckCommandOptions {
                    flags: FsckFlags::empty(),
                    block_group: None,
                },
            )
            .expect("build fsck output for clean ext4 image")
        });

        assert_eq!(output.scrub.scanned, 0);
        assert_eq!(output.scrub.corrupt, 0);
        assert_eq!(output.scrub.error_or_higher, 0);
        let scrub_phase = output
            .phases
            .iter()
            .find(|phase| phase.phase == "checksum_scrub")
            .expect("checksum_scrub phase should be present");
        assert_eq!(scrub_phase.status, "skipped");
        assert!(scrub_phase.detail.contains("pass --force for full scrub"));
        assert!(
            output
                .limitations
                .iter()
                .any(|limitation| limitation.contains("skipped block-level scrub"))
        );
    }

    #[test]
    fn fsck_force_runs_full_scrub_for_clean_ext4() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let output = with_temp_image_path(&image, |path| {
            build_fsck_output(
                &path,
                FsckCommandOptions {
                    flags: FsckFlags::empty().with_force(true),
                    block_group: None,
                },
            )
            .expect("build fsck output for forced clean ext4 image")
        });

        assert!(output.scrub.scanned > 0);
        let scrub_phase = output
            .phases
            .iter()
            .find(|phase| phase.phase == "checksum_scrub")
            .expect("checksum_scrub phase should be present");
        assert_eq!(scrub_phase.status, "ok");
        assert!(
            !output
                .limitations
                .iter()
                .any(|limitation| limitation.contains("skipped block-level scrub"))
        );
    }

    #[test]
    fn repair_selection_prefers_explicit_stale_groups() {
        let all = vec![0, 1, 2, 3];
        let staleness = vec![
            (0, Ext4RepairStaleness::Fresh),
            (1, Ext4RepairStaleness::Stale),
            (2, Ext4RepairStaleness::Untracked),
            (3, Ext4RepairStaleness::Stale),
        ];
        let selected = select_ext4_repair_groups(RepairFlags::empty(), true, &all, &staleness);
        assert_eq!(selected, vec![1, 3]);
    }

    #[test]
    fn repair_selection_skips_when_clean_and_untracked() {
        let all = vec![0, 1];
        let staleness = vec![
            (0, Ext4RepairStaleness::Untracked),
            (1, Ext4RepairStaleness::Untracked),
        ];
        let selected = select_ext4_repair_groups(RepairFlags::empty(), true, &all, &staleness);
        assert!(selected.is_empty());
    }

    #[test]
    fn repair_selection_runs_full_when_dirty_and_untracked() {
        let all = vec![0, 1, 2];
        let staleness = vec![
            (0, Ext4RepairStaleness::Untracked),
            (1, Ext4RepairStaleness::Untracked),
            (2, Ext4RepairStaleness::Untracked),
        ];
        let selected = select_ext4_repair_groups(RepairFlags::empty(), false, &all, &staleness);
        assert_eq!(selected, all);
    }

    #[test]
    fn repair_selection_rebuild_symbols_forces_full_scope() {
        let all = vec![0, 1, 2];
        let staleness = vec![
            (0, Ext4RepairStaleness::Fresh),
            (1, Ext4RepairStaleness::Stale),
            (2, Ext4RepairStaleness::Untracked),
        ];
        let selected = select_ext4_repair_groups(
            RepairFlags::empty().with_rebuild_symbols(true),
            true,
            &all,
            &staleness,
        );
        assert_eq!(selected, all);
    }

    #[test]
    fn repair_verify_only_ignores_rebuild_symbols_write_path() {
        const EXT4_VALID_FS: u16 = 0x0001;
        let image = build_test_ext4_image_with_state(EXT4_VALID_FS);
        let output = with_temp_image_path(&image, |path| {
            build_repair_output(
                &path,
                RepairCommandOptions {
                    flags: RepairFlags::empty()
                        .with_verify_only(true)
                        .with_rebuild_symbols(true),
                    block_group: None,
                    max_threads: None,
                },
            )
            .expect("build repair output for verify-only rebuild request")
        });

        assert!(
            output
                .limitations
                .iter()
                .any(|limitation| limitation.contains("ignored when --verify-only"))
        );
        assert!(matches!(
            output.action,
            super::RepairActionOutput::VerifyOnly
        ));
    }

    #[test]
    fn btrfs_super_mirror_offsets_follow_kernel_layout() {
        let image_len =
            ((16 * 1024_u64) << 24) + u64::try_from(super::BTRFS_SUPER_INFO_SIZE).unwrap_or(0);
        let offsets = btrfs_super_mirror_offsets(image_len);
        assert_eq!(
            offsets,
            vec![64 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024 * 1024]
        );
    }

    #[test]
    fn btrfs_superblock_normalization_retargets_primary_offset() {
        let mut region = build_test_btrfs_superblock(64 * 1024 * 1024, 3);
        normalize_btrfs_superblock_as_primary(&mut region)
            .expect("normalize mirror superblock for primary slot");
        let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(&region)
            .expect("normalized superblock should parse");
        assert_eq!(parsed.bytenr, super::BTRFS_SUPER_INFO_OFFSET as u64);
        ffs_ondisk::verify_btrfs_superblock_checksum(&region)
            .expect("normalized superblock checksum should be valid");
    }

    #[test]
    fn repair_restores_primary_btrfs_superblock_from_backup_mirror() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let mut primary = build_test_btrfs_superblock(primary_offset as u64, 5);
        primary[0] ^= 0xA5; // Corrupt checksum but keep structure parseable.
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let backup = build_test_btrfs_superblock(backup_offset as u64, 9);
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_primary) = with_temp_image_path(&image, |path| {
            build_repair_output(
                &path,
                RepairCommandOptions {
                    flags: RepairFlags::empty(),
                    block_group: None,
                    max_threads: Some(1),
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired primary checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired primary should parse");
                (output, parsed)
            })
            .expect("repair output for btrfs mirror recovery")
        });

        assert!(matches!(
            output.action,
            super::RepairActionOutput::RepairRequested
        ));
        assert_eq!(parsed_primary.bytenr, primary_offset as u64);
        assert!(
            output
                .limitations
                .iter()
                .any(|limitation| limitation.contains("restored primary btrfs superblock"))
        );
    }

    #[test]
    fn repair_restores_corrupt_btrfs_backup_superblock_from_primary() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let primary = build_test_btrfs_superblock(primary_offset as u64, 21);
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let mut backup = build_test_btrfs_superblock(backup_offset as u64, 17);
        backup[0] ^= 0x7E; // Corrupt checksum but keep structure parseable.
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_backup) = with_temp_image_path(&image, |path| {
            build_repair_output(
                &path,
                RepairCommandOptions {
                    flags: RepairFlags::empty(),
                    block_group: None,
                    max_threads: Some(1),
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired backup checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired backup should parse");
                (output, parsed)
            })
            .expect("repair output for btrfs backup mirror recovery")
        });

        assert!(matches!(
            output.action,
            super::RepairActionOutput::RepairRequested
        ));
        assert_eq!(parsed_backup.bytenr, backup_offset as u64);
        assert!(output.limitations.iter().any(|limitation| {
            limitation.contains("restored 1 btrfs superblock mirror(s) from primary superblock")
        }));
    }

    #[test]
    fn mirror_repair_skips_when_corrupt_backup_superblock_outside_scope() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let block_size = 4096_u32;
        let mut image = vec![0_u8; image_len];

        let primary = build_test_btrfs_superblock(primary_offset as u64, 53);
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let mut backup = build_test_btrfs_superblock(backup_offset as u64, 51);
        backup[0] ^= 0x55; // Corrupt checksum but keep structure parseable.
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (outcome, limitations, backup_still_corrupt) = with_temp_image_path(&image, |path| {
            let mut limitations = Vec::new();
            let outcome = super::repair_corrupt_btrfs_superblock_mirrors_from_primary(
                &path,
                block_size,
                super::BlockNumber(0),
                8,
                &mut limitations,
            )
            .expect("run scoped mirror repair outside backup scope");
            let repaired = std::fs::read(&path).expect("read image after scoped mirror repair");
            let sb_region = &repaired[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE];
            let backup_still_corrupt =
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region).is_err();
            (outcome, limitations, backup_still_corrupt)
        });

        assert!(!outcome.attempted);
        assert_eq!(outcome.repaired, 0);
        assert!(backup_still_corrupt);
        assert!(!limitations.iter().any(|limitation| {
            limitation.contains("restored 1 btrfs superblock mirror(s) from primary superblock")
        }));
    }

    #[test]
    fn mirror_repair_restores_corrupt_backup_superblock_when_in_scope() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let block_size = 4096_u32;
        let mut image = vec![0_u8; image_len];

        let primary = build_test_btrfs_superblock(primary_offset as u64, 63);
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let mut backup = build_test_btrfs_superblock(backup_offset as u64, 61);
        backup[0] ^= 0xAA; // Corrupt checksum but keep structure parseable.
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (outcome, limitations, parsed_backup) = with_temp_image_path(&image, |path| {
            let mut limitations = Vec::new();
            let backup_block = super::BlockNumber((backup_offset as u64) / u64::from(block_size));
            let outcome = super::repair_corrupt_btrfs_superblock_mirrors_from_primary(
                &path,
                block_size,
                backup_block,
                1,
                &mut limitations,
            )
            .expect("run scoped mirror repair for backup slot");
            let repaired = std::fs::read(&path).expect("read image after scoped mirror repair");
            let sb_region = &repaired[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE];
            ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                .expect("repaired backup checksum should be valid");
            let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                .expect("repaired backup should parse");
            (outcome, limitations, parsed)
        });

        assert!(outcome.attempted);
        assert_eq!(outcome.repaired, 1);
        assert_eq!(parsed_backup.bytenr, backup_offset as u64);
        assert!(limitations.iter().any(|limitation| {
            limitation.contains("restored 1 btrfs superblock mirror(s) from primary superblock")
        }));
    }

    #[test]
    fn fsck_repair_restores_primary_btrfs_superblock_from_backup_mirror() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let mut primary = build_test_btrfs_superblock(primary_offset as u64, 11);
        primary[0] ^= 0x3C; // Corrupt checksum but keep structure parseable.
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let backup = build_test_btrfs_superblock(backup_offset as u64, 13);
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_primary) = with_temp_image_path(&image, |path| {
            build_fsck_output(
                &path,
                FsckCommandOptions {
                    flags: FsckFlags::empty().with_repair(true),
                    block_group: None,
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired primary checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired primary should parse");
                (output, parsed)
            })
            .expect("fsck output for btrfs mirror recovery")
        });

        assert!(matches!(
            output.repair_status,
            super::FsckRepairStatus::RequestedPerformed
        ));
        let repair_phase = output
            .phases
            .iter()
            .find(|phase| phase.phase == "repair")
            .expect("repair phase should be present");
        assert_eq!(repair_phase.status, "ok");
        assert!(
            repair_phase
                .detail
                .contains("restored primary btrfs superblock")
        );
        assert_eq!(parsed_primary.bytenr, primary_offset as u64);
        assert!(output.scrub.scanned > 0);
    }

    #[test]
    fn fsck_repair_restores_corrupt_btrfs_backup_superblock_from_primary() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let primary = build_test_btrfs_superblock(primary_offset as u64, 29);
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let mut backup = build_test_btrfs_superblock(backup_offset as u64, 26);
        backup[0] ^= 0x19; // Corrupt checksum but keep structure parseable.
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_backup) = with_temp_image_path(&image, |path| {
            build_fsck_output(
                &path,
                FsckCommandOptions {
                    flags: FsckFlags::empty().with_repair(true),
                    block_group: None,
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired backup checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired backup should parse");
                (output, parsed)
            })
            .expect("fsck output for btrfs backup mirror recovery")
        });

        assert!(matches!(
            output.repair_status,
            super::FsckRepairStatus::RequestedPerformed
        ));
        let repair_phase = output
            .phases
            .iter()
            .find(|phase| phase.phase == "repair")
            .expect("repair phase should be present");
        assert_eq!(repair_phase.status, "ok");
        assert!(
            repair_phase
                .detail
                .contains("restored 1 btrfs superblock mirror(s) from primary superblock")
        );
        assert_eq!(parsed_backup.bytenr, backup_offset as u64);
    }

    #[test]
    fn repair_bootstraps_btrfs_detection_from_backup_superblock() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let mut primary = build_test_btrfs_superblock(primary_offset as u64, 31);
        primary[0x40..0x48].fill(0); // Break magic so initial filesystem detection fails.
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let backup = build_test_btrfs_superblock(backup_offset as u64, 33);
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_primary) = with_temp_image_path(&image, |path| {
            build_repair_output(
                &path,
                RepairCommandOptions {
                    flags: RepairFlags::empty(),
                    block_group: None,
                    max_threads: Some(1),
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired primary checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired primary should parse");
                (output, parsed)
            })
            .expect("repair output for btrfs bootstrap recovery")
        });

        assert!(matches!(
            output.action,
            super::RepairActionOutput::RepairRequested
        ));
        assert_eq!(parsed_primary.bytenr, primary_offset as u64);
        assert!(
            output.limitations.iter().any(
                |limitation| limitation.contains("bootstrap restored primary btrfs superblock")
            )
        );
    }

    #[test]
    fn fsck_repair_bootstraps_btrfs_detection_from_backup_superblock() {
        let primary_offset = super::BTRFS_SUPER_INFO_OFFSET;
        let backup_offset = 64 * 1024 * 1024_usize;
        let image_len = backup_offset + super::BTRFS_SUPER_INFO_SIZE + 4096;
        let mut image = vec![0_u8; image_len];

        let mut primary = build_test_btrfs_superblock(primary_offset as u64, 41);
        primary[0x40..0x48].fill(0); // Break magic so initial filesystem detection fails.
        image[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE]
            .copy_from_slice(&primary);

        let backup = build_test_btrfs_superblock(backup_offset as u64, 44);
        image[backup_offset..backup_offset + super::BTRFS_SUPER_INFO_SIZE].copy_from_slice(&backup);

        let (output, parsed_primary) = with_temp_image_path(&image, |path| {
            build_fsck_output(
                &path,
                FsckCommandOptions {
                    flags: FsckFlags::empty().with_repair(true),
                    block_group: None,
                },
            )
            .map(|output| {
                let repaired = std::fs::read(&path).expect("read repaired image");
                let sb_region =
                    &repaired[primary_offset..primary_offset + super::BTRFS_SUPER_INFO_SIZE];
                ffs_ondisk::verify_btrfs_superblock_checksum(sb_region)
                    .expect("repaired primary checksum should be valid");
                let parsed = ffs_ondisk::BtrfsSuperblock::parse_superblock_region(sb_region)
                    .expect("repaired primary should parse");
                (output, parsed)
            })
            .expect("fsck output for btrfs bootstrap recovery")
        });

        assert!(matches!(
            output.repair_status,
            super::FsckRepairStatus::RequestedPerformed
        ));
        let repair_phase = output
            .phases
            .iter()
            .find(|phase| phase.phase == "repair")
            .expect("repair phase should be present");
        assert_eq!(repair_phase.status, "ok");
        assert!(
            repair_phase
                .detail
                .contains("bootstrap restored primary btrfs superblock")
        );
        assert_eq!(parsed_primary.bytenr, primary_offset as u64);
        assert!(output.scrub.scanned > 0);
    }

    #[test]
    fn btrfs_repair_group_spec_maps_contiguous_single_chunk() {
        // Group must be large enough for the repair tail (>= 4 blocks):
        // REPAIR_DESC_SLOT_COUNT (2) + at least 1 repair block + source blocks.
        // Chunk covers [0x1000..0x20FFF], group occupies [0x2000..0x11FFF] (16 blocks).
        let chunks = vec![test_btrfs_chunk_entry(0x1000, 0x20000, 0x8000)];
        let spec = build_btrfs_repair_group_spec(2, 0x2000, 0x10000, 4096, &chunks)
            .expect("map contiguous btrfs group");
        assert_eq!(spec.group, 2);
        assert_eq!(spec.logical_start, 0x2000);
        assert_eq!(spec.logical_bytes, 0x10000);
        // physical_start = chunk_physical + (logical_start - chunk_logical) = 0x8000 + (0x2000 - 0x1000) = 0x9000
        // start_block = 0x9000 / 4096 = 9
        assert_eq!(spec.physical_start_block.0, 9);
        // block_count = 0x10000 / 4096 = 16
        assert_eq!(spec.physical_block_count, 16);
    }

    #[test]
    fn btrfs_repair_group_spec_rejects_unmapped_range() {
        let chunks = vec![test_btrfs_chunk_entry(0x1000, 0x1000, 0x8000)];
        let err = build_btrfs_repair_group_spec(0, 0x4000, 0x1000, 4096, &chunks)
            .expect_err("unmapped logical range should fail");
        let detail = format!("{err:#}");
        assert!(detail.contains("not covered by any chunk"));
    }

    #[test]
    fn btrfs_repair_group_spec_rejects_non_contiguous_chunk_mapping() {
        let chunks = vec![
            test_btrfs_chunk_entry(0x0, 0x1000, 0x10000),
            test_btrfs_chunk_entry(0x1000, 0x1000, 0x30000),
        ];
        let err = build_btrfs_repair_group_spec(1, 0x0, 0x2000, 4096, &chunks)
            .expect_err("discontiguous mapping should fail");
        let detail = format!("{err:#}");
        assert!(detail.contains("non-contiguous chunk mapping"));
    }
}
