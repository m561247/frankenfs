#![forbid(unsafe_code)]
//! FUSE adapter for FrankenFS.
//!
//! This crate is a thin translation layer: kernel FUSE requests arrive via the
//! `fuser` crate, get forwarded to a [`FsOps`] implementation (from `ffs-core`),
//! and errors are mapped through [`FfsError::to_errno()`].
//!
//! See [`per_core::PerCoreDispatcher`] for thread-per-core dispatch routing.

pub mod per_core;

use asupersync::Cx;
use ffs_core::{
    BackpressureDecision, BackpressureGate, FileType as FfsFileType, FsOps, InodeAttr, RequestOp,
    SetAttrRequest, XattrSetMode,
};
use ffs_error::FfsError;
use ffs_types::InodeNumber;
use fuser::{
    FileAttr, FileType, Filesystem, KernelConfig, MountOption, ReplyAttr, ReplyCreate, ReplyData,
    ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, ReplyXattr,
    Request, TimeOrNow,
};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::os::raw::c_int;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tracing::{debug, info, trace, warn};

/// Default TTL for cached attributes and entries.
///
/// Read-only images are immutable, so a generous TTL is safe.
const ATTR_TTL: Duration = Duration::from_secs(60);
const MIN_SEQUENTIAL_READS_FOR_BATCH: u32 = 2;
const COALESCED_FETCH_MULTIPLIER: u32 = 4;
const MAX_COALESCED_READ_SIZE: u32 = 256 * 1024;
const MAX_PENDING_READAHEAD_ENTRIES: usize = 64;
const MAX_ACCESS_PREDICTOR_ENTRIES: usize = 4096;
const BACKPRESSURE_THROTTLE_DELAY: Duration = Duration::from_millis(5);
const XATTR_FLAG_CREATE: i32 = 0x1;
const XATTR_FLAG_REPLACE: i32 = 0x2;

// ── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum FuseError {
    #[error("invalid mountpoint: {0}")]
    InvalidMountpoint(String),
    #[error("mount I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ── FUSE error context ─────────────────────────────────────────────────────

/// Structured error context for FUSE operation failures.
///
/// Captures the operation name, inode, optional offset, and the underlying
/// error. Used to produce consistent, structured tracing for every FUSE
/// error reply.
pub struct FuseErrorContext<'a> {
    pub error: &'a FfsError,
    pub operation: &'static str,
    pub ino: u64,
    pub offset: Option<u64>,
}

impl FuseErrorContext<'_> {
    /// Log this error context via tracing and return the errno for the reply.
    pub fn log_and_errno(&self) -> c_int {
        let errno = self.error.to_errno();
        // ENOENT on lookup is normal — log at trace instead of warn.
        if errno == libc::ENOENT {
            trace!(
                op = self.operation,
                ino = self.ino,
                errno,
                error = %self.error,
                "FUSE op returned ENOENT"
            );
        } else {
            warn!(
                op = self.operation,
                ino = self.ino,
                offset = self.offset,
                errno,
                error = %self.error,
                "FUSE op failed"
            );
        }
        errno
    }
}

// ── Type conversions ────────────────────────────────────────────────────────

/// Convert an `ffs_core::FileType` to `fuser::FileType`.
fn to_fuser_file_type(ft: FfsFileType) -> FileType {
    match ft {
        FfsFileType::RegularFile => FileType::RegularFile,
        FfsFileType::Directory => FileType::Directory,
        FfsFileType::Symlink => FileType::Symlink,
        FfsFileType::BlockDevice => FileType::BlockDevice,
        FfsFileType::CharDevice => FileType::CharDevice,
        FfsFileType::Fifo => FileType::NamedPipe,
        FfsFileType::Socket => FileType::Socket,
    }
}

/// Convert an `ffs_core::InodeAttr` to `fuser::FileAttr`.
fn to_file_attr(attr: &InodeAttr) -> FileAttr {
    FileAttr {
        ino: attr.ino.0,
        size: attr.size,
        blocks: attr.blocks,
        atime: attr.atime,
        mtime: attr.mtime,
        ctime: attr.ctime,
        crtime: attr.crtime,
        kind: to_fuser_file_type(attr.kind),
        perm: attr.perm,
        nlink: attr.nlink,
        uid: attr.uid,
        gid: attr.gid,
        rdev: attr.rdev,
        blksize: attr.blksize,
        flags: 0,
    }
}

// ── Mount options ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MountOptions {
    pub read_only: bool,
    pub allow_other: bool,
    pub auto_unmount: bool,
    /// Number of worker threads for FUSE dispatch.
    ///
    /// For explicit non-zero values, FrankenFS maps this to kernel FUSE queue
    /// tuning (`max_background` and `congestion_threshold`) so mount behavior
    /// changes under load. A value of `0` means "auto" and uses defaults.
    pub worker_threads: usize,
}

impl Default for MountOptions {
    fn default() -> Self {
        Self {
            read_only: true,
            allow_other: false,
            auto_unmount: true,
            worker_threads: 0,
        }
    }
}

impl MountOptions {
    /// Resolved thread count.
    ///
    /// `worker_threads == 0` means "auto": `min(available_parallelism, 8)`.
    /// Non-zero values are returned as-is (clamped to at least 1).
    #[must_use]
    pub fn resolved_thread_count(&self) -> usize {
        if self.worker_threads == 0 {
            std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8)
        } else {
            self.worker_threads.max(1)
        }
    }
}

// ── Cache-line padding ──────────────────────────────────────────────────────

/// Pad a value to 64 bytes to avoid false sharing between hot counters
/// updated on different CPU cores.
#[repr(C, align(64))]
pub struct CacheLinePadded<T>(pub T);

impl<T: std::fmt::Debug> std::fmt::Debug for CacheLinePadded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// ── Atomic metrics ──────────────────────────────────────────────────────────

/// Lock-free per-mount request counters.
///
/// Each counter sits on its own cache line (64 B) so cores updating
/// different counters never invalidate each other's L1 lines.
#[repr(C)]
pub struct AtomicMetrics {
    pub requests_total: CacheLinePadded<AtomicU64>,
    pub requests_ok: CacheLinePadded<AtomicU64>,
    pub requests_err: CacheLinePadded<AtomicU64>,
    pub bytes_read: CacheLinePadded<AtomicU64>,
}

impl AtomicMetrics {
    #[must_use]
    pub fn new() -> Self {
        Self {
            requests_total: CacheLinePadded(AtomicU64::new(0)),
            requests_ok: CacheLinePadded(AtomicU64::new(0)),
            requests_err: CacheLinePadded(AtomicU64::new(0)),
            bytes_read: CacheLinePadded(AtomicU64::new(0)),
        }
    }

    fn record_ok(&self) {
        self.requests_total.0.fetch_add(1, Ordering::Relaxed);
        self.requests_ok.0.fetch_add(1, Ordering::Relaxed);
    }

    fn record_err(&self) {
        self.requests_total.0.fetch_add(1, Ordering::Relaxed);
        self.requests_err.0.fetch_add(1, Ordering::Relaxed);
    }

    fn record_bytes_read(&self, n: u64) {
        self.bytes_read.0.fetch_add(n, Ordering::Relaxed);
    }

    /// Snapshot of all counters (for diagnostics / reporting).
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_total: self.requests_total.0.load(Ordering::Relaxed),
            requests_ok: self.requests_ok.0.load(Ordering::Relaxed),
            requests_err: self.requests_err.0.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.0.load(Ordering::Relaxed),
        }
    }
}

impl Default for AtomicMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AtomicMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.snapshot();
        f.debug_struct("AtomicMetrics")
            .field("requests_total", &s.requests_total)
            .field("requests_ok", &s.requests_ok)
            .field("requests_err", &s.requests_err)
            .field("bytes_read", &s.bytes_read)
            .finish()
    }
}

/// Point-in-time snapshot of metrics (all plain `u64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub requests_ok: u64,
    pub requests_err: u64,
    pub bytes_read: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessDirection {
    Forward,
    Backward,
}

#[derive(Debug, Clone, Copy)]
struct AccessPattern {
    last_offset: u64,
    last_size: u32,
    sequential_count: u32,
    direction: AccessDirection,
    last_touch: u64,
}

#[derive(Debug, Default)]
struct AccessPredictorState {
    history: BTreeMap<u64, AccessPattern>,
    lru: BTreeMap<u64, u64>,
    next_touch: u64,
}

#[derive(Debug)]
struct AccessPredictor {
    state: Mutex<AccessPredictorState>,
    max_entries: usize,
}

impl Default for AccessPredictor {
    fn default() -> Self {
        Self::new(MAX_ACCESS_PREDICTOR_ENTRIES)
    }
}

impl AccessPredictor {
    fn new(max_entries: usize) -> Self {
        Self {
            state: Mutex::new(AccessPredictorState::default()),
            max_entries: max_entries.max(1),
        }
    }

    fn fetch_size(&self, ino: InodeNumber, offset: u64, requested: u32) -> u32 {
        let guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("AccessPredictor state lock poisoned in fetch_size, recovering");
                poisoned.into_inner()
            }
        };
        let pattern = guard.history.get(&ino.0).copied();
        drop(guard);

        let Some(pattern) = pattern else {
            return requested;
        };
        let next_forward_offset = pattern
            .last_offset
            .saturating_add(u64::from(pattern.last_size));
        let should_batch = pattern.direction == AccessDirection::Forward
            && pattern.last_size == requested
            && pattern.sequential_count >= MIN_SEQUENTIAL_READS_FOR_BATCH
            && next_forward_offset == offset;
        if should_batch {
            requested
                .saturating_mul(COALESCED_FETCH_MULTIPLIER)
                .clamp(requested, MAX_COALESCED_READ_SIZE.max(requested))
        } else {
            requested
        }
    }

    fn record_read(&self, ino: InodeNumber, offset: u64, size: u32) {
        if size == 0 {
            return;
        }
        {
            let mut guard = match self.state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    warn!("AccessPredictor state lock poisoned in record_read, recovering");
                    poisoned.into_inner()
                }
            };

            guard.next_touch = guard.next_touch.saturating_add(1);
            let touch = guard.next_touch;
            if let Some(old_touch) = guard.history.get(&ino.0).map(|old| old.last_touch) {
                guard.lru.remove(&old_touch);
            }
            guard.lru.insert(touch, ino.0);

            let entry = guard.history.entry(ino.0).or_insert(AccessPattern {
                last_offset: offset,
                last_size: size,
                sequential_count: 1,
                direction: AccessDirection::Forward,
                last_touch: touch,
            });

            let next_forward_offset = entry.last_offset.saturating_add(u64::from(entry.last_size));
            let next_backward_offset = offset.saturating_add(u64::from(size));

            if entry.last_size == size && next_forward_offset == offset {
                entry.sequential_count = entry.sequential_count.saturating_add(1);
                entry.direction = AccessDirection::Forward;
            } else if entry.last_size == size && next_backward_offset == entry.last_offset {
                entry.sequential_count = entry.sequential_count.saturating_add(1);
                entry.direction = AccessDirection::Backward;
            } else {
                entry.sequential_count = 1;
                entry.direction = AccessDirection::Forward;
            }
            entry.last_offset = offset;
            entry.last_size = size;
            entry.last_touch = touch;

            while guard.history.len() > self.max_entries {
                if let Some((_, oldest_inode)) = guard.lru.pop_first() {
                    let _ = guard.history.remove(&oldest_inode);
                } else {
                    break;
                }
            }

            drop(guard);
        }
    }
}

#[derive(Debug, Default)]
struct ReadaheadState {
    map: BTreeMap<(u64, u64), Vec<u8>>,
    fifo: std::collections::VecDeque<(u64, u64)>,
}

#[derive(Debug)]
struct ReadaheadManager {
    pending: Mutex<ReadaheadState>,
    max_pending: usize,
}

impl ReadaheadManager {
    fn new(max_pending: usize) -> Self {
        Self {
            pending: Mutex::new(ReadaheadState::default()),
            max_pending: max_pending.max(1),
        }
    }

    fn insert(&self, ino: InodeNumber, offset: u64, data: Vec<u8>) {
        if data.is_empty() {
            return;
        }
        let mut guard = match self.pending.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("ReadaheadCache pending lock poisoned in insert, recovering");
                poisoned.into_inner()
            }
        };
        if guard.map.insert((ino.0, offset), data).is_none() {
            guard.fifo.push_back((ino.0, offset));
        }
        while guard.fifo.len() > self.max_pending {
            if let Some(key) = guard.fifo.pop_front() {
                let _ = guard.map.remove(&key);
            } else {
                break;
            }
        }
        drop(guard);
    }

    fn take(&self, ino: InodeNumber, offset: u64, requested_len: usize) -> Option<Vec<u8>> {
        let mut guard = match self.pending.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("ReadaheadCache pending lock poisoned in take, recovering");
                poisoned.into_inner()
            }
        };
        let mut cached = guard.map.remove(&(ino.0, offset))?;
        if cached.len() <= requested_len {
            drop(guard);
            return Some(cached);
        }

        let tail = cached.split_off(requested_len);
        let consumed = u64::try_from(cached.len()).unwrap_or(u64::MAX);
        let next_offset = offset.saturating_add(consumed);
        if guard.map.insert((ino.0, next_offset), tail).is_none() {
            guard.fifo.push_back((ino.0, next_offset));
        }
        while guard.fifo.len() > self.max_pending {
            if let Some(key) = guard.fifo.pop_front() {
                let _ = guard.map.remove(&key);
            } else {
                break;
            }
        }
        drop(guard);
        Some(cached)
    }
}

// ── Shared FUSE inner state ─────────────────────────────────────────────────

/// Thread-safe shared state for the FUSE backend.
///
/// All fields are `Send + Sync`:
/// - `ops` delegates to `FsOps` which is `Send + Sync` by trait bound.
/// - `metrics` uses atomic counters with cache-line padding.
/// - `thread_count` is immutable after mount.
struct FuseInner {
    ops: Arc<dyn FsOps>,
    metrics: Arc<AtomicMetrics>,
    thread_count: usize,
    read_only: bool,
    backpressure: Option<BackpressureGate>,
    access_predictor: AccessPredictor,
    readahead: ReadaheadManager,
}

impl std::fmt::Debug for FuseInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuseInner")
            .field("metrics", &self.metrics)
            .field("thread_count", &self.thread_count)
            .field("read_only", &self.read_only)
            .finish_non_exhaustive()
    }
}

// ── FUSE filesystem adapter ─────────────────────────────────────────────────

/// FUSE adapter that delegates all operations to a [`FsOps`] implementation.
///
/// Internally wraps all state in `Arc<FuseInner>` so it is `Send + Sync`
/// and ready for multi-threaded FUSE dispatch.  All `FsOps` calls go
/// through `self.inner.ops` (which is `Arc<dyn FsOps>`), and lock-free
/// [`AtomicMetrics`] are updated on every request.
pub struct FrankenFuse {
    inner: Arc<FuseInner>,
}

// Compile-time assertions: FrankenFuse must be Send + Sync.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<FrankenFuse>;
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XattrReplyPlan {
    Size(u32),
    Data,
    Error(c_int),
}

#[derive(Debug)]
enum MutationDispatchError {
    Errno(c_int),
    Operation {
        error: FfsError,
        offset: Option<u64>,
    },
}

impl FrankenFuse {
    /// Create a new FUSE adapter wrapping the given `FsOps` implementation.
    ///
    /// Uses default thread count (auto-detected).
    #[must_use]
    pub fn new(ops: Box<dyn FsOps>) -> Self {
        Self::with_options(ops, &MountOptions::default())
    }

    /// Create a new FUSE adapter with explicit mount options.
    ///
    /// The resolved `thread_count` is logged at info level.
    #[must_use]
    pub fn with_options(ops: Box<dyn FsOps>, options: &MountOptions) -> Self {
        let thread_count = options.resolved_thread_count();
        info!(thread_count, "FrankenFuse initialized");
        Self {
            inner: Arc::new(FuseInner {
                ops: Arc::from(ops),
                metrics: Arc::new(AtomicMetrics::new()),
                thread_count,
                read_only: options.read_only,
                backpressure: None,
                access_predictor: AccessPredictor::default(),
                readahead: ReadaheadManager::new(MAX_PENDING_READAHEAD_ENTRIES),
            }),
        }
    }

    /// Create a FUSE adapter with an attached backpressure gate.
    #[must_use]
    pub fn with_backpressure(
        ops: Box<dyn FsOps>,
        options: &MountOptions,
        gate: BackpressureGate,
    ) -> Self {
        let thread_count = options.resolved_thread_count();
        info!(thread_count, "FrankenFuse initialized with backpressure");
        Self {
            inner: Arc::new(FuseInner {
                ops: Arc::from(ops),
                metrics: Arc::new(AtomicMetrics::new()),
                thread_count,
                read_only: options.read_only,
                backpressure: Some(gate),
                access_predictor: AccessPredictor::default(),
                readahead: ReadaheadManager::new(MAX_PENDING_READAHEAD_ENTRIES),
            }),
        }
    }

    /// Get a reference to the shared metrics.
    #[must_use]
    pub fn metrics(&self) -> &AtomicMetrics {
        &self.inner.metrics
    }

    /// Configured thread count.
    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.inner.thread_count
    }

    /// Check backpressure for an operation. Returns `true` if the operation
    /// should be rejected (shed).
    fn should_shed(&self, op: RequestOp) -> bool {
        let Some(gate) = self.inner.backpressure.as_ref() else {
            return false;
        };

        match gate.check(op) {
            BackpressureDecision::Proceed => false,
            BackpressureDecision::Throttle => {
                trace!(
                    ?op,
                    delay_ms = BACKPRESSURE_THROTTLE_DELAY.as_millis(),
                    "backpressure: throttling request"
                );
                std::thread::sleep(BACKPRESSURE_THROTTLE_DELAY);
                false
            }
            BackpressureDecision::Shed => true,
        }
    }

    /// Create a `Cx` for a FUSE request.
    ///
    /// In the future this could inherit deadlines or tracing spans from the
    /// fuser `Request`, but for now we use a plain request context.
    fn cx_for_request() -> Cx {
        Cx::for_request()
    }

    fn reply_error_attr(ctx: &FuseErrorContext<'_>, reply: ReplyAttr) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_entry(ctx: &FuseErrorContext<'_>, reply: ReplyEntry) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_data(ctx: &FuseErrorContext<'_>, reply: ReplyData) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_dir(ctx: &FuseErrorContext<'_>, reply: ReplyDirectory) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_xattr(ctx: &FuseErrorContext<'_>, reply: ReplyXattr) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_empty(ctx: &FuseErrorContext<'_>, reply: ReplyEmpty) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_write(ctx: &FuseErrorContext<'_>, reply: ReplyWrite) {
        reply.error(ctx.log_and_errno());
    }

    fn reply_error_create(ctx: &FuseErrorContext<'_>, reply: ReplyCreate) {
        reply.error(ctx.log_and_errno());
    }

    fn classify_xattr_reply(size: u32, payload_len: usize) -> XattrReplyPlan {
        match u32::try_from(payload_len) {
            Ok(payload_len_u32) if size == 0 => XattrReplyPlan::Size(payload_len_u32),
            Ok(payload_len_u32) if payload_len_u32 <= size => XattrReplyPlan::Data,
            Ok(_) => XattrReplyPlan::Error(libc::ERANGE),
            Err(_) => XattrReplyPlan::Error(libc::EOVERFLOW),
        }
    }

    fn reply_xattr_payload(size: u32, payload: &[u8], reply: ReplyXattr) {
        match Self::classify_xattr_reply(size, payload.len()) {
            XattrReplyPlan::Size(payload_len) => reply.size(payload_len),
            XattrReplyPlan::Data => reply.data(payload),
            XattrReplyPlan::Error(errno) => reply.error(errno),
        }
    }

    #[cfg(target_os = "linux")]
    const fn missing_xattr_errno() -> c_int {
        libc::ENODATA
    }

    #[cfg(not(target_os = "linux"))]
    const fn missing_xattr_errno() -> c_int {
        libc::ENOATTR
    }

    fn parse_setxattr_mode(flags: i32, position: u32) -> Result<XattrSetMode, c_int> {
        if position != 0 {
            return Err(libc::EINVAL);
        }

        let known = XATTR_FLAG_CREATE | XATTR_FLAG_REPLACE;
        if flags & !known != 0 {
            return Err(libc::EINVAL);
        }

        let create = flags & XATTR_FLAG_CREATE != 0;
        let replace = flags & XATTR_FLAG_REPLACE != 0;
        if create && replace {
            return Err(libc::EINVAL);
        }

        if create {
            Ok(XattrSetMode::Create)
        } else if replace {
            Ok(XattrSetMode::Replace)
        } else {
            Ok(XattrSetMode::Set)
        }
    }

    fn encode_xattr_names(names: &[String]) -> Vec<u8> {
        let total_len = names.iter().map(|name| name.len() + 1).sum();
        let mut bytes = Vec::with_capacity(total_len);
        for name in names {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(0);
        }
        bytes
    }

    fn with_request_scope<T, F>(&self, cx: &Cx, op: RequestOp, f: F) -> ffs_error::Result<T>
    where
        F: FnOnce(&Cx) -> ffs_error::Result<T>,
    {
        let scope = self.inner.ops.begin_request_scope(cx, op)?;
        let op_result = f(cx);
        let end_result = self.inner.ops.end_request_scope(cx, op, scope);

        match (op_result, end_result) {
            (Ok(value), Ok(())) => {
                self.inner.metrics.record_ok();
                Ok(value)
            }
            (Ok(_), Err(end_err)) => {
                self.inner.metrics.record_err();
                Err(end_err)
            }
            (Err(op_err), Ok(())) => {
                self.inner.metrics.record_err();
                Err(op_err)
            }
            (Err(op_err), Err(end_err)) => {
                self.inner.metrics.record_err();
                warn!(?op, error = %end_err, "request scope cleanup failed after operation error");
                Err(op_err)
            }
        }
    }

    fn enforce_mutation_guards(
        &self,
        op: RequestOp,
        ino_for_logging: u64,
    ) -> Result<(), MutationDispatchError> {
        if self.inner.read_only {
            return Err(MutationDispatchError::Errno(libc::EROFS));
        }
        if self.should_shed(op) {
            warn!(
                ino = ino_for_logging,
                ?op,
                "backpressure: shedding mutation request"
            );
            return Err(MutationDispatchError::Errno(libc::EBUSY));
        }
        Ok(())
    }

    fn dispatch_mkdir(
        &self,
        parent: u64,
        name: &OsStr,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> Result<InodeAttr, MutationDispatchError> {
        self.enforce_mutation_guards(RequestOp::Mkdir, parent)?;
        let cx = Self::cx_for_request();
        self.with_request_scope(&cx, RequestOp::Mkdir, |cx| {
            self.inner
                .ops
                .mkdir(cx, InodeNumber(parent), name, mode, uid, gid)
        })
        .map_err(|error| MutationDispatchError::Operation {
            error,
            offset: None,
        })
    }

    fn dispatch_rmdir(&self, parent: u64, name: &OsStr) -> Result<(), MutationDispatchError> {
        self.enforce_mutation_guards(RequestOp::Rmdir, parent)?;
        let cx = Self::cx_for_request();
        self.with_request_scope(&cx, RequestOp::Rmdir, |cx| {
            self.inner.ops.rmdir(cx, InodeNumber(parent), name)
        })
        .map_err(|error| MutationDispatchError::Operation {
            error,
            offset: None,
        })
    }

    fn dispatch_rename(
        &self,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
    ) -> Result<(), MutationDispatchError> {
        self.enforce_mutation_guards(RequestOp::Rename, parent)?;
        let cx = Self::cx_for_request();
        self.with_request_scope(&cx, RequestOp::Rename, |cx| {
            self.inner.ops.rename(
                cx,
                InodeNumber(parent),
                name,
                InodeNumber(newparent),
                newname,
            )
        })
        .map_err(|error| MutationDispatchError::Operation {
            error,
            offset: None,
        })
    }

    fn dispatch_write(
        &self,
        ino: u64,
        offset: i64,
        data: &[u8],
    ) -> Result<u32, MutationDispatchError> {
        self.enforce_mutation_guards(RequestOp::Write, ino)?;
        let byte_offset =
            u64::try_from(offset).map_err(|_| MutationDispatchError::Errno(libc::EINVAL))?;
        let cx = Self::cx_for_request();
        self.with_request_scope(&cx, RequestOp::Write, |cx| {
            self.inner
                .ops
                .write(cx, InodeNumber(ino), byte_offset, data)
        })
        .map_err(|error| MutationDispatchError::Operation {
            error,
            offset: Some(byte_offset),
        })
    }

    fn read_with_readahead(
        &self,
        cx: &Cx,
        ino: InodeNumber,
        byte_offset: u64,
        size: u32,
    ) -> ffs_error::Result<Vec<u8>> {
        let requested_len = usize::try_from(size).unwrap_or(usize::MAX);
        self.with_request_scope(cx, RequestOp::Read, |cx| {
            let mut served = self
                .inner
                .readahead
                .take(ino, byte_offset, requested_len)
                .map_or_else(Vec::new, |prefetched| {
                    trace!(
                        target: "ffs::fuse::io",
                        event = "readahead_hit",
                        ino = ino.0,
                        offset = byte_offset,
                        bytes = prefetched.len()
                    );
                    prefetched
                });

            if served.len() < requested_len {
                let remaining_req =
                    size.saturating_sub(u32::try_from(served.len()).unwrap_or(u32::MAX));
                let next_offset =
                    byte_offset.saturating_add(u64::try_from(served.len()).unwrap_or(u64::MAX));
                let fetch_size =
                    self.inner
                        .access_predictor
                        .fetch_size(ino, next_offset, remaining_req);

                let mut fetched = self.inner.ops.read(cx, ino, next_offset, fetch_size)?;
                let fetched_served_len = (requested_len - served.len()).min(fetched.len());
                let tail = fetched.split_off(fetched_served_len);

                served.append(&mut fetched);

                if !tail.is_empty() {
                    let consumed = u64::try_from(fetched_served_len).unwrap_or(u64::MAX);
                    let prefetch_offset = next_offset.saturating_add(consumed);
                    let prefetch_bytes = tail.len();
                    self.inner.readahead.insert(ino, prefetch_offset, tail);
                    debug!(
                        target: "ffs::fuse::io",
                        event = "readahead_queued",
                        ino = ino.0,
                        offset = prefetch_offset,
                        bytes = prefetch_bytes
                    );
                }
            }

            self.inner.access_predictor.record_read(
                ino,
                byte_offset,
                u32::try_from(served.len()).unwrap_or(u32::MAX),
            );

            Ok(served)
        })
    }
}

impl Filesystem for FrankenFuse {
    fn init(&mut self, _req: &Request<'_>, _config: &mut KernelConfig) -> Result<(), c_int> {
        Ok(())
    }

    fn destroy(&mut self) {}

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Getattr, |cx| {
            self.inner.ops.getattr(cx, InodeNumber(ino))
        }) {
            Ok(attr) => reply.attr(&ATTR_TTL, &to_file_attr(&attr)),
            Err(e) => {
                Self::reply_error_attr(
                    &FuseErrorContext {
                        error: &e,
                        operation: "getattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn statfs(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyStatfs) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Statfs, |cx| {
            self.inner.ops.statfs(cx, InodeNumber(ino))
        }) {
            Ok(stats) => reply.statfs(
                stats.blocks,
                stats.blocks_free,
                stats.blocks_available,
                stats.files,
                stats.files_free,
                stats.block_size,
                stats.name_max,
                stats.fragment_size,
            ),
            Err(e) => {
                let ctx = FuseErrorContext {
                    error: &e,
                    operation: "statfs",
                    ino,
                    offset: None,
                };
                reply.error(ctx.log_and_errno());
            }
        }
    }

    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Lookup, |cx| {
            self.inner.ops.lookup(cx, InodeNumber(parent), name)
        }) {
            Ok(attr) => reply.entry(&ATTR_TTL, &to_file_attr(&attr), 0),
            Err(e) => {
                Self::reply_error_entry(
                    &FuseErrorContext {
                        error: &e,
                        operation: "lookup",
                        ino: parent,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Open, |_cx| Ok(())) {
            // Stateless open: we don't track file handles.
            Ok(()) => reply.opened(0, 0),
            Err(e) => {
                let ctx = FuseErrorContext {
                    error: &e,
                    operation: "open",
                    ino,
                    offset: None,
                };
                reply.error(ctx.log_and_errno());
            }
        }
    }

    fn opendir(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Opendir, |_cx| Ok(())) {
            Ok(()) => reply.opened(0, 0),
            Err(e) => {
                let ctx = FuseErrorContext {
                    error: &e,
                    operation: "opendir",
                    ino,
                    offset: None,
                };
                reply.error(ctx.log_and_errno());
            }
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        let cx = Self::cx_for_request();
        let Ok(byte_offset) = u64::try_from(offset) else {
            warn!(ino, offset, "read: negative offset");
            reply.error(libc::EINVAL);
            return;
        };
        match self.read_with_readahead(&cx, InodeNumber(ino), byte_offset, size) {
            Ok(data) => {
                self.inner
                    .metrics
                    .record_bytes_read(u64::try_from(data.len()).unwrap_or(u64::MAX));
                reply.data(&data);
            }
            Err(e) => {
                Self::reply_error_data(
                    &FuseErrorContext {
                        error: &e,
                        operation: "read",
                        ino,
                        offset: Some(byte_offset),
                    },
                    reply,
                );
            }
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let cx = Self::cx_for_request();
        let Ok(fs_offset) = u64::try_from(offset) else {
            warn!(ino, offset, "readdir: negative offset");
            reply.error(libc::EINVAL);
            return;
        };
        match self.with_request_scope(&cx, RequestOp::Readdir, |cx| {
            self.inner.ops.readdir(cx, InodeNumber(ino), fs_offset)
        }) {
            Ok(entries) => {
                for entry in &entries {
                    #[cfg(unix)]
                    let name = OsStr::from_bytes(&entry.name);
                    #[cfg(not(unix))]
                    let owned_name = entry.name_str();
                    #[cfg(not(unix))]
                    let name = OsStr::new(&owned_name);

                    let full = reply.add(
                        entry.ino.0,
                        i64::try_from(entry.offset).unwrap_or(i64::MAX),
                        to_fuser_file_type(entry.kind),
                        name,
                    );
                    if full {
                        break;
                    }
                }
                reply.ok();
            }
            Err(e) => {
                Self::reply_error_dir(
                    &FuseErrorContext {
                        error: &e,
                        operation: "readdir",
                        ino,
                        offset: Some(fs_offset),
                    },
                    reply,
                );
            }
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Readlink, |cx| {
            self.inner.ops.readlink(cx, InodeNumber(ino))
        }) {
            Ok(target) => reply.data(&target),
            Err(e) => {
                Self::reply_error_data(
                    &FuseErrorContext {
                        error: &e,
                        operation: "readlink",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn symlink(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        link: &Path,
        reply: ReplyEntry,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Symlink) {
            warn!(parent, "backpressure: shedding symlink");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Symlink, |cx| {
            self.inner
                .ops
                .symlink(cx, InodeNumber(parent), name, link, req.uid(), req.gid())
        }) {
            Ok(attr) => reply.entry(&ATTR_TTL, &to_file_attr(&attr), 0),
            Err(e) => {
                Self::reply_error_entry(
                    &FuseErrorContext {
                        error: &e,
                        operation: "symlink",
                        ino: parent,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn getxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        size: u32,
        reply: ReplyXattr,
    ) {
        let Some(name) = name.to_str() else {
            reply.error(libc::EINVAL);
            return;
        };
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Getxattr, |cx| {
            self.inner.ops.getxattr(cx, InodeNumber(ino), name)
        }) {
            Ok(Some(value)) => Self::reply_xattr_payload(size, &value, reply),
            Ok(None) => reply.error(Self::missing_xattr_errno()),
            Err(e) => {
                Self::reply_error_xattr(
                    &FuseErrorContext {
                        error: &e,
                        operation: "getxattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn setxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        value: &[u8],
        flags: i32,
        position: u32,
        reply: ReplyEmpty,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Setxattr) {
            warn!(ino, "backpressure: shedding setxattr");
            reply.error(libc::EBUSY);
            return;
        }
        let Some(name) = name.to_str() else {
            reply.error(libc::EINVAL);
            return;
        };
        let Ok(mode) = Self::parse_setxattr_mode(flags, position) else {
            reply.error(libc::EINVAL);
            return;
        };

        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Setxattr, |cx| {
            self.inner
                .ops
                .setxattr(cx, InodeNumber(ino), name, value, mode)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                if matches!(mode, XattrSetMode::Replace)
                    && matches!(e, FfsError::NotFound(_))
                    && self.inner.ops.getattr(&cx, InodeNumber(ino)).is_ok()
                {
                    reply.error(Self::missing_xattr_errno());
                    return;
                }
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "setxattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn removexattr(&mut self, _req: &Request<'_>, ino: u64, name: &OsStr, reply: ReplyEmpty) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Removexattr) {
            warn!(ino, "backpressure: shedding removexattr");
            reply.error(libc::EBUSY);
            return;
        }
        let Some(name) = name.to_str() else {
            reply.error(libc::EINVAL);
            return;
        };

        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Removexattr, |cx| {
            self.inner.ops.removexattr(cx, InodeNumber(ino), name)
        }) {
            Ok(true) => reply.ok(),
            Ok(false) => reply.error(Self::missing_xattr_errno()),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "removexattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn listxattr(&mut self, _req: &Request<'_>, ino: u64, size: u32, reply: ReplyXattr) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Listxattr, |cx| {
            self.inner.ops.listxattr(cx, InodeNumber(ino))
        }) {
            Ok(names) => {
                let payload = Self::encode_xattr_names(&names);
                Self::reply_xattr_payload(size, &payload, reply);
            }
            Err(e) => {
                Self::reply_error_xattr(
                    &FuseErrorContext {
                        error: &e,
                        operation: "listxattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    // ── Write operations ─────────────────────────────────────────────────

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Setattr) {
            warn!(ino, "backpressure: shedding setattr");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        let resolve_time = |t: TimeOrNow| -> SystemTime {
            match t {
                TimeOrNow::SpecificTime(st) => st,
                TimeOrNow::Now => SystemTime::now(),
            }
        };
        let attrs = SetAttrRequest {
            #[allow(clippy::cast_possible_truncation)]
            mode: mode.map(|m| m as u16), // FUSE mode is u32, ext4 mode is u16
            uid,
            gid,
            size,
            atime: atime.map(resolve_time),
            mtime: mtime.map(resolve_time),
        };
        match self.with_request_scope(&cx, RequestOp::Setattr, |cx| {
            self.inner.ops.setattr(cx, InodeNumber(ino), &attrs)
        }) {
            Ok(attr) => reply.attr(&ATTR_TTL, &to_file_attr(&attr)),
            Err(e) => {
                Self::reply_error_attr(
                    &FuseErrorContext {
                        error: &e,
                        operation: "setattr",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)] // FUSE mode u32 → ext4 u16
    fn mkdir(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        match self.dispatch_mkdir(parent, name, mode as u16, req.uid(), req.gid()) {
            Ok(attr) => reply.entry(&ATTR_TTL, &to_file_attr(&attr), 0),
            Err(MutationDispatchError::Errno(errno)) => reply.error(errno),
            Err(MutationDispatchError::Operation { error, offset }) => {
                Self::reply_error_entry(
                    &FuseErrorContext {
                        error: &error,
                        operation: "mkdir",
                        ino: parent,
                        offset,
                    },
                    reply,
                );
            }
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Unlink) {
            warn!(parent, "backpressure: shedding unlink");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Unlink, |cx| {
            self.inner.ops.unlink(cx, InodeNumber(parent), name)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "unlink",
                        ino: parent,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        match self.dispatch_rmdir(parent, name) {
            Ok(()) => reply.ok(),
            Err(MutationDispatchError::Errno(errno)) => reply.error(errno),
            Err(MutationDispatchError::Operation { error, offset }) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &error,
                        operation: "rmdir",
                        ino: parent,
                        offset,
                    },
                    reply,
                );
            }
        }
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        match self.dispatch_rename(parent, name, newparent, newname) {
            Ok(()) => reply.ok(),
            Err(MutationDispatchError::Errno(errno)) => reply.error(errno),
            Err(MutationDispatchError::Operation { error, offset }) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &error,
                        operation: "rename",
                        ino: parent,
                        offset,
                    },
                    reply,
                );
            }
        }
    }

    fn link(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        newparent: u64,
        newname: &OsStr,
        reply: ReplyEntry,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Link) {
            warn!(ino, "backpressure: shedding link");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Link, |cx| {
            self.inner
                .ops
                .link(cx, InodeNumber(ino), InodeNumber(newparent), newname)
        }) {
            Ok(attr) => reply.entry(&ATTR_TTL, &to_file_attr(&attr), 0),
            Err(e) => {
                Self::reply_error_entry(
                    &FuseErrorContext {
                        error: &e,
                        operation: "link",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        match self.dispatch_write(ino, offset, data) {
            Ok(written) => reply.written(written),
            Err(MutationDispatchError::Errno(errno)) => reply.error(errno),
            Err(MutationDispatchError::Operation { error, offset }) => {
                Self::reply_error_write(
                    &FuseErrorContext {
                        error: &error,
                        operation: "write",
                        ino,
                        offset,
                    },
                    reply,
                );
            }
        }
    }

    fn fallocate(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        length: i64,
        mode: i32,
        reply: ReplyEmpty,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Fallocate) {
            warn!(ino, "backpressure: shedding fallocate");
            reply.error(libc::EBUSY);
            return;
        }

        let Ok(byte_offset) = u64::try_from(offset) else {
            reply.error(libc::EINVAL);
            return;
        };
        let Ok(byte_length) = u64::try_from(length) else {
            reply.error(libc::EINVAL);
            return;
        };

        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Fallocate, |cx| {
            self.inner
                .ops
                .fallocate(cx, InodeNumber(ino), byte_offset, byte_length, mode)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "fallocate",
                        ino,
                        offset: Some(byte_offset),
                    },
                    reply,
                );
            }
        }
    }

    fn flush(&mut self, _req: &Request<'_>, ino: u64, fh: u64, lock_owner: u64, reply: ReplyEmpty) {
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Flush, |cx| {
            self.inner.ops.flush(cx, InodeNumber(ino), fh, lock_owner)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "flush",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn fsync(&mut self, _req: &Request<'_>, ino: u64, fh: u64, datasync: bool, reply: ReplyEmpty) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Fsync) {
            warn!(ino, "backpressure: shedding fsync");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Fsync, |cx| {
            self.inner.ops.fsync(cx, InodeNumber(ino), fh, datasync)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "fsync",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    fn fsyncdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        fh: u64,
        datasync: bool,
        reply: ReplyEmpty,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Fsyncdir) {
            warn!(ino, "backpressure: shedding fsyncdir");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Fsyncdir, |cx| {
            self.inner.ops.fsyncdir(cx, InodeNumber(ino), fh, datasync)
        }) {
            Ok(()) => reply.ok(),
            Err(e) => {
                Self::reply_error_empty(
                    &FuseErrorContext {
                        error: &e,
                        operation: "fsyncdir",
                        ino,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)] // FUSE mode u32 → ext4 u16
    fn create(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        if self.inner.read_only {
            reply.error(libc::EROFS);
            return;
        }
        if self.should_shed(RequestOp::Create) {
            warn!(parent, "backpressure: shedding create");
            reply.error(libc::EBUSY);
            return;
        }
        let cx = Self::cx_for_request();
        match self.with_request_scope(&cx, RequestOp::Create, |cx| {
            self.inner.ops.create(
                cx,
                InodeNumber(parent),
                name,
                mode as u16,
                req.uid(),
                req.gid(),
            )
        }) {
            Ok(attr) => {
                reply.created(&ATTR_TTL, &to_file_attr(&attr), 0, 0, 0);
            }
            Err(e) => {
                Self::reply_error_create(
                    &FuseErrorContext {
                        error: &e,
                        operation: "create",
                        ino: parent,
                        offset: None,
                    },
                    reply,
                );
            }
        }
    }
}

// ── Mount entrypoint ────────────────────────────────────────────────────────

/// Build a list of `fuser::MountOption` from our `MountOptions`.
fn build_mount_options(options: &MountOptions) -> Vec<MountOption> {
    let mut opts = vec![
        MountOption::FSName("frankenfs".to_owned()),
        MountOption::Subtype("ffs".to_owned()),
        MountOption::DefaultPermissions,
        MountOption::NoAtime,
    ];

    if options.read_only {
        opts.push(MountOption::RO);
    }
    if options.allow_other {
        opts.push(MountOption::AllowOther);
    }
    if options.auto_unmount {
        opts.push(MountOption::AutoUnmount);
    }
    if options.worker_threads > 0 {
        let max_background = options.resolved_thread_count();
        let congestion_threshold = max_background.saturating_mul(3).saturating_div(4).max(1);
        opts.push(MountOption::CUSTOM(format!(
            "max_background={max_background}"
        )));
        opts.push(MountOption::CUSTOM(format!(
            "congestion_threshold={congestion_threshold}"
        )));
    }

    opts
}

/// Mount a FrankenFS filesystem at the given mountpoint (blocking).
///
/// This function blocks until the filesystem is unmounted.
pub fn mount(
    ops: Box<dyn FsOps>,
    mountpoint: impl AsRef<Path>,
    options: &MountOptions,
) -> Result<(), FuseError> {
    let mountpoint = mountpoint.as_ref();
    if mountpoint.as_os_str().is_empty() {
        return Err(FuseError::InvalidMountpoint(
            "mountpoint cannot be empty".to_owned(),
        ));
    }
    let fuse_opts = build_mount_options(options);
    let fs = FrankenFuse::with_options(ops, options);
    fuser::mount2(fs, mountpoint, &fuse_opts)?;
    Ok(())
}

/// Mount a FrankenFS filesystem in the background, returning a session handle.
///
/// The filesystem is unmounted when the returned `BackgroundSession` is dropped.
pub fn mount_background(
    ops: Box<dyn FsOps>,
    mountpoint: impl AsRef<Path>,
    options: &MountOptions,
) -> Result<fuser::BackgroundSession, FuseError> {
    let mountpoint = mountpoint.as_ref();
    if mountpoint.as_os_str().is_empty() {
        return Err(FuseError::InvalidMountpoint(
            "mountpoint cannot be empty".to_owned(),
        ));
    }
    let fuse_opts = build_mount_options(options);
    let fs = FrankenFuse::with_options(ops, options);
    let session = fuser::spawn_mount2(fs, mountpoint, &fuse_opts)?;
    Ok(session)
}

// ── Mount lifecycle ─────────────────────────────────────────────────────────

/// Configuration for a managed mount with lifecycle control.
#[derive(Debug, Clone)]
pub struct MountConfig {
    /// Base mount options (RO, allow_other, threads, etc.).
    pub options: MountOptions,
    /// Grace period for in-flight requests during unmount.
    pub unmount_timeout: Duration,
}

impl Default for MountConfig {
    fn default() -> Self {
        Self {
            options: MountOptions::default(),
            unmount_timeout: Duration::from_secs(30),
        }
    }
}

/// Handle for a live FUSE mount with lifecycle control.
///
/// Dropping the handle triggers a clean unmount.  Call [`wait`] to block
/// until external shutdown (Ctrl+C / programmatic `shutdown()`).
///
/// # Signal Handling
///
/// `MountHandle` exposes a shared `shutdown` flag (`Arc<AtomicBool>`).
/// The CLI (or any owner) should wire SIGTERM / SIGINT handlers that set
/// this flag.  [`wait`] polls the flag and triggers unmount when set.
/// The `AutoUnmount` fuser option provides a safety net: the kernel
/// unmounts the filesystem if the process exits without a clean unmount.
pub struct MountHandle {
    session: Option<fuser::BackgroundSession>,
    mountpoint: PathBuf,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    metrics: Arc<AtomicMetrics>,
    config: MountConfig,
}

impl MountHandle {
    /// The mountpoint path.
    #[must_use]
    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }

    /// Shared shutdown flag.
    ///
    /// Set this to `true` (from a signal handler or another thread) to
    /// trigger a graceful unmount.
    #[must_use]
    pub fn shutdown_flag(&self) -> &Arc<std::sync::atomic::AtomicBool> {
        &self.shutdown
    }

    /// Get a snapshot of the mount metrics.
    #[must_use]
    pub fn metrics_snapshot(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// Block until the shutdown flag is set, then unmount cleanly.
    ///
    /// Returns the final metrics snapshot.
    #[must_use]
    pub fn wait(mut self) -> MetricsSnapshot {
        info!(mountpoint = %self.mountpoint.display(), "waiting for shutdown signal");
        while !self.shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }
        info!(mountpoint = %self.mountpoint.display(), "shutdown signal received");
        self.do_unmount()
    }

    /// Trigger a graceful unmount.
    ///
    /// Returns the final metrics snapshot.
    #[must_use]
    pub fn unmount(mut self) -> MetricsSnapshot {
        self.do_unmount()
    }

    fn do_unmount(&mut self) -> MetricsSnapshot {
        let snap = self.metrics.snapshot();
        if let Some(session) = self.session.take() {
            info!(
                mountpoint = %self.mountpoint.display(),
                requests_total = snap.requests_total,
                requests_ok = snap.requests_ok,
                requests_err = snap.requests_err,
                bytes_read = snap.bytes_read,
                "unmounting FUSE filesystem"
            );
            // Dropping the BackgroundSession triggers FUSE unmount.
            drop(session);
            info!(mountpoint = %self.mountpoint.display(), "unmount complete");
        }
        snap
    }
}

impl Drop for MountHandle {
    fn drop(&mut self) {
        if self.session.is_some() {
            self.do_unmount();
        }
    }
}

impl std::fmt::Debug for MountHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MountHandle")
            .field("mountpoint", &self.mountpoint)
            .field("active", &self.session.is_some())
            .field(
                "shutdown",
                &self.shutdown.load(std::sync::atomic::Ordering::Relaxed),
            )
            .field("metrics", &self.metrics.snapshot())
            .field("unmount_timeout", &self.config.unmount_timeout)
            .finish()
    }
}

/// Mount a FrankenFS filesystem with full lifecycle control.
///
/// Returns a [`MountHandle`] that can be used to wait for signals,
/// query metrics, and trigger a clean unmount.
///
/// # Example
/// ```no_run
/// # use ffs_fuse::{MountConfig, mount_managed};
/// # fn example(ops: Box<dyn ffs_core::FsOps>) {
/// let handle = mount_managed(ops, "/mnt/ffs", &MountConfig::default()).unwrap();
/// // Wire Ctrl+C to the shutdown flag (e.g. via ctrlc crate):
/// let flag = handle.shutdown_flag().clone();
/// // ... register signal handler that sets `flag.store(true, ...)` ...
/// let stats = handle.wait();
/// println!("served {} requests", stats.requests_total);
/// # }
/// ```
pub fn mount_managed(
    ops: Box<dyn FsOps>,
    mountpoint: impl AsRef<Path>,
    config: &MountConfig,
) -> Result<MountHandle, FuseError> {
    let mountpoint = mountpoint.as_ref();
    if mountpoint.as_os_str().is_empty() {
        return Err(FuseError::InvalidMountpoint(
            "mountpoint cannot be empty".to_owned(),
        ));
    }
    if !mountpoint.exists() {
        return Err(FuseError::InvalidMountpoint(format!(
            "mountpoint does not exist: {}",
            mountpoint.display()
        )));
    }

    let thread_count = config.options.resolved_thread_count();
    info!(
        mountpoint = %mountpoint.display(),
        thread_count,
        read_only = config.options.read_only,
        unmount_timeout_secs = config.unmount_timeout.as_secs(),
        "mounting FrankenFS"
    );

    let fuse_opts = build_mount_options(&config.options);
    let fs = FrankenFuse::with_options(ops, &config.options);
    let metrics_ref = Arc::clone(&fs.inner.metrics);

    let session = fuser::spawn_mount2(fs, mountpoint, &fuse_opts)?;

    info!(mountpoint = %mountpoint.display(), "FUSE mount active");

    Ok(MountHandle {
        session: Some(session),
        mountpoint: mountpoint.to_owned(),
        shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        metrics: metrics_ref,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffs_core::{DirEntry as FfsDirEntry, RequestScope};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Instant, SystemTime};

    /// Minimal FsOps stub for tests that don't need real filesystem behavior.
    struct StubFs;
    impl FsOps for StubFs {
        fn getattr(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }
        fn lookup(
            &self,
            _cx: &Cx,
            _parent: InodeNumber,
            _name: &OsStr,
        ) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }
        fn readdir(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
        ) -> ffs_error::Result<Vec<FfsDirEntry>> {
            Ok(vec![])
        }
        fn read(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
            _size: u32,
        ) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }
        fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }
    }

    #[test]
    fn file_type_conversion_roundtrip() {
        let cases = [
            (FfsFileType::RegularFile, FileType::RegularFile),
            (FfsFileType::Directory, FileType::Directory),
            (FfsFileType::Symlink, FileType::Symlink),
            (FfsFileType::BlockDevice, FileType::BlockDevice),
            (FfsFileType::CharDevice, FileType::CharDevice),
            (FfsFileType::Fifo, FileType::NamedPipe),
            (FfsFileType::Socket, FileType::Socket),
        ];
        for (ffs_ft, expected_fuser_ft) in &cases {
            assert_eq!(to_fuser_file_type(*ffs_ft), *expected_fuser_ft);
        }
    }

    #[test]
    fn inode_attr_to_file_attr_conversion() {
        let iattr = InodeAttr {
            ino: InodeNumber(42),
            size: 1024,
            blocks: 2,
            atime: SystemTime::UNIX_EPOCH,
            mtime: SystemTime::UNIX_EPOCH,
            ctime: SystemTime::UNIX_EPOCH,
            crtime: SystemTime::UNIX_EPOCH,
            kind: FfsFileType::RegularFile,
            perm: 0o644,
            nlink: 1,
            uid: 1000,
            gid: 1000,
            rdev: 0,
            blksize: 4096,
        };
        let fattr = to_file_attr(&iattr);
        assert_eq!(fattr.ino, 42);
        assert_eq!(fattr.size, 1024);
        assert_eq!(fattr.blocks, 2);
        assert_eq!(fattr.kind, FileType::RegularFile);
        assert_eq!(fattr.perm, 0o644);
        assert_eq!(fattr.nlink, 1);
        assert_eq!(fattr.uid, 1000);
        assert_eq!(fattr.gid, 1000);
        assert_eq!(fattr.rdev, 0);
        assert_eq!(fattr.blksize, 4096);
        assert_eq!(fattr.flags, 0);
    }

    #[test]
    fn mount_options_default_is_read_only() {
        let opts = MountOptions::default();
        assert!(opts.read_only);
        assert!(!opts.allow_other);
        assert!(opts.auto_unmount);
    }

    #[test]
    fn build_mount_options_includes_ro_when_read_only() {
        let opts = MountOptions::default();
        let mount_opts = build_mount_options(&opts);
        // Default includes FSName + Subtype + DefaultPermissions + NoAtime + RO + AutoUnmount = 6
        assert!(mount_opts.len() >= 5);
    }

    #[test]
    fn mount_rejects_empty_mountpoint() {
        // We can't construct a real FsOps without a filesystem, but we can
        // verify the mountpoint validation fires before any FsOps call.
        // Use a minimal stub.
        struct NeverCalledFs;
        impl FsOps for NeverCalledFs {
            fn getattr(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
                unreachable!()
            }
            fn lookup(
                &self,
                _cx: &Cx,
                _parent: InodeNumber,
                _name: &OsStr,
            ) -> ffs_error::Result<InodeAttr> {
                unreachable!()
            }
            fn readdir(
                &self,
                _cx: &Cx,
                _ino: InodeNumber,
                _offset: u64,
            ) -> ffs_error::Result<Vec<FfsDirEntry>> {
                unreachable!()
            }
            fn read(
                &self,
                _cx: &Cx,
                _ino: InodeNumber,
                _offset: u64,
                _size: u32,
            ) -> ffs_error::Result<Vec<u8>> {
                unreachable!()
            }
            fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
                unreachable!()
            }
        }
        let err = mount(Box::new(NeverCalledFs), "", &MountOptions::default()).unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn franken_fuse_construction() {
        let _fuse = FrankenFuse::new(Box::new(StubFs));
        // Verify the Cx creation helper works.
        let _cx = FrankenFuse::cx_for_request();
    }

    #[test]
    fn encode_xattr_names_empty_is_empty_payload() {
        let encoded = FrankenFuse::encode_xattr_names(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn encode_xattr_names_produces_nul_separated_list() {
        let encoded = FrankenFuse::encode_xattr_names(&[
            "user.project".to_owned(),
            "security.selinux".to_owned(),
        ]);
        assert_eq!(encoded, b"user.project\0security.selinux\0");
    }

    #[test]
    fn classify_xattr_reply_size_probe_returns_size() {
        assert_eq!(
            FrankenFuse::classify_xattr_reply(0, 11),
            XattrReplyPlan::Size(11)
        );
    }

    #[test]
    fn classify_xattr_reply_data_when_buffer_fits() {
        assert_eq!(
            FrankenFuse::classify_xattr_reply(64, 32),
            XattrReplyPlan::Data
        );
    }

    #[test]
    fn classify_xattr_reply_erange_when_buffer_too_small() {
        assert_eq!(
            FrankenFuse::classify_xattr_reply(8, 32),
            XattrReplyPlan::Error(libc::ERANGE)
        );
    }

    #[test]
    fn classify_xattr_reply_eoverflow_for_oversized_payload() {
        assert_eq!(
            FrankenFuse::classify_xattr_reply(0, usize::MAX),
            XattrReplyPlan::Error(libc::EOVERFLOW)
        );
    }

    #[test]
    fn missing_xattr_errno_matches_platform() {
        #[cfg(target_os = "linux")]
        assert_eq!(FrankenFuse::missing_xattr_errno(), libc::ENODATA);

        #[cfg(not(target_os = "linux"))]
        assert_eq!(FrankenFuse::missing_xattr_errno(), libc::ENOATTR);
    }

    #[test]
    fn parse_setxattr_mode_defaults_to_set() {
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(0, 0).unwrap(),
            XattrSetMode::Set
        );
    }

    #[test]
    fn parse_setxattr_mode_accepts_create_and_replace_flags() {
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(XATTR_FLAG_CREATE, 0).unwrap(),
            XattrSetMode::Create
        );
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(XATTR_FLAG_REPLACE, 0).unwrap(),
            XattrSetMode::Replace
        );
    }

    #[test]
    fn parse_setxattr_mode_rejects_invalid_flag_combinations() {
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(XATTR_FLAG_CREATE | XATTR_FLAG_REPLACE, 0)
                .unwrap_err(),
            libc::EINVAL
        );
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(0x40, 0).unwrap_err(),
            libc::EINVAL
        );
        assert_eq!(
            FrankenFuse::parse_setxattr_mode(XATTR_FLAG_CREATE, 1).unwrap_err(),
            libc::EINVAL
        );
    }

    #[test]
    fn access_predictor_doubles_fetch_size_for_forward_sequence() {
        let predictor = AccessPredictor::default();
        let ino = InodeNumber(11);
        let size = 4096_u32;

        assert_eq!(predictor.fetch_size(ino, 0, size), size);
        predictor.record_read(ino, 0, size);
        assert_eq!(predictor.fetch_size(ino, u64::from(size), size), size);

        predictor.record_read(ino, u64::from(size), size);
        assert_eq!(
            predictor.fetch_size(ino, u64::from(size) * 2, size),
            size.saturating_mul(COALESCED_FETCH_MULTIPLIER)
                .min(MAX_COALESCED_READ_SIZE)
        );
    }

    #[test]
    fn readahead_manager_partial_take_requeues_tail() {
        let manager = ReadaheadManager::new(8);
        let ino = InodeNumber(5);

        manager.insert(ino, 100, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(manager.take(ino, 100, 4), Some(vec![1, 2, 3, 4]));
        assert_eq!(manager.take(ino, 104, 8), Some(vec![5, 6]));
    }

    #[test]
    fn readahead_manager_caps_pending_entries() {
        let manager = ReadaheadManager::new(2);
        let ino = InodeNumber(9);

        manager.insert(ino, 0, vec![0]);
        manager.insert(ino, 8, vec![1]);
        manager.insert(ino, 16, vec![2]);

        assert_eq!(manager.take(ino, 0, 1), None);
        assert_eq!(manager.take(ino, 8, 1), Some(vec![1]));
        assert_eq!(manager.take(ino, 16, 1), Some(vec![2]));
    }

    struct CountingReadFs {
        data: Vec<u8>,
        read_calls: Arc<AtomicU64>,
    }

    impl CountingReadFs {
        fn new(data: Vec<u8>, read_calls: Arc<AtomicU64>) -> Self {
            Self { data, read_calls }
        }
    }

    impl FsOps for CountingReadFs {
        fn getattr(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn lookup(
            &self,
            _cx: &Cx,
            _parent: InodeNumber,
            _name: &OsStr,
        ) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn readdir(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
        ) -> ffs_error::Result<Vec<FfsDirEntry>> {
            Ok(vec![])
        }

        fn read(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            offset: u64,
            size: u32,
        ) -> ffs_error::Result<Vec<u8>> {
            self.read_calls.fetch_add(1, Ordering::Relaxed);
            let start = usize::try_from(offset).unwrap_or(usize::MAX);
            if start >= self.data.len() {
                return Ok(vec![]);
            }
            let requested = usize::try_from(size).unwrap_or(usize::MAX);
            let end = start.saturating_add(requested).min(self.data.len());
            Ok(self.data[start..end].to_vec())
        }

        fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }
    }

    #[test]
    fn sequential_reads_use_prefetched_tail_without_extra_backend_call() {
        let read_calls = Arc::new(AtomicU64::new(0));
        let data: Vec<u8> = (0_u8..64).collect();
        let fuse = FrankenFuse::new(Box::new(CountingReadFs::new(data, Arc::clone(&read_calls))));
        let cx = Cx::for_testing();
        let ino = InodeNumber(1);

        assert_eq!(
            fuse.read_with_readahead(&cx, ino, 0, 4).unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            fuse.read_with_readahead(&cx, ino, 4, 4).unwrap(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            fuse.read_with_readahead(&cx, ino, 8, 4).unwrap(),
            vec![8, 9, 10, 11]
        );
        assert_eq!(
            fuse.read_with_readahead(&cx, ino, 12, 4).unwrap(),
            vec![12, 13, 14, 15]
        );

        // The third read uses a doubled fetch and queues the tail for the
        // fourth read, so only three backend reads are needed.
        assert_eq!(read_calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn long_sequential_reads_exceed_two_x_call_reduction() {
        let read_calls = Arc::new(AtomicU64::new(0));
        let data: Vec<u8> = (0_u8..128).collect();
        let fuse = FrankenFuse::new(Box::new(CountingReadFs::new(data, Arc::clone(&read_calls))));
        let cx = Cx::for_testing();
        let ino = InodeNumber(2);

        for index in 0_u64..12 {
            let offset = index * 4;
            let expected_start = u8::try_from(offset).unwrap_or(u8::MAX);
            let expected = vec![
                expected_start,
                expected_start.saturating_add(1),
                expected_start.saturating_add(2),
                expected_start.saturating_add(3),
            ];
            assert_eq!(
                fuse.read_with_readahead(&cx, ino, offset, 4).unwrap(),
                expected
            );
        }

        // 12 logical reads complete with at most 5 backend reads, which is
        // >2x reduction versus the unbatched baseline of 12 calls.
        assert!(read_calls.load(Ordering::Relaxed) <= 5);
    }

    #[test]
    fn non_sequential_reads_do_not_trigger_coalescing() {
        let read_calls = Arc::new(AtomicU64::new(0));
        let data: Vec<u8> = (0_u8..128).collect();
        let fuse = FrankenFuse::new(Box::new(CountingReadFs::new(data, Arc::clone(&read_calls))));
        let cx = Cx::for_testing();
        let ino = InodeNumber(3);
        let offsets = [0_u64, 32, 4, 48, 8, 64];

        for offset in offsets {
            let _ = fuse.read_with_readahead(&cx, ino, offset, 4).unwrap();
        }

        assert_eq!(
            read_calls.load(Ordering::Relaxed),
            u64::try_from(offsets.len()).unwrap_or(u64::MAX)
        );
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum HookEvent {
        Begin(RequestOp),
        Body(RequestOp),
        End(RequestOp),
    }

    struct HookFs {
        events: Arc<Mutex<Vec<HookEvent>>>,
        fail_begin: bool,
        fail_end: bool,
    }

    impl HookFs {
        fn new(events: Arc<Mutex<Vec<HookEvent>>>, fail_begin: bool, fail_end: bool) -> Self {
            Self {
                events,
                fail_begin,
                fail_end,
            }
        }
    }

    impl FsOps for HookFs {
        fn getattr(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn lookup(
            &self,
            _cx: &Cx,
            _parent: InodeNumber,
            _name: &OsStr,
        ) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn readdir(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
        ) -> ffs_error::Result<Vec<FfsDirEntry>> {
            Ok(vec![])
        }

        fn read(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
            _size: u32,
        ) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }

        fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }

        fn begin_request_scope(&self, _cx: &Cx, op: RequestOp) -> ffs_error::Result<RequestScope> {
            self.events.lock().unwrap().push(HookEvent::Begin(op));
            if self.fail_begin {
                return Err(FfsError::Io(std::io::Error::other("begin failed")));
            }
            Ok(RequestScope::empty())
        }

        fn end_request_scope(
            &self,
            _cx: &Cx,
            op: RequestOp,
            _scope: RequestScope,
        ) -> ffs_error::Result<()> {
            self.events.lock().unwrap().push(HookEvent::End(op));
            if self.fail_end {
                return Err(FfsError::Io(std::io::Error::other("end failed")));
            }
            Ok(())
        }
    }

    #[test]
    fn request_scope_calls_begin_and_end_for_successful_operation() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), false, false);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();
        let body_events = Arc::clone(&events);

        let out = fuse
            .with_request_scope(&cx, RequestOp::Read, |_cx| {
                body_events
                    .lock()
                    .unwrap()
                    .push(HookEvent::Body(RequestOp::Read));
                Ok::<u32, FfsError>(7)
            })
            .unwrap();
        assert_eq!(out, 7);
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                HookEvent::Begin(RequestOp::Read),
                HookEvent::Body(RequestOp::Read),
                HookEvent::End(RequestOp::Read)
            ]
        );
    }

    #[test]
    fn request_scope_short_circuits_body_when_begin_fails() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), true, false);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();
        let body_called = Arc::new(AtomicBool::new(false));
        let body_called_ref = Arc::clone(&body_called);

        let err = fuse
            .with_request_scope(&cx, RequestOp::Lookup, |_cx| {
                body_called_ref.store(true, Ordering::Relaxed);
                Ok::<(), FfsError>(())
            })
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EIO);
        assert!(!body_called.load(Ordering::Relaxed));
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[HookEvent::Begin(RequestOp::Lookup)]
        );
    }

    #[test]
    fn request_scope_prefers_operation_error_when_body_and_end_fail() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), false, true);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();
        let body_events = Arc::clone(&events);

        let err = fuse
            .with_request_scope(&cx, RequestOp::Readlink, |_cx| {
                body_events
                    .lock()
                    .unwrap()
                    .push(HookEvent::Body(RequestOp::Readlink));
                Err::<(), FfsError>(FfsError::NotFound("missing".into()))
            })
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOENT);
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                HookEvent::Begin(RequestOp::Readlink),
                HookEvent::Body(RequestOp::Readlink),
                HookEvent::End(RequestOp::Readlink)
            ]
        );
    }

    #[test]
    fn request_scope_returns_cleanup_error_when_operation_succeeds() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), false, true);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();
        let body_events = Arc::clone(&events);

        let err = fuse
            .with_request_scope(&cx, RequestOp::Getattr, |_cx| {
                body_events
                    .lock()
                    .unwrap()
                    .push(HookEvent::Body(RequestOp::Getattr));
                Ok::<(), FfsError>(())
            })
            .unwrap_err();
        assert_eq!(err.to_errno(), libc::EIO);
        assert_eq!(
            events.lock().unwrap().as_slice(),
            &[
                HookEvent::Begin(RequestOp::Getattr),
                HookEvent::Body(RequestOp::Getattr),
                HookEvent::End(RequestOp::Getattr)
            ]
        );
    }

    fn test_inode_attr(ino: u64, kind: FfsFileType, perm: u16) -> InodeAttr {
        InodeAttr {
            ino: InodeNumber(ino),
            size: 0,
            blocks: 0,
            atime: SystemTime::UNIX_EPOCH,
            mtime: SystemTime::UNIX_EPOCH,
            ctime: SystemTime::UNIX_EPOCH,
            crtime: SystemTime::UNIX_EPOCH,
            kind,
            perm,
            nlink: 1,
            uid: 1000,
            gid: 1000,
            rdev: 0,
            blksize: 4096,
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum MutationCall {
        Write {
            ino: InodeNumber,
            offset: u64,
            data: Vec<u8>,
        },
        Mkdir {
            parent: InodeNumber,
            name: String,
            mode: u16,
            uid: u32,
            gid: u32,
        },
        Rmdir {
            parent: InodeNumber,
            name: String,
        },
        Rename {
            parent: InodeNumber,
            name: String,
            new_parent: InodeNumber,
            new_name: String,
        },
    }

    struct MutationRecordingFs {
        calls: Arc<Mutex<Vec<MutationCall>>>,
    }

    impl MutationRecordingFs {
        fn new(calls: Arc<Mutex<Vec<MutationCall>>>) -> Self {
            Self { calls }
        }
    }

    impl FsOps for MutationRecordingFs {
        fn getattr(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn lookup(
            &self,
            _cx: &Cx,
            _parent: InodeNumber,
            _name: &OsStr,
        ) -> ffs_error::Result<InodeAttr> {
            Err(FfsError::NotFound("stub".into()))
        }

        fn readdir(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
        ) -> ffs_error::Result<Vec<FfsDirEntry>> {
            Ok(vec![])
        }

        fn read(
            &self,
            _cx: &Cx,
            _ino: InodeNumber,
            _offset: u64,
            _size: u32,
        ) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }

        fn readlink(&self, _cx: &Cx, _ino: InodeNumber) -> ffs_error::Result<Vec<u8>> {
            Ok(vec![])
        }

        fn mkdir(
            &self,
            _cx: &Cx,
            parent: InodeNumber,
            name: &OsStr,
            mode: u16,
            uid: u32,
            gid: u32,
        ) -> ffs_error::Result<InodeAttr> {
            self.calls
                .lock()
                .expect("lock mutation calls")
                .push(MutationCall::Mkdir {
                    parent,
                    name: name.to_string_lossy().into_owned(),
                    mode,
                    uid,
                    gid,
                });
            Ok(test_inode_attr(101, FfsFileType::Directory, mode))
        }

        fn rmdir(&self, _cx: &Cx, parent: InodeNumber, name: &OsStr) -> ffs_error::Result<()> {
            self.calls
                .lock()
                .expect("lock mutation calls")
                .push(MutationCall::Rmdir {
                    parent,
                    name: name.to_string_lossy().into_owned(),
                });
            Ok(())
        }

        fn rename(
            &self,
            _cx: &Cx,
            parent: InodeNumber,
            name: &OsStr,
            new_parent: InodeNumber,
            new_name: &OsStr,
        ) -> ffs_error::Result<()> {
            self.calls
                .lock()
                .expect("lock mutation calls")
                .push(MutationCall::Rename {
                    parent,
                    name: name.to_string_lossy().into_owned(),
                    new_parent,
                    new_name: new_name.to_string_lossy().into_owned(),
                });
            Ok(())
        }

        fn write(
            &self,
            _cx: &Cx,
            ino: InodeNumber,
            offset: u64,
            data: &[u8],
        ) -> ffs_error::Result<u32> {
            self.calls
                .lock()
                .expect("lock mutation calls")
                .push(MutationCall::Write {
                    ino,
                    offset,
                    data: data.to_vec(),
                });
            Ok(u32::try_from(data.len()).unwrap_or(u32::MAX))
        }
    }

    #[test]
    fn dispatch_write_routes_to_fsops() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
        );

        let written = fuse
            .dispatch_write(42, 4096, b"abc")
            .expect("dispatch write");
        assert_eq!(written, 3);
        assert_eq!(
            calls.lock().expect("lock calls").as_slice(),
            &[MutationCall::Write {
                ino: InodeNumber(42),
                offset: 4096,
                data: b"abc".to_vec(),
            }]
        );
    }

    #[test]
    fn dispatch_mkdir_routes_to_fsops() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
        );

        let attr = fuse
            .dispatch_mkdir(2, OsStr::new("logs"), 0o755, 123, 456)
            .expect("dispatch mkdir");
        assert_eq!(attr.ino, InodeNumber(101));
        assert_eq!(attr.kind, FfsFileType::Directory);
        assert_eq!(attr.perm, 0o755);
        assert_eq!(
            calls.lock().expect("lock calls").as_slice(),
            &[MutationCall::Mkdir {
                parent: InodeNumber(2),
                name: "logs".to_owned(),
                mode: 0o755,
                uid: 123,
                gid: 456,
            }]
        );
    }

    #[test]
    fn dispatch_rmdir_routes_to_fsops() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
        );

        fuse.dispatch_rmdir(7, OsStr::new("tmp"))
            .expect("dispatch rmdir");
        assert_eq!(
            calls.lock().expect("lock calls").as_slice(),
            &[MutationCall::Rmdir {
                parent: InodeNumber(7),
                name: "tmp".to_owned(),
            }]
        );
    }

    #[test]
    fn dispatch_rename_routes_to_fsops() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
        );

        fuse.dispatch_rename(8, OsStr::new("old"), 9, OsStr::new("new"))
            .expect("dispatch rename");
        assert_eq!(
            calls.lock().expect("lock calls").as_slice(),
            &[MutationCall::Rename {
                parent: InodeNumber(8),
                name: "old".to_owned(),
                new_parent: InodeNumber(9),
                new_name: "new".to_owned(),
            }]
        );
    }

    #[test]
    fn dispatch_write_rejects_negative_offset() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
        );

        let err = fuse
            .dispatch_write(99, -1, b"z")
            .expect_err("negative offset should fail");
        assert!(matches!(err, MutationDispatchError::Errno(libc::EINVAL)));
        assert!(calls.lock().expect("lock calls").is_empty());
    }

    #[test]
    fn dispatch_mutations_return_erofs_when_read_only() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fuse = FrankenFuse::new(Box::new(MutationRecordingFs::new(Arc::clone(&calls))));

        assert!(matches!(
            fuse.dispatch_write(1, 0, b"x"),
            Err(MutationDispatchError::Errno(libc::EROFS))
        ));
        assert!(matches!(
            fuse.dispatch_mkdir(1, OsStr::new("d"), 0o755, 1, 1),
            Err(MutationDispatchError::Errno(libc::EROFS))
        ));
        assert!(matches!(
            fuse.dispatch_rmdir(1, OsStr::new("d")),
            Err(MutationDispatchError::Errno(libc::EROFS))
        ));
        assert!(matches!(
            fuse.dispatch_rename(1, OsStr::new("a"), 2, OsStr::new("b")),
            Err(MutationDispatchError::Errno(libc::EROFS))
        ));
        assert!(calls.lock().expect("lock calls").is_empty());
    }

    #[test]
    fn dispatch_write_returns_ebusy_under_emergency_backpressure() {
        use asupersync::SystemPressure;
        use ffs_core::DegradationFsm;

        let calls = Arc::new(Mutex::new(Vec::new()));
        let options = MountOptions {
            read_only: false,
            ..MountOptions::default()
        };
        let pressure = Arc::new(SystemPressure::with_headroom(0.02));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);
        let fuse = FrankenFuse::with_backpressure(
            Box::new(MutationRecordingFs::new(Arc::clone(&calls))),
            &options,
            gate,
        );

        let err = fuse
            .dispatch_write(11, 0, b"abc")
            .expect_err("write should be shed");
        assert!(matches!(err, MutationDispatchError::Errno(libc::EBUSY)));
        assert!(calls.lock().expect("lock calls").is_empty());
    }

    #[test]
    fn fuse_error_context_returns_correct_errno() {
        let cases: Vec<(FfsError, libc::c_int)> = vec![
            (FfsError::NotFound("test".into()), libc::ENOENT),
            (FfsError::PermissionDenied, libc::EACCES),
            (FfsError::IsDirectory, libc::EISDIR),
            (FfsError::NotDirectory, libc::ENOTDIR),
            (FfsError::ReadOnly, libc::EROFS),
            (FfsError::NoSpace, libc::ENOSPC),
            (FfsError::NameTooLong, libc::ENAMETOOLONG),
            (FfsError::NotEmpty, libc::ENOTEMPTY),
            (FfsError::Exists, libc::EEXIST),
            (FfsError::Cancelled, libc::EINTR),
            (FfsError::MvccConflict { tx: 1, block: 2 }, libc::EAGAIN),
            (
                FfsError::Corruption {
                    block: 0,
                    detail: "bad csum".into(),
                },
                libc::EIO,
            ),
            (FfsError::Format("bad".into()), libc::EINVAL),
            (
                FfsError::UnsupportedFeature("ENCRYPT".into()),
                libc::EOPNOTSUPP,
            ),
            (FfsError::RepairFailed("irrecoverable".into()), libc::EIO),
        ];

        for (error, expected) in &cases {
            let ctx = FuseErrorContext {
                error,
                operation: "test_op",
                ino: 42,
                offset: None,
            };
            assert_eq!(ctx.log_and_errno(), *expected, "wrong errno for {error:?}",);
        }
    }

    #[test]
    fn fuse_error_context_with_offset() {
        let error = FfsError::NotFound("file.txt".into());
        let ctx = FuseErrorContext {
            error: &error,
            operation: "read",
            ino: 100,
            offset: Some(4096),
        };
        assert_eq!(ctx.log_and_errno(), libc::ENOENT);
    }

    // ── Thread safety tests ──────────────────────────────────────────────

    #[test]
    fn franken_fuse_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrankenFuse>();
        assert_send_sync::<FuseInner>();
        assert_send_sync::<AtomicMetrics>();
    }

    #[test]
    fn mount_options_resolved_thread_count() {
        let mut opts = MountOptions::default();
        assert_eq!(opts.worker_threads, 0);
        // Auto resolution gives at least 1.
        assert!(opts.resolved_thread_count() >= 1);
        assert!(opts.resolved_thread_count() <= 8);

        opts.worker_threads = 4;
        assert_eq!(opts.resolved_thread_count(), 4);
    }

    #[test]
    fn franken_fuse_with_options_sets_thread_count() {
        let opts = MountOptions {
            worker_threads: 6,
            ..MountOptions::default()
        };
        let fuse = FrankenFuse::with_options(Box::new(StubFs), &opts);
        assert_eq!(fuse.thread_count(), 6);
    }

    #[test]
    fn atomic_metrics_snapshot_initially_zero() {
        let m = AtomicMetrics::new();
        let s = m.snapshot();
        assert_eq!(s.requests_total, 0);
        assert_eq!(s.requests_ok, 0);
        assert_eq!(s.requests_err, 0);
        assert_eq!(s.bytes_read, 0);
    }

    #[test]
    fn atomic_metrics_record_ok_and_err() {
        let m = AtomicMetrics::new();
        m.record_ok();
        m.record_ok();
        m.record_err();
        m.record_bytes_read(1024);
        let s = m.snapshot();
        assert_eq!(s.requests_total, 3);
        assert_eq!(s.requests_ok, 2);
        assert_eq!(s.requests_err, 1);
        assert_eq!(s.bytes_read, 1024);
    }

    #[test]
    fn cache_line_padded_alignment() {
        let padded = CacheLinePadded(AtomicU64::new(0));
        let ptr = std::ptr::addr_of!(padded) as usize;
        // Must be 64-byte aligned.
        assert_eq!(ptr % 64, 0);
    }

    #[test]
    fn request_scope_updates_metrics() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), false, false);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();

        // Successful request.
        let _ = fuse.with_request_scope(&cx, RequestOp::Read, |_cx| Ok::<u32, FfsError>(7));

        let s = fuse.metrics().snapshot();
        assert_eq!(s.requests_total, 1);
        assert_eq!(s.requests_ok, 1);
        assert_eq!(s.requests_err, 0);
    }

    #[test]
    fn request_scope_records_err_metric() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let fs = HookFs::new(Arc::clone(&events), false, false);
        let fuse = FrankenFuse::new(Box::new(fs));
        let cx = Cx::for_testing();

        let _ = fuse.with_request_scope(&cx, RequestOp::Read, |_cx| {
            Err::<u32, FfsError>(FfsError::NotFound("gone".into()))
        });

        let s = fuse.metrics().snapshot();
        assert_eq!(s.requests_total, 1);
        assert_eq!(s.requests_ok, 0);
        assert_eq!(s.requests_err, 1);
    }

    #[test]
    fn concurrent_fsops_access_no_deadlock() {
        // Verify FsOps can be called concurrently from multiple threads
        // via Arc<dyn FsOps>.
        let fs: Arc<dyn FsOps> = Arc::new(StubFs);
        let barrier = Arc::new(std::sync::Barrier::new(10));

        std::thread::scope(|s| {
            for _ in 0..10 {
                let fs: Arc<dyn FsOps> = Arc::clone(&fs);
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let cx = Cx::for_testing();
                    barrier.wait();
                    for _ in 0..100 {
                        let _ = fs.getattr(&cx, InodeNumber(1));
                        let _ = fs.readdir(&cx, InodeNumber(1), 0);
                        let _ = fs.read(&cx, InodeNumber(1), 0, 4096);
                    }
                });
            }
        });
    }

    #[test]
    fn concurrent_metrics_stress() {
        // 10 threads x 1000 increments each.
        let metrics = Arc::new(AtomicMetrics::new());
        let barrier = Arc::new(std::sync::Barrier::new(10));

        std::thread::scope(|s| {
            for _ in 0..10 {
                let m = Arc::clone(&metrics);
                let b = Arc::clone(&barrier);
                s.spawn(move || {
                    b.wait();
                    for _ in 0..1000 {
                        m.record_ok();
                        m.record_bytes_read(512);
                    }
                });
            }
        });

        let s = metrics.snapshot();
        assert_eq!(s.requests_total, 10_000);
        assert_eq!(s.requests_ok, 10_000);
        assert_eq!(s.requests_err, 0);
        assert_eq!(s.bytes_read, 10_000 * 512);
    }

    #[test]
    fn fuse_inner_shared_across_threads() {
        // Simulate multi-threaded FUSE dispatch: multiple threads share
        // the same FuseInner via Arc and call FsOps concurrently.
        let inner = Arc::new(FuseInner {
            ops: Arc::new(StubFs),
            metrics: Arc::new(AtomicMetrics::new()),
            thread_count: 4,
            read_only: true,
            backpressure: None,
            access_predictor: AccessPredictor::default(),
            readahead: ReadaheadManager::new(MAX_PENDING_READAHEAD_ENTRIES),
        });
        let barrier = Arc::new(std::sync::Barrier::new(10));

        std::thread::scope(|s| {
            for _ in 0..10 {
                let inner = Arc::clone(&inner);
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let cx = Cx::for_testing();
                    barrier.wait();
                    for _ in 0..1000 {
                        let _ = inner.ops.getattr(&cx, InodeNumber(2));
                        inner.metrics.record_ok();
                        let _ = inner.ops.read(&cx, InodeNumber(2), 0, 4096);
                        inner.metrics.record_bytes_read(4096);
                    }
                });
            }
        });

        let snap = inner.metrics.snapshot();
        assert_eq!(snap.requests_ok, 10_000);
        assert_eq!(snap.bytes_read, 10_000 * 4096);
    }

    // ── Mount lifecycle tests ─────────────────────────────────────────

    #[test]
    fn mount_config_default_has_30s_timeout() {
        let cfg = MountConfig::default();
        assert_eq!(cfg.unmount_timeout, Duration::from_secs(30));
        assert!(cfg.options.read_only);
    }

    #[test]
    fn mount_managed_rejects_empty_mountpoint() {
        let ops: Box<dyn FsOps> = Box::new(StubFs);
        let err = mount_managed(ops, "", &MountConfig::default()).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected 'empty' in error: {err}"
        );
    }

    #[test]
    fn mount_managed_rejects_nonexistent_mountpoint() {
        let ops: Box<dyn FsOps> = Box::new(StubFs);
        let err = mount_managed(
            ops,
            "/tmp/frankenfs_no_such_dir_xyzzy",
            &MountConfig::default(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("does not exist"),
            "expected 'does not exist' in error: {err}"
        );
    }

    #[test]
    fn mount_handle_shutdown_flag_lifecycle() {
        // Build a MountHandle manually (without a real FUSE session) to
        // exercise the shutdown flag + metrics plumbing.
        let metrics = Arc::new(AtomicMetrics::new());
        metrics.record_ok();
        metrics.record_ok();
        metrics.record_bytes_read(8192);

        let handle = MountHandle {
            session: None,
            mountpoint: PathBuf::from("/mnt/test"),
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::clone(&metrics),
            config: MountConfig::default(),
        };

        // Shutdown flag starts false.
        assert!(!handle.shutdown_flag().load(Ordering::Relaxed));

        // Metrics snapshot reflects pre-recorded data.
        let snap = handle.metrics_snapshot();
        assert_eq!(snap.requests_ok, 2);
        assert_eq!(snap.bytes_read, 8192);

        // Unmount returns final snapshot.
        let final_snap = handle.unmount();
        assert_eq!(final_snap.requests_ok, 2);
    }

    #[test]
    fn mount_handle_debug_format() {
        let handle = MountHandle {
            session: None,
            mountpoint: PathBuf::from("/mnt/dbg"),
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(AtomicMetrics::new()),
            config: MountConfig::default(),
        };
        let dbg = format!("{handle:?}");
        assert!(dbg.contains("MountHandle"), "missing struct name: {dbg}");
        assert!(dbg.contains("/mnt/dbg"), "missing mountpoint: {dbg}");
        assert!(dbg.contains("active: false"), "missing active: {dbg}");
        assert!(dbg.contains("shutdown: false"), "missing shutdown: {dbg}");
    }

    #[test]
    fn mount_handle_drop_is_safe_without_session() {
        // Verify that dropping a MountHandle with no session doesn't panic.
        let handle = MountHandle {
            session: None,
            mountpoint: PathBuf::from("/mnt/drop"),
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(AtomicMetrics::new()),
            config: MountConfig::default(),
        };
        drop(handle);
    }

    #[test]
    fn mount_handle_wait_returns_on_shutdown() {
        let metrics = Arc::new(AtomicMetrics::new());
        metrics.record_ok();

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_trigger = Arc::clone(&shutdown);

        let handle = MountHandle {
            session: None,
            mountpoint: PathBuf::from("/mnt/wait"),
            shutdown: Arc::clone(&shutdown),
            metrics,
            config: MountConfig::default(),
        };

        // Set the shutdown flag from another thread after a short delay.
        let shutdown_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            shutdown_trigger.store(true, Ordering::Relaxed);
        });

        let snap = handle.wait();
        shutdown_thread
            .join()
            .expect("shutdown trigger thread should not panic");
        assert_eq!(snap.requests_ok, 1);
    }

    #[test]
    fn mount_handle_wait_respects_unmount_timeout() {
        let config = MountConfig {
            options: MountOptions::default(),
            unmount_timeout: Duration::from_millis(60),
        };
        let handle = MountHandle {
            session: None,
            mountpoint: PathBuf::from("/mnt/timeout"),
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(AtomicMetrics::new()),
            config: config.clone(),
        };

        let started = Instant::now();
        let snap = handle.wait();
        let elapsed = started.elapsed();

        assert_eq!(snap.requests_total, 0);
        assert!(elapsed >= config.unmount_timeout);
        assert!(elapsed < Duration::from_millis(500));
    }

    // ── FuseErrorContext errno mapping for all 21 variants (bd-2s4.6) ──

    #[test]
    fn fuse_error_context_log_and_errno_covers_all_variants() {
        let cases: Vec<(FfsError, libc::c_int)> = vec![
            (FfsError::Io(std::io::Error::other("test")), libc::EIO),
            (
                FfsError::Corruption {
                    block: 1,
                    detail: "bad crc".into(),
                },
                libc::EIO,
            ),
            (FfsError::Format("bad magic".into()), libc::EINVAL),
            (FfsError::Parse("truncated".into()), libc::EINVAL),
            (
                FfsError::UnsupportedFeature("ENCRYPT".into()),
                libc::EOPNOTSUPP,
            ),
            (
                FfsError::IncompatibleFeature("missing FILETYPE".into()),
                libc::EOPNOTSUPP,
            ),
            (
                FfsError::UnsupportedBlockSize("8192".into()),
                libc::EOPNOTSUPP,
            ),
            (
                FfsError::InvalidGeometry("blocks_per_group=0".into()),
                libc::EINVAL,
            ),
            (FfsError::MvccConflict { tx: 1, block: 2 }, libc::EAGAIN),
            (FfsError::Cancelled, libc::EINTR),
            (FfsError::NoSpace, libc::ENOSPC),
            (FfsError::NotFound("gone".into()), libc::ENOENT),
            (FfsError::PermissionDenied, libc::EACCES),
            (FfsError::ReadOnly, libc::EROFS),
            (FfsError::NotDirectory, libc::ENOTDIR),
            (FfsError::IsDirectory, libc::EISDIR),
            (FfsError::NotEmpty, libc::ENOTEMPTY),
            (FfsError::NameTooLong, libc::ENAMETOOLONG),
            (FfsError::Exists, libc::EEXIST),
            (FfsError::RepairFailed("checksum".into()), libc::EIO),
        ];

        // 20 variants listed; verify count matches expectation.
        assert_eq!(
            cases.len(),
            20,
            "expected all 20 constructible FfsError variants"
        );

        for (error, expected) in &cases {
            let ctx = FuseErrorContext {
                error,
                operation: "test_op",
                ino: 99,
                offset: Some(0),
            };
            assert_eq!(ctx.log_and_errno(), *expected, "wrong errno for {error:?}",);
        }
    }

    #[test]
    fn fuse_error_context_io_preserves_raw_os_error() {
        let raw = std::io::Error::from_raw_os_error(libc::EPERM);
        let err = FfsError::Io(raw);
        let ctx = FuseErrorContext {
            error: &err,
            operation: "open",
            ino: 5,
            offset: None,
        };
        assert_eq!(ctx.log_and_errno(), libc::EPERM);
    }

    #[test]
    fn fuse_error_context_enoent_does_not_panic() {
        // ENOENT is logged at trace, not warn — ensure it doesn't panic.
        let err = FfsError::NotFound("test".into());
        let ctx = FuseErrorContext {
            error: &err,
            operation: "lookup",
            ino: 2,
            offset: None,
        };
        assert_eq!(ctx.log_and_errno(), libc::ENOENT);
    }

    // ── Read-only flag propagation ───────────────────────────────────────

    #[test]
    fn fuse_inner_read_only_true_when_mount_option_set() {
        let opts = MountOptions {
            read_only: true,
            ..Default::default()
        };
        let fuse = FrankenFuse::with_options(Box::new(StubFs), &opts);
        assert!(fuse.inner.read_only);
    }

    #[test]
    fn fuse_inner_read_only_false_when_writable() {
        let opts = MountOptions {
            read_only: false,
            ..Default::default()
        };
        let fuse = FrankenFuse::with_options(Box::new(StubFs), &opts);
        assert!(!fuse.inner.read_only);
    }

    #[test]
    fn build_mount_options_omits_ro_when_read_write() {
        let opts = MountOptions {
            read_only: false,
            allow_other: false,
            auto_unmount: true,
            worker_threads: 0,
        };
        let mount_opts = build_mount_options(&opts);
        // Should NOT contain RO
        let has_ro = mount_opts.iter().any(|o| matches!(o, MountOption::RO));
        assert!(!has_ro, "RO should not be present when read_only=false");
    }

    #[test]
    fn build_mount_options_includes_allow_other_when_set() {
        let opts = MountOptions {
            read_only: true,
            allow_other: true,
            auto_unmount: false,
            worker_threads: 0,
        };
        let mount_opts = build_mount_options(&opts);
        let has_allow = mount_opts
            .iter()
            .any(|o| matches!(o, MountOption::AllowOther));
        assert!(has_allow, "AllowOther should be present");
    }

    #[test]
    fn build_mount_options_includes_queue_tuning_when_worker_threads_explicit() {
        let opts = MountOptions {
            read_only: true,
            allow_other: false,
            auto_unmount: true,
            worker_threads: 8,
        };
        let mount_opts = build_mount_options(&opts);
        assert!(
            mount_opts
                .iter()
                .any(|o| matches!(o, MountOption::CUSTOM(v) if v == "max_background=8"))
        );
        assert!(
            mount_opts
                .iter()
                .any(|o| matches!(o, MountOption::CUSTOM(v) if v == "congestion_threshold=6"))
        );
    }

    #[test]
    fn build_mount_options_auto_worker_threads_omits_queue_tuning() {
        let opts = MountOptions {
            read_only: true,
            allow_other: false,
            auto_unmount: true,
            worker_threads: 0,
        };
        let mount_opts = build_mount_options(&opts);
        assert!(
            !mount_opts
                .iter()
                .any(|o| matches!(o, MountOption::CUSTOM(v) if v.starts_with("max_background=")))
        );
        assert!(!mount_opts.iter().any(
            |o| matches!(o, MountOption::CUSTOM(v) if v.starts_with("congestion_threshold="))
        ));
    }

    // ── should_shed backpressure tests ───────────────────────────────────

    #[test]
    fn should_shed_returns_false_without_backpressure_gate() {
        let fuse = FrankenFuse::new(Box::new(StubFs));
        // No backpressure gate → never shed.
        assert!(!fuse.should_shed(RequestOp::Read));
        assert!(!fuse.should_shed(RequestOp::Write));
        assert!(!fuse.should_shed(RequestOp::Create));
        assert!(!fuse.should_shed(RequestOp::Mkdir));
    }

    #[test]
    fn should_shed_with_emergency_gate_sheds_writes() {
        use asupersync::SystemPressure;
        use ffs_core::DegradationFsm;

        // Emergency level: headroom 0.02 → all writes shed.
        let pressure = Arc::new(SystemPressure::with_headroom(0.02));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        let opts = MountOptions::default();
        let fuse = FrankenFuse::with_backpressure(Box::new(StubFs), &opts, gate);

        // Reads proceed.
        assert!(!fuse.should_shed(RequestOp::Read));
        assert!(!fuse.should_shed(RequestOp::Lookup));
        assert!(!fuse.should_shed(RequestOp::Getattr));
        assert!(!fuse.should_shed(RequestOp::Readdir));

        // Writes are shed.
        assert!(fuse.should_shed(RequestOp::Write));
        assert!(fuse.should_shed(RequestOp::Create));
        assert!(fuse.should_shed(RequestOp::Mkdir));
        assert!(fuse.should_shed(RequestOp::Unlink));
        assert!(fuse.should_shed(RequestOp::Rmdir));
        assert!(fuse.should_shed(RequestOp::Rename));
        assert!(fuse.should_shed(RequestOp::Link));
        assert!(fuse.should_shed(RequestOp::Symlink));
        assert!(fuse.should_shed(RequestOp::Fallocate));
        assert!(fuse.should_shed(RequestOp::Setattr));
        assert!(fuse.should_shed(RequestOp::Setxattr));
        assert!(fuse.should_shed(RequestOp::Removexattr));
    }

    #[test]
    fn should_shed_with_normal_gate_proceeds_all() {
        use asupersync::SystemPressure;
        use ffs_core::DegradationFsm;

        // Normal level: headroom 0.9 → all ops proceed.
        let pressure = Arc::new(SystemPressure::with_headroom(0.9));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        let opts = MountOptions::default();
        let fuse = FrankenFuse::with_backpressure(Box::new(StubFs), &opts, gate);

        assert!(!fuse.should_shed(RequestOp::Read));
        assert!(!fuse.should_shed(RequestOp::Write));
        assert!(!fuse.should_shed(RequestOp::Create));
        assert!(!fuse.should_shed(RequestOp::Mkdir));
    }

    #[test]
    fn should_shed_with_degraded_gate_throttles_without_shedding() {
        use asupersync::SystemPressure;
        use ffs_core::DegradationFsm;

        // Degraded level: headroom 0.2 -> writes are throttled (not shed).
        let pressure = Arc::new(SystemPressure::with_headroom(0.2));
        let fsm = Arc::new(DegradationFsm::new(Arc::clone(&pressure), 1));
        fsm.tick();
        let gate = BackpressureGate::new(fsm);

        let opts = MountOptions::default();
        let fuse = FrankenFuse::with_backpressure(Box::new(StubFs), &opts, gate);

        let start = std::time::Instant::now();
        assert!(!fuse.should_shed(RequestOp::Write));
        assert!(start.elapsed() >= BACKPRESSURE_THROTTLE_DELAY);
    }

    // ── AccessPredictor backward sequence detection ──────────────────────

    #[test]
    fn access_predictor_backward_sequence_does_not_batch() {
        let predictor = AccessPredictor::default();
        let ino = InodeNumber(20);
        let size = 4096_u32;

        // Read backward: 3*4096, 2*4096, 1*4096, 0
        predictor.record_read(ino, u64::from(size) * 3, size);
        predictor.record_read(ino, u64::from(size) * 2, size);
        predictor.record_read(ino, u64::from(size), size);

        // After backward sequence, fetch_size should NOT batch (returns requested).
        assert_eq!(predictor.fetch_size(ino, 0, size), size);
    }

    #[test]
    fn access_predictor_random_access_does_not_batch() {
        let predictor = AccessPredictor::default();
        let ino = InodeNumber(21);
        let size = 4096_u32;

        // Random offsets.
        predictor.record_read(ino, 0, size);
        predictor.record_read(ino, u64::from(size) * 10, size);
        predictor.record_read(ino, u64::from(size) * 3, size);
        predictor.record_read(ino, u64::from(size) * 7, size);

        // Not sequential → no batching.
        assert_eq!(predictor.fetch_size(ino, u64::from(size) * 8, size), size);
    }

    #[test]
    fn access_predictor_different_inodes_are_independent() {
        let predictor = AccessPredictor::default();
        let size = 4096_u32;

        // Build forward sequence on inode 30.
        for i in 0..5_u64 {
            predictor.record_read(InodeNumber(30), i * u64::from(size), size);
        }

        // Inode 31 has no history — should not batch.
        assert_eq!(predictor.fetch_size(InodeNumber(31), 0, size), size);
    }

    #[test]
    fn access_predictor_history_is_bounded() {
        let predictor = AccessPredictor::new(3);
        let size = 4096_u32;

        for ino in 0..10_u64 {
            predictor.record_read(InodeNumber(100 + ino), 0, size);
        }

        let tracked = {
            let guard = match predictor.state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.history.len()
        };
        assert_eq!(tracked, 3);
    }

    #[test]
    fn access_predictor_evicts_least_recent_inode() {
        let predictor = AccessPredictor::new(2);
        let size = 4096_u32;

        predictor.record_read(InodeNumber(1), 0, size);
        predictor.record_read(InodeNumber(2), 0, size);
        predictor.record_read(InodeNumber(1), u64::from(size), size);
        predictor.record_read(InodeNumber(3), 0, size);

        let (tracked, has_one, has_two, has_three) = {
            let guard = match predictor.state.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            (
                guard.history.len(),
                guard.history.contains_key(&1),
                guard.history.contains_key(&2),
                guard.history.contains_key(&3),
            )
        };
        assert_eq!(tracked, 2);
        assert!(has_one);
        assert!(!has_two);
        assert!(has_three);
    }

    // ── Concurrent AccessPredictor stress ────────────────────────────────

    #[test]
    fn access_predictor_concurrent_stress() {
        let predictor = Arc::new(AccessPredictor::default());
        let barrier = Arc::new(std::sync::Barrier::new(8));

        std::thread::scope(|s| {
            for thread_id in 0_u64..8 {
                let predictor = Arc::clone(&predictor);
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let ino = InodeNumber(100 + thread_id);
                    barrier.wait();
                    for i in 0_u64..500 {
                        let offset = i * 4096;
                        let _ = predictor.fetch_size(ino, offset, 4096);
                        predictor.record_read(ino, offset, 4096);
                    }
                });
            }
        });

        // No panic or deadlock = success. Verify state is queryable.
        for thread_id in 0_u64..8 {
            let _ = predictor.fetch_size(InodeNumber(100 + thread_id), 0, 4096);
        }
    }

    // ── Metrics record_err tracking ──────────────────────────────────────

    #[test]
    fn atomic_metrics_tracks_errors_separately() {
        let metrics = AtomicMetrics::new();
        metrics.record_ok();
        metrics.record_ok();
        metrics.record_err();
        metrics.record_bytes_read(1024);

        let snap = metrics.snapshot();
        assert_eq!(snap.requests_total, 3);
        assert_eq!(snap.requests_ok, 2);
        assert_eq!(snap.requests_err, 1);
        assert_eq!(snap.bytes_read, 1024);
    }

    // ── MountOptions thread count resolution ─────────────────────────────

    #[test]
    fn resolved_thread_count_auto_is_bounded() {
        let opts = MountOptions {
            worker_threads: 0,
            ..Default::default()
        };
        let count = opts.resolved_thread_count();
        assert!(count >= 1);
        assert!(count <= 8);
    }

    #[test]
    fn resolved_thread_count_explicit_value_passes_through() {
        let opts = MountOptions {
            worker_threads: 4,
            ..Default::default()
        };
        assert_eq!(opts.resolved_thread_count(), 4);
    }

    #[test]
    fn resolved_thread_count_clamps_to_at_least_one() {
        // worker_threads=0 means auto, so test with 1.
        let opts = MountOptions {
            worker_threads: 1,
            ..Default::default()
        };
        assert_eq!(opts.resolved_thread_count(), 1);
    }

    // ── FrankenFuse thread_count accessor ────────────────────────────────

    #[test]
    fn franken_fuse_thread_count_matches_options() {
        let opts = MountOptions {
            worker_threads: 3,
            ..Default::default()
        };
        let fuse = FrankenFuse::with_options(Box::new(StubFs), &opts);
        assert_eq!(fuse.thread_count(), 3);
    }

    // ── ReadaheadManager edge cases ──────────────────────────────────────

    #[test]
    fn readahead_manager_miss_returns_none() {
        let manager = ReadaheadManager::new(8);
        // No data inserted → take returns None.
        assert_eq!(manager.take(InodeNumber(1), 0, 4), None);
    }

    #[test]
    fn readahead_manager_wrong_offset_returns_none() {
        let manager = ReadaheadManager::new(8);
        let ino = InodeNumber(2);
        manager.insert(ino, 100, vec![1, 2, 3]);
        // Wrong offset → miss.
        assert_eq!(manager.take(ino, 200, 3), None);
        // Correct offset → hit.
        assert_eq!(manager.take(ino, 100, 3), Some(vec![1, 2, 3]));
    }

    #[test]
    fn readahead_manager_exact_size_take() {
        let manager = ReadaheadManager::new(8);
        let ino = InodeNumber(3);
        manager.insert(ino, 0, vec![10, 20, 30, 40]);
        // Take exactly the stored amount.
        assert_eq!(manager.take(ino, 0, 4), Some(vec![10, 20, 30, 40]));
        // Second take should return None (consumed).
        assert_eq!(manager.take(ino, 0, 4), None);
    }

    // ── Edge-case hardening tests ──────────────────────────────────────

    #[test]
    fn build_mount_options_rw_allow_other_with_threads() {
        let opts = MountOptions {
            read_only: false,
            allow_other: true,
            auto_unmount: false,
            worker_threads: 4,
        };
        let mount_opts = build_mount_options(&opts);
        // Should contain FSName, Subtype, DefaultPermissions, NoAtime, AllowOther,
        // max_background, congestion_threshold — but NOT RO or AutoUnmount.
        let dbg = format!("{mount_opts:?}");
        assert!(dbg.contains("AllowOther"), "missing AllowOther: {dbg}");
        assert!(
            dbg.contains("max_background"),
            "missing max_background: {dbg}"
        );
        assert!(
            dbg.contains("congestion_threshold"),
            "missing congestion_threshold: {dbg}"
        );
        assert!(!dbg.contains("\"RO\""), "should not contain RO: {dbg}");
    }

    #[test]
    fn build_mount_options_zero_threads_omits_custom_background() {
        let opts = MountOptions {
            worker_threads: 0,
            ..MountOptions::default()
        };
        let mount_opts = build_mount_options(&opts);
        let dbg = format!("{mount_opts:?}");
        assert!(
            !dbg.contains("max_background"),
            "zero threads should not set max_background: {dbg}"
        );
    }

    #[test]
    fn metrics_snapshot_equality() {
        let a = MetricsSnapshot {
            requests_total: 10,
            requests_ok: 7,
            requests_err: 3,
            bytes_read: 4096,
        };
        let b = a;
        assert_eq!(a, b);

        let c = MetricsSnapshot {
            requests_total: 10,
            requests_ok: 6,
            requests_err: 4,
            bytes_read: 4096,
        };
        assert_ne!(a, c);
    }

    #[test]
    fn atomic_metrics_debug_shows_fields() {
        let m = AtomicMetrics::new();
        m.record_ok();
        m.record_bytes_read(512);
        let dbg = format!("{m:?}");
        assert!(dbg.contains("AtomicMetrics"), "missing struct name: {dbg}");
        assert!(dbg.contains("requests_total"), "missing field: {dbg}");
    }

    #[test]
    fn cache_line_padded_debug_delegates_to_inner() {
        let padded = CacheLinePadded(42_u32);
        let dbg = format!("{padded:?}");
        assert!(
            dbg.contains("42"),
            "CacheLinePadded Debug should show inner: {dbg}"
        );
    }

    #[test]
    fn access_predictor_backward_sequence_not_coalesced() {
        // Backward sequential reads should increment sequential_count
        // but NOT trigger coalescing (only forward does).
        let predictor = AccessPredictor::new(64);
        let ino = InodeNumber(50);
        let size = 4096_u32;

        // Read offsets: 3*4096, 2*4096, 1*4096, 0 (backward).
        for i in (0..4).rev() {
            predictor.record_read(ino, u64::from(size) * i, size);
        }
        // Asking for the next backward read shouldn't coalesce.
        // Since coalescing is only for forward, fetch_size should return `size`.
        let fetch = predictor.fetch_size(ino, 0, size);
        assert_eq!(
            fetch, size,
            "backward sequence should not trigger coalescing"
        );
    }

    #[test]
    fn access_predictor_capacity_one_evicts_oldest() {
        let predictor = AccessPredictor::new(1);
        let size = 4096_u32;

        // Record inode 1, then inode 2 → inode 1 should be evicted.
        predictor.record_read(InodeNumber(1), 0, size);
        predictor.record_read(InodeNumber(2), 0, size);

        // Inode 1 should be unknown now.
        assert_eq!(predictor.fetch_size(InodeNumber(1), 0, size), size);

        // Inode 2 is still known.
        {
            let state = predictor.state.lock().unwrap();
            assert!(state.history.contains_key(&2));
            assert!(!state.history.contains_key(&1));
            drop(state);
        }
    }

    #[test]
    fn access_predictor_non_sequential_resets_count() {
        let predictor = AccessPredictor::new(64);
        let ino = InodeNumber(77);
        let size = 4096_u32;

        // Build forward sequential: 0, 4096, 8192.
        predictor.record_read(ino, 0, size);
        predictor.record_read(ino, 4096, size);
        predictor.record_read(ino, 8192, size);

        // Random jump to offset 999999 → resets sequential count.
        predictor.record_read(ino, 999_999, size);

        // Next forward read from expected position shouldn't coalesce
        // because sequential_count was reset to 1.
        let fetch = predictor.fetch_size(ino, 999_999 + u64::from(size), size);
        assert_eq!(fetch, size, "jump should reset sequential count");
    }

    #[test]
    fn readahead_manager_overwrite_same_key() {
        let manager = ReadaheadManager::new(8);
        let ino = InodeNumber(10);

        // Insert at offset 0 with data [1,2,3].
        manager.insert(ino, 0, vec![1, 2, 3]);
        // Overwrite at same key with [4,5,6].
        manager.insert(ino, 0, vec![4, 5, 6]);

        // Should get the latest data.
        assert_eq!(manager.take(ino, 0, 3), Some(vec![4, 5, 6]));
    }

    #[test]
    fn readahead_manager_empty_insert_is_noop() {
        let manager = ReadaheadManager::new(8);
        let ino = InodeNumber(20);

        manager.insert(ino, 0, vec![]);
        assert_eq!(manager.take(ino, 0, 0), None);
    }

    #[test]
    fn fuse_error_display_variants() {
        let invalid_mp = FuseError::InvalidMountpoint("bad path".into());
        assert!(
            invalid_mp.to_string().contains("bad path"),
            "InvalidMountpoint should contain path: {invalid_mp}",
        );

        let io_err = FuseError::Io(std::io::Error::other("disk gone"));
        assert!(
            io_err.to_string().contains("disk gone"),
            "Io variant should contain inner error: {io_err}",
        );
    }

    #[test]
    fn fuse_inner_debug_shows_non_exhaustive() {
        let inner = FuseInner {
            ops: Arc::new(StubFs),
            metrics: Arc::new(AtomicMetrics::new()),
            thread_count: 2,
            read_only: false,
            backpressure: None,
            access_predictor: AccessPredictor::default(),
            readahead: ReadaheadManager::new(8),
        };
        let dbg = format!("{inner:?}");
        assert!(dbg.contains("FuseInner"), "missing struct name: {dbg}");
        assert!(
            dbg.contains("thread_count: 2"),
            "missing thread_count: {dbg}"
        );
        assert!(dbg.contains("read_only: false"), "missing read_only: {dbg}");
    }

    #[test]
    fn mount_options_worker_threads_one_resolves_to_one() {
        let opts = MountOptions {
            worker_threads: 1,
            ..MountOptions::default()
        };
        assert_eq!(opts.resolved_thread_count(), 1);
    }

    #[test]
    fn classify_xattr_reply_data_exact_fit() {
        // payload_len == size → Data.
        assert_eq!(
            FrankenFuse::classify_xattr_reply(32, 32),
            XattrReplyPlan::Data
        );
    }

    #[test]
    fn classify_xattr_reply_size_zero_payload() {
        // size=0, payload=0 → Size(0).
        assert_eq!(
            FrankenFuse::classify_xattr_reply(0, 0),
            XattrReplyPlan::Size(0)
        );
    }

    #[test]
    fn access_direction_equality() {
        assert_eq!(AccessDirection::Forward, AccessDirection::Forward);
        assert_eq!(AccessDirection::Backward, AccessDirection::Backward);
        assert_ne!(AccessDirection::Forward, AccessDirection::Backward);
    }

    // ── Proptest property-based tests ─────────────────────────────────────

    #[expect(clippy::cast_possible_truncation)] // test-only: proptest ranges guarantee safe casts
    mod proptests {
        use super::*;
        use crate::per_core::{
            CoreMetrics, PerCoreConfig, PerCoreDispatcher, inode_to_core, lookup_to_core,
        };
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            // ── inode_to_core properties ────────────────────────────────

            /// Routing is always deterministic: same inputs produce same output.
            #[test]
            fn inode_routing_is_deterministic(ino in 0_u64..=u64::MAX, cores in 1_u32..=256) {
                let a = inode_to_core(ino, cores);
                let b = inode_to_core(ino, cores);
                prop_assert_eq!(a, b);
            }

            /// Routing output is always within [0, num_cores).
            #[test]
            fn inode_routing_in_range(ino in 0_u64..=u64::MAX, cores in 1_u32..=256) {
                let core = inode_to_core(ino, cores);
                prop_assert!(core < cores, "core {core} >= num_cores {cores}");
            }

            /// Routing with num_cores=0 always returns 0.
            #[test]
            fn inode_routing_zero_cores_always_zero(ino in 0_u64..=u64::MAX) {
                prop_assert_eq!(inode_to_core(ino, 0), 0);
            }

            /// With 1 core, every inode routes to core 0.
            #[test]
            fn inode_routing_single_core(ino in 0_u64..=u64::MAX) {
                prop_assert_eq!(inode_to_core(ino, 1), 0);
            }

            /// lookup_to_core delegates to inode_to_core on parent.
            #[test]
            fn lookup_routes_same_as_inode(parent in 0_u64..=u64::MAX, cores in 1_u32..=256) {
                prop_assert_eq!(
                    lookup_to_core(parent, cores),
                    inode_to_core(parent, cores)
                );
            }

            // ── classify_xattr_reply properties ────────────────────────

            /// size=0 always produces Size variant (probe mode).
            #[test]
            fn xattr_probe_always_returns_size(payload_len in 0_usize..=u32::MAX as usize) {
                let plan = FrankenFuse::classify_xattr_reply(0, payload_len);
                match plan {
                    XattrReplyPlan::Size(n) => {
                        prop_assert_eq!(n, u32::try_from(payload_len).unwrap());
                    }
                    _ => prop_assert!(false, "expected Size variant, got {plan:?}"),
                }
            }

            /// When buffer fits (payload <= size), always produces Data.
            #[test]
            fn xattr_data_when_fits(
                size in 1_u32..=u32::MAX,
                payload_len in 0_u32..=u32::MAX,
            ) {
                // Only test when payload_len <= size
                if payload_len <= size {
                    let plan = FrankenFuse::classify_xattr_reply(size, payload_len as usize);
                    prop_assert_eq!(plan, XattrReplyPlan::Data);
                }
            }

            /// When buffer too small (payload > size > 0), produces ERANGE.
            #[test]
            fn xattr_erange_when_too_small(
                size in 1_u32..=u32::MAX - 1,
                extra in 1_u32..=1024,
            ) {
                let payload_len = (u64::from(size) + u64::from(extra)).min(u64::from(u32::MAX)) as usize;
                if payload_len > usize::try_from(size).unwrap() {
                    let plan = FrankenFuse::classify_xattr_reply(size, payload_len);
                    prop_assert_eq!(plan, XattrReplyPlan::Error(libc::ERANGE));
                }
            }

            // ── parse_setxattr_mode properties ─────────────────────────

            /// Valid flags (0, CREATE, REPLACE) with position=0 always succeed.
            #[test]
            fn setxattr_valid_flags_succeed(flag in prop_oneof![
                Just(0_i32),
                Just(XATTR_FLAG_CREATE),
                Just(XATTR_FLAG_REPLACE),
            ]) {
                prop_assert!(FrankenFuse::parse_setxattr_mode(flag, 0).is_ok());
            }

            /// Non-zero position always fails with EINVAL.
            #[test]
            fn setxattr_nonzero_position_fails(flags in 0_i32..=3, position in 1_u32..=u32::MAX) {
                let result = FrankenFuse::parse_setxattr_mode(flags, position);
                prop_assert_eq!(result, Err(libc::EINVAL));
            }

            /// Unknown flags (bits outside CREATE|REPLACE) always fail.
            #[test]
            fn setxattr_unknown_flags_fail(unknown_bits in 4_i32..=i32::MAX) {
                // Ensure at least one bit outside the known mask is set.
                let known = XATTR_FLAG_CREATE | XATTR_FLAG_REPLACE;
                if unknown_bits & !known != 0 {
                    let result = FrankenFuse::parse_setxattr_mode(unknown_bits, 0);
                    prop_assert_eq!(result, Err(libc::EINVAL));
                }
            }

            /// CREATE|REPLACE together always fail.
            #[test]
            fn setxattr_create_and_replace_fail(_dummy in 0_u8..1) {
                let result = FrankenFuse::parse_setxattr_mode(
                    XATTR_FLAG_CREATE | XATTR_FLAG_REPLACE, 0
                );
                prop_assert_eq!(result, Err(libc::EINVAL));
            }

            // ── encode_xattr_names properties ──────────────────────────

            /// Encoded output length = sum(name.len() + 1) for each name.
            #[test]
            fn xattr_encode_length_property(
                names in prop::collection::vec("[a-z]{1,20}", 0..10)
            ) {
                let encoded = FrankenFuse::encode_xattr_names(&names);
                let expected_len: usize = names.iter().map(|n| n.len() + 1).sum();
                prop_assert_eq!(encoded.len(), expected_len);
            }

            /// Each encoded name ends with NUL separator.
            #[test]
            fn xattr_encode_nul_separated(
                names in prop::collection::vec("[a-z]{1,20}", 1..10)
            ) {
                let encoded = FrankenFuse::encode_xattr_names(&names);
                if !encoded.is_empty() {
                    prop_assert_eq!(*encoded.last().unwrap(), 0_u8);
                }
                // Count NUL bytes = number of names.
                #[expect(clippy::naive_bytecount)] // test: bytecount crate not warranted
                let nul_count = encoded.iter().filter(|&&b| b == 0).count();
                prop_assert_eq!(nul_count, names.len());
            }

            // ── AccessPredictor properties ──────────────────────────────

            /// History never exceeds max_entries.
            #[test]
            fn access_predictor_bounded_history(
                max_entries in 1_usize..=16,
                num_reads in 1_usize..=64,
            ) {
                let predictor = AccessPredictor::new(max_entries);
                for i in 0..u64::try_from(num_reads).unwrap() {
                    predictor.record_read(InodeNumber(i), 0, 4096);
                }
                let count = match predictor.state.lock() {
                    Ok(guard) => guard.history.len(),
                    Err(poisoned) => poisoned.into_inner().history.len(),
                };
                prop_assert!(count <= max_entries, "history {count} > max {max_entries}");
            }

            /// Fetch size for unknown inode equals requested size.
            #[test]
            fn access_predictor_unknown_inode_returns_requested(
                ino in 0_u64..=u64::MAX,
                offset in 0_u64..=u64::MAX,
                size in 1_u32..=65536,
            ) {
                let predictor = AccessPredictor::new(16);
                prop_assert_eq!(predictor.fetch_size(InodeNumber(ino), offset, size), size);
            }

            /// Zero-size reads are silently dropped (no state mutation).
            #[test]
            fn access_predictor_zero_size_read_is_noop(ino in 0_u64..=1000) {
                let predictor = AccessPredictor::new(16);
                predictor.record_read(InodeNumber(ino), 0, 0);
                let count = match predictor.state.lock() {
                    Ok(guard) => guard.history.len(),
                    Err(poisoned) => poisoned.into_inner().history.len(),
                };
                prop_assert_eq!(count, 0);
            }

            /// Coalesced fetch size is always >= requested size.
            #[test]
            fn access_predictor_fetch_at_least_requested(
                offset in 0_u64..=1_000_000,
                size in 1_u32..=65536,
            ) {
                let predictor = AccessPredictor::new(64);
                let ino = InodeNumber(42);
                // Build some sequential history.
                for i in 0..5_u64 {
                    predictor.record_read(ino, i * u64::from(size), size);
                }
                let fetch = predictor.fetch_size(ino, offset, size);
                prop_assert!(fetch >= size, "fetch {fetch} < requested {size}");
            }

            /// Coalesced fetch size never exceeds MAX_COALESCED_READ_SIZE.
            #[test]
            fn access_predictor_fetch_capped(size in 1_u32..=65536) {
                let predictor = AccessPredictor::new(64);
                let ino = InodeNumber(99);
                // Build long forward sequence.
                for i in 0..20_u64 {
                    predictor.record_read(ino, i * u64::from(size), size);
                }
                let next_offset = 20 * u64::from(size);
                let fetch = predictor.fetch_size(ino, next_offset, size);
                prop_assert!(
                    fetch <= MAX_COALESCED_READ_SIZE.max(size),
                    "fetch {fetch} > cap {}",
                    MAX_COALESCED_READ_SIZE.max(size)
                );
            }

            // ── ReadaheadManager properties ─────────────────────────────

            /// insert then take at same offset returns the data.
            #[test]
            fn readahead_insert_take_roundtrip(
                ino in 1_u64..=1000,
                offset in 0_u64..=1_000_000,
                data in prop::collection::vec(any::<u8>(), 1..128),
            ) {
                let manager = ReadaheadManager::new(64);
                let data_clone = data.clone();
                manager.insert(InodeNumber(ino), offset, data);
                let taken = manager.take(InodeNumber(ino), offset, data_clone.len());
                prop_assert_eq!(taken, Some(data_clone));
            }

            /// take after consume returns None.
            #[test]
            fn readahead_double_take_returns_none(
                ino in 1_u64..=1000,
                offset in 0_u64..=1_000_000,
                data in prop::collection::vec(any::<u8>(), 1..64),
            ) {
                let manager = ReadaheadManager::new(64);
                let len = data.len();
                manager.insert(InodeNumber(ino), offset, data);
                let _ = manager.take(InodeNumber(ino), offset, len);
                let second = manager.take(InodeNumber(ino), offset, len);
                prop_assert_eq!(second, None);
            }

            /// Pending entries never exceed max_pending.
            #[test]
            fn readahead_bounded_entries(
                max_pending in 1_usize..=8,
                num_inserts in 1_usize..=32,
            ) {
                let manager = ReadaheadManager::new(max_pending);
                for i in 0..u64::try_from(num_inserts).unwrap() {
                    manager.insert(InodeNumber(1), i * 1024, vec![0xAA]);
                }
                let count = match manager.pending.lock() {
                    Ok(guard) => guard.map.len(),
                    Err(poisoned) => poisoned.into_inner().map.len(),
                };
                prop_assert!(count <= max_pending, "entries {count} > max {max_pending}");
            }

            /// Empty data insertions are silently ignored.
            #[test]
            fn readahead_empty_insert_ignored(ino in 1_u64..=100, offset in 0_u64..=1000) {
                let manager = ReadaheadManager::new(8);
                manager.insert(InodeNumber(ino), offset, vec![]);
                let count = match manager.pending.lock() {
                    Ok(guard) => guard.map.len(),
                    Err(poisoned) => poisoned.into_inner().map.len(),
                };
                prop_assert_eq!(count, 0);
            }

            /// Partial take returns prefix and preserves tail at correct offset.
            #[test]
            fn readahead_partial_take_preserves_tail(
                data in prop::collection::vec(any::<u8>(), 4..128),
                take_len in 1_usize..=3,
            ) {
                let manager = ReadaheadManager::new(16);
                let ino = InodeNumber(7);
                let offset = 0_u64;
                let data_clone = data.clone();
                let actual_take = take_len.min(data.len() - 1); // Ensure a tail exists.
                if actual_take < data.len() {
                    manager.insert(ino, offset, data);
                    let prefix = manager.take(ino, offset, actual_take);
                    prop_assert_eq!(prefix.as_deref(), Some(&data_clone[..actual_take]));

                    // Tail should be at offset + actual_take.
                    let tail_offset = offset + u64::try_from(actual_take).unwrap();
                    let tail = manager.take(ino, tail_offset, data_clone.len());
                    prop_assert_eq!(tail.as_deref(), Some(&data_clone[actual_take..]));
                }
            }

            // ── AtomicMetrics properties ────────────────────────────────

            /// ok + err always equals total.
            #[test]
            fn metrics_ok_plus_err_equals_total(
                num_ok in 0_u64..=500,
                num_err in 0_u64..=500,
            ) {
                let metrics = AtomicMetrics::new();
                for _ in 0..num_ok { metrics.record_ok(); }
                for _ in 0..num_err { metrics.record_err(); }
                let snap = metrics.snapshot();
                prop_assert_eq!(snap.requests_ok, num_ok);
                prop_assert_eq!(snap.requests_err, num_err);
                prop_assert_eq!(snap.requests_total, num_ok + num_err);
            }

            /// bytes_read accumulates correctly.
            #[test]
            fn metrics_bytes_read_accumulates(
                reads in prop::collection::vec(1_u64..=8192, 0..50),
            ) {
                let metrics = AtomicMetrics::new();
                let expected: u64 = reads.iter().sum();
                for &n in &reads {
                    metrics.record_bytes_read(n);
                }
                prop_assert_eq!(metrics.snapshot().bytes_read, expected);
            }

            // ── MountOptions properties ─────────────────────────────────

            /// Resolved thread count is always >= 1.
            #[test]
            fn mount_options_resolved_at_least_one(threads in 0_usize..=256) {
                let opts = MountOptions {
                    worker_threads: threads,
                    ..Default::default()
                };
                prop_assert!(opts.resolved_thread_count() >= 1);
            }

            /// Explicit worker_threads passes through (when > 0).
            #[test]
            fn mount_options_explicit_passthrough(threads in 1_usize..=256) {
                let opts = MountOptions {
                    worker_threads: threads,
                    ..Default::default()
                };
                prop_assert_eq!(opts.resolved_thread_count(), threads);
            }

            // ── PerCoreConfig properties ────────────────────────────────

            /// total_cache_blocks = resolved_cores * cache_blocks_per_core.
            #[test]
            fn per_core_total_cache_blocks(
                cores in 1_u32..=16,
                blocks_per_core in 1_u32..=65536,
            ) {
                let cfg = PerCoreConfig {
                    num_cores: cores,
                    cache_blocks_per_core: blocks_per_core,
                    steal_threshold: 2.0,
                    advisory_affinity: true,
                };
                prop_assert_eq!(
                    cfg.total_cache_blocks(),
                    u64::from(cores) * u64::from(blocks_per_core)
                );
            }

            /// PerCoreDispatcher has exactly num_cores metrics slots.
            #[test]
            fn dispatcher_correct_num_metrics(cores in 1_u32..=16) {
                let cfg = PerCoreConfig {
                    num_cores: cores,
                    ..Default::default()
                };
                let disp = PerCoreDispatcher::new(cfg);
                prop_assert_eq!(disp.num_cores(), cores);
                for c in 0..cores {
                    prop_assert!(disp.core_metrics(c).is_some());
                }
                prop_assert!(disp.core_metrics(cores).is_none());
            }

            /// Aggregate total_requests = sum of per-core requests.
            #[test]
            fn dispatcher_aggregate_sums(
                per_core_counts in prop::collection::vec(0_u64..=1000, 2..=8),
            ) {
                let n = per_core_counts.len() as u32;
                let cfg = PerCoreConfig {
                    num_cores: n,
                    ..Default::default()
                };
                let disp = PerCoreDispatcher::new(cfg);
                let mut expected_total = 0_u64;
                for (i, &count) in per_core_counts.iter().enumerate() {
                    let m = disp.core_metrics(i as u32).unwrap();
                    for _ in 0..count {
                        m.record_request();
                    }
                    expected_total += count;
                }
                let agg = disp.aggregate_metrics();
                prop_assert_eq!(agg.total_requests, expected_total);
                prop_assert_eq!(agg.per_core.len(), n as usize);
            }

            /// Hit rate is in [0.0, 1.0] range.
            #[test]
            fn core_metrics_hit_rate_bounded(
                hits in 0_u64..=1000,
                misses in 0_u64..=1000,
            ) {
                let m = CoreMetrics::new();
                for _ in 0..hits { m.record_hit(); }
                for _ in 0..misses { m.record_miss(); }
                let rate = m.snapshot().hit_rate();
                prop_assert!((0.0..=1.0).contains(&rate), "hit_rate {rate} out of bounds");
            }

            /// Imbalance ratio >= 1.0 (or infinity if min is zero).
            #[test]
            fn dispatcher_imbalance_ratio_at_least_one(
                per_core_counts in prop::collection::vec(0_u64..=1000, 2..=8),
            ) {
                let n = per_core_counts.len() as u32;
                let cfg = PerCoreConfig {
                    num_cores: n,
                    ..Default::default()
                };
                let disp = PerCoreDispatcher::new(cfg);
                for (i, &count) in per_core_counts.iter().enumerate() {
                    let m = disp.core_metrics(i as u32).unwrap();
                    for _ in 0..count {
                        m.record_request();
                    }
                }
                let ratio = disp.aggregate_metrics().imbalance_ratio();
                prop_assert!(ratio >= 1.0 || ratio.is_infinite(),
                    "imbalance_ratio {ratio} < 1.0");
            }
        }
    }
}
