//! VFS semantics layer: filesystem-agnostic types and operations trait.
//!
//! This module defines the core VFS abstractions that higher layers (FUSE,
//! test harness) consume. Format-specific implementations (ext4, btrfs) live
//! behind the [`FsOps`] trait so that callers are filesystem-agnostic.

use asupersync::Cx;
use ffs_error::FfsError;
use ffs_types::{InodeNumber, Snapshot, TxnId};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::Path;
use std::time::SystemTime;

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
    /// Create a scope with no snapshot or transaction attached.
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
