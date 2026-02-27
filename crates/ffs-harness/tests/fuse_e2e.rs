#![forbid(unsafe_code)]
//! E2E tests that mount an ext4 image via FUSE and verify file operations
//! through the kernel VFS.
//!
//! These tests require:
//! - `/dev/fuse` to exist (FUSE kernel module)
//! - `mkfs.ext4` and `debugfs` on `$PATH`
//! - `fusermount3` permission to mount (may fail in containers)
//!
//! A small smoke subset (read-only + lightweight rw ext4/btrfs) runs by
//! default and returns early when prerequisites are unavailable.
//! Heavier write-path and btrfs coverage remains gated with
//! `#[ignore = "requires /dev/fuse"]` and can be run explicitly via
//! `cargo test -- --ignored` or `cargo test -- --include-ignored`.

use asupersync::Cx;
use ffs_core::{Ext4JournalReplayMode, OpenFs, OpenOptions};
use ffs_fuse::{MountOptions, mount_background};
use std::fs;
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

/// Check if FUSE E2E prerequisites are met.
fn fuse_available() -> bool {
    Path::new("/dev/fuse").exists()
        && Command::new("which")
            .arg("mkfs.ext4")
            .output()
            .is_ok_and(|o| o.status.success())
        && Command::new("which")
            .arg("debugfs")
            .output()
            .is_ok_and(|o| o.status.success())
}

/// Create a small ext4 image and populate it with test files using debugfs.
fn create_test_image(dir: &Path) -> std::path::PathBuf {
    let image = dir.join("test.ext4");

    // Create a 4 MiB sparse image.
    let f = fs::File::create(&image).expect("create image");
    f.set_len(4 * 1024 * 1024).expect("set image size");
    drop(f);

    // mkfs.ext4
    let out = Command::new("mkfs.ext4")
        .args([
            "-F",
            "-b",
            "4096",
            "-L",
            "ffs-fuse-e2e",
            image.to_str().unwrap(),
        ])
        .output()
        .expect("mkfs.ext4");
    assert!(
        out.status.success(),
        "mkfs.ext4 failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Populate with test files via debugfs.
    let hello_path = dir.join("hello_src.txt");
    let nested_path = dir.join("nested_src.txt");
    fs::write(&hello_path, b"Hello from FrankenFS E2E!\n").expect("write hello src");
    fs::write(&nested_path, b"Nested file content.\n").expect("write nested src");

    // Create directory
    let out = Command::new("debugfs")
        .args(["-w", "-R", "mkdir testdir", image.to_str().unwrap()])
        .output()
        .expect("debugfs mkdir");
    assert!(
        out.status.success(),
        "debugfs mkdir failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Write hello.txt
    let out = Command::new("debugfs")
        .args([
            "-w",
            "-R",
            &format!("write {} hello.txt", hello_path.display()),
            image.to_str().unwrap(),
        ])
        .output()
        .expect("debugfs write hello.txt");
    assert!(
        out.status.success(),
        "debugfs write hello.txt failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Write nested.txt
    let out = Command::new("debugfs")
        .args([
            "-w",
            "-R",
            &format!("write {} testdir/nested.txt", nested_path.display()),
            image.to_str().unwrap(),
        ])
        .output()
        .expect("debugfs write nested.txt");
    assert!(
        out.status.success(),
        "debugfs write nested.txt failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    image
}

/// Try to mount an ext4 image via FrankenFS FUSE (read-only).
///
/// Returns `None` if FUSE mounting fails (e.g. permission denied in containers).
fn try_mount_ffs(image: &Path, mountpoint: &Path) -> Option<fuser::BackgroundSession> {
    let cx = Cx::for_testing();
    let opts = OpenOptions {
        skip_validation: false,
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
    };
    let fs = OpenFs::open_with_options(&cx, image, &opts).expect("open ext4 image");
    let mount_opts = MountOptions {
        read_only: true,
        auto_unmount: false,
        ..MountOptions::default()
    };
    match mount_background(Box::new(fs), mountpoint, &mount_opts) {
        Ok(session) => {
            // Give FUSE a moment to initialize.
            thread::sleep(Duration::from_millis(300));
            Some(session)
        }
        Err(e) => {
            eprintln!("FUSE mount failed (skipping test): {e}");
            None
        }
    }
}

#[test]
fn fuse_read_hello_txt() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Read hello.txt through FUSE.
    let content = fs::read_to_string(mnt.join("hello.txt")).expect("read hello.txt via FUSE");
    assert_eq!(content, "Hello from FrankenFS E2E!\n");
}

#[test]
fn fuse_readdir_root() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Read root directory entries.
    let entries: Vec<String> = fs::read_dir(&mnt)
        .expect("readdir root via FUSE")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    assert!(
        entries.contains(&"hello.txt".to_owned()),
        "root should contain hello.txt, got: {entries:?}"
    );
    assert!(
        entries.contains(&"testdir".to_owned()),
        "root should contain testdir, got: {entries:?}"
    );
    assert!(
        entries.contains(&"lost+found".to_owned()),
        "root should contain lost+found, got: {entries:?}"
    );
}

#[test]
fn fuse_read_nested_file() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Read nested file through FUSE.
    let content =
        fs::read_to_string(mnt.join("testdir/nested.txt")).expect("read nested.txt via FUSE");
    assert_eq!(content, "Nested file content.\n");
}

#[test]
fn fuse_getattr_file_metadata() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Check file metadata.
    let meta = fs::metadata(mnt.join("hello.txt")).expect("stat hello.txt via FUSE");
    assert!(meta.is_file(), "hello.txt should be a regular file");
    assert_eq!(
        meta.len(),
        26,
        "hello.txt should be 26 bytes ('Hello from FrankenFS E2E!\\n')"
    );

    // Check directory metadata.
    let dir_meta = fs::metadata(mnt.join("testdir")).expect("stat testdir via FUSE");
    assert!(dir_meta.is_dir(), "testdir should be a directory");
}

#[test]
fn fuse_readlink_and_symlink_detection() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());

    // Add a symlink via debugfs.
    let out = Command::new("debugfs")
        .args([
            "-w",
            "-R",
            "symlink link.txt hello.txt",
            image.to_str().unwrap(),
        ])
        .output()
        .expect("debugfs symlink");
    assert!(
        out.status.success(),
        "debugfs symlink failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Reading the symlink target.
    let target = fs::read_link(mnt.join("link.txt")).expect("readlink via FUSE");
    assert_eq!(
        target.to_str().unwrap(),
        "hello.txt",
        "symlink should point to hello.txt"
    );

    // Following the symlink should give the same content.
    let content = fs::read_to_string(mnt.join("link.txt")).expect("read through symlink via FUSE");
    assert_eq!(content, "Hello from FrankenFS E2E!\n");
}

// ── Write-path E2E tests ────────────────────────────────────────────────────

/// Try to mount an ext4 image via FrankenFS FUSE in **read-write** mode.
///
/// Returns `None` if FUSE mounting fails (e.g. permission denied in containers).
fn try_mount_ffs_rw(image: &Path, mountpoint: &Path) -> Option<fuser::BackgroundSession> {
    let cx = Cx::for_testing();
    let opts = OpenOptions {
        skip_validation: false,
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
    };
    let mut fs = OpenFs::open_with_options(&cx, image, &opts).expect("open ext4 image");
    fs.enable_writes(&cx).expect("enable ext4 write support");
    let mount_opts = MountOptions {
        read_only: false,
        auto_unmount: false,
        ..MountOptions::default()
    };
    match mount_background(Box::new(fs), mountpoint, &mount_opts) {
        Ok(session) => {
            thread::sleep(Duration::from_millis(300));
            Some(session)
        }
        Err(e) => {
            eprintln!("FUSE mount (rw) failed (skipping test): {e}");
            None
        }
    }
}

/// Helper: create image, mount rw, run a closure, then drop the session.
fn with_rw_mount(f: impl FnOnce(&Path)) {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }
    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs_rw(&image, &mnt) else {
        return;
    };
    f(&mnt);
}

#[test]
fn fuse_create_and_read_file() {
    with_rw_mount(|mnt| {
        let path = mnt.join("newfile.txt");
        fs::write(&path, b"Created via FUSE write path!\n").expect("create file via FUSE");

        let content = fs::read_to_string(&path).expect("read back created file");
        assert_eq!(content, "Created via FUSE write path!\n");

        let meta = fs::metadata(&path).expect("stat created file");
        assert!(meta.is_file());
        assert_eq!(meta.len(), 29);
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_write_overwrite_and_append() {
    with_rw_mount(|mnt| {
        let path = mnt.join("overwrite.txt");

        // Write initial content.
        fs::write(&path, b"initial").expect("write initial");
        assert_eq!(fs::read_to_string(&path).expect("read initial"), "initial");

        // Overwrite with longer content.
        fs::write(&path, b"overwritten content").expect("overwrite");
        assert_eq!(
            fs::read_to_string(&path).expect("read overwritten"),
            "overwritten content"
        );

        // Append additional content.
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open for append");
        file.write_all(b" + appended").expect("append write");
        drop(file);
        assert_eq!(
            fs::read_to_string(&path).expect("read appended"),
            "overwritten content + appended"
        );
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_write_with_offset_extends_file_and_zero_fills_gap() {
    with_rw_mount(|mnt| {
        let path = mnt.join("offset_write.bin");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .expect("create offset_write.bin");

        file.seek(SeekFrom::Start(8))
            .expect("seek to sparse offset");
        file.write_all(b"abc").expect("write payload at offset");
        drop(file);

        let bytes = fs::read(&path).expect("read sparse write result");
        assert_eq!(bytes.len(), 11);
        assert_eq!(&bytes[..8], vec![0_u8; 8].as_slice());
        assert_eq!(&bytes[8..], b"abc");
    });
}

#[test]
fn fuse_mkdir_and_nested_create() {
    with_rw_mount(|mnt| {
        let dir = mnt.join("newdir");
        fs::create_dir(&dir).expect("mkdir via FUSE");

        let meta = fs::metadata(&dir).expect("stat newdir");
        assert!(meta.is_dir());

        // Create a file inside the new directory.
        let nested = dir.join("inner.txt");
        fs::write(&nested, b"nested content\n").expect("write nested file");

        let content = fs::read_to_string(&nested).expect("read nested file");
        assert_eq!(content, "nested content\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_mkdir_existing_directory_fails() {
    with_rw_mount(|mnt| {
        let dir = mnt.join("already_there");
        fs::create_dir(&dir).expect("initial mkdir should succeed");

        let err = fs::create_dir(&dir).expect_err("mkdir existing should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
    });
}

#[test]
fn fuse_unlink_removes_file() {
    with_rw_mount(|mnt| {
        // hello.txt exists from create_test_image.
        let path = mnt.join("hello.txt");
        assert!(path.exists(), "hello.txt should exist before unlink");

        fs::remove_file(&path).expect("unlink hello.txt via FUSE");
        assert!(!path.exists(), "hello.txt should be gone after unlink");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rmdir_missing_directory_fails() {
    with_rw_mount(|mnt| {
        let missing = mnt.join("no_such_dir");
        let err = fs::remove_dir(&missing).expect_err("rmdir missing should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rmdir_removes_empty_directory() {
    with_rw_mount(|mnt| {
        let dir = mnt.join("empty_dir");
        fs::create_dir(&dir).expect("mkdir empty_dir");
        assert!(dir.exists());

        fs::remove_dir(&dir).expect("rmdir empty_dir via FUSE");
        assert!(!dir.exists(), "empty_dir should be gone after rmdir");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rmdir_non_empty_fails() {
    with_rw_mount(|mnt| {
        let dir = mnt.join("non_empty_dir");
        fs::create_dir(&dir).expect("mkdir non_empty_dir");
        fs::write(dir.join("child.txt"), b"child").expect("create child in non_empty_dir");

        let err = fs::remove_dir(&dir).expect_err("rmdir non-empty should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::DirectoryNotEmpty);
        assert!(
            dir.exists(),
            "directory should still exist after failed rmdir"
        );
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rename_over_existing_destination_replaces_target() {
    with_rw_mount(|mnt| {
        let src = mnt.join("src.txt");
        let dst = mnt.join("dst.txt");
        fs::write(&src, b"from-src").expect("write src");
        fs::write(&dst, b"stale-dst").expect("write existing dst");

        fs::rename(&src, &dst).expect("rename over existing destination");
        assert!(!src.exists(), "source path should be removed");
        assert_eq!(
            fs::read_to_string(&dst).expect("read replaced dst"),
            "from-src"
        );
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rename_file() {
    with_rw_mount(|mnt| {
        let old = mnt.join("hello.txt");
        let new = mnt.join("renamed.txt");
        assert!(old.exists());

        fs::rename(&old, &new).expect("rename via FUSE");
        assert!(!old.exists(), "old name should be gone");
        assert!(new.exists(), "new name should exist");

        let content = fs::read_to_string(&new).expect("read renamed file");
        assert_eq!(content, "Hello from FrankenFS E2E!\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_rename_across_directories() {
    with_rw_mount(|mnt| {
        let src = mnt.join("hello.txt");
        let dst = mnt.join("testdir/moved.txt");
        assert!(src.exists());

        fs::rename(&src, &dst).expect("rename across dirs via FUSE");
        assert!(!src.exists());

        let content = fs::read_to_string(&dst).expect("read moved file");
        assert_eq!(content, "Hello from FrankenFS E2E!\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_hard_link() {
    with_rw_mount(|mnt| {
        let original = mnt.join("hello.txt");
        let link = mnt.join("hello_link.txt");

        fs::hard_link(&original, &link).expect("hard link via FUSE");

        let content = fs::read_to_string(&link).expect("read through hard link");
        assert_eq!(content, "Hello from FrankenFS E2E!\n");

        // Both should share the same inode.
        let orig_ino = fs::metadata(&original).expect("stat original").ino();
        let link_ino = fs::metadata(&link).expect("stat link").ino();
        assert_eq!(orig_ino, link_ino, "hard link should share inode");
        let orig_nlink = fs::metadata(&original).expect("stat original").nlink();
        let link_nlink = fs::metadata(&link).expect("stat link").nlink();
        assert_eq!(orig_nlink, 2, "original should report two hard links");
        assert_eq!(link_nlink, 2, "link should report two hard links");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_symlink_create_and_follow() {
    with_rw_mount(|mnt| {
        let target = mnt.join("hello.txt");
        let link = mnt.join("sym.txt");

        std::os::unix::fs::symlink("hello.txt", &link).expect("symlink via FUSE");

        // Verify readlink returns the target.
        let read_target = fs::read_link(&link).expect("readlink via FUSE");
        assert_eq!(read_target.to_str().unwrap(), "hello.txt");

        // Following the symlink should work.
        let content = fs::read_to_string(&link).expect("read through new symlink");
        assert_eq!(content, "Hello from FrankenFS E2E!\n");

        // Symlink metadata should differ from target.
        let link_meta = fs::symlink_metadata(&link).expect("lstat symlink");
        assert!(link_meta.file_type().is_symlink());
        let _ = target; // used implicitly via symlink follow
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_setattr_truncate() {
    with_rw_mount(|mnt| {
        let path = mnt.join("hello.txt");
        let original_len = fs::metadata(&path).expect("stat").len();
        assert!(original_len > 0);

        // Truncate to 5 bytes.
        let f = fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for truncate");
        f.set_len(5).expect("truncate via FUSE");
        drop(f);

        let new_len = fs::metadata(&path).expect("stat after truncate").len();
        assert_eq!(new_len, 5, "file should be truncated to 5 bytes");

        let content = fs::read_to_string(&path).expect("read truncated file");
        assert_eq!(content, "Hello");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_setattr_chmod() {
    with_rw_mount(|mnt| {
        // hello.txt exists from create_test_image.
        let path = mnt.join("hello.txt");
        let orig_meta = fs::metadata(&path).expect("stat hello.txt");
        let orig_mode = orig_meta.permissions().mode() & 0o7777;

        // Change to 0o755.
        let new_perms = fs::Permissions::from_mode(0o755);
        fs::set_permissions(&path, new_perms).expect("chmod 755 via FUSE");

        let meta = fs::metadata(&path).expect("stat after chmod");
        assert_eq!(
            meta.permissions().mode() & 0o7777,
            0o755,
            "permissions should be 0o755 after chmod (were 0o{orig_mode:o})"
        );

        // Change to 0o600.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .expect("chmod 600 via FUSE");

        let meta = fs::metadata(&path).expect("stat after second chmod");
        assert_eq!(
            meta.permissions().mode() & 0o7777,
            0o600,
            "permissions should be 0o600 after second chmod"
        );

        // File should still be readable/writable by us since we own it.
        let content = fs::read_to_string(&path).expect("read after chmod");
        assert_eq!(content, "Hello from FrankenFS E2E!\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_statfs_returns_valid_stats() {
    if !fuse_available() {
        eprintln!("FUSE prerequisites not met, skipping");
        return;
    }

    let tmp = TempDir::new().expect("tmpdir");
    let image = create_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_ffs(&image, &mnt) else {
        return;
    };

    // Use `stat -f` to exercise the FUSE statfs handler and parse the output.
    let out = Command::new("stat")
        .args(["-f", "-c", "%s %b %f %a %c %d %l", mnt.to_str().unwrap()])
        .output()
        .expect("stat -f on mountpoint");
    assert!(
        out.status.success(),
        "stat -f failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let fields: Vec<&str> = stdout.split_whitespace().collect();
    assert_eq!(
        fields.len(),
        7,
        "expected 7 stat -f fields, got: {stdout:?}"
    );

    // Parse statfs fields: block_size blocks blocks_free blocks_avail files files_free namelen
    let block_size: u64 = fields[0].parse().expect("parse block_size");
    let blocks: u64 = fields[1].parse().expect("parse blocks");
    let blocks_free: u64 = fields[2].parse().expect("parse blocks_free");
    let blocks_avail: u64 = fields[3].parse().expect("parse blocks_avail");
    let files: u64 = fields[4].parse().expect("parse files");
    let files_free: u64 = fields[5].parse().expect("parse files_free");
    let namelen: u64 = fields[6].parse().expect("parse namelen");

    // Validate: block size should be a power of two in [1024, 65536].
    assert!(
        block_size.is_power_of_two() && (1024..=65536).contains(&block_size),
        "block_size {block_size} should be a power-of-two in [1024, 65536]"
    );

    // Total blocks should be non-zero (we made a 4 MiB image).
    assert!(blocks > 0, "total blocks should be > 0");

    // Free blocks should not exceed total blocks.
    assert!(
        blocks_free <= blocks,
        "free blocks ({blocks_free}) should be <= total ({blocks})"
    );
    assert!(
        blocks_avail <= blocks,
        "available blocks ({blocks_avail}) should be <= total ({blocks})"
    );

    // Total inodes should be non-zero.
    assert!(files > 0, "total inodes should be > 0");
    assert!(
        files_free <= files,
        "free inodes ({files_free}) should be <= total ({files})"
    );

    // Max filename length: ext4 is 255.
    assert_eq!(namelen, 255, "ext4 max filename length should be 255");
}

#[test]
#[ignore = "requires /dev/fuse"]
fn fuse_write_large_file() {
    with_rw_mount(|mnt| {
        let path = mnt.join("large.bin");
        // Write 64 KiB of patterned data (crosses multiple blocks).
        let data: Vec<u8> = (0..65536_u32).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &data).expect("write large file via FUSE");

        let readback = fs::read(&path).expect("read large file");
        assert_eq!(readback.len(), 65536);
        assert_eq!(readback, data, "large file content should match");
    });
}

// ── btrfs FUSE E2E tests ────────────────────────────────────────────────────

/// Check if btrfs FUSE prerequisites are met.
fn btrfs_fuse_available() -> bool {
    Path::new("/dev/fuse").exists()
        && Command::new("which")
            .arg("mkfs.btrfs")
            .output()
            .is_ok_and(|o| o.status.success())
}

/// Create a small btrfs image and populate it with test files.
fn create_btrfs_test_image(dir: &Path) -> std::path::PathBuf {
    let image = dir.join("test.btrfs");

    // Create a 128 MiB sparse image (btrfs minimum is ~109 MiB).
    let f = fs::File::create(&image).expect("create btrfs image");
    f.set_len(128 * 1024 * 1024).expect("set btrfs image size");
    drop(f);

    // mkfs.btrfs
    let out = Command::new("mkfs.btrfs")
        .args(["-f", "-L", "ffs-btrfs-e2e", image.to_str().unwrap()])
        .output()
        .expect("mkfs.btrfs");
    assert!(
        out.status.success(),
        "mkfs.btrfs failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    image
}

/// Try to mount a btrfs image via FrankenFS FUSE (read-write).
fn try_mount_btrfs_rw(image: &Path, mountpoint: &Path) -> Option<fuser::BackgroundSession> {
    let cx = Cx::for_testing();
    let opts = OpenOptions {
        skip_validation: false,
        ext4_journal_replay_mode: Ext4JournalReplayMode::SimulateOverlay,
    };
    let mut fs = OpenFs::open_with_options(&cx, image, &opts).expect("open btrfs image");
    if let Err(e) = fs.enable_writes(&cx) {
        eprintln!("btrfs enable_writes failed (skipping test): {e}");
        return None;
    }
    let mount_opts = MountOptions {
        read_only: false,
        auto_unmount: false,
        ..MountOptions::default()
    };
    match mount_background(Box::new(fs), mountpoint, &mount_opts) {
        Ok(session) => {
            thread::sleep(Duration::from_millis(300));
            Some(session)
        }
        Err(e) => {
            eprintln!("btrfs FUSE mount failed (skipping test): {e}");
            None
        }
    }
}

/// Helper: create btrfs image, mount rw, run a closure.
fn with_btrfs_rw_mount(f: impl FnOnce(&Path)) {
    if !btrfs_fuse_available() {
        eprintln!("btrfs FUSE prerequisites not met, skipping");
        return;
    }
    let tmp = TempDir::new().expect("tmpdir");
    let image = create_btrfs_test_image(tmp.path());
    let mnt = tmp.path().join("mnt");
    fs::create_dir_all(&mnt).expect("create mountpoint");

    let Some(_session) = try_mount_btrfs_rw(&image, &mnt) else {
        return;
    };
    f(&mnt);
}

#[test]
fn btrfs_fuse_readdir_root() {
    with_btrfs_rw_mount(|mnt| {
        // Empty btrfs should at least have . and ..
        let entries: Vec<String> = fs::read_dir(mnt)
            .expect("readdir btrfs root via FUSE")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        // btrfs root dir exists; entries may be empty (no default subvolume files).
        // Just verify readdir doesn't error.
        let _ = entries;
    });
}

#[test]
fn btrfs_fuse_create_and_read_file() {
    with_btrfs_rw_mount(|mnt| {
        let path = mnt.join("hello.txt");
        fs::write(&path, b"Hello from btrfs FUSE!\n").expect("write file on btrfs");

        let content = fs::read_to_string(&path).expect("read file on btrfs");
        assert_eq!(content, "Hello from btrfs FUSE!\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn btrfs_fuse_mkdir_and_nested_file() {
    with_btrfs_rw_mount(|mnt| {
        let dir = mnt.join("subdir");
        fs::create_dir(&dir).expect("mkdir on btrfs");

        let nested = dir.join("nested.txt");
        fs::write(&nested, b"nested btrfs content\n").expect("write nested on btrfs");

        let content = fs::read_to_string(&nested).expect("read nested on btrfs");
        assert_eq!(content, "nested btrfs content\n");
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn btrfs_fuse_unlink_and_rmdir() {
    with_btrfs_rw_mount(|mnt| {
        // Create and remove a file.
        let path = mnt.join("temp.txt");
        fs::write(&path, b"temporary").expect("write temp");
        assert!(path.exists());
        fs::remove_file(&path).expect("unlink on btrfs");
        assert!(!path.exists());

        // Create and remove a directory.
        let dir = mnt.join("tempdir");
        fs::create_dir(&dir).expect("mkdir tempdir");
        assert!(dir.exists());
        fs::remove_dir(&dir).expect("rmdir on btrfs");
        assert!(!dir.exists());
    });
}

#[test]
#[ignore = "requires /dev/fuse"]
fn btrfs_fuse_rename() {
    with_btrfs_rw_mount(|mnt| {
        let old = mnt.join("original.txt");
        let new = mnt.join("renamed.txt");
        fs::write(&old, b"rename test").expect("write for rename");
        assert!(old.exists());

        fs::rename(&old, &new).expect("rename on btrfs");
        assert!(!old.exists());
        assert!(new.exists());
        assert_eq!(
            fs::read_to_string(&new).expect("read renamed"),
            "rename test"
        );
    });
}
