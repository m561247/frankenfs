#![forbid(unsafe_code)]

use asupersync::Cx;
use ffs_core::{OpenFs, OpenOptions};
use ffs_harness::{
    e2e::{CrashReplaySuiteConfig, FsxStressConfig, run_crash_replay_suite, run_fsx_stress},
    GoldenReference, ParityReport, validate_btrfs_chunk_fixture, validate_btrfs_fixture,
    validate_btrfs_leaf_fixture, validate_dir_block_fixture, validate_ext4_fixture,
    validate_group_desc_fixture, validate_inode_fixture,
};
use serde_json::Value;
use std::path::Path;

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("conformance")
        .join("fixtures")
        .join(name)
}

#[test]
fn ext4_and_btrfs_fixtures_conform() {
    let ext4_sparse = validate_ext4_fixture(&fixture_path("ext4_superblock_sparse.json"))
        .expect("ext4 sparse fixture");
    let ext4_mkfs = validate_ext4_fixture(&fixture_path("ext4_superblock_mkfs_4096.json"))
        .expect("ext4 mkfs fixture");
    let btrfs = validate_btrfs_fixture(&fixture_path("btrfs_superblock_sparse.json"))
        .expect("btrfs fixture");

    assert_eq!(ext4_sparse.block_size, 4096);
    assert_eq!(ext4_mkfs.block_size, 4096);
    assert_eq!(ext4_mkfs.log_cluster_size, 2);
    assert_eq!(ext4_mkfs.cluster_size, 4096);
    assert_eq!(ext4_mkfs.blocks_per_group, ext4_mkfs.clusters_per_group);
    assert_eq!(ext4_mkfs.volume_name, "ffs-mkfs");
    assert_eq!(btrfs.sectorsize, 4096);
}

#[test]
fn ext4_group_desc_fixtures_conform() {
    let gd32 = validate_group_desc_fixture(&fixture_path("ext4_group_desc_32byte.json"), 32)
        .expect("32-byte group desc");
    assert_eq!(gd32.block_bitmap, 5);
    assert_eq!(gd32.inode_bitmap, 6);
    assert_eq!(gd32.inode_table, 7);
    assert_eq!(gd32.free_blocks_count, 200);

    let gd64 = validate_group_desc_fixture(&fixture_path("ext4_group_desc_64byte.json"), 64)
        .expect("64-byte group desc");
    assert!(
        gd64.block_bitmap > u64::from(u32::MAX),
        "64-bit path should set high bits"
    );
}

#[test]
fn ext4_inode_fixtures_conform() {
    let file_inode = validate_inode_fixture(&fixture_path("ext4_inode_regular_file.json"))
        .expect("regular file inode");
    assert_eq!(
        file_inode.mode & 0o17_0000,
        0o10_0000,
        "should be regular file"
    );
    assert_eq!(file_inode.size, 1024);

    let dir_inode = validate_inode_fixture(&fixture_path("ext4_inode_directory.json"))
        .expect("directory inode");
    assert_eq!(dir_inode.mode & 0o17_0000, 0o4_0000, "should be directory");
    assert_eq!(dir_inode.links_count, 2);
}

#[test]
fn ext4_dir_block_fixture_conforms() {
    let entries =
        validate_dir_block_fixture(&fixture_path("ext4_dir_block.json"), 4096).expect("dir block");
    assert!(entries.len() >= 3, "should have at least 3 entries");
    assert!(entries.iter().any(|e| e.name_str() == "hello.txt"));
}

#[test]
fn btrfs_chunk_mapping_fixture_conforms() {
    let (sb, chunks) =
        validate_btrfs_chunk_fixture(&fixture_path("btrfs_superblock_with_chunks.json"))
            .expect("btrfs chunk fixture");
    assert!(!chunks.is_empty(), "should have at least one chunk entry");
    // root and chunk_root should be mappable
    let root_map = ffs_ondisk::map_logical_to_physical(&chunks, sb.root)
        .expect("mapping ok")
        .expect("root covered");
    assert_eq!(root_map.devid, 1);
    let cr_map = ffs_ondisk::map_logical_to_physical(&chunks, sb.chunk_root)
        .expect("mapping ok")
        .expect("chunk_root covered");
    assert_eq!(cr_map.devid, 1);
}

#[test]
fn btrfs_leaf_fixture_conforms() {
    let (header, items) = validate_btrfs_leaf_fixture(&fixture_path("btrfs_leaf_node.json"))
        .expect("btrfs leaf fixture");
    assert_eq!(header.level, 0, "should be a leaf");
    assert!(items.len() >= 3, "should have at least 3 items");
    // Items should be sorted by key (objectid then type)
    for pair in items.windows(2) {
        let a = &pair[0].key;
        let b = &pair[1].key;
        assert!(
            (a.objectid, a.item_type) <= (b.objectid, b.item_type),
            "items should be sorted by key"
        );
    }
}

/// btrfs item type constants for fixture validation
mod btrfs_item_types {
    pub const INODE_ITEM: u8 = 1;
    pub const DIR_ITEM: u8 = 84;
    pub const DIR_INDEX: u8 = 96;
    pub const EXTENT_DATA: u8 = 108;
    pub const ROOT_ITEM: u8 = 132;
}

/// Validate the fs-tree leaf fixture (bd-2jk.2 deliverable).
///
/// This fixture contains the minimum item types needed to support btrfs
/// read-only operations: INODE_ITEM, DIR_ITEM, DIR_INDEX, EXTENT_DATA.
#[test]
fn btrfs_fstree_leaf_fixture_conforms() {
    let (header, items) = validate_btrfs_leaf_fixture(&fixture_path("btrfs_fstree_leaf.json"))
        .expect("btrfs fs-tree leaf fixture");

    // Verify header
    assert_eq!(header.level, 0, "should be a leaf");
    assert_eq!(header.owner, 5, "owner should be FS_TREE (5)");
    assert!(items.len() >= 5, "should have at least 5 items");

    // Verify items are sorted
    for pair in items.windows(2) {
        let a = &pair[0].key;
        let b = &pair[1].key;
        assert!(
            (a.objectid, a.item_type, a.offset) <= (b.objectid, b.item_type, b.offset),
            "items should be sorted by key: {a:?} vs {b:?}"
        );
    }

    // Verify required item types are present
    let has_inode = items
        .iter()
        .any(|i| i.key.item_type == btrfs_item_types::INODE_ITEM);
    let has_dir_item = items
        .iter()
        .any(|i| i.key.item_type == btrfs_item_types::DIR_ITEM);
    let has_dir_index = items
        .iter()
        .any(|i| i.key.item_type == btrfs_item_types::DIR_INDEX);
    let has_extent_data = items
        .iter()
        .any(|i| i.key.item_type == btrfs_item_types::EXTENT_DATA);

    assert!(has_inode, "fixture should contain INODE_ITEM (type 1)");
    assert!(has_dir_item, "fixture should contain DIR_ITEM (type 84)");
    assert!(has_dir_index, "fixture should contain DIR_INDEX (type 96)");
    assert!(
        has_extent_data,
        "fixture should contain EXTENT_DATA (type 108)"
    );
}

/// Validate the root-tree leaf fixture (bd-2jk.2 deliverable).
///
/// This fixture contains ROOT_ITEM entries for the core btrfs trees,
/// needed to bootstrap tree traversal from the superblock.
#[test]
fn btrfs_roottree_leaf_fixture_conforms() {
    let (header, items) = validate_btrfs_leaf_fixture(&fixture_path("btrfs_roottree_leaf.json"))
        .expect("btrfs root-tree leaf fixture");

    // Verify header
    assert_eq!(header.level, 0, "should be a leaf");
    assert_eq!(header.owner, 1, "owner should be ROOT_TREE (1)");
    assert!(items.len() >= 3, "should have at least 3 ROOT_ITEM entries");

    // Verify items are sorted
    for pair in items.windows(2) {
        let a = &pair[0].key;
        let b = &pair[1].key;
        assert!(
            (a.objectid, a.item_type, a.offset) <= (b.objectid, b.item_type, b.offset),
            "items should be sorted by key: {a:?} vs {b:?}"
        );
    }

    // All items should be ROOT_ITEM (type 132)
    for item in &items {
        assert_eq!(
            item.key.item_type,
            btrfs_item_types::ROOT_ITEM,
            "root tree should only contain ROOT_ITEM entries"
        );
    }

    // Should have entries for standard trees: EXTENT_TREE (2), CHUNK_TREE (3), FS_TREE (5)
    let tree_ids: Vec<u64> = items.iter().map(|i| i.key.objectid).collect();
    assert!(
        tree_ids.contains(&2),
        "should have ROOT_ITEM for EXTENT_TREE (2)"
    );
    assert!(
        tree_ids.contains(&3),
        "should have ROOT_ITEM for CHUNK_TREE (3)"
    );
    assert!(
        tree_ids.contains(&5),
        "should have ROOT_ITEM for FS_TREE (5)"
    );
}

#[test]
fn parity_report_totals_are_consistent() {
    let report = ParityReport::current();
    let implemented_sum: u32 = report.domains.iter().map(|d| d.implemented).sum();
    let total_sum: u32 = report.domains.iter().map(|d| d.total).sum();

    assert_eq!(implemented_sum, report.overall_implemented);
    assert_eq!(total_sum, report.overall_total);
}

#[test]
fn ext4_reference_image_opens_with_journal_replay_segments() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let image_path = workspace.join("conformance/golden/ext4_8mb_reference.ext4");
    assert!(
        image_path.exists(),
        "expected golden ext4 image at {}",
        image_path.display()
    );

    let cx = Cx::for_testing();
    let fs = OpenFs::open_with_options(&cx, &image_path, &OpenOptions::default())
        .expect("open ext4 golden image with journal replay");
    let replay = fs
        .ext4_journal_replay()
        .expect("journal-enabled reference image should expose replay outcome");

    assert!(
        replay.stats.scanned_blocks > 0,
        "journal replay should scan at least one journal block"
    );
}

/// CI gate: verify that every fixture listed in checksums.sha256 exists,
/// is non-empty, and parses successfully. The actual SHA-256 comparison
/// is done by `scripts/verify_golden.sh` (which calls `sha256sum -c`).
#[test]
fn fixture_checksum_manifest_is_complete() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let checksums_path = workspace.join("conformance/fixtures/checksums.sha256");
    let checksums_text = std::fs::read_to_string(&checksums_path)
        .expect("read conformance/fixtures/checksums.sha256");

    let listed_files: Vec<&str> = checksums_text
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| l.split_once("  ").map(|(_, f)| f))
        .collect();

    assert!(
        !listed_files.is_empty(),
        "checksums.sha256 should list fixture files"
    );

    let fixtures_dir = workspace.join("conformance/fixtures");
    for filename in &listed_files {
        let path = fixtures_dir.join(filename);
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("fixture {filename} missing or unreadable: {e}"));
        assert!(!data.is_empty(), "fixture {filename} should be non-empty");
    }

    // Verify all .json fixture files are listed in the manifest
    let actual_jsons: Vec<_> = std::fs::read_dir(&fixtures_dir)
        .expect("read fixtures dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    for json_file in &actual_jsons {
        assert!(
            listed_files.contains(&json_file.as_str()),
            "fixture {json_file} exists but is not listed in checksums.sha256"
        );
    }
}

/// CI gate: verify that every golden file listed in checksums.sha256 exists,
/// is non-empty, and that every golden JSON is present in the manifest.
#[test]
fn golden_checksum_manifest_is_complete() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let checksums_path = workspace.join("conformance/golden/checksums.sha256");
    let checksums_text =
        std::fs::read_to_string(&checksums_path).expect("read conformance/golden/checksums.sha256");

    let listed_files: Vec<&str> = checksums_text
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|l| l.split_once("  ").map(|(_, f)| f))
        .collect();

    assert!(
        !listed_files.is_empty(),
        "checksums.sha256 should list golden files"
    );

    let golden_dir = workspace.join("conformance/golden");
    for filename in &listed_files {
        assert!(
            Path::new(filename)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json")),
            "golden checksum manifest should only track .json files: {filename}"
        );
        let path = golden_dir.join(filename);
        let data = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("golden {filename} missing or unreadable: {e}"));
        assert!(!data.is_empty(), "golden {filename} should be non-empty");
    }

    let mut actual_jsons: Vec<_> = std::fs::read_dir(&golden_dir)
        .expect("read golden dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    actual_jsons.sort();

    for json_file in &actual_jsons {
        assert!(
            listed_files.contains(&json_file.as_str()),
            "golden {json_file} exists but is not listed in checksums.sha256"
        );
    }
}

/// Full conformance gate pass (bd-2jk.14).
///
/// This is the single CI gate test that exercises every conformance
/// surface in one deterministic pass: all fixture parsers, checksum
/// manifests, golden references, fuzz corpus, and parity report.
/// Must complete in < 60 seconds.
#[test]
fn full_conformance_gate_pass() {
    let start = std::time::Instant::now();

    // 1) All fixture parsers + spot checks (12 total fixture JSONs).
    ext4_and_btrfs_fixtures_conform();
    ext4_group_desc_fixtures_conform();
    ext4_inode_fixtures_conform();
    ext4_dir_block_fixture_conforms();
    btrfs_chunk_mapping_fixture_conforms();
    btrfs_leaf_fixture_conforms();
    btrfs_fstree_leaf_fixture_conforms();
    btrfs_roottree_leaf_fixture_conforms();

    // 2) Checksum manifests are bidirectionally complete.
    fixture_checksum_manifest_is_complete();
    golden_checksum_manifest_is_complete();

    // 3) Goldens deserialize successfully and satisfy basic invariants.
    let workspace = workspace_root();
    validate_golden_jsons(workspace);

    // 4) Fuzz corpus is populated.
    assert_fuzz_corpus_populated(workspace);

    // 5) Parity report at 100%, internally consistent, and deterministic.
    assert_parity_report_100_percent();
    parity_report_totals_are_consistent();

    // 6) Time bound.
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 60,
        "conformance gate should complete in < 60s, took {elapsed:?}"
    );
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
}

fn validate_golden_jsons(workspace: &Path) {
    let golden_dir = workspace.join("conformance/golden");

    let ext4_golden_names = [
        "ext4_64mb_sparse_super.json",
        "ext4_htree_dirindex.json",
        "ext4_64mb_reference.json",
        "ext4_dir_index_reference.json",
        "ext4_8mb_reference.json",
    ];
    for name in &ext4_golden_names {
        let text = std::fs::read_to_string(golden_dir.join(name))
            .unwrap_or_else(|e| panic!("golden {name} unreadable: {e}"));
        let golden: GoldenReference =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("golden {name} invalid: {e}"));
        assert!(golden.version >= 1, "golden {name} version should be >= 1");
        assert!(
            !golden.source.is_empty(),
            "golden {name} source should be non-empty"
        );
    }

    let btrfs_golden_names = ["btrfs_small.json", "btrfs_medium.json", "btrfs_large.json"];
    for name in &btrfs_golden_names {
        let text = std::fs::read_to_string(golden_dir.join(name))
            .unwrap_or_else(|e| panic!("golden {name} unreadable: {e}"));
        let golden: Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("golden {name} invalid: {e}"));
        assert_eq!(
            golden.get("filesystem").and_then(Value::as_str),
            Some("btrfs"),
            "golden {name} filesystem should be btrfs"
        );
        for numeric in ["sectorsize", "nodesize", "generation"] {
            assert!(
                golden.get(numeric).and_then(Value::as_u64).is_some(),
                "golden {name} missing numeric field {numeric}"
            );
        }
        assert!(
            golden
                .get("label")
                .and_then(Value::as_str)
                .is_some_and(|label| !label.is_empty()),
            "golden {name} label should be non-empty"
        );
    }
}

fn assert_fuzz_corpus_populated(workspace: &Path) {
    let corpus_dir = workspace.join("tests/fuzz_corpus");
    let corpus_count = std::fs::read_dir(&corpus_dir)
        .expect("read fuzz_corpus dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
        .count();
    assert!(
        corpus_count >= 50,
        "expected >= 50 fuzz corpus samples, found {corpus_count}"
    );
}

fn assert_parity_report_100_percent() {
    let report = ParityReport::current();
    assert_eq!(report.overall_implemented, report.overall_total);
    assert!(
        (report.overall_coverage_percent - 100.0).abs() < f64::EPSILON,
        "overall coverage should be 100%, got {}%",
        report.overall_coverage_percent
    );
    for domain in &report.domains {
        assert_eq!(
            domain.implemented, domain.total,
            "domain '{}' not at 100%: {}/{}",
            domain.domain, domain.implemented, domain.total
        );
    }

    // Deterministic: second run yields identical report.
    let report2 = ParityReport::current();
    assert_eq!(report.overall_implemented, report2.overall_implemented);
    assert_eq!(report.overall_total, report2.overall_total);
    assert_eq!(report.domains.len(), report2.domains.len());
}

#[test]
#[ignore = "stress integration; run explicitly with --ignored"]
fn crash_replay_suite_short_integration() {
    let config = CrashReplaySuiteConfig {
        schedule_count: 20,
        min_operations: 100,
        max_operations: 1000,
        base_seed: 0xFF5E_ED00_0000_0001,
        output_dir: None,
    };
    let report = run_crash_replay_suite(&config).expect("run crash replay suite");
    assert_eq!(report.schedule_count, config.schedule_count);
    assert_eq!(report.failed_schedules, 0);
    assert_eq!(report.passed_schedules, config.schedule_count);
}

#[test]
#[ignore = "stress integration; run explicitly with --ignored"]
fn fsx_stress_short_integration() {
    let config = FsxStressConfig {
        operation_count: 500,
        seed: 0xF5A5_7E55_0000_0001,
        max_file_size_bytes: 8 * 1024 * 1024,
        corruption_every_ops: 100,
        full_verify_every_ops: 100,
        output_dir: None,
    };
    let report = run_fsx_stress(&config).expect("run fsx stress");
    assert!(report.passed, "failure: {:#?}", report.failure);
    assert_eq!(report.operations_executed, config.operation_count);
    assert!(report.failure.is_none());
}
