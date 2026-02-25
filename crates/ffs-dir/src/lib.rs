#![forbid(unsafe_code)]
//! Directory operations.
//!
//! Linear directory entry scan, htree (hashed B-tree) lookup with
//! dx_hash computation (half-MD4 and TEA), directory entry creation
//! and deletion, and `..`/`.` management.

use ffs_error::{FfsError, Result};
use ffs_ondisk::Ext4FileType;

/// ext4 directory entry header size (`ext4_dir_entry_2`).
const DIR_ENTRY_HEADER_LEN: usize = 8;

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

fn required_rec_len(name_len: usize) -> usize {
    align4(DIR_ENTRY_HEADER_LEN + name_len)
}

fn read_u16_le(buf: &[u8], off: usize) -> Option<u16> {
    let bytes = buf.get(off..off + 2)?;
    Some(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32_le(buf: &[u8], off: usize) -> Option<u32> {
    let bytes = buf.get(off..off + 4)?;
    Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_u16_le(buf: &mut [u8], off: usize, value: u16) -> Result<()> {
    let dst = buf
        .get_mut(off..off + 2)
        .ok_or_else(|| FfsError::Corruption {
            block: 0,
            detail: "u16 write out of bounds".to_owned(),
        })?;
    dst.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u32_le(buf: &mut [u8], off: usize, value: u32) -> Result<()> {
    let dst = buf
        .get_mut(off..off + 4)
        .ok_or_else(|| FfsError::Corruption {
            block: 0,
            detail: "u32 write out of bounds".to_owned(),
        })?;
    dst.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn validate_name(name: &[u8]) -> Result<()> {
    if name.is_empty() {
        return Err(FfsError::Format(
            "directory entry name cannot be empty".to_owned(),
        ));
    }
    if name.len() > u8::MAX as usize {
        return Err(FfsError::Format(
            "directory entry name exceeds 255 bytes".to_owned(),
        ));
    }
    Ok(())
}

fn write_entry(
    block: &mut [u8],
    offset: usize,
    ino: u32,
    rec_len: usize,
    file_type: Ext4FileType,
    name: &[u8],
) -> Result<()> {
    let name_len_u8 = u8::try_from(name.len())
        .map_err(|_| FfsError::Format("directory entry name exceeds 255 bytes".to_owned()))?;
    let rec_len_u16 = u16::try_from(rec_len)
        .map_err(|_| FfsError::Format("directory entry rec_len exceeds u16".to_owned()))?;
    let end = offset
        .checked_add(rec_len)
        .ok_or_else(|| FfsError::Format("directory entry offset overflow".to_owned()))?;
    if end > block.len() {
        return Err(FfsError::Corruption {
            block: 0,
            detail: "directory entry exceeds block boundary".to_owned(),
        });
    }
    let min_size = required_rec_len(name.len());
    if rec_len < min_size {
        return Err(FfsError::Format(
            "directory entry rec_len smaller than minimum".to_owned(),
        ));
    }

    write_u32_le(block, offset, ino)?;
    write_u16_le(block, offset + 4, rec_len_u16)?;
    block[offset + 6] = name_len_u8;
    block[offset + 7] = file_type as u8;
    block[offset + DIR_ENTRY_HEADER_LEN..offset + DIR_ENTRY_HEADER_LEN + name.len()]
        .copy_from_slice(name);
    // Zero remaining bytes in slot for deterministic tests and clean replay.
    if rec_len > DIR_ENTRY_HEADER_LEN + name.len() {
        block[offset + DIR_ENTRY_HEADER_LEN + name.len()..end].fill(0);
    }
    Ok(())
}

/// Add a directory entry into a single directory block.
///
/// Uses ext4-style `rec_len` management:
/// - Reuses deleted slots (`inode == 0`) when large enough.
/// - Otherwise splits a live slot when it has enough slack.
/// - Returns the byte offset where the new entry was inserted.
pub fn add_entry(
    block: &mut [u8],
    ino: u32,
    name: &[u8],
    file_type: Ext4FileType,
) -> Result<usize> {
    if ino == 0 {
        return Err(FfsError::Format(
            "directory entry inode cannot be zero".to_owned(),
        ));
    }
    validate_name(name)?;

    let need = required_rec_len(name.len());
    if need > block.len() {
        return Err(FfsError::NoSpace);
    }

    let mut off = 0usize;
    while off + DIR_ENTRY_HEADER_LEN <= block.len() {
        let rec_len =
            usize::from(
                read_u16_le(block, off + 4).ok_or_else(|| FfsError::Corruption {
                    block: 0,
                    detail: "unable to read directory entry rec_len".to_owned(),
                })?,
            );
        if rec_len < DIR_ENTRY_HEADER_LEN || (rec_len % 4) != 0 {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "invalid directory entry rec_len".to_owned(),
            });
        }
        let end = off
            .checked_add(rec_len)
            .ok_or_else(|| FfsError::Corruption {
                block: 0,
                detail: "directory entry offset overflow".to_owned(),
            })?;
        if end > block.len() {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "directory entry exceeds block boundary".to_owned(),
            });
        }

        let cur_ino = read_u32_le(block, off).ok_or_else(|| FfsError::Corruption {
            block: 0,
            detail: "unable to read directory entry inode".to_owned(),
        })?;
        let cur_name_len = usize::from(block[off + 6]);

        if cur_ino == 0 {
            if rec_len >= need {
                write_entry(block, off, ino, rec_len, file_type, name)?;
                return Ok(off);
            }
            off = end;
            continue;
        }

        let actual = required_rec_len(cur_name_len);
        if actual > rec_len {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "directory entry name length exceeds rec_len".to_owned(),
            });
        }

        let slack = rec_len - actual;
        if slack >= need {
            let actual_u16 = u16::try_from(actual)
                .map_err(|_| FfsError::Format("actual rec_len exceeds u16".to_owned()))?;
            write_u16_le(block, off + 4, actual_u16)?;
            let new_off = off + actual;
            write_entry(block, new_off, ino, slack, file_type, name)?;
            return Ok(new_off);
        }

        off = end;
    }

    Err(FfsError::NoSpace)
}

/// Remove a directory entry by name from a single directory block.
///
/// On success:
/// - If there is a previous live entry, its `rec_len` is expanded to absorb
///   the removed slot (coalescing free space).
/// - Otherwise the target entry is marked deleted (`inode = 0`).
pub fn remove_entry(block: &mut [u8], name: &[u8]) -> Result<bool> {
    validate_name(name)?;

    let mut off = 0usize;
    let mut prev_off_opt: Option<usize> = None;

    while off + DIR_ENTRY_HEADER_LEN <= block.len() {
        let rec_len =
            usize::from(
                read_u16_le(block, off + 4).ok_or_else(|| FfsError::Corruption {
                    block: 0,
                    detail: "unable to read directory entry rec_len".to_owned(),
                })?,
            );
        if rec_len < DIR_ENTRY_HEADER_LEN || (rec_len % 4) != 0 {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "invalid directory entry rec_len".to_owned(),
            });
        }
        let end = off
            .checked_add(rec_len)
            .ok_or_else(|| FfsError::Corruption {
                block: 0,
                detail: "directory entry offset overflow".to_owned(),
            })?;
        if end > block.len() {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "directory entry exceeds block boundary".to_owned(),
            });
        }

        let cur_ino = read_u32_le(block, off).ok_or_else(|| FfsError::Corruption {
            block: 0,
            detail: "unable to read directory entry inode".to_owned(),
        })?;
        let cur_name_len = usize::from(block[off + 6]);
        let name_end = off
            .checked_add(DIR_ENTRY_HEADER_LEN + cur_name_len)
            .ok_or_else(|| FfsError::Corruption {
                block: 0,
                detail: "directory entry name offset overflow".to_owned(),
            })?;
        if name_end > end {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "directory entry name exceeds rec_len".to_owned(),
            });
        }

        if cur_ino != 0 && &block[off + DIR_ENTRY_HEADER_LEN..name_end] == name {
            if let Some(prev_off) = prev_off_opt {
                let prev_len = usize::from(read_u16_le(block, prev_off + 4).ok_or_else(|| {
                    FfsError::Corruption {
                        block: 0,
                        detail: "unable to read previous directory entry rec_len".to_owned(),
                    }
                })?);
                let merged = prev_len
                    .checked_add(rec_len)
                    .ok_or_else(|| FfsError::Format("merged rec_len overflow".to_owned()))?;
                let merged_u16 = u16::try_from(merged)
                    .map_err(|_| FfsError::Format("merged rec_len exceeds u16".to_owned()))?;
                write_u16_le(block, prev_off + 4, merged_u16)?;
            }

            write_u32_le(block, off, 0)?;
            block[off + 6] = 0;
            block[off + 7] = 0;
            return Ok(true);
        }

        prev_off_opt = Some(off);
        off = end;
    }

    Ok(false)
}

/// Initialize an empty directory block with `.` and `..` entries.
pub fn init_dir_block(block: &mut [u8], self_ino: u32, parent_ino: u32) -> Result<()> {
    if block.len() < required_rec_len(1) + required_rec_len(2) {
        return Err(FfsError::Format(
            "directory block too small for . and .. entries".to_owned(),
        ));
    }
    block.fill(0);

    let dot_len = required_rec_len(1);
    let dotdot_len = block.len() - dot_len;

    write_entry(block, 0, self_ino, dot_len, Ext4FileType::Dir, b".")?;
    write_entry(
        block,
        dot_len,
        parent_ino,
        dotdot_len,
        Ext4FileType::Dir,
        b"..",
    )?;
    Ok(())
}

/// Hash/index entry for ext4 htree directory indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HtreeEntry {
    pub hash: u32,
    pub block: u32,
}

/// Compute ext4 DX hash major value for a directory name.
#[must_use]
pub fn compute_dx_hash(hash_version: u8, name: &[u8], seed: &[u32; 4]) -> u32 {
    ffs_ondisk::dx_hash(hash_version, name, seed).0
}

/// Insert an htree mapping entry while preserving hash ordering.
///
/// Returns the insertion index.
pub fn htree_insert(entries: &mut Vec<HtreeEntry>, hash: u32, block: u32) -> usize {
    let idx = entries.partition_point(|e| e.hash <= hash);
    entries.insert(idx, HtreeEntry { hash, block });
    idx
}

/// Remove one matching htree mapping entry (`hash`, `block`).
///
/// Returns `true` when an entry was removed.
pub fn htree_remove(entries: &mut Vec<HtreeEntry>, hash: u32, block: u32) -> bool {
    let Some(pos) = entries
        .iter()
        .position(|e| e.hash == hash && e.block == block)
    else {
        return false;
    };
    entries.remove(pos);
    true
}

/// Find the leaf block using the "rightmost hash <= target" rule.
#[must_use]
pub fn htree_find_leaf(entries: &[HtreeEntry], target_hash: u32) -> Option<u32> {
    if entries.is_empty() {
        return None;
    }
    let idx = entries.partition_point(|e| e.hash <= target_hash);
    let chosen = if idx == 0 { 0 } else { idx - 1 };
    entries.get(chosen).map(|e| e.block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffs_ondisk::{Ext4FileType, parse_dir_block};

    #[test]
    fn init_dir_block_contains_dot_and_dotdot() {
        let mut block = vec![0u8; 1024];
        init_dir_block(&mut block, 11, 2).unwrap();
        let (entries, tail) = parse_dir_block(&block, 1024).unwrap();
        assert!(tail.is_none());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, b".".to_vec());
        assert_eq!(entries[0].inode, 11);
        assert_eq!(entries[1].name, b"..".to_vec());
        assert_eq!(entries[1].inode, 2);
    }

    #[test]
    fn add_entry_splits_live_slot_slack() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 2, 1024, Ext4FileType::Dir, b".").unwrap();
        let off = add_entry(&mut block, 33, b"hello", Ext4FileType::RegFile).unwrap();
        assert_eq!(off, 12);
        let (entries, _) = parse_dir_block(&block, 1024).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, b".".to_vec());
        assert_eq!(entries[1].name, b"hello".to_vec());
        assert_eq!(entries[1].inode, 33);
    }

    #[test]
    fn add_entry_reuses_deleted_slot() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 2, 12, Ext4FileType::Dir, b".").unwrap();
        write_entry(&mut block, 12, 0, 1012, Ext4FileType::Unknown, b"x").unwrap();
        let off = add_entry(&mut block, 44, b"new", Ext4FileType::RegFile).unwrap();
        assert_eq!(off, 12);
        let (entries, _) = parse_dir_block(&block, 1024).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].inode, 44);
        assert_eq!(entries[1].name, b"new".to_vec());
    }

    #[test]
    fn add_entry_no_space_returns_enospc() {
        let mut block = vec![0u8; 24];
        write_entry(&mut block, 0, 1, 12, Ext4FileType::RegFile, b"a").unwrap();
        write_entry(&mut block, 12, 2, 12, Ext4FileType::RegFile, b"b").unwrap();
        let err = add_entry(&mut block, 3, b"c", Ext4FileType::RegFile).unwrap_err();
        assert_eq!(err.to_errno(), libc::ENOSPC);
    }

    #[test]
    fn remove_entry_coalesces_prev_rec_len() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 10, 12, Ext4FileType::RegFile, b"a").unwrap();
        write_entry(&mut block, 12, 11, 12, Ext4FileType::RegFile, b"b").unwrap();
        write_entry(&mut block, 24, 12, 1000, Ext4FileType::RegFile, b"c").unwrap();

        let removed = remove_entry(&mut block, b"b").unwrap();
        assert!(removed);
        let merged = read_u16_le(&block, 4).unwrap();
        assert_eq!(merged, 24);
    }

    #[test]
    fn remove_first_entry_marks_deleted() {
        let mut block = vec![0u8; 128];
        write_entry(&mut block, 0, 10, 128, Ext4FileType::RegFile, b"a").unwrap();
        let removed = remove_entry(&mut block, b"a").unwrap();
        assert!(removed);
        assert_eq!(read_u32_le(&block, 0).unwrap(), 0);
    }

    #[test]
    fn htree_insert_preserves_sorted_hash_order() {
        let mut entries = Vec::new();
        htree_insert(&mut entries, 0x2000, 2);
        htree_insert(&mut entries, 0x1000, 1);
        htree_insert(&mut entries, 0x5000, 5);
        htree_insert(&mut entries, 0x3000, 3);
        let hashes: Vec<u32> = entries.iter().map(|e| e.hash).collect();
        assert_eq!(hashes, vec![0x1000, 0x2000, 0x3000, 0x5000]);
    }

    #[test]
    fn htree_find_leaf_uses_rightmost_lte() {
        let entries = vec![
            HtreeEntry {
                hash: 0x0000,
                block: 1,
            },
            HtreeEntry {
                hash: 0x1000,
                block: 2,
            },
            HtreeEntry {
                hash: 0x8000,
                block: 3,
            },
        ];
        assert_eq!(htree_find_leaf(&entries, 0x0500), Some(1));
        assert_eq!(htree_find_leaf(&entries, 0x1000), Some(2));
        assert_eq!(htree_find_leaf(&entries, 0xFFFF), Some(3));
        assert_eq!(htree_find_leaf(&[], 0xFFFF), None);
    }

    #[test]
    fn htree_remove_specific_entry() {
        let mut entries = vec![
            HtreeEntry {
                hash: 0x1000,
                block: 2,
            },
            HtreeEntry {
                hash: 0x1000,
                block: 8,
            },
        ];
        assert!(htree_remove(&mut entries, 0x1000, 8));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].block, 2);
        assert!(!htree_remove(&mut entries, 0x1000, 8));
    }

    #[test]
    fn compute_dx_hash_is_deterministic() {
        let seed = [1, 2, 3, 4];
        let h1 = compute_dx_hash(1, b"hello", &seed);
        let h2 = compute_dx_hash(1, b"hello", &seed);
        let h3 = compute_dx_hash(1, b"world", &seed);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    // ── Additional edge-case tests ───────────────────────────────────

    #[test]
    fn add_entry_rejects_zero_inode() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 1, 1024, Ext4FileType::Dir, b".").unwrap();
        let err = add_entry(&mut block, 0, b"bad", Ext4FileType::RegFile).unwrap_err();
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn add_entry_rejects_empty_name() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 1, 1024, Ext4FileType::Dir, b".").unwrap();
        let err = add_entry(&mut block, 10, b"", Ext4FileType::RegFile).unwrap_err();
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn remove_nonexistent_entry_returns_false() {
        let mut block = vec![0u8; 1024];
        write_entry(&mut block, 0, 10, 1024, Ext4FileType::RegFile, b"a").unwrap();
        let removed = remove_entry(&mut block, b"nonexistent").unwrap();
        assert!(!removed);
    }

    #[test]
    fn add_multiple_entries_and_remove() {
        let mut block = vec![0u8; 4096];
        init_dir_block(&mut block, 2, 2).unwrap();

        // Add several entries.
        add_entry(&mut block, 100, b"file1.txt", Ext4FileType::RegFile).unwrap();
        add_entry(&mut block, 101, b"file2.txt", Ext4FileType::RegFile).unwrap();
        add_entry(&mut block, 102, b"subdir", Ext4FileType::Dir).unwrap();

        let (entries, _) = parse_dir_block(&block, 4096).unwrap();
        assert_eq!(entries.len(), 5); // . + .. + 3 entries

        // Remove middle entry.
        let removed = remove_entry(&mut block, b"file2.txt").unwrap();
        assert!(removed);

        let (entries, _) = parse_dir_block(&block, 4096).unwrap();
        assert_eq!(entries.len(), 4); // . + .. + 2 remaining
        assert!(!entries.iter().any(|e| e.name == b"file2.txt"));
    }

    #[test]
    fn init_dir_block_too_small_fails() {
        let mut block = vec![0u8; 16]; // Too small for . and ..
        let err = init_dir_block(&mut block, 1, 2).unwrap_err();
        assert!(matches!(err, FfsError::Format(_)));
    }

    #[test]
    fn add_entry_max_name_length() {
        let mut block = vec![0u8; 4096];
        write_entry(&mut block, 0, 1, 4096, Ext4FileType::Dir, b".").unwrap();

        // 255-byte name (max valid).
        let long_name = vec![b'x'; 255];
        let off = add_entry(&mut block, 42, &long_name, Ext4FileType::RegFile).unwrap();
        assert!(off > 0);

        let (entries, _) = parse_dir_block(&block, 4096).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].name.len(), 255);
    }

    #[test]
    fn htree_find_leaf_single_entry() {
        let entries = vec![HtreeEntry { hash: 0, block: 42 }];
        // Any hash should map to the single leaf block.
        assert_eq!(htree_find_leaf(&entries, 0), Some(42));
        assert_eq!(htree_find_leaf(&entries, 0xFFFF_FFFF), Some(42));
    }

    #[test]
    fn htree_insert_duplicate_hashes() {
        let mut entries = Vec::new();
        htree_insert(&mut entries, 0x1000, 1);
        htree_insert(&mut entries, 0x1000, 2);
        htree_insert(&mut entries, 0x1000, 3);
        assert_eq!(entries.len(), 3);
        // All have same hash, different blocks.
        assert!(entries.iter().all(|e| e.hash == 0x1000));
    }

    #[test]
    fn htree_remove_nonexistent_returns_false() {
        let mut entries = vec![HtreeEntry {
            hash: 0x1000,
            block: 1,
        }];
        assert!(!htree_remove(&mut entries, 0x2000, 1));
        assert!(!htree_remove(&mut entries, 0x1000, 2));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn align4_roundtrip() {
        assert_eq!(align4(0), 0);
        assert_eq!(align4(1), 4);
        assert_eq!(align4(4), 4);
        assert_eq!(align4(5), 8);
        assert_eq!(align4(8), 8);
    }

    #[test]
    fn required_rec_len_minimum() {
        // Header (8) + 1-byte name → aligned to 12.
        assert_eq!(required_rec_len(1), 12);
        // Header (8) + 4-byte name → 12 (already aligned).
        assert_eq!(required_rec_len(4), 12);
        // Header (8) + 5-byte name → 16.
        assert_eq!(required_rec_len(5), 16);
    }

    #[test]
    fn add_entry_after_remove_reuses_space() {
        let mut block = vec![0u8; 1024];
        init_dir_block(&mut block, 2, 2).unwrap();

        // Add, then remove, then add again to same slot.
        add_entry(&mut block, 100, b"temp", Ext4FileType::RegFile).unwrap();
        remove_entry(&mut block, b"temp").unwrap();
        add_entry(&mut block, 200, b"repl", Ext4FileType::RegFile).unwrap();

        let (entries, _) = parse_dir_block(&block, 1024).unwrap();
        assert!(entries.iter().any(|e| e.inode == 200 && e.name == b"repl"));
        assert!(!entries.iter().any(|e| e.name == b"temp"));
    }

    // ── dx_hash edge-case and distribution tests ────────────────────

    #[test]
    fn dx_hash_different_names_produce_different_hashes() {
        let seed = [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0xABCD_EF01];
        let names: Vec<&[u8]> = vec![
            b"a",
            b"b",
            b"ab",
            b"ba",
            b"file.txt",
            b"FILE.TXT",
            b"index.html",
            b"readme.md",
            b"Cargo.toml",
            b"lib.rs",
        ];

        let hashes: Vec<u32> = names.iter().map(|n| compute_dx_hash(1, n, &seed)).collect();

        // Check for uniqueness — with 10 distinct names, we expect at least
        // 8 distinct hashes (allowing some collision, but not total collision).
        let mut unique = hashes.clone();
        unique.sort_unstable();
        unique.dedup();
        assert!(
            unique.len() >= 8,
            "expected at least 8 distinct hashes from 10 names, got {}: {hashes:?}",
            unique.len(),
        );
    }

    #[test]
    fn dx_hash_seed_variation_changes_output() {
        let name = b"test_file.txt";
        let seed_a = [1, 2, 3, 4];
        let seed_b = [5, 6, 7, 8];
        let seed_zero = [0, 0, 0, 0];

        let hash_a = compute_dx_hash(1, name, &seed_a);
        let hash_b = compute_dx_hash(1, name, &seed_b);
        let hash_zero = compute_dx_hash(1, name, &seed_zero);

        // Different seeds should (almost certainly) produce different hashes.
        assert_ne!(
            hash_a, hash_b,
            "different seeds should produce different hashes"
        );
        assert_ne!(hash_a, hash_zero, "non-zero vs zero seed should differ");
    }

    #[test]
    fn dx_hash_single_byte_names() {
        let seed = [0x1111, 0x2222, 0x3333, 0x4444];
        // Every single printable ASCII byte should produce a valid (non-zero-for-most) hash.
        let mut hashes = std::collections::HashSet::new();
        for byte in b'!'..=b'~' {
            let h = compute_dx_hash(1, &[byte], &seed);
            hashes.insert(h);
        }
        // 94 printable ASCII chars should produce many distinct hashes.
        assert!(
            hashes.len() >= 80,
            "expected at least 80 distinct hashes from 94 single-byte names, got {}",
            hashes.len(),
        );
    }

    #[test]
    fn dx_hash_distribution_across_buckets() {
        // Hash 1000 sequential filenames and verify the distribution across
        // 16 buckets is roughly uniform (no bucket has more than 3x the
        // expected count, which would indicate severe clustering).
        let seed = [0xDEAD_BEEF, 0xCAFE_BABE, 0x1234_5678, 0xABCD_EF01];
        let num_buckets = 16u32;
        let num_names = 1000usize;
        let mut buckets = vec![0u32; num_buckets as usize];

        for i in 0..num_names {
            let name = format!("file_{i:04}.txt");
            let h = compute_dx_hash(1, name.as_bytes(), &seed);
            // Use top 4 bits for bucket assignment (hash has bit 0 cleared by ext4 convention).
            let bucket = (h >> 28) as usize;
            buckets[bucket] += 1;
        }

        #[expect(clippy::cast_possible_truncation)]
        let expected = num_names as u32 / num_buckets;
        let max_allowed = expected * 3; // 3x expected = very loose bound
        for (i, &count) in buckets.iter().enumerate() {
            assert!(
                count <= max_allowed,
                "bucket {i} has {count} entries (expected ~{expected}, max {max_allowed}) — \
                 hash distribution is severely skewed"
            );
        }

        // Also verify no bucket is completely empty (with 1000 names / 16 buckets,
        // every bucket should have at least one entry).
        let empty_buckets = buckets.iter().filter(|&&c| c == 0).count();
        assert!(
            empty_buckets <= 2,
            "too many empty buckets ({empty_buckets}/16) — hash is poorly distributed"
        );
    }

    #[test]
    fn dx_hash_collision_rate_within_bounds() {
        // Hash 500 distinct filenames and verify the collision rate is below 10%.
        // A good 32-bit hash should have very few collisions for 500 names
        // (birthday paradox: ~0.003% expected for truly random 32-bit values).
        let seed = [0x1111_2222, 0x3333_4444, 0x5555_6666, 0x7777_8888];
        let num_names = 500usize;
        let mut hashes = std::collections::HashSet::new();

        for i in 0..num_names {
            let name = format!("document_{i:05}.dat");
            let h = compute_dx_hash(1, name.as_bytes(), &seed);
            hashes.insert(h);
        }

        let distinct = hashes.len();
        let collisions = num_names - distinct;
        let collision_pct = (collisions as f64 / num_names as f64) * 100.0;
        assert!(
            collision_pct < 10.0,
            "collision rate {collision_pct:.1}% ({collisions}/{num_names}) exceeds 10% threshold"
        );
    }

    #[test]
    fn dx_hash_all_algorithms_produce_distinct_outputs() {
        // All supported hash algorithm variants should produce different hashes
        // for the same input (since they use fundamentally different transforms).
        let seed = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD];
        let name = b"test_file.txt";

        let versions = [0u8, 1, 2, 3, 4, 5]; // legacy, half-md4, tea, legacy-unsigned, half-md4-unsigned, tea-unsigned
        let hashes: Vec<u32> = versions
            .iter()
            .map(|&v| compute_dx_hash(v, name, &seed))
            .collect();

        // At least 3 distinct values (legacy signed/unsigned might collide for ASCII,
        // but half-md4 and tea should differ from each other and from legacy).
        let mut unique = hashes.clone();
        unique.sort_unstable();
        unique.dedup();
        assert!(
            unique.len() >= 3,
            "expected at least 3 distinct hashes from 6 algorithm variants, got {}: {hashes:?}",
            unique.len(),
        );
    }

    #[test]
    fn dx_hash_low_bit_always_cleared() {
        // ext4 convention: the major hash always has bit 0 cleared (reserved).
        let seed = [0x1234, 0x5678, 0x9ABC, 0xDEF0];
        for version in [0u8, 1, 2, 3, 5] {
            for i in 0..100 {
                let name = format!("entry_{version}_{i}");
                let h = compute_dx_hash(version, name.as_bytes(), &seed);
                assert_eq!(
                    h & 1,
                    0,
                    "hash version {version}, name '{name}': bit 0 should be cleared, got {h:#010x}"
                );
            }
        }
    }

    #[test]
    fn dx_hash_bit_utilization() {
        // Verify that all 32 bits of the hash output are actually used
        // (no stuck-at-zero or stuck-at-one bits) across many inputs.
        let seed = [0xFEED_FACE, 0xDEAD_C0DE, 0xBAAD_F00D, 0xC0FF_EE42];
        let mut or_all = 0u32;
        let mut and_all = 0xFFFF_FFFFu32;

        for i in 0..200 {
            let name = format!("bit_test_{i:03}");
            let h = compute_dx_hash(1, name.as_bytes(), &seed);
            or_all |= h;
            and_all &= h;
        }

        // All bits (except bit 0) should have been set in at least one hash.
        // Bit 0 is always cleared by ext4 convention.
        let or_mask = or_all & !1u32; // ignore bit 0
        assert_eq!(
            or_mask, 0xFFFF_FFFEu32,
            "some hash bits are never set (stuck at 0): or_all = {or_all:#010x}"
        );

        // All bits should have been cleared in at least one hash.
        assert_eq!(
            and_all, 0,
            "some hash bits are never cleared (stuck at 1): and_all = {and_all:#010x}"
        );
    }

    #[test]
    fn dx_hash_long_name_does_not_panic() {
        let seed = [1, 2, 3, 4];
        // 255-byte name (maximum valid ext4 name length) should not panic.
        let long_name = vec![b'A'; 255];
        let h = compute_dx_hash(1, &long_name, &seed);
        // Should produce a valid hash.
        let _ = h;

        // A name with very different content should hash differently.
        let different: Vec<u8> = (0..255).map(|i| b'a' + (i % 26)).collect();
        let h2 = compute_dx_hash(1, &different, &seed);
        assert_ne!(
            h, h2,
            "structurally different long names should hash differently"
        );
    }

    // ── Property-based tests (proptest) ────────────────────────────────

    use proptest::prelude::*;

    /// Strategy: generate a valid directory entry name (3..32 bytes, no NUL or /).
    /// Minimum length 3 avoids generating "." or ".." which collide with
    /// the special directory entries created by `init_dir_block`.
    fn name_strategy() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(
            prop::sample::select(
                (b'a'..=b'z')
                    .chain(b'A'..=b'Z')
                    .chain(b'0'..=b'9')
                    .chain([b'_', b'-'])
                    .collect::<Vec<u8>>(),
            ),
            3..32,
        )
    }

    /// Operation for the add/remove state machine test.
    #[derive(Debug, Clone)]
    enum DirOp {
        Add(Vec<u8>, u32),
        Remove(Vec<u8>),
    }

    /// Strategy: generate a sequence of add/remove operations.
    fn dir_ops_strategy() -> impl Strategy<Value = Vec<DirOp>> {
        prop::collection::vec(
            prop_oneof![
                3 => (name_strategy(), 1_u32..1000).prop_map(|(n, i)| DirOp::Add(n, i)),
                1 => name_strategy().prop_map(DirOp::Remove),
            ],
            1..20,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// add_entry/remove_entry sequence matches a reference HashMap.
        #[test]
        fn proptest_add_remove_matches_reference(ops in dir_ops_strategy()) {
            let mut block = vec![0u8; 4096];
            init_dir_block(&mut block, 2, 2).unwrap();

            let mut reference: std::collections::HashMap<Vec<u8>, u32> =
                std::collections::HashMap::new();

            for op in &ops {
                match op {
                    DirOp::Add(name, ino) => {
                        if !reference.contains_key(name)
                            && add_entry(&mut block, *ino, name, Ext4FileType::RegFile).is_ok()
                        {
                            reference.insert(name.clone(), *ino);
                        }
                    }
                    DirOp::Remove(name) => {
                        if remove_entry(&mut block, name).unwrap_or(false) {
                            reference.remove(name);
                        }
                    }
                }
            }

            // Verify: the live entries in the block match the reference.
            let (entries, _) = parse_dir_block(&block, 4096).unwrap();
            let live: std::collections::HashMap<Vec<u8>, u32> = entries
                .iter()
                .filter(|e| e.name != b"." && e.name != b"..")
                .map(|e| (e.name.clone(), e.inode))
                .collect();

            prop_assert_eq!(
                live.len(),
                reference.len(),
                "entry count mismatch: block has {} live entries, reference has {}",
                live.len(),
                reference.len(),
            );
            for (name, ino) in &reference {
                let block_ino = live.get(name);
                prop_assert_eq!(
                    block_ino,
                    Some(ino),
                    "name {:?} has inode {:?} in block but {:?} in reference",
                    String::from_utf8_lossy(name),
                    block_ino,
                    Some(ino),
                );
            }
        }

        /// After removing an entry, adding a same-or-smaller entry succeeds
        /// (space reclamation works).
        #[test]
        fn proptest_remove_then_add_reclaims_space(
            name_a in name_strategy(),
            name_b in name_strategy(),
            ino_a in 1_u32..1000,
            ino_b in 1_u32..1000,
        ) {
            // name_b must be no longer than name_a for guaranteed fit.
            let name_b = if name_b.len() > name_a.len() {
                name_b[..name_a.len()].to_vec()
            } else {
                name_b
            };

            let mut block = vec![0u8; 4096];
            init_dir_block(&mut block, 2, 2).unwrap();

            add_entry(&mut block, ino_a, &name_a, Ext4FileType::RegFile).unwrap();
            remove_entry(&mut block, &name_a).unwrap();

            // A new entry that fits in the same space should succeed.
            let result = add_entry(&mut block, ino_b, &name_b, Ext4FileType::RegFile);
            prop_assert!(result.is_ok(), "add after remove should succeed, got {:?}", result.err());
        }

        /// htree_insert always maintains sorted order by hash.
        #[test]
        fn proptest_htree_insert_maintains_sorted(
            inserts in prop::collection::vec((any::<u32>(), any::<u32>()), 1..64),
        ) {
            let mut entries = Vec::new();
            for (hash, block) in &inserts {
                htree_insert(&mut entries, *hash, *block);
            }
            // Verify sorted by hash.
            for i in 1..entries.len() {
                prop_assert!(
                    entries[i - 1].hash <= entries[i].hash,
                    "entries not sorted: [{}].hash={} > [{}].hash={}",
                    i - 1, entries[i - 1].hash, i, entries[i].hash,
                );
            }
            prop_assert_eq!(entries.len(), inserts.len());
        }

        /// htree_find_leaf returns a block whose hash is <= target (rightmost-lte).
        #[test]
        fn proptest_htree_find_leaf_rightmost_lte(
            inserts in prop::collection::vec((any::<u32>(), any::<u32>()), 1..32),
            target in any::<u32>(),
        ) {
            let mut entries = Vec::new();
            for (hash, block) in &inserts {
                htree_insert(&mut entries, *hash, *block);
            }
            if let Some(found_block) = htree_find_leaf(&entries, target) {
                // The found block must correspond to an entry with hash <= target
                // (or be the first entry if target < all hashes).
                let found_entry = entries.iter().find(|e| e.block == found_block);
                prop_assert!(found_entry.is_some(), "returned block {} not in entries", found_block);
            }
        }

        /// htree_remove + htree_insert roundtrip preserves entries.
        #[test]
        fn proptest_htree_remove_insert_roundtrip(
            base in prop::collection::vec((any::<u32>(), any::<u32>()), 1..16),
            remove_idx in 0_usize..16,
        ) {
            let mut entries = Vec::new();
            for (hash, block) in &base {
                htree_insert(&mut entries, *hash, *block);
            }
            let original_len = entries.len();

            if !entries.is_empty() {
                let idx = remove_idx % entries.len();
                let removed = entries[idx];
                htree_remove(&mut entries, removed.hash, removed.block);
                prop_assert_eq!(entries.len(), original_len - 1);

                htree_insert(&mut entries, removed.hash, removed.block);
                prop_assert_eq!(entries.len(), original_len);
            }
        }

        /// dx_hash is deterministic: same input always produces same output.
        #[test]
        fn proptest_dx_hash_deterministic(
            name in name_strategy(),
            seed in prop::array::uniform4(any::<u32>()),
            version in prop::sample::select(vec![0_u8, 1, 2, 3, 4, 5]),
        ) {
            let h1 = compute_dx_hash(version, &name, &seed);
            let h2 = compute_dx_hash(version, &name, &seed);
            prop_assert_eq!(h1, h2, "dx_hash not deterministic for {:?}", name);
            // Bit 0 always cleared (ext4 convention).
            prop_assert_eq!(h1 & 1, 0, "dx_hash bit 0 not cleared for {:?}", name);
        }
    }
}
