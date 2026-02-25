#![forbid(unsafe_code)]
//! Inode management.
//!
//! Read, write, create, and delete inodes. Permission checks,
//! timestamp management (atime/ctime/mtime/crtime), flag handling,
//! and inode table I/O.

use asupersync::Cx;
use ffs_alloc::{FsGeometry, GroupStats};
use ffs_block::BlockDevice;
use ffs_error::{FfsError, Result};
use ffs_ondisk::Ext4Inode;
use ffs_types::{BlockNumber, GroupNumber, InodeNumber};

// ── Constants ────────────────────────────────────────────────────────────────

/// Ext4 extent header magic.
const EXT4_EXTENT_MAGIC: u16 = 0xF30A;

/// Inode flag: uses extents (EXT4_EXTENTS_FL).
const EXT4_EXTENTS_FL: u32 = 0x0008_0000;

/// Checksum field offsets within the raw inode bytes.
const INODE_CHECKSUM_LO_OFFSET: usize = 0x7C;
const INODE_CHECKSUM_HI_OFFSET: usize = 0x82;

// ── Inode location ──────────────────────────────────────────────────────────

/// Computed on-disk location for an inode.
#[derive(Debug, Clone, Copy)]
pub struct InodeLocation {
    pub block: BlockNumber,
    pub byte_offset: usize,
}

/// Compute the disk location of an inode within the inode table.
#[must_use]
#[expect(clippy::cast_possible_truncation)]
pub fn locate_inode(
    ino: InodeNumber,
    geo: &FsGeometry,
    groups: &[GroupStats],
) -> Option<InodeLocation> {
    let group = ffs_types::inode_to_group(ino, geo.inodes_per_group);
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return None;
    }
    let index = ffs_types::inode_index_in_group(ino, geo.inodes_per_group);
    let byte_in_table = u64::from(index) * u64::from(geo.inode_size);
    let block_offset = byte_in_table / u64::from(geo.block_size);
    let byte_offset = (byte_in_table % u64::from(geo.block_size)) as usize;
    let block = BlockNumber(groups[gidx].inode_table_block.0.checked_add(block_offset)?);
    Some(InodeLocation { block, byte_offset })
}

// ── Read ────────────────────────────────────────────────────────────────────

/// Read and parse an inode from the block device.
pub fn read_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &[GroupStats],
    ino: InodeNumber,
) -> Result<Ext4Inode> {
    cx_checkpoint(cx)?;

    let loc = locate_inode(ino, geo, groups).ok_or_else(|| FfsError::Corruption {
        block: 0,
        detail: format!("inode {ino} out of range"),
    })?;

    let buf = dev.read_block(cx, loc.block)?;
    let data = buf.as_slice();
    let inode_size = usize::from(geo.inode_size);

    if loc.byte_offset + inode_size > data.len() {
        return Err(FfsError::Corruption {
            block: loc.block.0,
            detail: "inode extends beyond block boundary".into(),
        });
    }

    let raw = &data[loc.byte_offset..loc.byte_offset + inode_size];
    Ext4Inode::parse_from_bytes(raw).map_err(|e| FfsError::Format(format!("{e}")))
}

// ── Write ───────────────────────────────────────────────────────────────────

/// Serialize an inode and write it to the block device.
///
/// Computes the CRC32C checksum before writing.
pub fn write_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &[GroupStats],
    ino: InodeNumber,
    inode: &Ext4Inode,
    csum_seed: u32,
) -> Result<()> {
    cx_checkpoint(cx)?;

    let loc = locate_inode(ino, geo, groups).ok_or_else(|| FfsError::Corruption {
        block: 0,
        detail: format!("inode {ino} out of range"),
    })?;

    let inode_size = usize::from(geo.inode_size);
    let mut raw = serialize_inode(inode, inode_size);

    // Compute and write checksum.
    #[expect(clippy::cast_possible_truncation)]
    let ino32 = ino.0 as u32;
    compute_and_set_checksum(&mut raw, csum_seed, ino32);

    // Read the block, patch the inode bytes, write back.
    let buf = dev.read_block(cx, loc.block)?;
    let mut block_data = buf.as_slice().to_vec();
    block_data[loc.byte_offset..loc.byte_offset + inode_size].copy_from_slice(&raw);
    dev.write_block(cx, loc.block, &block_data)?;

    Ok(())
}

/// Serialize an `Ext4Inode` into raw bytes of the given `inode_size`.
#[expect(clippy::cast_possible_truncation)]
fn serialize_inode(inode: &Ext4Inode, inode_size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; inode_size];

    // Mode (0x00).
    buf[0x00..0x02].copy_from_slice(&inode.mode.to_le_bytes());
    // UID low (0x02).
    buf[0x02..0x04].copy_from_slice(&(inode.uid as u16).to_le_bytes());
    // Size low (0x04).
    buf[0x04..0x08].copy_from_slice(&(inode.size as u32).to_le_bytes());
    // atime (0x08).
    buf[0x08..0x0C].copy_from_slice(&inode.atime.to_le_bytes());
    // ctime (0x0C).
    buf[0x0C..0x10].copy_from_slice(&inode.ctime.to_le_bytes());
    // mtime (0x10).
    buf[0x10..0x14].copy_from_slice(&inode.mtime.to_le_bytes());
    // dtime (0x14).
    buf[0x14..0x18].copy_from_slice(&inode.dtime.to_le_bytes());
    // GID low (0x18).
    buf[0x18..0x1A].copy_from_slice(&(inode.gid as u16).to_le_bytes());
    // Links count (0x1A).
    buf[0x1A..0x1C].copy_from_slice(&inode.links_count.to_le_bytes());
    // Blocks low (0x1C).
    buf[0x1C..0x20].copy_from_slice(&(inode.blocks as u32).to_le_bytes());
    // Flags (0x20).
    buf[0x20..0x24].copy_from_slice(&inode.flags.to_le_bytes());
    // i_block / extent bytes (0x28, 60 bytes).
    let copy_len = inode.extent_bytes.len().min(60);
    buf[0x28..0x28 + copy_len].copy_from_slice(&inode.extent_bytes[..copy_len]);
    // Generation (0x64).
    buf[0x64..0x68].copy_from_slice(&inode.generation.to_le_bytes());
    // File ACL low (0x68).
    buf[0x68..0x6C].copy_from_slice(&(inode.file_acl as u32).to_le_bytes());
    // Size high (0x6C).
    buf[0x6C..0x70].copy_from_slice(&((inode.size >> 32) as u32).to_le_bytes());
    // Blocks high (0x74, 2 bytes).
    buf[0x74..0x76].copy_from_slice(&((inode.blocks >> 32) as u16).to_le_bytes());
    // File ACL high (0x76, 2 bytes).
    buf[0x76..0x78].copy_from_slice(&((inode.file_acl >> 32) as u16).to_le_bytes());
    // UID high (0x78).
    buf[0x78..0x7A].copy_from_slice(&((inode.uid >> 16) as u16).to_le_bytes());
    // GID high (0x7A).
    buf[0x7A..0x7C].copy_from_slice(&((inode.gid >> 16) as u16).to_le_bytes());
    // checksum_lo (0x7C) — will be set by compute_and_set_checksum.

    // Extended area (when inode_size > 128).
    if inode_size > 128 {
        // extra_isize (0x80).
        buf[0x80..0x82].copy_from_slice(&inode.extra_isize.to_le_bytes());
        // checksum_hi (0x82) — will be set by compute_and_set_checksum.

        // Extended timestamps.
        if inode_size >= 0x88 {
            buf[0x84..0x88].copy_from_slice(&inode.ctime_extra.to_le_bytes());
        }
        if inode_size >= 0x8C {
            buf[0x88..0x8C].copy_from_slice(&inode.mtime_extra.to_le_bytes());
        }
        if inode_size >= 0x90 {
            buf[0x8C..0x90].copy_from_slice(&inode.atime_extra.to_le_bytes());
        }
        if inode_size >= 0x98 {
            buf[0x90..0x94].copy_from_slice(&inode.crtime.to_le_bytes());
            buf[0x94..0x98].copy_from_slice(&inode.crtime_extra.to_le_bytes());
        }
        if inode_size >= 0xA0 {
            buf[0x9C..0xA0].copy_from_slice(&inode.projid.to_le_bytes());
        }

        // Inline xattrs go after 128 + extra_isize.
        let xattr_start = 128 + usize::from(inode.extra_isize);
        let xattr_copy = inode
            .xattr_ibody
            .len()
            .min(inode_size.saturating_sub(xattr_start));
        if xattr_start < inode_size && xattr_copy > 0 {
            buf[xattr_start..xattr_start + xattr_copy]
                .copy_from_slice(&inode.xattr_ibody[..xattr_copy]);
        }
    }

    buf
}

/// Compute CRC32C checksum and store it in the raw inode buffer.
fn compute_and_set_checksum(raw: &mut [u8], csum_seed: u32, ino: u32) {
    let is = raw.len();
    if is < 128 {
        return;
    }

    // Per-inode seed: ext4_chksum(csum_seed, le_ino) then ext4_chksum(ino_seed, le_gen).
    let ino_seed = ffs_ondisk::ext4_chksum(csum_seed, &ino.to_le_bytes());
    let generation = u32::from_le_bytes([raw[0x64], raw[0x65], raw[0x66], raw[0x67]]);
    let ino_seed = ffs_ondisk::ext4_chksum(ino_seed, &generation.to_le_bytes());

    // Zero out checksum fields before computing.
    raw[INODE_CHECKSUM_LO_OFFSET] = 0;
    raw[INODE_CHECKSUM_LO_OFFSET + 1] = 0;
    if is >= INODE_CHECKSUM_HI_OFFSET + 2 {
        raw[INODE_CHECKSUM_HI_OFFSET] = 0;
        raw[INODE_CHECKSUM_HI_OFFSET + 1] = 0;
    }

    // CRC the entire raw inode.
    let csum = ffs_ondisk::ext4_chksum(ino_seed, raw);

    // Store checksum.
    let lo = (csum & 0xFFFF) as u16;
    raw[INODE_CHECKSUM_LO_OFFSET..INODE_CHECKSUM_LO_OFFSET + 2].copy_from_slice(&lo.to_le_bytes());
    if is >= INODE_CHECKSUM_HI_OFFSET + 2 {
        let hi = ((csum >> 16) & 0xFFFF) as u16;
        raw[INODE_CHECKSUM_HI_OFFSET..INODE_CHECKSUM_HI_OFFSET + 2]
            .copy_from_slice(&hi.to_le_bytes());
    }
}

// ── Create ──────────────────────────────────────────────────────────────────

/// File type constants for inode mode.
pub mod file_type {
    pub const S_IFREG: u16 = 0o100_000;
    pub const S_IFDIR: u16 = 0o040_000;
    pub const S_IFLNK: u16 = 0o120_000;
}

/// Create a new inode on disk.
///
/// Allocates an inode number via `ffs-alloc`, initializes fields, writes to disk.
/// Returns `(InodeNumber, Ext4Inode)`.
#[expect(clippy::too_many_arguments)]
pub fn create_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    mode: u16,
    uid: u32,
    gid: u32,
    parent_group: GroupNumber,
    csum_seed: u32,
    now_secs: u64,
    now_nsec: u32,
    pctx: &ffs_alloc::PersistCtx,
) -> Result<(InodeNumber, Ext4Inode)> {
    cx_checkpoint(cx)?;

    let is_dir = (mode & 0xF000) == file_type::S_IFDIR;
    let alloc = ffs_alloc::alloc_inode_persist(cx, dev, geo, groups, parent_group, is_dir, pctx)?;

    if is_dir {
        let gidx = alloc.group.0 as usize;
        if gidx < groups.len() {
            groups[gidx].used_dirs = groups[gidx].used_dirs.saturating_add(1);
        }
    }

    // Read old generation from the on-disk inode slot so we can bump it.
    // This is the NFS-style generation counter: when an inode number is reused,
    // incrementing the generation lets the FUSE/NFS layer detect stale handles.
    let old_generation = locate_inode(alloc.ino, geo, groups)
        .and_then(|loc| {
            let buf = dev.read_block(cx, loc.block).ok()?;
            let data = buf.as_slice();
            let off = loc.byte_offset;
            // generation lives at offset 0x64 in the raw inode (4 bytes LE).
            if off + 0x68 <= data.len() {
                Some(u32::from_le_bytes([
                    data[off + 0x64],
                    data[off + 0x65],
                    data[off + 0x66],
                    data[off + 0x67],
                ]))
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Initialize extent tree root (empty tree: magic + 0 entries, max 4, depth 0).
    let mut extent_bytes = vec![0u8; 60];
    extent_bytes[0] = (EXT4_EXTENT_MAGIC & 0xFF) as u8;
    extent_bytes[1] = (EXT4_EXTENT_MAGIC >> 8) as u8;
    // entries = 0.
    extent_bytes[4] = 4; // max_entries = 4.
    // depth = 0 (already zero).

    let extra_time = encode_extra_timestamp(now_secs, now_nsec);

    #[allow(clippy::cast_possible_truncation)]
    let now_lo = now_secs as u32;
    let inode = Ext4Inode {
        mode,
        uid,
        gid,
        size: 0,
        links_count: if is_dir { 2 } else { 1 },
        blocks: 0,
        flags: EXT4_EXTENTS_FL,
        generation: old_generation.wrapping_add(1),
        file_acl: 0,
        atime: now_lo,
        ctime: now_lo,
        mtime: now_lo,
        dtime: 0,
        atime_extra: extra_time,
        ctime_extra: extra_time,
        mtime_extra: extra_time,
        crtime: now_lo,
        crtime_extra: extra_time,
        extra_isize: 32,
        checksum: 0,
        projid: 0,
        extent_bytes,
        xattr_ibody: Vec::new(),
    };

    write_inode(cx, dev, geo, groups, alloc.ino, &inode, csum_seed)?;

    Ok((alloc.ino, inode))
}

// ── Delete ──────────────────────────────────────────────────────────────────

/// Delete an inode: truncate all extents, free the inode, zero the on-disk data.
#[expect(clippy::too_many_arguments)]
pub fn delete_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    ino: InodeNumber,
    inode: &mut Ext4Inode,
    csum_seed: u32,
    now_secs: u64,
    pctx: &ffs_alloc::PersistCtx,
) -> Result<()> {
    cx_checkpoint(cx)?;

    // Truncate all extents if the inode uses extents.
    if inode.flags & EXT4_EXTENTS_FL != 0 && inode.extent_bytes.len() >= 60 {
        let mut root_buf = [0u8; 60];
        root_buf.copy_from_slice(&inode.extent_bytes[..60]);
        ffs_extent::truncate_extents(cx, dev, &mut root_buf, geo, groups, 0, pctx)?;
        inode.extent_bytes[..60].copy_from_slice(&root_buf);
    }

    // Free the external xattr block if present.
    if inode.file_acl != 0 {
        let acl_block = BlockNumber(inode.file_acl);
        let buf = dev.read_block(cx, acl_block)?;
        let mut data = buf.as_slice().to_vec();
        if data.len() >= 32 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == 0xEA02_0000 {
                let refcount = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                if refcount > 1 {
                    let new_refcount = refcount - 1;
                    data[4..8].copy_from_slice(&new_refcount.to_le_bytes());
                    dev.write_block(cx, acl_block, &data)?;
                } else {
                    ffs_alloc::free_blocks_persist(cx, dev, geo, groups, acl_block, 1, pctx)?;
                }
            } else {
                // Not a valid xattr block, but we still free it to prevent leaks.
                ffs_alloc::free_blocks_persist(cx, dev, geo, groups, acl_block, 1, pctx)?;
            }
        }
        inode.file_acl = 0;
    }

    // Set deletion time.
    #[allow(clippy::cast_possible_truncation)]
    {
        inode.dtime = now_secs as u32;
    }
    inode.links_count = 0;
    inode.size = 0;
    inode.blocks = 0;

    // Write the zeroed-out inode to disk.
    write_inode(cx, dev, geo, groups, ino, inode, csum_seed)?;

    // Free the inode in the bitmap.
    ffs_alloc::free_inode_persist(cx, dev, geo, groups, ino, pctx)?;

    Ok(())
}

// ── Timestamps ──────────────────────────────────────────────────────────────

/// Encode nanoseconds and epoch extension into the `_extra` timestamp field.
///
/// Layout: bits 0-1 = epoch extension (seconds >> 32), bits 2-31 = nanoseconds.
#[must_use]
pub fn encode_extra_timestamp(secs: u64, nsec: u32) -> u32 {
    let epoch = ((secs >> 32) & 0x3) as u32; // upper 2 bits of seconds
    let nsec_bits = nsec << 2; // nanoseconds shifted to bits 2-31
    epoch | nsec_bits
}

/// Touch atime on an inode.
#[allow(clippy::cast_possible_truncation)]
pub fn touch_atime(inode: &mut Ext4Inode, secs: u64, nsec: u32) {
    inode.atime = secs as u32; // lower 32 bits per ext4 spec
    inode.atime_extra = encode_extra_timestamp(secs, nsec);
}

/// Touch mtime and ctime on an inode.
#[allow(clippy::cast_possible_truncation)]
pub fn touch_mtime_ctime(inode: &mut Ext4Inode, secs: u64, nsec: u32) {
    inode.mtime = secs as u32;
    inode.mtime_extra = encode_extra_timestamp(secs, nsec);
    inode.ctime = secs as u32;
    inode.ctime_extra = encode_extra_timestamp(secs, nsec);
}

/// Touch ctime only on an inode (e.g., for chmod, chown).
#[allow(clippy::cast_possible_truncation)]
pub fn touch_ctime(inode: &mut Ext4Inode, secs: u64, nsec: u32) {
    inode.ctime = secs as u32;
    inode.ctime_extra = encode_extra_timestamp(secs, nsec);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn cx_checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|_| FfsError::Cancelled)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use ffs_block::BlockBuf;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct MemBlockDevice {
        block_size: u32,
        blocks: Mutex<HashMap<u64, Vec<u8>>>,
    }

    impl MemBlockDevice {
        fn new(block_size: u32) -> Self {
            Self {
                block_size,
                blocks: Mutex::new(HashMap::new()),
            }
        }
    }

    impl BlockDevice for MemBlockDevice {
        fn read_block(&self, _cx: &Cx, block: BlockNumber) -> Result<BlockBuf> {
            let blocks = self.blocks.lock().unwrap();
            blocks.get(&block.0).map_or_else(
                || Ok(BlockBuf::new(vec![0u8; self.block_size as usize])),
                |data| Ok(BlockBuf::new(data.clone())),
            )
        }

        fn write_block(&self, _cx: &Cx, block: BlockNumber, data: &[u8]) -> Result<()> {
            self.blocks.lock().unwrap().insert(block.0, data.to_vec());
            Ok(())
        }

        fn block_size(&self) -> u32 {
            self.block_size
        }

        fn block_count(&self) -> u64 {
            1_000_000
        }

        fn sync(&self, _cx: &Cx) -> Result<()> {
            Ok(())
        }
    }

    fn test_cx() -> Cx {
        Cx::for_testing()
    }

    fn mock_pctx() -> ffs_alloc::PersistCtx {
        ffs_alloc::PersistCtx {
            gdt_block: BlockNumber(1),
            desc_size: 32,
            has_metadata_csum: false,
            csum_seed: 0,
        }
    }

    fn make_geometry() -> FsGeometry {
        FsGeometry {
            blocks_per_group: 8192,
            inodes_per_group: 2048,
            block_size: 4096,
            total_blocks: 32768,
            total_inodes: 8192,
            first_data_block: 0,
            group_count: 4,
            inode_size: 256,
        }
    }

    fn make_groups(geo: &FsGeometry) -> Vec<GroupStats> {
        (0..geo.group_count)
            .map(|g| GroupStats {
                group: GroupNumber(g),
                free_blocks: geo.blocks_per_group,
                free_inodes: geo.inodes_per_group,
                used_dirs: 0,
                block_bitmap_block: BlockNumber(u64::from(g) * 100 + 1),
                inode_bitmap_block: BlockNumber(u64::from(g) * 100 + 2),
                inode_table_block: BlockNumber(u64::from(g) * 100 + 3),
                flags: 0,
            })
            .collect()
    }

    #[test]
    fn locate_inode_basic() {
        let geo = make_geometry();
        let groups = make_groups(&geo);

        // Inode 1 → group 0, index 0.
        let loc = locate_inode(InodeNumber(1), &geo, &groups).unwrap();
        assert_eq!(loc.block, BlockNumber(3)); // inode_table_block for group 0
        assert_eq!(loc.byte_offset, 0);

        // Inode 2 → group 0, index 1.
        let loc = locate_inode(InodeNumber(2), &geo, &groups).unwrap();
        assert_eq!(loc.block, BlockNumber(3));
        assert_eq!(loc.byte_offset, 256); // 1 * 256

        // Inode at group boundary: 2049 → group 1, index 0.
        let loc = locate_inode(InodeNumber(2049), &geo, &groups).unwrap();
        assert_eq!(loc.block, BlockNumber(103)); // group 1 table
        assert_eq!(loc.byte_offset, 0);
    }

    #[test]
    fn serialize_roundtrip() {
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 1000,
            gid: 1000,
            size: 4096,
            links_count: 1,
            blocks: 8,
            flags: EXT4_EXTENTS_FL,
            generation: 42,
            file_acl: 0,
            atime: 1_700_000_000,
            ctime: 1_700_000_000,
            mtime: 1_700_000_000,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 1_700_000_000,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        let raw = serialize_inode(&inode, 256);
        assert_eq!(raw.len(), 256);

        // Parse back.
        let parsed = Ext4Inode::parse_from_bytes(&raw).unwrap();
        assert_eq!(parsed.mode, inode.mode);
        assert_eq!(parsed.uid, inode.uid);
        assert_eq!(parsed.gid, inode.gid);
        assert_eq!(parsed.size, inode.size);
        assert_eq!(parsed.links_count, inode.links_count);
        assert_eq!(parsed.flags, inode.flags);
        assert_eq!(parsed.generation, inode.generation);
        assert_eq!(parsed.atime, inode.atime);
        assert_eq!(parsed.mtime, inode.mtime);
        assert_eq!(parsed.ctime, inode.ctime);
    }

    #[test]
    fn checksum_roundtrip() {
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 1000,
            gid: 1000,
            size: 0,
            links_count: 1,
            blocks: 0,
            flags: EXT4_EXTENTS_FL,
            generation: 1,
            file_acl: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        let mut raw = serialize_inode(&inode, 256);
        compute_and_set_checksum(&mut raw, 0xDEAD_BEEF, 42);

        // Verify the checksum using ffs-ondisk's verifier.
        let result = ffs_ondisk::verify_inode_checksum(&raw, 0xDEAD_BEEF, 42, 256);
        assert!(result.is_ok(), "checksum verification failed: {result:?}");
    }

    #[test]
    fn create_and_read_inode() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, created) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            1000,
            1000,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        assert_eq!(ino, InodeNumber(1));
        assert_eq!(created.mode, 0o100_644);
        assert_eq!(created.uid, 1000);
        assert_eq!(created.links_count, 1);

        // Read it back.
        let read_back = read_inode(&cx, &dev, &geo, &groups, ino).unwrap();
        assert_eq!(read_back.mode, 0o100_644);
        assert_eq!(read_back.uid, 1000);
    }

    #[test]
    fn create_directory_inode() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (_, created) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            file_type::S_IFDIR | 0o755,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        assert_eq!(created.links_count, 2);
        assert_eq!(created.mode, file_type::S_IFDIR | 0o755);
    }

    #[test]
    fn delete_inode_frees_resources() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, mut inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        let free_before = groups[0].free_inodes;

        delete_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            ino,
            &mut inode,
            0,
            1_700_000_001,
            &mock_pctx(),
        )
        .unwrap();

        assert_eq!(inode.links_count, 0);
        assert_eq!(inode.dtime, 1_700_000_001);
        assert_eq!(groups[0].free_inodes, free_before + 1);
    }

    #[test]
    fn write_and_verify_checksum() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0x1234_5678,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        // Read raw bytes and verify checksum.
        let loc = locate_inode(ino, &geo, &groups).unwrap();
        let buf = dev.read_block(&cx, loc.block).unwrap();
        let raw = &buf.as_slice()[loc.byte_offset..loc.byte_offset + 256];

        let result = ffs_ondisk::verify_inode_checksum(raw, 0x1234_5678, ino.0 as u32, 256);
        assert!(result.is_ok(), "checksum verification failed: {result:?}");

        // Verify the inode fields are correct.
        let parsed = Ext4Inode::parse_from_bytes(raw).unwrap();
        assert_eq!(parsed.mode, inode.mode);
    }

    #[test]
    fn touch_timestamps() {
        let mut inode = Ext4Inode {
            mode: 0o100_644,
            uid: 0,
            gid: 0,
            size: 0,
            links_count: 1,
            blocks: 0,
            flags: 0,
            generation: 0,
            file_acl: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        touch_atime(&mut inode, 100, 500_000_000);
        assert_eq!(inode.atime, 100);

        touch_mtime_ctime(&mut inode, 200, 0);
        assert_eq!(inode.mtime, 200);
        assert_eq!(inode.ctime, 200);

        touch_ctime(&mut inode, 300, 0);
        assert_eq!(inode.ctime, 300);
        assert_eq!(inode.mtime, 200); // mtime unchanged.
    }

    #[test]
    fn encode_extra_timestamp_nsec() {
        let extra = encode_extra_timestamp(0, 999_999_999);
        // Nanoseconds stored in bits 2-31.
        let nsec = extra >> 2;
        // Should preserve the nanosecond value.
        assert_eq!(nsec, 999_999_999);
    }

    #[test]
    fn locate_inode_out_of_range() {
        let geo = make_geometry();
        let groups = make_groups(&geo);

        // Inode 0 is invalid (ext4 inodes are 1-based).
        // Inode way beyond total → group index out of range.
        let result = locate_inode(InodeNumber(100_000), &geo, &groups);
        assert!(result.is_none());
    }

    #[test]
    fn locate_inode_block_offset_within_table() {
        let geo = make_geometry();
        let groups = make_groups(&geo);

        // 4096-byte block / 256-byte inode = 16 inodes per block.
        // Inode 17 → group 0, index 16 → byte 16*256 = 4096 → second block.
        let loc = locate_inode(InodeNumber(17), &geo, &groups).unwrap();
        assert_eq!(loc.block, BlockNumber(4)); // inode_table_block(3) + 1
        assert_eq!(loc.byte_offset, 0);

        // Inode 18 → index 17 → byte 17*256 = 4352 → block 4, offset 256.
        let loc = locate_inode(InodeNumber(18), &geo, &groups).unwrap();
        assert_eq!(loc.block, BlockNumber(4));
        assert_eq!(loc.byte_offset, 256);
    }

    #[test]
    fn serialize_large_uid_gid() {
        // UID/GID > 65535 should be split across low and high fields.
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 0x0012_3456, // high=0x0012, low=0x3456
            gid: 0x00AB_CDEF, // high=0x00AB, low=0xCDEF
            size: 0,
            links_count: 1,
            blocks: 0,
            flags: 0,
            generation: 0,
            file_acl: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        let raw = serialize_inode(&inode, 256);
        let parsed = Ext4Inode::parse_from_bytes(&raw).unwrap();
        assert_eq!(parsed.uid, 0x0012_3456);
        assert_eq!(parsed.gid, 0x00AB_CDEF);
    }

    #[test]
    fn serialize_large_file_size() {
        // Size > 4GB should be split across low and high fields.
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 0,
            gid: 0,
            size: 0x1_0000_1234, // > 4GB
            links_count: 1,
            blocks: 0,
            flags: 0,
            generation: 0,
            file_acl: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        let raw = serialize_inode(&inode, 256);
        let parsed = Ext4Inode::parse_from_bytes(&raw).unwrap();
        assert_eq!(parsed.size, 0x1_0000_1234);
    }

    #[test]
    fn serialize_xattr_ibody_preserved() {
        let xattr_data = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 0,
            gid: 0,
            size: 0,
            links_count: 1,
            blocks: 0,
            flags: 0,
            generation: 0,
            file_acl: 0,
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: xattr_data.clone(),
        };

        let raw = serialize_inode(&inode, 256);
        // Inline xattrs start at offset 128 + extra_isize = 160.
        assert_eq!(&raw[160..164], &xattr_data);
    }

    #[test]
    fn checksum_skipped_for_small_inode() {
        // Inode < 128 bytes should not panic.
        let mut raw = vec![0u8; 64];
        compute_and_set_checksum(&mut raw, 0, 1);
        // Should just return without modifying (no checksum field to set).
        assert_eq!(raw, vec![0u8; 64]);
    }

    #[test]
    fn create_multiple_inodes_sequential() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = mock_pctx();

        let mut inos = Vec::new();
        for _ in 0..5 {
            let (ino, _) = create_inode(
                &cx,
                &dev,
                &geo,
                &mut groups,
                0o100_644,
                0,
                0,
                GroupNumber(0),
                0,
                1_700_000_000,
                0,
                &pctx,
            )
            .unwrap();
            inos.push(ino);
        }

        // All inodes should be unique.
        for i in 0..inos.len() {
            for j in (i + 1)..inos.len() {
                assert_ne!(inos[i], inos[j], "duplicate inode numbers");
            }
        }

        // All should be readable.
        for &ino in &inos {
            let inode = read_inode(&cx, &dev, &geo, &groups, ino).unwrap();
            assert_eq!(inode.mode, 0o100_644);
        }
    }

    #[test]
    fn read_inode_out_of_range_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let groups = make_groups(&geo);

        let result = read_inode(&cx, &dev, &geo, &groups, InodeNumber(100_000));
        assert!(result.is_err());
    }

    #[test]
    fn write_inode_updates_on_disk() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, mut inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        // Modify and write back.
        inode.size = 8192;
        inode.uid = 500;
        touch_mtime_ctime(&mut inode, 1_700_000_100, 0);
        write_inode(&cx, &dev, &geo, &groups, ino, &inode, 0).unwrap();

        // Read back and verify.
        let read_back = read_inode(&cx, &dev, &geo, &groups, ino).unwrap();
        assert_eq!(read_back.size, 8192);
        assert_eq!(read_back.uid, 500);
        assert_eq!(read_back.mtime, 1_700_000_100);
        assert_eq!(read_back.ctime, 1_700_000_100);
    }

    #[test]
    fn delete_directory_zeroes_fields() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, mut inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            file_type::S_IFDIR | 0o755,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();

        // Directory creation should increment used_dirs in some group.
        let total_dirs: u32 = groups.iter().map(|g| g.used_dirs).sum();
        assert_eq!(total_dirs, 1, "one directory should be tracked");

        delete_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            ino,
            &mut inode,
            0,
            1_700_000_001,
            &mock_pctx(),
        )
        .unwrap();

        assert_eq!(inode.links_count, 0);
        assert_eq!(inode.size, 0);
        assert_eq!(inode.dtime, 1_700_000_001);
    }

    #[test]
    fn encode_extra_timestamp_zero_nsec() {
        let extra = encode_extra_timestamp(0, 0);
        assert_eq!(extra, 0);
    }

    #[test]
    fn encode_extra_timestamp_max_nsec() {
        // 999_999_999 ns is the maximum valid nanosecond value.
        let extra = encode_extra_timestamp(0, 999_999_999);
        let nsec_back = extra >> 2;
        assert_eq!(nsec_back, 999_999_999);
        // Verify the epoch bits (0-1) are zero for 32-bit timestamps.
        assert_eq!(extra & 0x3, 0);
    }

    #[test]
    fn timestamp_nsec_roundtrip_through_write_read() {
        // Create an inode with specific nanosecond timestamps, write it, read back,
        // and verify the nanosecond fields survive the roundtrip.
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (ino, mut inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            1000,
            1000,
            GroupNumber(0),
            0,
            0,
            0,
            &mock_pctx(),
        )
        .expect("create inode");

        // Set timestamps with specific nanosecond values.
        touch_atime(&mut inode, 1_700_000_000, 123_456_788);
        touch_mtime_ctime(&mut inode, 1_700_000_001, 999_999_996);

        write_inode(&cx, &dev, &geo, &groups, ino, &inode, 0).expect("write");
        let read_back = read_inode(&cx, &dev, &geo, &groups, ino).expect("read");

        assert_eq!(read_back.atime, 1_700_000_000);
        assert_eq!(read_back.atime_extra, inode.atime_extra);
        assert_eq!(read_back.mtime, 1_700_000_001);
        assert_eq!(read_back.mtime_extra, inode.mtime_extra);
        assert_eq!(read_back.ctime, 1_700_000_001);
        assert_eq!(read_back.ctime_extra, inode.ctime_extra);
    }

    #[test]
    fn touch_ctime_preserves_mtime() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let (_ino, mut inode) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0,
            0,
            0,
            &mock_pctx(),
        )
        .expect("create");

        touch_mtime_ctime(&mut inode, 100, 500_000_000);
        touch_ctime(&mut inode, 200, 750_000_000);

        // ctime should be updated
        assert_eq!(inode.ctime, 200);
        // mtime should remain unchanged
        assert_eq!(inode.mtime, 100);
        assert_eq!(inode.mtime_extra, encode_extra_timestamp(100, 500_000_000));
    }

    #[test]
    fn encode_extra_timestamp_epoch_extension() {
        // Timestamp at 2^32 (epoch 1): after Feb 2106.
        let secs: u64 = 1 << 32; // 4_294_967_296
        let extra = encode_extra_timestamp(secs, 0);
        // Epoch bits (bits 0-1) should be 1, nsec bits should be 0.
        assert_eq!(extra & 0x3, 1);
        assert_eq!(extra >> 2, 0);

        // Timestamp at 2^33 (epoch 2).
        let secs: u64 = 1 << 33;
        let extra = encode_extra_timestamp(secs, 123_456_788);
        assert_eq!(extra & 0x3, 2);
        assert_eq!(extra >> 2, 123_456_788);

        // Timestamp at 3 * 2^32 (epoch 3).
        let secs: u64 = 3 << 32;
        let extra = encode_extra_timestamp(secs, 0);
        assert_eq!(extra & 0x3, 3);
    }

    #[test]
    fn serialize_blocks_high_bits() {
        // Test 48-bit blocks field split.
        let inode = Ext4Inode {
            mode: 0o100_644,
            uid: 0,
            gid: 0,
            size: 0,
            links_count: 1,
            blocks: 0x1_2345_6789, // needs high bits
            flags: 0,
            generation: 0,
            file_acl: 0x2_0000_0000, // needs file_acl high bits
            atime: 0,
            ctime: 0,
            mtime: 0,
            dtime: 0,
            atime_extra: 0,
            ctime_extra: 0,
            mtime_extra: 0,
            crtime: 0,
            crtime_extra: 0,
            extra_isize: 32,
            checksum: 0,
            projid: 0,
            extent_bytes: vec![0u8; 60],
            xattr_ibody: Vec::new(),
        };

        let raw = serialize_inode(&inode, 256);
        let parsed = Ext4Inode::parse_from_bytes(&raw).unwrap();
        assert_eq!(parsed.blocks, 0x1_2345_6789);
        assert_eq!(parsed.file_acl, 0x2_0000_0000);
    }

    #[test]
    fn create_inode_bumps_generation_on_reuse() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Create first inode — on a zeroed device the old generation is 0,
        // so the new generation should be 1.
        let (ino1, inode1) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_000,
            0,
            &mock_pctx(),
        )
        .unwrap();
        assert_eq!(inode1.generation, 1, "first alloc: 0 → 1");

        // Delete then re-create at the same inode slot.
        let mut del = inode1;
        delete_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            ino1,
            &mut del,
            0,
            1_700_000_001,
            &mock_pctx(),
        )
        .unwrap();
        let (ino2, inode2) = create_inode(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0o100_644,
            0,
            0,
            GroupNumber(0),
            0,
            1_700_000_002,
            0,
            &mock_pctx(),
        )
        .unwrap();
        assert_eq!(ino2, ino1, "should reuse the freed inode number");
        assert_eq!(inode2.generation, 2, "reuse: 1 → 2");
    }

    // ── Proptest property-based tests ─────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// encode_extra_timestamp preserves epoch bits (lower 2 bits of secs >> 32).
        #[test]
        fn proptest_encode_extra_timestamp_epoch_bits(
            secs in 0_u64..=0x3_FFFF_FFFF_u64,
            nsec in 0_u32..1_000_000_000,
        ) {
            let extra = encode_extra_timestamp(secs, nsec);
            let epoch = ((secs >> 32) & 0x3) as u32;
            prop_assert_eq!(extra & 0x3, epoch, "epoch bits mismatch");
        }

        /// encode_extra_timestamp preserves nanoseconds in bits 2-31.
        #[test]
        fn proptest_encode_extra_timestamp_nsec_preserved(
            secs in 0_u64..=0x3_FFFF_FFFF_u64,
            nsec in 0_u32..1_000_000_000,
        ) {
            let extra = encode_extra_timestamp(secs, nsec);
            let decoded_nsec = extra >> 2;
            prop_assert_eq!(decoded_nsec, nsec, "nanoseconds not preserved");
        }

        /// touch_atime only modifies atime fields, not mtime/ctime.
        #[test]
        fn proptest_touch_atime_isolation(
            secs in 0_u64..=0x3_FFFF_FFFF_u64,
            nsec in 0_u32..1_000_000_000,
        ) {
            let mut inode = Ext4Inode {
                mode: 0o100_644, uid: 0, gid: 0, size: 0,
                links_count: 1, blocks: 0, flags: 0, generation: 0,
                file_acl: 0, atime: 100, ctime: 200, mtime: 300, dtime: 0,
                atime_extra: 0, ctime_extra: 999, mtime_extra: 888,
                crtime: 0, crtime_extra: 0, extra_isize: 32,
                checksum: 0, projid: 0,
                extent_bytes: vec![0u8; 60], xattr_ibody: Vec::new(),
            };

            let orig_ctime = inode.ctime;
            let orig_ctime_extra = inode.ctime_extra;
            let orig_mtime = inode.mtime;
            let orig_mtime_extra = inode.mtime_extra;

            touch_atime(&mut inode, secs, nsec);

            prop_assert_eq!(inode.ctime, orig_ctime, "ctime must not change");
            prop_assert_eq!(inode.ctime_extra, orig_ctime_extra, "ctime_extra must not change");
            prop_assert_eq!(inode.mtime, orig_mtime, "mtime must not change");
            prop_assert_eq!(inode.mtime_extra, orig_mtime_extra, "mtime_extra must not change");
            prop_assert_eq!(inode.atime, secs as u32, "atime lower 32 bits");
        }

        /// touch_mtime_ctime sets both mtime and ctime identically.
        #[test]
        fn proptest_touch_mtime_ctime_symmetric(
            secs in 0_u64..=0x3_FFFF_FFFF_u64,
            nsec in 0_u32..1_000_000_000,
        ) {
            let mut inode = Ext4Inode {
                mode: 0o100_644, uid: 0, gid: 0, size: 0,
                links_count: 1, blocks: 0, flags: 0, generation: 0,
                file_acl: 0, atime: 100, ctime: 0, mtime: 0, dtime: 0,
                atime_extra: 42, ctime_extra: 0, mtime_extra: 0,
                crtime: 0, crtime_extra: 0, extra_isize: 32,
                checksum: 0, projid: 0,
                extent_bytes: vec![0u8; 60], xattr_ibody: Vec::new(),
            };

            let orig_atime = inode.atime;
            let orig_atime_extra = inode.atime_extra;

            touch_mtime_ctime(&mut inode, secs, nsec);

            prop_assert_eq!(inode.mtime, inode.ctime, "mtime and ctime must be equal");
            prop_assert_eq!(inode.mtime_extra, inode.ctime_extra, "mtime_extra and ctime_extra must be equal");
            prop_assert_eq!(inode.atime, orig_atime, "atime must not change");
            prop_assert_eq!(inode.atime_extra, orig_atime_extra, "atime_extra must not change");
        }

        /// serialize → parse roundtrip preserves core fields.
        #[test]
        fn proptest_serialize_parse_roundtrip(
            mode_bits in 0_u16..0o7777,
            uid in any::<u32>(),
            gid in any::<u32>(),
            size in 0_u64..(1_u64 << 48),
            links_count in any::<u16>(),
            blocks in 0_u64..(1_u64 << 48),
            flags in any::<u32>(),
            generation in any::<u32>(),
        ) {
            // Use S_IFREG so size_hi is parsed back correctly.
            let mode = 0o100_000 | mode_bits;
            let inode = Ext4Inode {
                mode, uid, gid, size, links_count, blocks, flags, generation,
                file_acl: 0, atime: 0, ctime: 0, mtime: 0, dtime: 0,
                atime_extra: 0, ctime_extra: 0, mtime_extra: 0,
                crtime: 0, crtime_extra: 0, extra_isize: 32,
                checksum: 0, projid: 0,
                extent_bytes: vec![0u8; 60], xattr_ibody: Vec::new(),
            };

            let raw = serialize_inode(&inode, 256);
            let parsed = Ext4Inode::parse_from_bytes(&raw).unwrap();

            prop_assert_eq!(parsed.mode, mode);
            prop_assert_eq!(parsed.uid, uid);
            prop_assert_eq!(parsed.gid, gid);
            prop_assert_eq!(parsed.size, size);
            prop_assert_eq!(parsed.links_count, links_count);
            prop_assert_eq!(parsed.blocks, blocks);
            prop_assert_eq!(parsed.flags, flags);
            prop_assert_eq!(parsed.generation, generation);
        }

        /// locate_inode returns None for out-of-range inodes.
        #[test]
        fn proptest_locate_inode_out_of_range(
            ino_raw in 8193_u64..=100_000,
        ) {
            let geo = make_geometry();
            let groups = make_groups(&geo);
            // total_inodes = 8192, so any ino > 8192 should be out of range.
            let result = locate_inode(InodeNumber(ino_raw), &geo, &groups);
            prop_assert!(result.is_none(), "ino {} should be out of range", ino_raw);
        }
    }
}
