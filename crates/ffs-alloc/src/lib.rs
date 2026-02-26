#![forbid(unsafe_code)]
//! Block and inode allocation.
//!
//! See [`succinct::SuccinctBitmap`] for O(1) rank / O(log n) select over bitmaps.
//!
//! mballoc-style multi-block allocator (buddy system, best-fit,
//! per-inode and per-locality-group preallocation) and Orlov
//! inode allocator for directory spreading.
//!
//! ## Design
//!
//! The allocator is layered:
//!
//! 1. **Bitmap** — raw bit manipulation on block/inode bitmaps.
//! 2. **GroupStats** — cached per-group free counts.
//! 3. **BlockAllocator** — goal-directed block allocation across groups.
//! 4. **InodeAllocator** — Orlov-style inode placement.

pub mod succinct;

use asupersync::Cx;
use ffs_block::BlockDevice;
use ffs_error::{FfsError, Result};
use ffs_ondisk::{Ext4GroupDesc, Ext4Superblock};
use ffs_types::{BlockNumber, GroupNumber, InodeNumber};

// ── Bitmap operations ───────────────────────────────────────────────────────

/// Group flags from `bg_flags` field.
const GD_FLAG_INODE_UNINIT: u16 = 0x0001;
const GD_FLAG_BLOCK_UNINIT: u16 = 0x0002;

/// Get bit `idx` from a bitmap byte slice.
#[must_use]
pub fn bitmap_get(bitmap: &[u8], idx: u32) -> bool {
    let byte_idx = (idx / 8) as usize;
    let bit_idx = idx % 8;
    if byte_idx >= bitmap.len() {
        return false;
    }
    (bitmap[byte_idx] >> bit_idx) & 1 == 1
}

/// Set bit `idx` in a bitmap byte slice.
pub fn bitmap_set(bitmap: &mut [u8], idx: u32) {
    let byte_idx = (idx / 8) as usize;
    let bit_idx = idx % 8;
    if byte_idx < bitmap.len() {
        bitmap[byte_idx] |= 1 << bit_idx;
    }
}

/// Clear bit `idx` in a bitmap byte slice.
pub fn bitmap_clear(bitmap: &mut [u8], idx: u32) {
    let byte_idx = (idx / 8) as usize;
    let bit_idx = idx % 8;
    if byte_idx < bitmap.len() {
        bitmap[byte_idx] &= !(1 << bit_idx);
    }
}

/// Count free (zero) bits in the first `count` bits of `bitmap`.
#[must_use]
pub fn bitmap_count_free(bitmap: &[u8], count: u32) -> u32 {
    let full_bytes = (count / 8) as usize;
    let remainder = count % 8;
    let mut free = 0u32;

    for &byte in bitmap.iter().take(full_bytes) {
        // Each zero bit is a free slot.
        // count_zeros() on a u8 returns at most 8, fits in u8.
        #[expect(clippy::cast_possible_truncation)]
        let zeros = byte.count_zeros() as u8;
        free += u32::from(zeros);
    }

    if remainder > 0 && full_bytes < bitmap.len() {
        let byte = bitmap[full_bytes];
        for bit in 0..remainder {
            if (byte >> bit) & 1 == 0 {
                free += 1;
            }
        }
    }

    free
}

/// Find the first free (zero) bit in the first `count` bits of `bitmap`,
/// starting from `start`.
#[must_use]
pub fn bitmap_find_free(bitmap: &[u8], count: u32, start: u32) -> Option<u32> {
    for idx in start..count {
        if !bitmap_get(bitmap, idx) {
            return Some(idx);
        }
    }
    // Wrap around: search from 0 to start.
    (0..start).find(|&idx| !bitmap_get(bitmap, idx))
}

/// Find `n` contiguous free bits in the first `count` bits of `bitmap`,
/// starting from `start`.
#[must_use]
pub fn bitmap_find_contiguous(bitmap: &[u8], count: u32, n: u32, start: u32) -> Option<u32> {
    if n == 0 {
        return Some(0);
    }

    // Pass 1: from `start` to `count`
    let mut run_start = start;
    let mut run_len = 0u32;
    for idx in start..count {
        if bitmap_get(bitmap, idx) {
            run_start = idx + 1;
            run_len = 0;
        } else {
            run_len += 1;
            if run_len >= n {
                return Some(run_start);
            }
        }
    }

    // Pass 2: wrap around from 0 to `start + n - 1` (to allow runs overlapping `start`)
    run_start = 0;
    run_len = 0;
    let pass2_end = start.saturating_add(n).saturating_sub(1).min(count);
    for idx in 0..pass2_end {
        if bitmap_get(bitmap, idx) {
            run_start = idx + 1;
            run_len = 0;
        } else {
            run_len += 1;
            if run_len >= n {
                return Some(run_start);
            }
        }
    }

    None
}

// ── Group stats ─────────────────────────────────────────────────────────────

/// Cached per-group statistics loaded from group descriptors.
#[derive(Debug, Clone)]
pub struct GroupStats {
    pub group: GroupNumber,
    pub free_blocks: u32,
    pub free_inodes: u32,
    pub used_dirs: u32,
    pub block_bitmap_block: BlockNumber,
    pub inode_bitmap_block: BlockNumber,
    pub inode_table_block: BlockNumber,
    pub flags: u16,
}

impl GroupStats {
    /// Create from a parsed group descriptor.
    #[must_use]
    pub fn from_group_desc(group: GroupNumber, gd: &Ext4GroupDesc) -> Self {
        Self {
            group,
            free_blocks: gd.free_blocks_count,
            free_inodes: gd.free_inodes_count,
            used_dirs: gd.used_dirs_count,
            block_bitmap_block: BlockNumber(gd.block_bitmap),
            inode_bitmap_block: BlockNumber(gd.inode_bitmap),
            inode_table_block: BlockNumber(gd.inode_table),
            flags: gd.flags,
        }
    }

    /// Whether the block bitmap is uninitialized (all free).
    #[must_use]
    pub fn block_bitmap_uninit(&self) -> bool {
        self.flags & GD_FLAG_BLOCK_UNINIT != 0
    }

    /// Whether the inode bitmap is uninitialized (all free).
    #[must_use]
    pub fn inode_bitmap_uninit(&self) -> bool {
        self.flags & GD_FLAG_INODE_UNINIT != 0
    }
}

// ── Allocation hint ─────────────────────────────────────────────────────────

/// Hint for the block allocator to guide placement decisions.
#[derive(Debug, Clone, Default)]
pub struct AllocHint {
    /// Preferred block group (e.g., same as parent inode).
    pub goal_group: Option<GroupNumber>,
    /// Preferred block number (e.g., adjacent to last allocated extent).
    pub goal_block: Option<BlockNumber>,
}

// ── Allocation result ───────────────────────────────────────────────────────

/// Result of a block allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAlloc {
    /// First allocated block.
    pub start: BlockNumber,
    /// Number of contiguous blocks allocated.
    pub count: u32,
}

/// Result of an inode allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InodeAlloc {
    /// Allocated inode number.
    pub ino: InodeNumber,
    /// Group the inode was allocated in.
    pub group: GroupNumber,
}

// ── Filesystem geometry ─────────────────────────────────────────────────────

/// Cached filesystem geometry needed by the allocator.
#[derive(Debug, Clone)]
pub struct FsGeometry {
    pub blocks_per_group: u32,
    pub inodes_per_group: u32,
    pub block_size: u32,
    pub total_blocks: u64,
    pub total_inodes: u32,
    pub first_data_block: u32,
    pub group_count: u32,
    pub inode_size: u16,
}

impl FsGeometry {
    /// Derive geometry from a parsed superblock.
    #[must_use]
    pub fn from_superblock(sb: &Ext4Superblock) -> Self {
        let group_count = if sb.blocks_per_group > 0 {
            let full = sb.blocks_count / u64::from(sb.blocks_per_group);
            let remainder = sb.blocks_count % u64::from(sb.blocks_per_group);
            let count = full + u64::from(remainder > 0);
            // Saturate at u32::MAX; geometry validation catches oversized values.
            u32::try_from(count).unwrap_or(u32::MAX)
        } else {
            0
        };
        Self {
            blocks_per_group: sb.blocks_per_group,
            inodes_per_group: sb.inodes_per_group,
            block_size: sb.block_size,
            total_blocks: sb.blocks_count,
            total_inodes: sb.inodes_count,
            first_data_block: sb.first_data_block,
            group_count,
            inode_size: sb.inode_size,
        }
    }

    /// Number of blocks in a specific group (last group may be shorter).
    #[must_use]
    #[expect(clippy::cast_possible_truncation)]
    pub fn blocks_in_group(&self, group: GroupNumber) -> u32 {
        let group_start = u64::from(self.first_data_block)
            + u64::from(group.0) * u64::from(self.blocks_per_group);
        let remaining = self.total_blocks.saturating_sub(group_start);
        if remaining >= u64::from(self.blocks_per_group) {
            self.blocks_per_group
        } else {
            remaining as u32
        }
    }

    /// Number of inodes in a specific group (last group may be shorter).
    #[must_use]
    #[expect(clippy::cast_possible_truncation)]
    pub fn inodes_in_group(&self, group: GroupNumber) -> u32 {
        let inode_start = u64::from(group.0) * u64::from(self.inodes_per_group);
        let remaining = u64::from(self.total_inodes).saturating_sub(inode_start);
        if remaining >= u64::from(self.inodes_per_group) {
            self.inodes_per_group
        } else {
            remaining as u32
        }
    }

    /// Absolute block number for a relative block within a group.
    #[must_use]
    pub fn group_block_to_absolute(&self, group: GroupNumber, rel_block: u32) -> BlockNumber {
        let abs = u64::from(self.first_data_block)
            + u64::from(group.0) * u64::from(self.blocks_per_group)
            + u64::from(rel_block);
        BlockNumber(abs)
    }

    /// Convert absolute block to (group, relative_block).
    #[must_use]
    #[expect(clippy::cast_possible_truncation)]
    pub fn absolute_to_group_block(&self, block: BlockNumber) -> (GroupNumber, u32) {
        let rel = block.0.saturating_sub(u64::from(self.first_data_block));
        let group = (rel / u64::from(self.blocks_per_group)) as u32;
        let offset = (rel % u64::from(self.blocks_per_group)) as u32;
        (GroupNumber(group), offset)
    }
}

// ── On-disk persistence context ─────────────────────────────────────────────

/// Context needed to persist allocator accounting changes to disk.
///
/// When provided to allocation/free operations, group descriptor counters are
/// written back to the device after bitmap updates, keeping on-disk metadata
/// self-consistent.
#[derive(Debug, Clone)]
pub struct PersistCtx {
    /// Block number of the first group descriptor table block.
    /// Group descriptors are packed contiguously starting here.
    pub gdt_block: BlockNumber,
    /// On-disk group descriptor size (32 or 64).
    pub desc_size: u16,
    /// Whether metadata_csum is enabled (triggers checksum stamping).
    pub has_metadata_csum: bool,
    /// CRC32C seed for metadata_csum (from superblock).
    pub csum_seed: u32,
}

/// Determine which relative block offsets within a group are reserved metadata
/// and must never be allocated as data blocks.
///
/// Returns a sorted `Vec` of relative block offsets within the group that are
/// occupied by: the superblock copy, the group descriptor table, the block
/// bitmap, the inode bitmap, and the inode table.
#[must_use]
pub fn reserved_blocks_in_group(
    geo: &FsGeometry,
    groups: &[GroupStats],
    group: GroupNumber,
) -> Vec<u32> {
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Vec::new();
    }

    let gs = &groups[gidx];
    let group_start =
        u64::from(geo.first_data_block) + u64::from(group.0) * u64::from(geo.blocks_per_group);
    let blocks_in_group = geo.blocks_in_group(group);
    let mut reserved = Vec::new();

    // Helper: convert absolute block to relative offset in this group,
    // and add to reserved if it falls within the group.
    let mut add_abs = |abs: u64| {
        if abs >= group_start {
            let rel = abs - group_start;
            if rel < u64::from(blocks_in_group) {
                #[expect(clippy::cast_possible_truncation)]
                reserved.push(rel as u32);
            }
        }
    };

    // Block bitmap, inode bitmap.
    add_abs(gs.block_bitmap_block.0);
    add_abs(gs.inode_bitmap_block.0);

    // Inode table spans multiple blocks.
    if geo.inodes_per_group > 0 && geo.inode_size > 0 {
        let inode_table_blocks = (u64::from(geo.inodes_per_group) * u64::from(geo.inode_size))
            .div_ceil(u64::from(geo.block_size));
        for i in 0..inode_table_blocks {
            add_abs(gs.inode_table_block.0 + i);
        }
    }

    reserved.sort_unstable();
    reserved.dedup();
    reserved
}

/// Check if a relative block offset in a group is reserved.
#[must_use]
fn is_reserved(reserved: &[u32], rel_block: u32) -> bool {
    reserved.binary_search(&rel_block).is_ok()
}

/// Persist a group descriptor's counter fields back to the on-disk GDT.
///
/// Reads the GDT block containing `group`, patches the free_blocks/inodes/dirs
/// fields, recomputes the checksum (if enabled), and writes the block back.
fn persist_group_desc(
    cx: &Cx,
    dev: &dyn BlockDevice,
    pctx: &PersistCtx,
    group: GroupNumber,
    stats: &GroupStats,
) -> Result<()> {
    let ds = usize::from(pctx.desc_size);
    let descs_per_block = dev.block_size() as usize / ds;
    let gdt_block_idx = group.0 as usize / descs_per_block;
    let offset_in_block = (group.0 as usize % descs_per_block) * ds;

    let block_num = BlockNumber(
        pctx.gdt_block
            .0
            .checked_add(gdt_block_idx as u64)
            .ok_or_else(|| FfsError::InvalidGeometry("GDT block number overflow".into()))?,
    );
    let raw = dev.read_block(cx, block_num)?;
    let mut buf = raw.as_slice().to_vec();

    // Build a temporary Ext4GroupDesc with updated counters and serialize.
    // Read existing descriptor to preserve fields we don't track.
    let existing = Ext4GroupDesc::parse_from_bytes(&buf[offset_in_block..], pctx.desc_size)
        .map_err(|e| FfsError::Format(format!("GDT parse: {e}")))?;

    let updated = Ext4GroupDesc {
        free_blocks_count: stats.free_blocks,
        free_inodes_count: stats.free_inodes,
        used_dirs_count: stats.used_dirs,
        ..existing
    };

    updated
        .write_to_bytes(&mut buf[offset_in_block..], pctx.desc_size)
        .map_err(|e| FfsError::Format(format!("GDT write: {e}")))?;

    if pctx.has_metadata_csum {
        ffs_ondisk::ext4::stamp_group_desc_checksum(
            &mut buf[offset_in_block..offset_in_block + ds],
            pctx.csum_seed,
            group.0,
            pctx.desc_size,
        );
    }

    dev.write_block(cx, block_num, &buf)?;
    Ok(())
}

// ── Block allocator ─────────────────────────────────────────────────────────

/// Allocate `count` contiguous blocks, using `hint` for goal-directed placement.
///
/// Strategy:
/// 1. Try the goal group/block if specified.
/// 2. Try nearby groups.
/// 3. Scan all groups for best fit.
pub fn alloc_blocks(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    count: u32,
    hint: &AllocHint,
) -> Result<BlockAlloc> {
    cx_checkpoint(cx)?;

    if count == 0 {
        return Err(FfsError::Format("cannot allocate 0 blocks".into()));
    }

    // Determine goal group.
    let goal_group = hint
        .goal_group
        .or_else(|| hint.goal_block.map(|b| geo.absolute_to_group_block(b).0))
        .unwrap_or(GroupNumber(0));

    // Try goal group first.
    if let Some(alloc) = try_alloc_in_group(cx, dev, geo, groups, goal_group, count, hint)? {
        return Ok(alloc);
    }

    // Try nearby groups (within 8 groups of goal).
    for delta in 1..=8u32 {
        for dir in [1i64, -1i64] {
            let g = i64::from(goal_group.0) + dir * i64::from(delta);
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            if g >= 0 && (g as u32) < geo.group_count {
                let group = GroupNumber(g as u32);
                if let Some(alloc) = try_alloc_in_group(cx, dev, geo, groups, group, count, hint)? {
                    return Ok(alloc);
                }
            }
        }
    }

    // Scan all groups.
    for g in 0..geo.group_count {
        let group = GroupNumber(g);
        if group == goal_group {
            continue;
        }
        if let Some(alloc) = try_alloc_in_group(cx, dev, geo, groups, group, count, hint)? {
            return Ok(alloc);
        }
    }

    Err(FfsError::NoSpace)
}

/// Try to allocate `count` blocks in a specific group.
fn try_alloc_in_group(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    group: GroupNumber,
    count: u32,
    hint: &AllocHint,
) -> Result<Option<BlockAlloc>> {
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Ok(None);
    }

    let gs = &groups[gidx];
    if gs.free_blocks < count {
        return Ok(None);
    }

    let blocks_in_group = geo.blocks_in_group(group);

    // Read the block bitmap.
    let bitmap_buf = dev.read_block(cx, gs.block_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    // Determine start position for search.
    let start = hint.goal_block.map_or(0, |goal| {
        let (g, off) = geo.absolute_to_group_block(goal);
        if g == group { off } else { 0 }
    });

    // Try to find contiguous free blocks.
    let found = if count == 1 {
        bitmap_find_free(&bitmap, blocks_in_group, start).map(|idx| (idx, 1))
    } else {
        bitmap_find_contiguous(&bitmap, blocks_in_group, count, start).map(|idx| (idx, count))
    };

    if let Some((rel_start, alloc_count)) = found {
        // Mark blocks as allocated.
        for i in rel_start..rel_start + alloc_count {
            bitmap_set(&mut bitmap, i);
        }

        // Write bitmap back.
        dev.write_block(cx, gs.block_bitmap_block, &bitmap)?;

        // Update group stats.
        groups[gidx].free_blocks = groups[gidx].free_blocks.saturating_sub(alloc_count);

        let abs_start = geo.group_block_to_absolute(group, rel_start);
        Ok(Some(BlockAlloc {
            start: abs_start,
            count: alloc_count,
        }))
    } else {
        Ok(None)
    }
}

/// Free `count` contiguous blocks starting at `start`.
pub fn free_blocks(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    start: BlockNumber,
    count: u32,
) -> Result<()> {
    cx_checkpoint(cx)?;

    let (group, rel_start) = geo.absolute_to_group_block(start);
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Err(FfsError::Corruption {
            block: start.0,
            detail: "free_blocks: group out of range".into(),
        });
    }

    if rel_start.saturating_add(count) > geo.blocks_in_group(group) {
        return Err(FfsError::Corruption {
            block: start.0,
            detail: "free_blocks: extent crosses block group boundary".into(),
        });
    }

    let gs = &groups[gidx];
    let bitmap_buf = dev.read_block(cx, gs.block_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    for i in rel_start..rel_start + count {
        bitmap_clear(&mut bitmap, i);
    }

    dev.write_block(cx, gs.block_bitmap_block, &bitmap)?;
    groups[gidx].free_blocks = groups[gidx].free_blocks.saturating_add(count);
    Ok(())
}

// ── Persistent block allocator ──────────────────────────────────────────────

/// Allocate `count` contiguous data blocks with full on-disk accounting.
///
/// Like [`alloc_blocks`], but additionally:
/// - Skips reserved metadata blocks (bitmaps, inode tables, GDT blocks).
/// - Writes updated group descriptor counters back to the device.
///
/// Returns the total number of free blocks delta for the caller to update
/// superblock counters at commit time.
pub fn alloc_blocks_persist(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    count: u32,
    hint: &AllocHint,
    pctx: &PersistCtx,
) -> Result<BlockAlloc> {
    cx_checkpoint(cx)?;

    if count == 0 {
        return Err(FfsError::Format("cannot allocate 0 blocks".into()));
    }

    let goal_group = hint
        .goal_group
        .or_else(|| hint.goal_block.map(|b| geo.absolute_to_group_block(b).0))
        .unwrap_or(GroupNumber(0));

    // Try goal group first.
    if let Some(alloc) = try_alloc_safe(cx, dev, geo, groups, goal_group, count, hint, pctx)? {
        return Ok(alloc);
    }

    // Try nearby groups (within 8 groups of goal).
    for delta in 1..=8u32 {
        for dir in [1i64, -1i64] {
            let g = i64::from(goal_group.0) + dir * i64::from(delta);
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            if g >= 0 && (g as u32) < geo.group_count {
                let group = GroupNumber(g as u32);
                if let Some(alloc) = try_alloc_safe(cx, dev, geo, groups, group, count, hint, pctx)?
                {
                    return Ok(alloc);
                }
            }
        }
    }

    // Scan all groups.
    for g in 0..geo.group_count {
        let group = GroupNumber(g);
        if group == goal_group {
            continue;
        }
        if let Some(alloc) = try_alloc_safe(cx, dev, geo, groups, group, count, hint, pctx)? {
            return Ok(alloc);
        }
    }

    Err(FfsError::NoSpace)
}

/// Try to allocate `count` blocks in a group, skipping reserved blocks and
/// persisting group descriptor updates.
#[expect(clippy::too_many_arguments)]
fn try_alloc_safe(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    group: GroupNumber,
    count: u32,
    hint: &AllocHint,
    pctx: &PersistCtx,
) -> Result<Option<BlockAlloc>> {
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Ok(None);
    }

    if groups[gidx].free_blocks < count {
        return Ok(None);
    }

    let blocks_in_group = geo.blocks_in_group(group);
    let reserved = reserved_blocks_in_group(geo, groups, group);

    let bitmap_buf = dev.read_block(cx, groups[gidx].block_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    // Ensure all reserved blocks are marked as allocated in the bitmap.
    for &r in &reserved {
        bitmap_set(&mut bitmap, r);
    }

    let start = hint.goal_block.map_or(0, |goal| {
        let (g, off) = geo.absolute_to_group_block(goal);
        if g == group { off } else { 0 }
    });

    // Find free blocks, respecting reserved bits now set in the bitmap.
    let found = if count == 1 {
        bitmap_find_free(&bitmap, blocks_in_group, start).map(|idx| (idx, 1))
    } else {
        bitmap_find_contiguous(&bitmap, blocks_in_group, count, start).map(|idx| (idx, count))
    };

    if let Some((rel_start, alloc_count)) = found {
        // Verify no allocated block is reserved.
        for i in rel_start..rel_start + alloc_count {
            if is_reserved(&reserved, i) {
                return Err(FfsError::Corruption {
                    block: geo.group_block_to_absolute(group, i).0,
                    detail: "alloc would overlap reserved metadata block".into(),
                });
            }
        }

        // Mark blocks as allocated.
        for i in rel_start..rel_start + alloc_count {
            bitmap_set(&mut bitmap, i);
        }

        dev.write_block(cx, groups[gidx].block_bitmap_block, &bitmap)?;
        groups[gidx].free_blocks = groups[gidx].free_blocks.saturating_sub(alloc_count);

        // Persist group descriptor.
        persist_group_desc(cx, dev, pctx, group, &groups[gidx])?;

        let abs_start = geo.group_block_to_absolute(group, rel_start);
        Ok(Some(BlockAlloc {
            start: abs_start,
            count: alloc_count,
        }))
    } else {
        Ok(None)
    }
}

/// Free `count` contiguous blocks with full on-disk accounting.
///
/// Like [`free_blocks`], but additionally:
/// - Validates that freed blocks are not reserved metadata.
/// - Validates that freed blocks are currently allocated.
/// - Writes updated group descriptor counters back to the device.
pub fn free_blocks_persist(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    start: BlockNumber,
    count: u32,
    pctx: &PersistCtx,
) -> Result<()> {
    cx_checkpoint(cx)?;

    let (group, rel_start) = geo.absolute_to_group_block(start);
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Err(FfsError::Corruption {
            block: start.0,
            detail: "free_blocks_persist: group out of range".into(),
        });
    }

    if rel_start.saturating_add(count) > geo.blocks_in_group(group) {
        return Err(FfsError::Corruption {
            block: start.0,
            detail: "free_blocks_persist: extent crosses block group boundary".into(),
        });
    }

    let reserved = reserved_blocks_in_group(geo, groups, group);

    // Validate none of the blocks being freed are reserved.
    for i in rel_start..rel_start + count {
        if is_reserved(&reserved, i) {
            return Err(FfsError::Corruption {
                block: geo.group_block_to_absolute(group, i).0,
                detail: "attempt to free reserved metadata block".into(),
            });
        }
    }

    let bitmap_buf = dev.read_block(cx, groups[gidx].block_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    // Validate all blocks are currently allocated (double-free detection).
    for i in rel_start..rel_start + count {
        if !bitmap_get(&bitmap, i) {
            return Err(FfsError::Corruption {
                block: geo.group_block_to_absolute(group, i).0,
                detail: "double-free: block already free in bitmap".into(),
            });
        }
    }

    for i in rel_start..rel_start + count {
        bitmap_clear(&mut bitmap, i);
    }

    dev.write_block(cx, groups[gidx].block_bitmap_block, &bitmap)?;
    groups[gidx].free_blocks = groups[gidx].free_blocks.saturating_add(count);

    // Persist group descriptor.
    persist_group_desc(cx, dev, pctx, group, &groups[gidx])?;

    Ok(())
}

// ── Inode allocator (Orlov) ─────────────────────────────────────────────────

/// Allocate an inode using the Orlov strategy.
///
/// - Directories: spread across groups (prefer groups with above-average free
///   inodes AND free blocks, fewest directories).
/// - Files: co-locate with parent directory's group.
pub fn alloc_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    parent_group: GroupNumber,
    is_directory: bool,
) -> Result<InodeAlloc> {
    cx_checkpoint(cx)?;

    let target_group = if is_directory {
        orlov_choose_group_for_dir(geo, groups)?
    } else {
        // Files: try parent group first, then nearby.
        parent_group
    };

    // Try target group, then scan.
    if let Some(alloc) = try_alloc_inode_in_group(cx, dev, geo, groups, target_group)? {
        return Ok(alloc);
    }

    // Scan all groups.
    for g in 0..geo.group_count {
        let group = GroupNumber(g);
        if group == target_group {
            continue;
        }
        if let Some(alloc) = try_alloc_inode_in_group(cx, dev, geo, groups, group)? {
            return Ok(alloc);
        }
    }

    Err(FfsError::NoSpace)
}

/// Orlov: choose a group for a new directory.
fn orlov_choose_group_for_dir(_geo: &FsGeometry, groups: &[GroupStats]) -> Result<GroupNumber> {
    if groups.is_empty() {
        return Err(FfsError::NoSpace);
    }

    // Compute averages.
    let total_free_inodes: u64 = groups.iter().map(|g| u64::from(g.free_inodes)).sum();
    let total_free_blocks: u64 = groups.iter().map(|g| u64::from(g.free_blocks)).sum();
    let total_dirs: u64 = groups.iter().map(|g| u64::from(g.used_dirs)).sum();
    let n = groups.len() as u64;
    let avg_free_inodes = total_free_inodes / n;
    let avg_free_blocks = total_free_blocks / n;
    let avg_dirs = total_dirs / n;

    // Find best group: above-average free inodes AND blocks, fewest dirs.
    let mut best_group = GroupNumber(0);
    let mut best_score = u64::MAX;

    for gs in groups {
        if u64::from(gs.free_inodes) < avg_free_inodes {
            continue;
        }
        if u64::from(gs.free_blocks) < avg_free_blocks {
            continue;
        }
        let score = u64::from(gs.used_dirs);
        if score < best_score || (score == best_score && score <= avg_dirs) {
            best_score = score;
            best_group = gs.group;
        }
    }

    // Fallback: any group with free inodes.
    if best_score == u64::MAX {
        for gs in groups {
            if gs.free_inodes > 0 {
                return Ok(gs.group);
            }
        }
        return Err(FfsError::NoSpace);
    }

    Ok(best_group)
}

/// Try to allocate an inode in a specific group.
fn try_alloc_inode_in_group(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    group: GroupNumber,
) -> Result<Option<InodeAlloc>> {
    let gidx = group.0 as usize;
    if gidx >= groups.len() {
        return Ok(None);
    }

    let gs = &groups[gidx];
    if gs.free_inodes == 0 {
        return Ok(None);
    }

    let bitmap_buf = dev.read_block(cx, gs.inode_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    let found = bitmap_find_free(&bitmap, geo.inodes_per_group, 0);
    if let Some(idx) = found {
        bitmap_set(&mut bitmap, idx);
        dev.write_block(cx, gs.inode_bitmap_block, &bitmap)?;

        groups[gidx].free_inodes = groups[gidx].free_inodes.saturating_sub(1);

        // Compute absolute inode number: group * inodes_per_group + idx + 1.
        let ino = u64::from(group.0) * u64::from(geo.inodes_per_group) + u64::from(idx) + 1;
        Ok(Some(InodeAlloc {
            ino: InodeNumber(ino),
            group,
        }))
    } else {
        Ok(None)
    }
}

/// Allocate an inode using the Orlov strategy with full on-disk accounting.
pub fn alloc_inode_persist(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    parent_group: GroupNumber,
    is_directory: bool,
    pctx: &PersistCtx,
) -> Result<InodeAlloc> {
    cx_checkpoint(cx)?;

    let target_group = if is_directory {
        orlov_choose_group_for_dir(geo, groups)?
    } else {
        parent_group
    };

    if let Some(alloc) = try_alloc_inode_in_group_persist(cx, dev, geo, groups, target_group, pctx)?
    {
        return Ok(alloc);
    }

    for g in 0..geo.group_count {
        let group = GroupNumber(g);
        if group == target_group {
            continue;
        }
        if let Some(alloc) = try_alloc_inode_in_group_persist(cx, dev, geo, groups, group, pctx)? {
            return Ok(alloc);
        }
    }

    Err(FfsError::NoSpace)
}

/// Try to allocate an inode in a specific group with full on-disk accounting.
fn try_alloc_inode_in_group_persist(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    group: GroupNumber,
    pctx: &PersistCtx,
) -> Result<Option<InodeAlloc>> {
    let alloc = try_alloc_inode_in_group(cx, dev, geo, groups, group)?;
    if let Some(a) = alloc {
        persist_group_desc(cx, dev, pctx, group, &groups[group.0 as usize])?;
        Ok(Some(a))
    } else {
        Ok(None)
    }
}

/// Free an inode.
pub fn free_inode(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    ino: InodeNumber,
) -> Result<()> {
    cx_checkpoint(cx)?;

    // Compute group and index.
    let ino_zero = ino.0.checked_sub(1).ok_or_else(|| FfsError::Corruption {
        block: 0,
        detail: "inode number 0 is invalid".into(),
    })?;
    let group_idx_u64 = ino_zero / u64::from(geo.inodes_per_group);
    let group_idx = u32::try_from(group_idx_u64).map_err(|_| FfsError::Corruption {
        block: 0,
        detail: format!("free_inode: group index {group_idx_u64} exceeds u32"),
    })?;
    #[expect(clippy::cast_possible_truncation)]
    let bit_idx = (ino_zero % u64::from(geo.inodes_per_group)) as u32;
    let gidx = group_idx as usize;

    if gidx >= groups.len() {
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!("free_inode: group {group_idx} out of range"),
        });
    }

    let gs = &groups[gidx];
    let bitmap_buf = dev.read_block(cx, gs.inode_bitmap_block)?;
    let mut bitmap = bitmap_buf.as_slice().to_vec();

    if !bitmap_get(&bitmap, bit_idx) {
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!("double-free: inode {} already free in bitmap", ino.0),
        });
    }

    bitmap_clear(&mut bitmap, bit_idx);
    dev.write_block(cx, gs.inode_bitmap_block, &bitmap)?;
    groups[gidx].free_inodes = groups[gidx].free_inodes.saturating_add(1);
    Ok(())
}

/// Free an inode with full on-disk accounting.
pub fn free_inode_persist(
    cx: &Cx,
    dev: &dyn BlockDevice,
    geo: &FsGeometry,
    groups: &mut [GroupStats],
    ino: InodeNumber,
    pctx: &PersistCtx,
) -> Result<()> {
    free_inode(cx, dev, geo, groups, ino)?;
    let ino_zero = ino.0.saturating_sub(1);
    let group_idx_u64 = ino_zero / u64::from(geo.inodes_per_group);
    let group_idx = u32::try_from(group_idx_u64).map_err(|_| FfsError::Corruption {
        block: 0,
        detail: format!("free_inode_persist: group index {group_idx_u64} exceeds u32"),
    })?;
    persist_group_desc(
        cx,
        dev,
        pctx,
        GroupNumber(group_idx),
        &groups[group_idx as usize],
    )?;
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn cx_checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|_| FfsError::Cancelled)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(clippy::option_if_let_else)]
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
            if let Some(data) = blocks.get(&block.0) {
                Ok(BlockBuf::new(data.clone()))
            } else {
                Ok(BlockBuf::new(vec![0u8; self.block_size as usize]))
            }
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

    // ── Bitmap tests ────────────────────────────────────────────────────

    #[test]
    fn bitmap_get_set_clear() {
        let mut bm = vec![0u8; 4];
        assert!(!bitmap_get(&bm, 0));
        bitmap_set(&mut bm, 0);
        assert!(bitmap_get(&bm, 0));
        bitmap_clear(&mut bm, 0);
        assert!(!bitmap_get(&bm, 0));

        bitmap_set(&mut bm, 7);
        assert!(bitmap_get(&bm, 7));
        assert_eq!(bm[0], 0x80);

        bitmap_set(&mut bm, 8);
        assert!(bitmap_get(&bm, 8));
        assert_eq!(bm[1], 0x01);
    }

    #[test]
    fn bitmap_count_free_all_free() {
        let bm = vec![0u8; 2]; // 16 bits, all free
        assert_eq!(bitmap_count_free(&bm, 16), 16);
    }

    #[test]
    fn bitmap_count_free_some_allocated() {
        let mut bm = vec![0u8; 2];
        bitmap_set(&mut bm, 0);
        bitmap_set(&mut bm, 5);
        bitmap_set(&mut bm, 15);
        assert_eq!(bitmap_count_free(&bm, 16), 13);
    }

    #[test]
    fn bitmap_find_free_basic() {
        let mut bm = vec![0u8; 2];
        bitmap_set(&mut bm, 0);
        bitmap_set(&mut bm, 1);
        assert_eq!(bitmap_find_free(&bm, 16, 0), Some(2));
    }

    #[test]
    fn bitmap_find_free_wraps() {
        let mut bm = vec![0xFFu8; 2];
        bitmap_clear(&mut bm, 3);
        assert_eq!(bitmap_find_free(&bm, 16, 5), Some(3));
    }

    #[test]
    fn bitmap_find_contiguous_basic() {
        let mut bm = vec![0u8; 4];
        bitmap_set(&mut bm, 0);
        bitmap_set(&mut bm, 1);
        // Free: 2,3,4,5,... contiguous from 2
        assert_eq!(bitmap_find_contiguous(&bm, 32, 4, 0), Some(2));
    }

    #[test]
    fn bitmap_find_contiguous_none() {
        let mut bm = vec![0u8; 2];
        // Set every other bit: 0,2,4,6,8,10,12,14
        for i in (0..16).step_by(2) {
            bitmap_set(&mut bm, i);
        }
        // No 2-contiguous free bits.
        assert_eq!(bitmap_find_contiguous(&bm, 16, 2, 0), None);
    }

    // ── Geometry tests ──────────────────────────────────────────────────

    #[test]
    fn geometry_group_block_conversion() {
        let geo = make_geometry();
        let abs = geo.group_block_to_absolute(GroupNumber(1), 42);
        assert_eq!(abs, BlockNumber(8192 + 42));
        let (g, off) = geo.absolute_to_group_block(abs);
        assert_eq!(g, GroupNumber(1));
        assert_eq!(off, 42);
    }

    #[test]
    fn geometry_blocks_in_group() {
        let mut geo = make_geometry();
        assert_eq!(geo.blocks_in_group(GroupNumber(0)), 8192);
        // Last group might be shorter: 32768 - 3*8192 = 8192 (exact fit).
        assert_eq!(geo.blocks_in_group(GroupNumber(3)), 8192);

        // Make total not evenly divisible.
        geo.total_blocks = 30000;
        // Groups 0,1,2 have 8192 each = 24576. Group 3 has 30000-24576 = 5424.
        assert_eq!(geo.blocks_in_group(GroupNumber(3)), 5424);
    }

    #[test]
    fn geometry_inodes_in_group() {
        let mut geo = make_geometry();
        // 4 groups * 2048 inodes_per_group = 8192 total_inodes (exact fit)
        assert_eq!(geo.inodes_in_group(GroupNumber(0)), 2048);
        assert_eq!(geo.inodes_in_group(GroupNumber(3)), 2048);

        // Make total not evenly divisible: 7000 total inodes
        // Groups 0,1,2 have 2048 each = 6144. Group 3 has 7000-6144 = 856.
        geo.total_inodes = 7000;
        assert_eq!(geo.inodes_in_group(GroupNumber(0)), 2048);
        assert_eq!(geo.inodes_in_group(GroupNumber(2)), 2048);
        assert_eq!(geo.inodes_in_group(GroupNumber(3)), 856);
    }

    // ── Block allocation tests ──────────────────────────────────────────

    #[test]
    fn alloc_single_block() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default());
        assert!(result.is_ok());
        let alloc = result.unwrap();
        assert_eq!(alloc.count, 1);
        assert_eq!(groups[0].free_blocks, 8191);
    }

    #[test]
    fn alloc_contiguous_blocks() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let hint = AllocHint {
            goal_group: Some(GroupNumber(1)),
            ..Default::default()
        };
        let alloc = alloc_blocks(&cx, &dev, &geo, &mut groups, 4, &hint).unwrap();
        assert_eq!(alloc.count, 4);
        // Should be in group 1.
        let (g, _) = geo.absolute_to_group_block(alloc.start);
        assert_eq!(g, GroupNumber(1));
        assert_eq!(groups[1].free_blocks, 8188);
    }

    #[test]
    fn alloc_and_free_roundtrip() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let alloc = alloc_blocks(&cx, &dev, &geo, &mut groups, 3, &AllocHint::default()).unwrap();
        assert_eq!(groups[0].free_blocks, 8189);

        free_blocks(&cx, &dev, &geo, &mut groups, alloc.start, alloc.count).unwrap();
        assert_eq!(groups[0].free_blocks, 8192);
    }

    #[test]
    fn alloc_no_space_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Mark all groups as having 0 free blocks.
        for g in &mut groups {
            g.free_blocks = 0;
        }

        let result = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default());
        assert!(matches!(result, Err(FfsError::NoSpace)));
    }

    // ── Inode allocation tests ──────────────────────────────────────────

    #[test]
    fn alloc_inode_basic() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(0), false).unwrap();
        assert_eq!(result.ino, InodeNumber(1));
        assert_eq!(result.group, GroupNumber(0));
        assert_eq!(groups[0].free_inodes, 2047);
    }

    #[test]
    fn alloc_inode_directory_orlov() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Make group 0 have many dirs, group 2 have fewest.
        groups[0].used_dirs = 100;
        groups[1].used_dirs = 50;
        groups[2].used_dirs = 10;
        groups[3].used_dirs = 30;

        let result = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(0), true).unwrap();
        // Orlov should prefer group 2 (fewest dirs, above-average free).
        assert_eq!(result.group, GroupNumber(2));
    }

    #[test]
    fn alloc_and_free_inode_roundtrip() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(1), false).unwrap();
        assert_eq!(groups[1].free_inodes, 2047);

        free_inode(&cx, &dev, &geo, &mut groups, result.ino).unwrap();
        assert_eq!(groups[1].free_inodes, 2048);
    }

    #[test]
    fn alloc_inode_no_space() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        for g in &mut groups {
            g.free_inodes = 0;
        }

        let result = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(0), false);
        assert!(matches!(result, Err(FfsError::NoSpace)));
    }

    #[test]
    fn alloc_multiple_blocks_same_group() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let a1 = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()).unwrap();
        let a2 = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()).unwrap();
        // Second allocation should get the next free block.
        assert_eq!(a2.start.0, a1.start.0 + 1);
    }

    // ── Reserved block tests ───────────────────────────────────────────

    #[test]
    fn reserved_blocks_includes_bitmaps_and_inode_table() {
        let geo = make_geometry();
        let groups = make_groups(&geo);

        // Group 0: bitmap at relative 1, inode bitmap at 2, inode table at 3.
        // Inode table: 2048 inodes * 256 bytes / 4096 bytes = 128 blocks.
        let reserved = reserved_blocks_in_group(&geo, &groups, GroupNumber(0));

        // Should contain bitmap block (rel 1), inode bitmap (rel 2),
        // and inode table blocks (rel 3..3+128).
        assert!(reserved.contains(&1), "block bitmap should be reserved");
        assert!(reserved.contains(&2), "inode bitmap should be reserved");
        assert!(
            reserved.contains(&3),
            "inode table start should be reserved"
        );
        assert!(
            reserved.contains(&130),
            "inode table end (3+127) should be reserved"
        );
        assert!(
            !reserved.contains(&131),
            "block after inode table should NOT be reserved"
        );
        // Total: 1 (block bitmap) + 1 (inode bitmap) + 128 (inode table) = 130
        assert_eq!(reserved.len(), 130);
    }

    // ── Persistent allocator tests ─────────────────────────────────────

    fn make_persist_ctx() -> PersistCtx {
        PersistCtx {
            gdt_block: BlockNumber(50), // arbitrary GDT location
            desc_size: 32,
            has_metadata_csum: false,
            csum_seed: 0,
        }
    }

    fn seed_gdt_block(dev: &MemBlockDevice, pctx: &PersistCtx, groups: &[GroupStats]) {
        // Write a GDT block with group descriptors packed at desc_size intervals.
        let block_size = dev.block_size() as usize;
        let ds = usize::from(pctx.desc_size);
        let mut buf = vec![0u8; block_size];
        for (i, gs) in groups.iter().enumerate() {
            let offset = i * ds;
            if offset + ds > block_size {
                break;
            }
            let gd = Ext4GroupDesc {
                block_bitmap: gs.block_bitmap_block.0,
                inode_bitmap: gs.inode_bitmap_block.0,
                inode_table: gs.inode_table_block.0,
                free_blocks_count: gs.free_blocks,
                free_inodes_count: gs.free_inodes,
                used_dirs_count: gs.used_dirs,
                itable_unused: 0,
                flags: gs.flags,
                checksum: 0,
            };
            gd.write_to_bytes(&mut buf[offset..], pctx.desc_size)
                .unwrap();
        }
        let cx = test_cx();
        dev.write_block(&cx, pctx.gdt_block, &buf).unwrap();
    }

    #[test]
    fn alloc_persist_skips_reserved_and_updates_gdt() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();
        seed_gdt_block(&dev, &pctx, &groups);

        let alloc = alloc_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            1,
            &AllocHint::default(),
            &pctx,
        )
        .unwrap();

        // The first non-reserved block in group 0 should be allocated.
        // Reserved blocks: 1,2,3..130 (bitmap+inode bitmap+inode table).
        // Block 0 is free and not reserved, so it should be allocated first.
        assert_eq!(alloc.start, BlockNumber(0));

        // In-memory stats should be decremented.
        assert_eq!(groups[0].free_blocks, 8191);

        // On-disk GDT should also be updated.
        let gdt_raw = dev.read_block(&cx, pctx.gdt_block).unwrap();
        let gd = Ext4GroupDesc::parse_from_bytes(gdt_raw.as_slice(), pctx.desc_size).unwrap();
        assert_eq!(gd.free_blocks_count, 8191);
    }

    #[test]
    fn alloc_persist_never_allocates_reserved_metadata() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();
        seed_gdt_block(&dev, &pctx, &groups);

        // Pre-mark block 0 as allocated in the bitmap so the allocator
        // must skip it and find the next non-reserved free block.
        let mut bitmap = vec![0u8; 4096];
        bitmap_set(&mut bitmap, 0);
        dev.write_block(&cx, groups[0].block_bitmap_block, &bitmap)
            .unwrap();

        let alloc = alloc_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            1,
            &AllocHint::default(),
            &pctx,
        )
        .unwrap();

        // Blocks 1..130 are reserved (bitmap, inode bitmap, inode table).
        // The allocator should skip them and return block 131.
        let reserved = reserved_blocks_in_group(&geo, &groups, GroupNumber(0));
        let (_, rel) = geo.absolute_to_group_block(alloc.start);
        assert!(
            !is_reserved(&reserved, rel),
            "allocated block {} (rel {}) is reserved",
            alloc.start.0,
            rel
        );
        assert_eq!(rel, 131, "should allocate first non-reserved free block");
    }

    #[test]
    fn free_persist_detects_double_free() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();
        seed_gdt_block(&dev, &pctx, &groups);

        // Allocate a block.
        let alloc = alloc_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            1,
            &AllocHint::default(),
            &pctx,
        )
        .unwrap();

        // Free it.
        free_blocks_persist(&cx, &dev, &geo, &mut groups, alloc.start, 1, &pctx).unwrap();

        // Double-free should fail.
        let result = free_blocks_persist(&cx, &dev, &geo, &mut groups, alloc.start, 1, &pctx);
        assert!(result.is_err(), "double-free should return error");
    }

    #[test]
    fn free_persist_rejects_reserved_block() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();
        seed_gdt_block(&dev, &pctx, &groups);

        // Try to free the block bitmap block (reserved).
        let bitmap_block = groups[0].block_bitmap_block;
        let result = free_blocks_persist(&cx, &dev, &geo, &mut groups, bitmap_block, 1, &pctx);
        assert!(
            result.is_err(),
            "freeing a reserved metadata block should fail"
        );
    }

    #[test]
    fn alloc_and_free_persist_roundtrip() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();
        seed_gdt_block(&dev, &pctx, &groups);

        let original_free = groups[0].free_blocks;

        let alloc = alloc_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            3,
            &AllocHint::default(),
            &pctx,
        )
        .unwrap();
        assert_eq!(groups[0].free_blocks, original_free - 3);

        free_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            alloc.start,
            alloc.count,
            &pctx,
        )
        .unwrap();
        assert_eq!(groups[0].free_blocks, original_free);

        // Verify on-disk GDT matches.
        let gdt_raw = dev.read_block(&cx, pctx.gdt_block).unwrap();
        let gd = Ext4GroupDesc::parse_from_bytes(gdt_raw.as_slice(), pctx.desc_size).unwrap();
        assert_eq!(gd.free_blocks_count, original_free);
    }

    // ── bd-1xe.5: ext4 read path allocator bitmap tests ─────────────────

    // Allocator Bitmap Test 6: Read block bitmap — free/used status correct
    #[test]
    fn readpath_block_bitmap_free_used_correct() {
        let mut bm = vec![0u8; 128]; // 1024 bits

        // Mark blocks 0-9 as used (typical for metadata reservation).
        for i in 0..10 {
            bitmap_set(&mut bm, i);
        }

        // Verify used blocks report correctly.
        for i in 0..10 {
            assert!(bitmap_get(&bm, i), "block {i} should be used (allocated)");
        }

        // Verify free blocks report correctly.
        for i in 10..64 {
            assert!(!bitmap_get(&bm, i), "block {i} should be free");
        }
    }

    // Allocator Bitmap Test 7: Read inode bitmap — free/used status correct
    #[test]
    fn readpath_inode_bitmap_free_used_correct() {
        let mut bm = vec![0u8; 32]; // 256 bits (inodes_per_group)

        // Mark inodes 0-10 as allocated (root inode + reserved + first user inodes).
        for i in 0..11 {
            bitmap_set(&mut bm, i);
        }

        // Verify allocated inodes.
        for i in 0..11 {
            assert!(bitmap_get(&bm, i), "inode {i} should be allocated");
        }

        // Verify free inodes.
        for i in 11..32 {
            assert!(!bitmap_get(&bm, i), "inode {i} should be free");
        }

        // Free count should match.
        assert_eq!(
            bitmap_count_free(&bm, 256),
            256 - 11,
            "free inode count should be total minus allocated"
        );
    }

    // Allocator Bitmap Test 8: Free block count matches bitmap popcount
    #[test]
    fn readpath_free_block_count_matches_popcount() {
        let blocks_per_group: u32 = 8192;
        let mut bm = vec![0u8; (blocks_per_group / 8) as usize];

        // Allocate specific blocks: 0, 1, 2 (superblock/GDT), 100, 200, 500
        let allocated = [0, 1, 2, 100, 200, 500];
        for &b in &allocated {
            bitmap_set(&mut bm, b);
        }

        let free = bitmap_count_free(&bm, blocks_per_group);
        let expected_free = blocks_per_group - u32::try_from(allocated.len()).unwrap();
        assert_eq!(
            free,
            expected_free,
            "free count ({free}) should equal blocks_per_group ({blocks_per_group}) minus allocated ({})",
            allocated.len()
        );

        // Double-check by counting set bits manually.
        let used: u32 = (0..blocks_per_group)
            .filter(|&i| bitmap_get(&bm, i))
            .count()
            .try_into()
            .unwrap();
        assert_eq!(used, u32::try_from(allocated.len()).unwrap());
        assert_eq!(free + used, blocks_per_group);
    }

    // Allocator Bitmap Test 9: Reserved blocks excluded from free count
    #[test]
    fn readpath_reserved_blocks_excluded_from_free() {
        let blocks_per_group: u32 = 64;
        let mut bm = vec![0u8; (blocks_per_group / 8) as usize]; // 8 bytes

        // Reserve first 5 blocks (superblock, GDT, bitmaps, inode table).
        let reserved_count = 5_u32;
        for i in 0..reserved_count {
            bitmap_set(&mut bm, i);
        }

        let free = bitmap_count_free(&bm, blocks_per_group);
        assert_eq!(
            free,
            blocks_per_group - reserved_count,
            "reserved blocks should not count as free"
        );

        // Find first free block — should skip reserved.
        let first_free = bitmap_find_free(&bm, blocks_per_group, 0);
        assert_eq!(
            first_free,
            Some(reserved_count),
            "first free block should be after reserved area"
        );
    }

    // ── Error-path and boundary hardening tests ────────────────────────

    #[test]
    fn reserved_blocks_out_of_range_group_returns_empty() {
        let geo = make_geometry();
        let groups = make_groups(&geo);
        // Group 99 is well beyond the 4 groups we created.
        let reserved = reserved_blocks_in_group(&geo, &groups, GroupNumber(99));
        assert!(
            reserved.is_empty(),
            "out-of-range group should return empty"
        );
    }

    #[test]
    fn reserved_blocks_includes_bitmap_and_inode_table() {
        let geo = make_geometry();
        let groups = make_groups(&geo);

        let reserved = reserved_blocks_in_group(&geo, &groups, GroupNumber(0));
        // Group 0 has block_bitmap at 1, inode_bitmap at 2, inode_table at 3..130.
        // (2048 inodes * 256 bytes / 4096 block_size = 128 blocks for inode table)
        assert!(
            reserved.contains(&1),
            "block bitmap should be reserved, got: {reserved:?}"
        );
        assert!(
            reserved.contains(&2),
            "inode bitmap should be reserved, got: {reserved:?}"
        );
        assert!(
            reserved.contains(&3),
            "first inode table block should be reserved"
        );
        assert!(
            reserved.contains(&130),
            "last inode table block should be reserved"
        );
        assert!(
            !reserved.contains(&131),
            "block past inode table should not be reserved"
        );
    }

    #[test]
    fn free_blocks_group_out_of_range_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Block far beyond total_blocks → group out of range.
        let result = free_blocks(&cx, &dev, &geo, &mut groups, BlockNumber(1_000_000), 1);
        assert!(result.is_err());
    }

    #[test]
    fn free_blocks_persist_cross_boundary_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();

        // Try to free blocks that span across a group boundary.
        // Group 0 has 8192 blocks (0..8191). Block 8190 + count=5 crosses into group 1.
        let result = free_blocks_persist(&cx, &dev, &geo, &mut groups, BlockNumber(8190), 5, &pctx);
        assert!(result.is_err(), "cross-boundary free should fail");
    }

    #[test]
    fn alloc_blocks_zero_count_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = alloc_blocks(&cx, &dev, &geo, &mut groups, 0, &AllocHint::default());
        assert!(result.is_err(), "allocating 0 blocks should fail");
    }

    #[test]
    fn alloc_blocks_persist_zero_count_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);
        let pctx = make_persist_ctx();

        let result = alloc_blocks_persist(
            &cx,
            &dev,
            &geo,
            &mut groups,
            0,
            &AllocHint::default(),
            &pctx,
        );
        assert!(result.is_err(), "allocating 0 blocks (persist) should fail");
    }

    #[test]
    fn alloc_blocks_goal_block_in_different_group_falls_back() {
        // When goal_block maps to a different group than goal_group,
        // the search start should fall back to 0 within the goal group.
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let hint = AllocHint {
            goal_group: Some(GroupNumber(0)),
            goal_block: Some(BlockNumber(10000)), // This is in group 1
        };

        let alloc = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &hint).unwrap();
        // Should still allocate in group 0 (goal_group), starting from 0.
        let (group, _) = geo.absolute_to_group_block(alloc.start);
        assert_eq!(
            group,
            GroupNumber(0),
            "should allocate in goal group even when goal_block is in a different group"
        );
    }

    #[test]
    fn alloc_blocks_no_hint_uses_group_0() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let alloc = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()).unwrap();

        let (group, _) = geo.absolute_to_group_block(alloc.start);
        assert_eq!(group, GroupNumber(0), "no hint should default to group 0");
    }

    #[test]
    fn free_inode_zero_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = free_inode(&cx, &dev, &geo, &mut groups, InodeNumber(0));
        assert!(result.is_err(), "freeing inode 0 should fail");
    }

    #[test]
    fn free_inode_out_of_range_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        let result = free_inode(&cx, &dev, &geo, &mut groups, InodeNumber(100_000));
        assert!(result.is_err(), "freeing out-of-range inode should fail");
    }

    #[test]
    fn free_inode_double_free_returns_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Allocate then free twice.
        let alloc = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(0), false).unwrap();
        free_inode(&cx, &dev, &geo, &mut groups, alloc.ino).unwrap();
        let result = free_inode(&cx, &dev, &geo, &mut groups, alloc.ino);
        assert!(
            result.is_err(),
            "double-freeing an inode should return error"
        );
    }

    #[test]
    fn bitmap_find_free_start_at_count_wraps_to_zero() {
        // When start == count, the forward search is empty.
        // The backward search 0..start should find a free bit.
        let mut bm = vec![0u8; 4];
        let count = 32;
        // Fill bits 0..15, leave 16..31 free.
        for i in 0..16 {
            bitmap_set(&mut bm, i);
        }
        // start=count=32 → forward loop is empty, wrap-around finds bit 16.
        let result = bitmap_find_free(&bm, count, count);
        assert_eq!(result, Some(16), "should wrap around and find bit 16");
    }

    #[test]
    fn bitmap_find_free_start_beyond_count_returns_wrap_result() {
        let bm = vec![0u8; 4]; // all free
        // start > count → forward loop is empty, wraps to find 0.
        let result = bitmap_find_free(&bm, 32, 100);
        assert_eq!(result, Some(0), "start > count should wrap to bit 0");
    }

    #[test]
    fn bitmap_find_contiguous_n_zero_returns_zero() {
        let bm = vec![0xFF; 4]; // all set
        let result = bitmap_find_contiguous(&bm, 32, 0, 0);
        assert_eq!(result, Some(0), "finding 0 contiguous bits always succeeds");
    }

    #[test]
    fn bitmap_count_free_zero_count_returns_zero() {
        let bm = vec![0u8; 4]; // all free
        assert_eq!(bitmap_count_free(&bm, 0), 0, "count=0 should return 0");
    }

    #[test]
    fn geometry_blocks_in_last_group_shorter() {
        // When total_blocks is not evenly divisible by blocks_per_group,
        // the last group should be shorter.
        let geo = FsGeometry {
            blocks_per_group: 8192,
            inodes_per_group: 2048,
            block_size: 4096,
            total_blocks: 30000, // not evenly divisible: 3 full groups + partial
            total_inodes: 8192,
            first_data_block: 0,
            group_count: 4,
            inode_size: 256,
        };

        assert_eq!(geo.blocks_in_group(GroupNumber(0)), 8192);
        assert_eq!(geo.blocks_in_group(GroupNumber(1)), 8192);
        assert_eq!(geo.blocks_in_group(GroupNumber(2)), 8192);
        // Last group: 30000 - 3*8192 = 5424.
        assert_eq!(geo.blocks_in_group(GroupNumber(3)), 5424);
    }

    #[test]
    fn geometry_absolute_to_group_with_first_data_block() {
        let geo = FsGeometry {
            blocks_per_group: 8192,
            inodes_per_group: 2048,
            block_size: 4096,
            total_blocks: 32768,
            total_inodes: 8192,
            first_data_block: 1, // ext4 with 1K blocks has first_data_block=1
            group_count: 4,
            inode_size: 256,
        };

        // Block 1 should be in group 0, relative 0.
        let (g, off) = geo.absolute_to_group_block(BlockNumber(1));
        assert_eq!(g, GroupNumber(0));
        assert_eq!(off, 0);

        // Block 8193 should be in group 1, relative 0.
        let (g, off) = geo.absolute_to_group_block(BlockNumber(8193));
        assert_eq!(g, GroupNumber(1));
        assert_eq!(off, 0);
    }

    #[test]
    fn orlov_all_groups_exhausted_returns_nospace() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let geo = make_geometry();
        let mut groups = make_groups(&geo);

        // Exhaust all inodes.
        for g in &mut groups {
            g.free_inodes = 0;
        }

        let result = alloc_inode(&cx, &dev, &geo, &mut groups, GroupNumber(0), true);
        assert!(
            result.is_err(),
            "all groups exhausted should return NoSpace"
        );
    }

    // ── Edge-case hardening tests ──────────────────────────────────────

    #[test]
    fn group_stats_uninit_flags() {
        let mut gs = GroupStats {
            group: GroupNumber(0),
            free_blocks: 100,
            free_inodes: 50,
            used_dirs: 0,
            block_bitmap_block: BlockNumber(1),
            inode_bitmap_block: BlockNumber(2),
            inode_table_block: BlockNumber(3),
            flags: 0,
        };
        assert!(!gs.block_bitmap_uninit());
        assert!(!gs.inode_bitmap_uninit());

        gs.flags = 0x0001; // GD_FLAG_INODE_UNINIT
        assert!(!gs.block_bitmap_uninit());
        assert!(gs.inode_bitmap_uninit());

        gs.flags = 0x0002; // GD_FLAG_BLOCK_UNINIT
        assert!(gs.block_bitmap_uninit());
        assert!(!gs.inode_bitmap_uninit());

        gs.flags = 0x0003; // both
        assert!(gs.block_bitmap_uninit());
        assert!(gs.inode_bitmap_uninit());
    }

    #[test]
    fn geometry_coordinate_roundtrip_with_first_data_block() {
        let geo = FsGeometry {
            blocks_per_group: 32768,
            inodes_per_group: 8192,
            block_size: 4096,
            total_blocks: 131072,
            total_inodes: 32768,
            first_data_block: 1,
            group_count: 4,
            inode_size: 256,
        };
        // Group 0, rel 0 -> absolute = first_data_block + 0 = 1
        let abs = geo.group_block_to_absolute(GroupNumber(0), 0);
        assert_eq!(abs, BlockNumber(1));

        let (g, r) = geo.absolute_to_group_block(BlockNumber(1));
        assert_eq!(g, GroupNumber(0));
        assert_eq!(r, 0);

        // Group 1, rel 5 -> absolute = 1 + 32768 + 5 = 32774
        let abs2 = geo.group_block_to_absolute(GroupNumber(1), 5);
        assert_eq!(abs2, BlockNumber(32774));
        let (g2, r2) = geo.absolute_to_group_block(abs2);
        assert_eq!(g2, GroupNumber(1));
        assert_eq!(r2, 5);
    }

    #[test]
    fn geometry_inodes_in_last_group_may_be_smaller() {
        let geo = FsGeometry {
            blocks_per_group: 8192,
            inodes_per_group: 2048,
            block_size: 4096,
            total_blocks: 20000,
            total_inodes: 5000, // 2 full groups + partial
            first_data_block: 0,
            group_count: 3,
            inode_size: 256,
        };
        assert_eq!(geo.inodes_in_group(GroupNumber(0)), 2048);
        assert_eq!(geo.inodes_in_group(GroupNumber(1)), 2048);
        assert_eq!(geo.inodes_in_group(GroupNumber(2)), 904); // 5000 - 4096
    }

    #[test]
    fn alloc_hint_default_has_no_preferences() {
        let hint = AllocHint::default();
        assert!(hint.goal_group.is_none());
        assert!(hint.goal_block.is_none());
    }

    #[test]
    fn block_alloc_and_inode_alloc_equality() {
        let a = BlockAlloc {
            start: BlockNumber(10),
            count: 3,
        };
        let b = BlockAlloc {
            start: BlockNumber(10),
            count: 3,
        };
        assert_eq!(a, b);

        let ia = InodeAlloc {
            ino: InodeNumber(100),
            group: GroupNumber(2),
        };
        let ib = InodeAlloc {
            ino: InodeNumber(100),
            group: GroupNumber(2),
        };
        assert_eq!(ia, ib);
    }

    #[test]
    fn bitmap_full_count_free_is_zero() {
        let bitmap = [0xFF_u8; 4];
        assert_eq!(bitmap_count_free(&bitmap, 32), 0);
    }

    #[test]
    fn bitmap_empty_count_free_is_count() {
        let bitmap = [0x00_u8; 4];
        assert_eq!(bitmap_count_free(&bitmap, 32), 32);
        assert_eq!(bitmap_count_free(&bitmap, 16), 16);
    }

    #[test]
    fn bitmap_find_free_on_full_returns_none() {
        let bitmap = [0xFF_u8; 4];
        assert!(bitmap_find_free(&bitmap, 32, 0).is_none());
    }

    #[test]
    fn bitmap_find_contiguous_larger_than_available_returns_none() {
        let bitmap = [0x00_u8; 2]; // 16 bits free
        assert!(bitmap_find_contiguous(&bitmap, 16, 17, 0).is_none());
    }

    // ── Property-based tests (proptest) ────────────────────────────────

    use proptest::prelude::*;

    /// Strategy: generate a bitmap of 1..128 bytes with a valid bit count.
    fn bitmap_strat() -> impl Strategy<Value = (Vec<u8>, u32)> {
        (1_usize..128).prop_flat_map(|byte_len| {
            let max_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            (prop::collection::vec(any::<u8>(), byte_len), 1..=max_bits)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn proptest_bitmap_set_get_roundtrip(
            byte_len in 1_usize..64,
            idx_seed in any::<u32>(),
        ) {
            let total_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            let idx = idx_seed % total_bits;
            let mut bm = vec![0u8; byte_len];
            prop_assert!(!bitmap_get(&bm, idx));
            bitmap_set(&mut bm, idx);
            prop_assert!(bitmap_get(&bm, idx));
            bitmap_clear(&mut bm, idx);
            prop_assert!(!bitmap_get(&bm, idx));
        }

        #[test]
        fn proptest_bitmap_count_free_consistency((ref bm, count) in bitmap_strat()) {
            let free = bitmap_count_free(bm, count);
            // Manual count for verification.
            let manual_free =
                (0..count).fold(0_u32, |acc, i| acc + u32::from(!bitmap_get(bm, i)));
            prop_assert_eq!(free, manual_free);
            // Ones + zeros = total.
            let used = (0..count).fold(0_u32, |acc, i| acc + u32::from(bitmap_get(bm, i)));
            prop_assert_eq!(free + used, count);
        }

        #[test]
        fn proptest_bitmap_find_free_returns_zero_bit(
            (ref bm, count) in bitmap_strat(),
            start_seed in any::<u32>(),
        ) {
            let start = start_seed % count;
            if let Some(pos) = bitmap_find_free(bm, count, start) {
                prop_assert!(pos < count, "found pos {} >= count {}", pos, count);
                prop_assert!(!bitmap_get(bm, pos), "bit {} is set but find_free returned it", pos);
            } else {
                // All bits should be set.
                let free = bitmap_count_free(bm, count);
                prop_assert_eq!(free, 0, "find_free returned None but {} bits are free", free);
            }
        }

        #[test]
        fn proptest_bitmap_find_contiguous_valid_run(
            (ref bm, count) in bitmap_strat(),
            n in 1_u32..32,
            start_seed in any::<u32>(),
        ) {
            let start = start_seed % count;
            if let Some(pos) = bitmap_find_contiguous(bm, count, n, start) {
                prop_assert!(pos + n <= count, "run [{}, {}) exceeds count {}", pos, pos + n, count);
                for i in pos..pos + n {
                    prop_assert!(
                        !bitmap_get(bm, i),
                        "bit {} in contiguous run [{}, {}) is set",
                        i, pos, pos + n,
                    );
                }
            }
        }

        #[test]
        fn proptest_alloc_free_roundtrip_preserves_free_count(
            num_allocs in 1_u32..8,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);
            let original_free: u32 = groups.iter().map(|g| g.free_blocks).sum();

            let mut allocations = Vec::new();
            for _ in 0..num_allocs {
                match alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()) {
                    Ok(a) => allocations.push(a),
                    Err(_) => break,
                }
            }

            let after_alloc_free: u32 = groups.iter().map(|g| g.free_blocks).sum();
            let allocated_count = u32::try_from(allocations.len())
                .expect("allocation count fits in u32 for test bounds");
            prop_assert_eq!(
                after_alloc_free,
                original_free - allocated_count,
                "free count after alloc: expected {}, got {}",
                original_free - allocated_count,
                after_alloc_free,
            );

            // Free all allocated blocks.
            for a in &allocations {
                free_blocks(&cx, &dev, &geo, &mut groups, a.start, a.count).unwrap();
            }

            let final_free: u32 = groups.iter().map(|g| g.free_blocks).sum();
            prop_assert_eq!(
                final_free,
                original_free,
                "free count after free: expected {}, got {}",
                original_free,
                final_free,
            );
        }

        /// Multi-block alloc/free roundtrip with varying block counts.
        #[test]
        fn proptest_multi_block_alloc_free_roundtrip(
            block_count in 1_u32..16,
            num_allocs in 1_u32..5,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);
            let original_free: u32 = groups.iter().map(|g| g.free_blocks).sum();

            let mut allocations = Vec::new();
            let mut total_allocated = 0_u32;
            for _ in 0..num_allocs {
                match alloc_blocks(&cx, &dev, &geo, &mut groups, block_count, &AllocHint::default()) {
                    Ok(a) => {
                        total_allocated += a.count;
                        allocations.push(a);
                    }
                    Err(_) => break,
                }
            }

            let after_free: u32 = groups.iter().map(|g| g.free_blocks).sum();
            prop_assert_eq!(
                after_free,
                original_free - total_allocated,
            );

            for a in &allocations {
                free_blocks(&cx, &dev, &geo, &mut groups, a.start, a.count).unwrap();
            }

            let final_free: u32 = groups.iter().map(|g| g.free_blocks).sum();
            prop_assert_eq!(final_free, original_free);
        }

        /// Allocated blocks are actually marked in the bitmap.
        #[test]
        fn proptest_alloc_marks_bitmap_bits(
            block_count in 1_u32..8,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);

            let alloc = alloc_blocks(
                &cx, &dev, &geo, &mut groups, block_count, &AllocHint::default(),
            ).unwrap();

            // Read the bitmap for the group that owns the allocation.
            let (group, rel_start) = geo.absolute_to_group_block(alloc.start);
            let bitmap_block = groups[group.0 as usize].block_bitmap_block;
            let bm = dev.read_block(&cx, bitmap_block).unwrap();

            for i in 0..alloc.count {
                let bit = rel_start + i;
                prop_assert!(
                    bitmap_get(bm.as_slice(), bit),
                    "bit {} in group {} should be set after alloc",
                    bit, group.0,
                );
            }
        }

        /// No two allocations ever overlap.
        #[test]
        fn proptest_alloc_no_overlaps(
            num_allocs in 2_u32..12,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);

            let mut allocations = Vec::new();
            for _ in 0..num_allocs {
                match alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()) {
                    Ok(a) => allocations.push(a),
                    Err(_) => break,
                }
            }

            // Check no two allocations share the same block.
            for i in 0..allocations.len() {
                for j in (i + 1)..allocations.len() {
                    let a = &allocations[i];
                    let b = &allocations[j];
                    let a_end = a.start.0 + u64::from(a.count);
                    let b_end = b.start.0 + u64::from(b.count);
                    let overlaps = a.start.0 < b_end && b.start.0 < a_end;
                    prop_assert!(
                        !overlaps,
                        "allocations {} [{}, {}) and {} [{}, {}) overlap",
                        i, a.start.0, a_end,
                        j, b.start.0, b_end,
                    );
                }
            }
        }

        /// bitmap_find_contiguous with wraparound: if a run is found, it must
        /// consist of contiguous zero bits regardless of the start position.
        #[test]
        fn proptest_find_contiguous_wraparound_valid(
            byte_len in 1_usize..32,
            fill_percent in 0_u8..80,
            n in 1_u32..8,
            start_seed in any::<u32>(),
        ) {
            let total_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            let mut bm = vec![0u8; byte_len];

            // Randomly fill some fraction of bits.
            let bits_to_set = (u32::from(fill_percent) * total_bits) / 100;
            // Deterministic pattern based on fill fraction.
            for i in 0..bits_to_set.min(total_bits) {
                let bit = (i.wrapping_mul(7) + i.wrapping_mul(13)) % total_bits;
                bitmap_set(&mut bm, bit);
            }

            let start = start_seed % total_bits;
            if let Some(pos) = bitmap_find_contiguous(&bm, total_bits, n, start) {
                // Entire run must be within bounds and all bits clear.
                prop_assert!(pos + n <= total_bits);
                for i in pos..pos + n {
                    prop_assert!(
                        !bitmap_get(&bm, i),
                        "bit {} in run [{}, {}) should be clear (start={})",
                        i, pos, pos + n, start,
                    );
                }
            }
        }

        /// GroupStats flag methods correctly detect UNINIT flags.
        #[test]
        fn proptest_groupstats_uninit_flags(flags in any::<u16>()) {
            let gs = GroupStats {
                group: GroupNumber(0),
                free_blocks: 100,
                free_inodes: 100,
                used_dirs: 0,
                block_bitmap_block: BlockNumber(1),
                inode_bitmap_block: BlockNumber(2),
                inode_table_block: BlockNumber(3),
                flags,
            };
            prop_assert_eq!(gs.block_bitmap_uninit(), flags & GD_FLAG_BLOCK_UNINIT != 0);
            prop_assert_eq!(gs.inode_bitmap_uninit(), flags & GD_FLAG_INODE_UNINIT != 0);
        }

        /// Alloc fails with NoSpace when all groups are completely full.
        #[test]
        fn proptest_alloc_fails_when_all_full(
            block_count in 1_u32..8,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups: Vec<GroupStats> = (0..geo.group_count)
                .map(|g| GroupStats {
                    group: GroupNumber(g),
                    free_blocks: 0,
                    free_inodes: 0,
                    used_dirs: 100,
                    block_bitmap_block: BlockNumber(u64::from(g) * 100 + 1),
                    inode_bitmap_block: BlockNumber(u64::from(g) * 100 + 2),
                    inode_table_block: BlockNumber(u64::from(g) * 100 + 3),
                    flags: 0,
                })
                .collect();

            let result = alloc_blocks(
                &cx, &dev, &geo, &mut groups, block_count, &AllocHint::default(),
            );
            prop_assert!(result.is_err(), "alloc should fail when all groups are full");
        }

        /// Inode alloc fails when all inodes exhausted.
        #[test]
        fn proptest_inode_alloc_fails_when_exhausted(is_dir in any::<bool>()) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups: Vec<GroupStats> = (0..geo.group_count)
                .map(|g| GroupStats {
                    group: GroupNumber(g),
                    free_blocks: 1000,
                    free_inodes: 0,
                    used_dirs: 100,
                    block_bitmap_block: BlockNumber(u64::from(g) * 100 + 1),
                    inode_bitmap_block: BlockNumber(u64::from(g) * 100 + 2),
                    inode_table_block: BlockNumber(u64::from(g) * 100 + 3),
                    flags: 0,
                })
                .collect();

            let result = alloc_inode(
                &cx, &dev, &geo, &mut groups, GroupNumber(0), is_dir,
            );
            prop_assert!(result.is_err(), "inode alloc should fail when all inodes exhausted");
        }

        /// Bitmap with all bits set: find_free returns None.
        #[test]
        fn proptest_bitmap_full_find_free_none(
            byte_len in 1_usize..64,
            start_seed in any::<u32>(),
        ) {
            let total_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            let bm = vec![0xFF_u8; byte_len];
            let start = start_seed % total_bits;
            prop_assert_eq!(bitmap_find_free(&bm, total_bits, start), None);
            prop_assert_eq!(bitmap_count_free(&bm, total_bits), 0);
        }

        /// Bitmap with all bits clear: find_free returns start (no wrap needed).
        #[test]
        fn proptest_bitmap_empty_find_free_at_start(
            byte_len in 1_usize..64,
            start_seed in any::<u32>(),
        ) {
            let total_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            let bm = vec![0_u8; byte_len];
            let start = start_seed % total_bits;
            prop_assert_eq!(bitmap_find_free(&bm, total_bits, start), Some(start));
            prop_assert_eq!(bitmap_count_free(&bm, total_bits), total_bits);
        }

        /// AllocHint goal_group is respected when the group has free space.
        #[test]
        fn proptest_alloc_respects_goal_group(
            goal_group in 0_u32..4,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);

            let hint = AllocHint {
                goal_group: Some(GroupNumber(goal_group)),
                ..AllocHint::default()
            };

            let alloc = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &hint).unwrap();
            let (allocated_group, _) = geo.absolute_to_group_block(alloc.start);
            prop_assert_eq!(
                allocated_group.0, goal_group,
                "allocation should land in goal group {} but landed in {}",
                goal_group, allocated_group.0,
            );
        }

        // ── FsGeometry coordinate conversion properties ─────────────

        /// group_block_to_absolute → absolute_to_group_block roundtrip.
        #[test]
        fn proptest_geo_coordinate_roundtrip(
            group_idx in 0_u32..4,
            rel_block_seed in 0_u32..8192,
        ) {
            let geo = make_geometry();
            let group = GroupNumber(group_idx);
            let blocks_in = geo.blocks_in_group(group);
            let rel_block = rel_block_seed % blocks_in;

            let abs = geo.group_block_to_absolute(group, rel_block);
            let (back_group, back_rel) = geo.absolute_to_group_block(abs);
            prop_assert_eq!(back_group, group, "group mismatch");
            prop_assert_eq!(back_rel, rel_block, "relative block mismatch");
        }

        /// blocks_in_group is always <= blocks_per_group.
        #[test]
        fn proptest_blocks_in_group_bounded(group_idx in 0_u32..4) {
            let geo = make_geometry();
            let blocks = geo.blocks_in_group(GroupNumber(group_idx));
            prop_assert!(blocks <= geo.blocks_per_group);
            prop_assert!(blocks > 0);
        }

        /// inodes_in_group is always <= inodes_per_group.
        #[test]
        fn proptest_inodes_in_group_bounded(group_idx in 0_u32..4) {
            let geo = make_geometry();
            let inodes = geo.inodes_in_group(GroupNumber(group_idx));
            prop_assert!(inodes <= geo.inodes_per_group);
            prop_assert!(inodes > 0);
        }

        /// Sum of blocks_in_group across all groups = total_blocks - first_data_block.
        #[test]
        fn proptest_blocks_sum_equals_total(
            bpg in prop::sample::select(vec![1024_u32, 2048, 4096, 8192]),
            total_blocks_mult in 1_u64..=8,
        ) {
            let total_blocks = u64::from(bpg) * total_blocks_mult;
            let group_count = total_blocks.div_ceil(u64::from(bpg));
            if group_count > u64::from(u32::MAX) { return Ok(()); }
            #[expect(clippy::cast_possible_truncation)]
            let gc = group_count as u32;
            let geo = FsGeometry {
                blocks_per_group: bpg,
                inodes_per_group: 256,
                block_size: 4096,
                total_blocks,
                total_inodes: gc * 256,
                first_data_block: 0,
                group_count: gc,
                inode_size: 256,
            };
            let sum: u64 = (0..gc).map(|g| u64::from(geo.blocks_in_group(GroupNumber(g)))).sum();
            prop_assert_eq!(sum, total_blocks);
        }

        // ── Bitmap edge case properties ─────────────────────────────

        /// bitmap_find_contiguous with n=0 always returns Some(0).
        #[test]
        fn proptest_find_contiguous_zero_always_succeeds(
            (ref bm, count) in bitmap_strat(),
            start_seed in any::<u32>(),
        ) {
            let start = start_seed % count;
            let result = bitmap_find_contiguous(bm, count, 0, start);
            prop_assert_eq!(result, Some(0));
        }

        /// bitmap_get beyond bitmap length returns false.
        #[test]
        fn proptest_bitmap_get_oob_is_false(
            byte_len in 1_usize..32,
            beyond in 0_u32..100,
        ) {
            let bm = vec![0xFF_u8; byte_len];
            let total = u32::try_from(byte_len * 8).unwrap();
            let oob_idx = total + beyond;
            prop_assert!(!bitmap_get(&bm, oob_idx));
        }

        /// bitmap_set/clear beyond length is a no-op (no panic).
        #[test]
        fn proptest_bitmap_set_clear_oob_noop(
            byte_len in 1_usize..32,
            beyond in 0_u32..100,
        ) {
            let mut bm = vec![0_u8; byte_len];
            let original = bm.clone();
            let total = u32::try_from(byte_len * 8).unwrap();
            let oob_idx = total + beyond;
            bitmap_set(&mut bm, oob_idx);
            prop_assert!(bm == original, "set beyond bounds should not modify bitmap");
            bitmap_clear(&mut bm, oob_idx);
            prop_assert!(bm == original, "clear beyond bounds should not modify bitmap");
        }

        /// Setting a single bit increases count_free by exactly -1.
        #[test]
        fn proptest_set_decreases_free_by_one(
            byte_len in 1_usize..64,
            idx_seed in any::<u32>(),
        ) {
            let total_bits = u32::try_from(byte_len * 8).unwrap();
            let idx = idx_seed % total_bits;
            let mut bm = vec![0u8; byte_len];
            let before = bitmap_count_free(&bm, total_bits);
            bitmap_set(&mut bm, idx);
            let after = bitmap_count_free(&bm, total_bits);
            prop_assert_eq!(after, before - 1);
        }

        /// Clearing a set bit increases count_free by exactly +1.
        #[test]
        fn proptest_clear_increases_free_by_one(
            byte_len in 1_usize..64,
            idx_seed in any::<u32>(),
        ) {
            let total_bits = u32::try_from(byte_len * 8).unwrap();
            let idx = idx_seed % total_bits;
            let mut bm = vec![0xFF_u8; byte_len];
            let before = bitmap_count_free(&bm, total_bits);
            bitmap_clear(&mut bm, idx);
            let after = bitmap_count_free(&bm, total_bits);
            prop_assert_eq!(after, before + 1);
        }

        // ── Alloc/free interleaving property ────────────────────────

        /// Interleaved alloc/free operations maintain free count invariant.
        #[test]
        fn proptest_interleaved_alloc_free_consistent(
            ops in prop::collection::vec(prop::bool::ANY, 1..20),
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);
            let original_free: u32 = groups.iter().map(|g| g.free_blocks).sum();

            let mut allocated = Vec::new();
            let mut net_allocated = 0_u32;

            for do_alloc in &ops {
                if *do_alloc {
                    // Alloc 1 block
                    if let Ok(a) = alloc_blocks(&cx, &dev, &geo, &mut groups, 1, &AllocHint::default()) {
                        net_allocated += a.count;
                        allocated.push(a);
                    }
                } else if let Some(a) = allocated.pop() {
                    // Free last allocated block
                    free_blocks(&cx, &dev, &geo, &mut groups, a.start, a.count).unwrap();
                    net_allocated -= a.count;
                }

                // Invariant: current free = original - net_allocated
                let current_free: u32 = groups.iter().map(|g| g.free_blocks).sum();
                prop_assert_eq!(
                    current_free, original_free - net_allocated,
                    "free count mismatch after op"
                );
            }
        }

        // ── Succinct bitmap properties ──────────────────────────────

        /// SuccinctBitmap rank0 + rank1 = position for all valid positions.
        #[test]
        fn proptest_succinct_rank_sum(
            (ref bm, count) in bitmap_strat(),
        ) {
            let sb = succinct::SuccinctBitmap::build(bm, count);
            // Check at a few positions within the range.
            for pos in [0, count / 4, count / 2, count.saturating_sub(1), count] {
                if pos <= count {
                    let r0 = sb.rank0(pos);
                    let r1 = sb.rank1(pos);
                    prop_assert_eq!(
                        r0 + r1, pos,
                        "rank0({}) + rank1({}) = {} != {}",
                        pos, pos, r0 + r1, pos,
                    );
                }
            }
        }

        /// SuccinctBitmap total ones matches manual popcount.
        #[test]
        fn proptest_succinct_ones_matches_popcount(
            (ref bm, count) in bitmap_strat(),
        ) {
            let sb = succinct::SuccinctBitmap::build(bm, count);
            let manual_ones = (0..count).filter(|&i| bitmap_get(bm, i)).count();
            prop_assert_eq!(
                sb.count_ones() as usize, manual_ones,
                "SuccinctBitmap.count_ones() mismatch"
            );
        }

        /// SuccinctBitmap select0 returns a valid zero-bit position.
        #[test]
        fn proptest_succinct_select0_valid(
            (ref bm, count) in bitmap_strat(),
        ) {
            let sb = succinct::SuccinctBitmap::build(bm, count);
            let zeros = sb.count_zeros();
            if zeros > 0 {
                // Check first and last zero.
                if let Some(pos) = sb.select0(0) {
                    prop_assert!(pos < count, "select0(0) = {} >= count {}", pos, count);
                    prop_assert!(!bitmap_get(bm, pos), "select0(0) points to a set bit");
                }
                if let Some(pos) = sb.select0(zeros - 1) {
                    prop_assert!(pos < count, "select0(last) = {} >= count", pos);
                    prop_assert!(!bitmap_get(bm, pos), "select0(last) points to a set bit");
                }
            }
            // select0 beyond zeros count returns None.
            prop_assert_eq!(sb.select0(zeros), None);
        }

        /// SuccinctBitmap select1 returns a valid one-bit position.
        #[test]
        fn proptest_succinct_select1_valid(
            (ref bm, count) in bitmap_strat(),
        ) {
            let sb = succinct::SuccinctBitmap::build(bm, count);
            let ones = sb.count_ones();
            if ones > 0 {
                if let Some(pos) = sb.select1(0) {
                    prop_assert!(pos < count);
                    prop_assert!(bitmap_get(bm, pos), "select1(0) points to a zero bit");
                }
                if let Some(pos) = sb.select1(ones - 1) {
                    prop_assert!(pos < count);
                    prop_assert!(bitmap_get(bm, pos), "select1(last) points to a zero bit");
                }
            }
            prop_assert_eq!(sb.select1(ones), None);
        }

        /// SuccinctBitmap find_free agrees with bitmap_find_free for start=0.
        #[test]
        fn proptest_succinct_find_free_matches_linear(
            (ref bm, count) in bitmap_strat(),
        ) {
            let sb = succinct::SuccinctBitmap::build(bm, count);
            let linear = bitmap_find_free(bm, count, 0);
            let succinct_result = sb.find_free(0);
            prop_assert_eq!(
                succinct_result, linear,
                "find_free mismatch: succinct={:?}, linear={:?}",
                succinct_result, linear,
            );
        }

        // ── Free blocks validation properties ───────────────────────

        /// Freeing blocks at an out-of-range group returns Corruption error.
        #[test]
        fn proptest_free_blocks_oob_group_errors(
            group_offset in 4_u32..100,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);

            // Build an absolute block in a non-existent group.
            let bad_start = BlockNumber(u64::from(group_offset) * u64::from(geo.blocks_per_group));
            let result = free_blocks(&cx, &dev, &geo, &mut groups, bad_start, 1);
            prop_assert!(result.is_err(), "free_blocks should reject out-of-range group");
        }

        /// Freeing blocks that cross group boundary returns error.
        #[test]
        fn proptest_free_blocks_cross_boundary_errors(
            group_idx in 0_u32..3,
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let geo = make_geometry();
            let mut groups = make_groups(&geo);

            let blocks_in = geo.blocks_in_group(GroupNumber(group_idx));
            // Start near end of group, count crosses boundary.
            let rel_start = blocks_in.saturating_sub(1);
            let abs = geo.group_block_to_absolute(GroupNumber(group_idx), rel_start);
            let result = free_blocks(&cx, &dev, &geo, &mut groups, abs, 2);
            prop_assert!(
                result.is_err(),
                "free_blocks should reject extent crossing group boundary"
            );
        }
    }
}
