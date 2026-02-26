#![forbid(unsafe_code)]
//! ext4 extent B+tree operations.
//!
//! Search, insert, split, merge, and tree walk over the extent tree
//! stored in inode `i_block[15]` fields and internal/leaf nodes.
//!
//! The extent tree maps logical file blocks to physical device blocks via a
//! B+tree rooted in the inode's `i_block[0..14]` (60 bytes). External nodes
//! occupy full disk blocks.
//!
//! Depth 0: root contains up to 4 leaf extents (60-12=48 bytes / 12 = 4).
//! Depth 1: root indexes up to 4 child blocks, each holding up to 340 leaf
//!           extents (at 4K blocks: (4096-12-4)/12 = 340).
//! Depth 2: up to 4 × 340 = 1360 child blocks, each with 340 leaf extents.

pub mod bw_tree;

use asupersync::Cx;
use ffs_block::BlockDevice;
use ffs_error::{FfsError, Result};
use ffs_ondisk::{Ext4Extent, Ext4ExtentHeader, Ext4ExtentIndex};
use ffs_types::BlockNumber;
use tracing::{debug, error, trace};

// ── Constants ───────────────────────────────────────────────────────────────

/// Extent header magic (Section 11.4.1).
const EXT4_EXTENT_MAGIC: u16 = 0xF30A;

/// Header size in bytes.
const HEADER_SIZE: usize = 12;

/// Entry size in bytes (both leaf extents and index entries are 12 bytes).
const ENTRY_SIZE: usize = 12;

/// Tail checksum size.
const TAIL_SIZE: usize = 4;

/// Max entries in root node (60 bytes of i_block: (60-12)/12 = 4).
const ROOT_MAX_ENTRIES: u16 = 4;

/// Bit 15 in `ee_len` indicates unwritten extent.
const EXT_INIT_MAX_LEN: u16 = 1_u16 << 15;

// ── Search ──────────────────────────────────────────────────────────────────

/// Result of searching for a logical block in the extent tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchResult {
    /// Found an extent covering the target logical block.
    Found {
        extent: Ext4Extent,
        /// Offset within the extent (target - extent.logical_block).
        offset_in_extent: u32,
    },
    /// Target falls in a hole (unmapped region).
    Hole {
        /// Number of unmapped blocks from target until the next extent
        /// (or end of address space if no subsequent extent).
        hole_len: u32,
    },
}

/// Search the extent tree rooted in `root_bytes` (the 60-byte `i_block` area)
/// for the given `target` logical block.
///
/// Reads child blocks from `dev` as needed for multi-level trees.
pub fn search(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &[u8; 60],
    target: u32,
) -> Result<SearchResult> {
    let (header, _) = parse_header(root_bytes)?;
    validate_header(&header, ROOT_MAX_ENTRIES)?;

    if header.depth == 0 {
        // Leaf: search extents directly in root.
        let extents = parse_leaf_entries(root_bytes, &header)?;
        return search_leaf(&extents, target);
    }

    // Internal: descend through index levels.
    let indexes = parse_index_entries(root_bytes, &header)?;
    let child_block = find_index_child(&indexes, target)?;

    descend_search(cx, dev, child_block, header.depth - 1, target)
}

/// Descend from an internal node at the given depth to find the target.
fn descend_search(
    cx: &Cx,
    dev: &dyn BlockDevice,
    block: u64,
    depth: u16,
    target: u32,
) -> Result<SearchResult> {
    cx_checkpoint(cx)?;

    let buf = dev.read_block(cx, BlockNumber(block))?;
    let data = buf.as_slice();
    let max_entries = max_entries_external(dev.block_size());
    let (header, _) = parse_header(data)?;
    validate_header(&header, max_entries)?;

    if header.depth != depth {
        return Err(FfsError::Corruption {
            block,
            detail: format!(
                "extent tree depth mismatch: expected {depth}, got {}",
                header.depth
            ),
        });
    }

    if depth == 0 {
        let extents = parse_leaf_entries(data, &header)?;
        search_leaf(&extents, target)
    } else {
        let indexes = parse_index_entries(data, &header)?;
        let child = find_index_child(&indexes, target)?;
        descend_search(cx, dev, child, depth - 1, target)
    }
}

/// Binary search leaf extents for the target logical block.
fn search_leaf(extents: &[Ext4Extent], target: u32) -> Result<SearchResult> {
    if extents.is_empty() {
        return Ok(SearchResult::Hole { hole_len: u32::MAX });
    }

    // Validate: reject zero-length extents (on-disk corruption).
    for ext in extents {
        if actual_len(ext.raw_len) == 0 {
            return Err(FfsError::Corruption {
                block: 0,
                detail: format!(
                    "zero-length extent traversal at logical block {}",
                    ext.logical_block,
                ),
            });
        }
    }

    // Binary search: find the last extent with logical_block <= target.
    let pos = extents.partition_point(|e| e.logical_block <= target);

    if pos > 0 {
        let ext = &extents[pos - 1];
        let len = actual_len(ext.raw_len);
        let end = ext.logical_block.saturating_add(u32::from(len));
        if target < end {
            return Ok(SearchResult::Found {
                extent: *ext,
                offset_in_extent: target - ext.logical_block,
            });
        }
    }

    // Target is in a hole. Determine hole length.
    let next_start = if pos < extents.len() {
        extents[pos].logical_block
    } else {
        u32::MAX
    };
    let hole_len = next_start.saturating_sub(target);
    Ok(SearchResult::Hole { hole_len })
}

/// Find the child block to descend into for the given target.
///
/// Returns the leaf_block of the last index entry with `logical_block <= target`.
fn find_index_child(indexes: &[Ext4ExtentIndex], target: u32) -> Result<u64> {
    if indexes.is_empty() {
        return Err(FfsError::Corruption {
            block: 0,
            detail: "extent tree index node has no entries".into(),
        });
    }

    let pos = indexes.partition_point(|idx| idx.logical_block <= target);
    let child_idx = if pos > 0 { pos - 1 } else { 0 };
    Ok(indexes[child_idx].leaf_block)
}

// ── Walk ────────────────────────────────────────────────────────────────────

/// Walk all extents in the tree in logical order.
///
/// Yields extents via `visitor`. Returns the total number of extents visited.
pub fn walk<F>(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &[u8; 60],
    visitor: &mut F,
) -> Result<usize>
where
    F: FnMut(&Ext4Extent) -> Result<()>,
{
    let (header, _) = parse_header(root_bytes)?;
    validate_header(&header, ROOT_MAX_ENTRIES)?;

    if header.depth == 0 {
        let extents = parse_leaf_entries(root_bytes, &header)?;
        for ext in &extents {
            visitor(ext)?;
        }
        return Ok(extents.len());
    }

    let indexes = parse_index_entries(root_bytes, &header)?;
    let mut count = 0;
    for idx in &indexes {
        count += walk_subtree(cx, dev, idx.leaf_block, header.depth - 1, visitor)?;
    }
    Ok(count)
}

fn walk_subtree<F>(
    cx: &Cx,
    dev: &dyn BlockDevice,
    block: u64,
    depth: u16,
    visitor: &mut F,
) -> Result<usize>
where
    F: FnMut(&Ext4Extent) -> Result<()>,
{
    cx_checkpoint(cx)?;

    let buf = dev.read_block(cx, BlockNumber(block))?;
    let data = buf.as_slice();
    let max_entries = max_entries_external(dev.block_size());
    let (header, _) = parse_header(data)?;
    validate_header(&header, max_entries)?;

    if header.depth != depth {
        return Err(FfsError::Corruption {
            block,
            detail: format!(
                "walk: depth mismatch: expected {depth}, got {}",
                header.depth
            ),
        });
    }

    if depth == 0 {
        let extents = parse_leaf_entries(data, &header)?;
        for ext in &extents {
            visitor(ext)?;
        }
        Ok(extents.len())
    } else {
        let indexes = parse_index_entries(data, &header)?;
        let mut count = 0;
        for idx in &indexes {
            count += walk_subtree(cx, dev, idx.leaf_block, depth - 1, visitor)?;
        }
        Ok(count)
    }
}

// ── Insert ──────────────────────────────────────────────────────────────────

/// Allocator callback for obtaining new blocks during tree modifications.
///
/// The caller provides a function that allocates a fresh block and returns
/// its [`BlockNumber`]. The block should be zeroed or the caller should
/// be prepared for the btree code to overwrite it entirely.
pub trait BlockAllocator {
    /// Allocate a single block for tree use (index or leaf node).
    fn alloc_block(&mut self, cx: &Cx) -> Result<BlockNumber>;

    /// Free a previously allocated tree block.
    fn free_block(&mut self, cx: &Cx, block: BlockNumber) -> Result<()>;
}

/// Insert a new extent into the tree rooted at `root_bytes`.
///
/// The new extent must not overlap any existing extent. The caller is
/// responsible for ensuring no overlap before calling insert.
///
/// Returns the (possibly modified) root bytes. If the tree needed to grow
/// in depth, new blocks are allocated via `alloc`.
pub fn insert(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &mut [u8; 60],
    extent: Ext4Extent,
    alloc: &mut dyn BlockAllocator,
) -> Result<()> {
    let (header, _) = parse_header(root_bytes)?;
    validate_header(&header, ROOT_MAX_ENTRIES)?;
    trace!(
        logical_start = extent.logical_block,
        len = u32::from(actual_len(extent.raw_len)),
        physical_block = extent.physical_start,
        tree_depth = header.depth,
        "extent_insert"
    );

    if header.depth == 0 {
        // Leaf root: try to insert directly.
        let mut extents = parse_leaf_entries(root_bytes, &header)?;
        let insert_pos = extents.partition_point(|e| e.logical_block < extent.logical_block);

        if usize::from(header.entries) < usize::from(header.max_entries) {
            // Space available: insert in sorted position.
            extents.insert(insert_pos, extent);
            write_leaf_root(root_bytes, &header, &extents);
            return Ok(());
        }

        // Root is full: grow tree by one level.
        extents.insert(insert_pos, extent);
        grow_root_leaf(cx, dev, root_bytes, &extents, alloc)?;
        return Ok(());
    }

    // Internal root: descend to find the right leaf.
    let indexes = parse_index_entries(root_bytes, &header)?;
    let child_pos = find_index_pos(&indexes, extent.logical_block);
    let child_block = indexes[child_pos].leaf_block;

    let split = insert_descend(cx, dev, child_block, header.depth - 1, extent, alloc)?;

    if let Some(new_entry) = split {
        // Child was split, need to insert new index entry in root.
        let mut indexes = parse_index_entries(root_bytes, &header)?;

        if usize::from(header.entries) < usize::from(header.max_entries) {
            // Space in root for new index entry.
            let pos = indexes.partition_point(|e| e.logical_block < new_entry.logical_block);
            indexes.insert(pos, new_entry);
            write_index_root(root_bytes, &header, &indexes);
            Ok(())
        } else {
            // Root is full: grow tree.
            indexes.insert(
                indexes.partition_point(|e| e.logical_block < new_entry.logical_block),
                new_entry,
            );
            grow_root_index(cx, dev, root_bytes, &header, &indexes, alloc)
        }
    } else {
        Ok(())
    }
}

/// Descend into a child node for insertion. Returns `Some(new_index_entry)`
/// if the child was split and the parent needs to accommodate a new entry.
fn insert_descend(
    cx: &Cx,
    dev: &dyn BlockDevice,
    block: u64,
    depth: u16,
    extent: Ext4Extent,
    alloc: &mut dyn BlockAllocator,
) -> Result<Option<Ext4ExtentIndex>> {
    cx_checkpoint(cx)?;

    let buf = dev.read_block(cx, BlockNumber(block))?;
    let data = buf.as_slice();
    let block_size = dev.block_size();
    let max = max_entries_external(block_size);
    let (header, _) = parse_header(data)?;
    validate_header(&header, max)?;

    if depth == 0 {
        // Leaf node.
        let mut extents = parse_leaf_entries(data, &header)?;
        let insert_pos = extents.partition_point(|e| e.logical_block < extent.logical_block);

        if usize::from(header.entries) < usize::from(max) {
            // Space available.
            extents.insert(insert_pos, extent);
            let new_data = serialize_leaf_block(block_size, &extents);
            dev.write_block(cx, BlockNumber(block), &new_data)?;
            return Ok(None);
        }

        // Leaf full: split.
        extents.insert(insert_pos, extent);
        let mid = extents.len() / 2;
        let right_extents = extents.split_off(mid);
        let left_extents = extents;

        // Allocate new block for right half first to prevent data loss on allocation failure.
        let new_block = alloc.alloc_block(cx)?;
        trace!(
            new_block_num = new_block.0,
            purpose = "leaf_split",
            "extent_block_alloc"
        );
        let right_data = serialize_leaf_block(block_size, &right_extents);
        dev.write_block(cx, new_block, &right_data)?;

        // Write left half back to original block.
        let left_data = serialize_leaf_block(block_size, &left_extents);
        dev.write_block(cx, BlockNumber(block), &left_data)?;
        debug!(
            old_node = block,
            new_node = new_block.0,
            separator_key = right_extents[0].logical_block,
            "extent_leaf_split"
        );

        // Return new index entry pointing to the right half.
        Ok(Some(Ext4ExtentIndex {
            logical_block: right_extents[0].logical_block,
            leaf_block: new_block.0,
        }))
    } else {
        // Internal node: descend further.
        let indexes = parse_index_entries(data, &header)?;
        let child_pos = find_index_pos(&indexes, extent.logical_block);
        let child_block = indexes[child_pos].leaf_block;

        let split = insert_descend(cx, dev, child_block, depth - 1, extent, alloc)?;

        if let Some(new_entry) = split {
            let mut indexes = parse_index_entries(data, &header)?;

            if usize::from(header.entries) < usize::from(max) {
                // Space in this node.
                let pos = indexes.partition_point(|e| e.logical_block < new_entry.logical_block);
                indexes.insert(pos, new_entry);
                let new_data = serialize_index_block(block_size, depth, &indexes);
                dev.write_block(cx, BlockNumber(block), &new_data)?;
                Ok(None)
            } else {
                // This node is also full: split it.
                let pos = indexes.partition_point(|e| e.logical_block < new_entry.logical_block);
                indexes.insert(pos, new_entry);
                let mid = indexes.len() / 2;
                let right_indexes = indexes.split_off(mid);
                let left_indexes = indexes;

                let new_block = alloc.alloc_block(cx)?;
                trace!(
                    new_block_num = new_block.0,
                    purpose = "index_split",
                    "extent_block_alloc"
                );
                let right_data = serialize_index_block(block_size, depth, &right_indexes);
                dev.write_block(cx, new_block, &right_data)?;

                let left_data = serialize_index_block(block_size, depth, &left_indexes);
                dev.write_block(cx, BlockNumber(block), &left_data)?;
                debug!(
                    old_node = block,
                    new_node = new_block.0,
                    separator_key = right_indexes[0].logical_block,
                    "extent_index_split"
                );

                Ok(Some(Ext4ExtentIndex {
                    logical_block: right_indexes[0].logical_block,
                    leaf_block: new_block.0,
                }))
            }
        } else {
            Ok(None)
        }
    }
}

/// Grow the tree when the root leaf is full.
/// All extents (including the new one, already inserted) are split into
/// two child leaf blocks, and the root becomes a depth-1 index node.
fn grow_root_leaf(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &mut [u8; 60],
    all_extents: &[Ext4Extent],
    alloc: &mut dyn BlockAllocator,
) -> Result<()> {
    let block_size = dev.block_size();
    let mid = all_extents.len() / 2;
    let left = &all_extents[..mid];
    let right = &all_extents[mid..];

    let left_block = alloc.alloc_block(cx)?;
    trace!(
        new_block_num = left_block.0,
        purpose = "root_grow_leaf_left",
        "extent_block_alloc"
    );
    let left_data = serialize_leaf_block(block_size, left);
    dev.write_block(cx, left_block, &left_data)?;

    let right_block = alloc.alloc_block(cx)?;
    trace!(
        new_block_num = right_block.0,
        purpose = "root_grow_leaf_right",
        "extent_block_alloc"
    );
    let right_data = serialize_leaf_block(block_size, right);
    dev.write_block(cx, right_block, &right_data)?;

    let indexes = vec![
        Ext4ExtentIndex {
            logical_block: left[0].logical_block,
            leaf_block: left_block.0,
        },
        Ext4ExtentIndex {
            logical_block: right[0].logical_block,
            leaf_block: right_block.0,
        },
    ];

    let header = Ext4ExtentHeader {
        magic: EXT4_EXTENT_MAGIC,
        entries: 2,
        max_entries: ROOT_MAX_ENTRIES,
        depth: 1,
        generation: 0,
    };
    write_index_root(root_bytes, &header, &indexes);
    debug!(
        old_node = "inode_root",
        new_node = format!("{},{}", left_block.0, right_block.0),
        separator_key = right[0].logical_block,
        "extent_root_split_leaf"
    );
    Ok(())
}

/// Grow the tree when the root index node is full.
fn grow_root_index(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &mut [u8; 60],
    header: &Ext4ExtentHeader,
    all_indexes: &[Ext4ExtentIndex],
    alloc: &mut dyn BlockAllocator,
) -> Result<()> {
    let block_size = dev.block_size();
    let mid = all_indexes.len() / 2;
    let left = &all_indexes[..mid];
    let right = &all_indexes[mid..];

    let left_block = alloc.alloc_block(cx)?;
    trace!(
        new_block_num = left_block.0,
        purpose = "root_grow_index_left",
        "extent_block_alloc"
    );
    let left_data = serialize_index_block(block_size, header.depth, left);
    dev.write_block(cx, left_block, &left_data)?;

    let right_block = alloc.alloc_block(cx)?;
    trace!(
        new_block_num = right_block.0,
        purpose = "root_grow_index_right",
        "extent_block_alloc"
    );
    let right_data = serialize_index_block(block_size, header.depth, right);
    dev.write_block(cx, right_block, &right_data)?;

    let new_indexes = vec![
        Ext4ExtentIndex {
            logical_block: left[0].logical_block,
            leaf_block: left_block.0,
        },
        Ext4ExtentIndex {
            logical_block: right[0].logical_block,
            leaf_block: right_block.0,
        },
    ];

    let new_header = Ext4ExtentHeader {
        magic: EXT4_EXTENT_MAGIC,
        entries: 2,
        max_entries: ROOT_MAX_ENTRIES,
        depth: header.depth + 1,
        generation: 0,
    };
    write_index_root(root_bytes, &new_header, &new_indexes);
    debug!(
        old_node = "inode_root",
        new_node = format!("{},{}", left_block.0, right_block.0),
        separator_key = right[0].logical_block,
        "extent_root_split_index"
    );
    Ok(())
}

// ── Delete ──────────────────────────────────────────────────────────────────

/// Remove all extents that overlap the range `[logical_start, logical_start + count)`.
///
/// Extents that partially overlap the range are trimmed. Extents fully within
/// the range are removed entirely. Returns the list of physical block ranges
/// that were freed (the caller is responsible for deallocating them).
#[expect(clippy::cast_possible_truncation)]
pub fn delete_range(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &mut [u8; 60],
    logical_start: u32,
    count: u32,
    alloc: &mut dyn BlockAllocator,
) -> Result<Vec<FreedRange>> {
    let logical_end = logical_start.saturating_add(count);
    let (header, _) = parse_header(root_bytes)?;
    validate_header(&header, ROOT_MAX_ENTRIES)?;
    trace!(
        logical_start,
        count,
        tree_depth = header.depth,
        "extent_delete_start"
    );

    if header.depth == 0 {
        let extents = parse_leaf_entries(root_bytes, &header)?;
        let (remaining, freed) = trim_extents(extents, logical_start, logical_end);
        write_leaf_root(root_bytes, &header, &remaining);
        trace!(
            logical_start,
            count,
            freed_blocks_count = freed.len(),
            "extent_delete_done"
        );
        return Ok(freed);
    }

    // Multi-level tree: descend, collapse empty children, and refresh separators.
    let mut indexes = parse_index_entries(root_bytes, &header)?;
    let mut all_freed = Vec::new();
    let mut empty_children = Vec::new();

    let mut idx_pos = 0;
    while idx_pos < indexes.len() {
        let child_block = indexes[idx_pos].leaf_block;
        let child_result = delete_range_subtree(
            cx,
            dev,
            child_block,
            header.depth - 1,
            logical_start,
            logical_end,
            alloc,
        )?;
        all_freed.extend(child_result.freed_ranges);

        if child_result.empty {
            empty_children.push(child_block);
            debug!(
                merged_nodes = 1_u8,
                resulting_node = "root",
                "extent_delete_collapse_empty_child"
            );
            indexes.remove(idx_pos);
            continue;
        }

        if let Some(new_first) = child_result.first_logical {
            indexes[idx_pos].logical_block = new_first;
        }
        idx_pos += 1;
    }

    if indexes.is_empty() {
        let empty_header = Ext4ExtentHeader {
            magic: EXT4_EXTENT_MAGIC,
            entries: 0,
            max_entries: ROOT_MAX_ENTRIES,
            depth: 0,
            generation: header.generation,
        };
        write_leaf_root(root_bytes, &empty_header, &[]);
    } else {
        let new_header = Ext4ExtentHeader {
            magic: EXT4_EXTENT_MAGIC,
            entries: indexes.len() as u16,
            max_entries: ROOT_MAX_ENTRIES,
            depth: header.depth,
            generation: header.generation,
        };
        write_index_root(root_bytes, &new_header, &indexes);
        maybe_shrink_root(cx, dev, root_bytes, alloc)?;
    }

    for child_block in empty_children {
        trace!(
            freed_block_num = child_block,
            reason = "empty_subtree",
            "extent_block_free"
        );
        alloc.free_block(cx, BlockNumber(child_block))?;
    }

    trace!(
        logical_start,
        count,
        freed_blocks_count = all_freed.len(),
        "extent_delete_done"
    );

    Ok(all_freed)
}

/// A range of physical blocks that were freed by a delete operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreedRange {
    pub physical_start: u64,
    pub count: u16,
}

fn delete_range_subtree(
    cx: &Cx,
    dev: &dyn BlockDevice,
    block: u64,
    depth: u16,
    logical_start: u32,
    logical_end: u32,
    alloc: &mut dyn BlockAllocator,
) -> Result<DeleteSubtreeResult> {
    cx_checkpoint(cx)?;

    let buf = dev.read_block(cx, BlockNumber(block))?;
    let data = buf.as_slice();
    let block_size = dev.block_size();
    let max = max_entries_external(block_size);
    let (header, _) = parse_header(data)?;
    validate_header(&header, max)?;
    if header.depth != depth {
        return Err(FfsError::Corruption {
            block,
            detail: format!(
                "delete: depth mismatch: expected {depth}, got {}",
                header.depth
            ),
        });
    }

    if depth == 0 {
        let extents = parse_leaf_entries(data, &header)?;
        let (remaining, freed) = trim_extents(extents, logical_start, logical_end);
        let new_data = serialize_leaf_block(block_size, &remaining);
        dev.write_block(cx, BlockNumber(block), &new_data)?;
        Ok(DeleteSubtreeResult {
            freed_ranges: freed,
            first_logical: remaining.first().map(|ext| ext.logical_block),
            empty: remaining.is_empty(),
        })
    } else {
        let mut indexes = parse_index_entries(data, &header)?;
        let mut all_freed = Vec::new();
        let mut empty_children = Vec::new();
        let mut idx_pos = 0;
        while idx_pos < indexes.len() {
            let child_block = indexes[idx_pos].leaf_block;
            let child_result = delete_range_subtree(
                cx,
                dev,
                child_block,
                depth - 1,
                logical_start,
                logical_end,
                alloc,
            )?;
            all_freed.extend(child_result.freed_ranges);

            if child_result.empty {
                empty_children.push(child_block);
                debug!(
                    merged_nodes = 1_u8,
                    resulting_node = block,
                    "extent_delete_collapse_empty_child"
                );
                indexes.remove(idx_pos);
                continue;
            }

            if let Some(new_first) = child_result.first_logical {
                indexes[idx_pos].logical_block = new_first;
            }
            idx_pos += 1;
        }

        if !indexes.is_empty() {
            let new_data = serialize_index_block(block_size, depth, &indexes);
            dev.write_block(cx, BlockNumber(block), &new_data)?;
        }

        for child_block in empty_children {
            trace!(
                freed_block_num = child_block,
                reason = "empty_subtree",
                "extent_block_free"
            );
            alloc.free_block(cx, BlockNumber(child_block))?;
        }

        Ok(DeleteSubtreeResult {
            freed_ranges: all_freed,
            first_logical: indexes.first().map(|idx| idx.logical_block),
            empty: indexes.is_empty(),
        })
    }
}

struct DeleteSubtreeResult {
    freed_ranges: Vec<FreedRange>,
    first_logical: Option<u32>,
    empty: bool,
}

#[expect(clippy::cast_possible_truncation)]
fn maybe_shrink_root(
    cx: &Cx,
    dev: &dyn BlockDevice,
    root_bytes: &mut [u8; 60],
    alloc: &mut dyn BlockAllocator,
) -> Result<()> {
    loop {
        let (root_header, _) = parse_header(root_bytes)?;
        validate_header(&root_header, ROOT_MAX_ENTRIES)?;
        if root_header.depth == 0 || root_header.entries != 1 {
            return Ok(());
        }

        let root_indexes = parse_index_entries(root_bytes, &root_header)?;
        let only_child = root_indexes[0].leaf_block;

        let child_buf = dev.read_block(cx, BlockNumber(only_child))?;
        let child_data = child_buf.as_slice();
        let child_max = max_entries_external(dev.block_size());
        let (child_header, _) = parse_header(child_data)?;
        validate_header(&child_header, child_max)?;

        if child_header.depth + 1 != root_header.depth || child_header.entries > ROOT_MAX_ENTRIES {
            return Ok(());
        }

        if child_header.depth == 0 {
            let child_extents = parse_leaf_entries(child_data, &child_header)?;
            let new_header = Ext4ExtentHeader {
                magic: EXT4_EXTENT_MAGIC,
                entries: child_extents.len() as u16,
                max_entries: ROOT_MAX_ENTRIES,
                depth: 0,
                generation: root_header.generation,
            };
            write_leaf_root(root_bytes, &new_header, &child_extents);
        } else {
            let child_indexes = parse_index_entries(child_data, &child_header)?;
            let new_header = Ext4ExtentHeader {
                magic: EXT4_EXTENT_MAGIC,
                entries: child_indexes.len() as u16,
                max_entries: ROOT_MAX_ENTRIES,
                depth: child_header.depth,
                generation: root_header.generation,
            };
            write_index_root(root_bytes, &new_header, &child_indexes);
        }

        trace!(
            freed_block_num = only_child,
            reason = "root_shrink",
            "extent_block_free"
        );
        alloc.free_block(cx, BlockNumber(only_child))?;
        debug!(
            merged_nodes = 1_u8,
            resulting_node = "inode_root",
            "extent_root_shrink"
        );
    }
}

/// Trim/remove extents that overlap `[start, end)`.
/// Returns (remaining_extents, freed_ranges).
#[expect(clippy::cast_possible_truncation)]
fn trim_extents(
    extents: Vec<Ext4Extent>,
    start: u32,
    end: u32,
) -> (Vec<Ext4Extent>, Vec<FreedRange>) {
    let mut remaining = Vec::new();
    let mut freed = Vec::new();

    for ext in extents {
        let ext_len = actual_len(ext.raw_len);
        let ext_start = ext.logical_block;
        let ext_end = ext_start.saturating_add(u32::from(ext_len));

        if ext_end <= start || ext_start >= end {
            // No overlap: keep extent as-is.
            remaining.push(ext);
            continue;
        }

        if ext_start >= start && ext_end <= end {
            // Fully within range: remove entirely.
            freed.push(FreedRange {
                physical_start: ext.physical_start,
                count: ext_len,
            });
            continue;
        }

        // Partial overlap: may produce one or two trimmed extents.

        // Left portion: extent starts before the delete range.
        if ext_start < start {
            let keep_len = start - ext_start;
            remaining.push(Ext4Extent {
                logical_block: ext_start,
                raw_len: encode_len(keep_len as u16, ext.is_unwritten()),
                physical_start: ext.physical_start,
            });
            let freed_start = u64::from(keep_len);
            // The freed portion might be further trimmed by the right side.
            if ext_end > end {
                // Delete range is in the middle: free the middle, keep right.
                let middle_len = end - start;
                freed.push(FreedRange {
                    physical_start: ext.physical_start + freed_start,
                    count: middle_len as u16,
                });
                let right_offset = end - ext_start;
                remaining.push(Ext4Extent {
                    logical_block: end,
                    raw_len: encode_len((ext_end - end) as u16, ext.is_unwritten()),
                    physical_start: ext.physical_start + u64::from(right_offset),
                });
            } else {
                freed.push(FreedRange {
                    physical_start: ext.physical_start + freed_start,
                    count: (ext_end - start) as u16,
                });
            }
        } else {
            // ext_start >= start, ext_end > end: trim from the left.
            let trim_count = end - ext_start;
            freed.push(FreedRange {
                physical_start: ext.physical_start,
                count: trim_count as u16,
            });
            remaining.push(Ext4Extent {
                logical_block: end,
                raw_len: encode_len((ext_end - end) as u16, ext.is_unwritten()),
                physical_start: ext.physical_start + u64::from(trim_count),
            });
        }
    }

    (remaining, freed)
}

// ── Parsing helpers ─────────────────────────────────────────────────────────

fn cx_checkpoint(cx: &Cx) -> Result<()> {
    cx.checkpoint().map_err(|_| FfsError::Cancelled)
}

fn parse_header(data: &[u8]) -> Result<(Ext4ExtentHeader, usize)> {
    if data.len() < HEADER_SIZE {
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!("extent header too small: {} < {HEADER_SIZE}", data.len()),
        });
    }

    let magic = u16::from_le_bytes([data[0], data[1]]);
    let entries = u16::from_le_bytes([data[2], data[3]]);
    let max_entries = u16::from_le_bytes([data[4], data[5]]);
    let depth = u16::from_le_bytes([data[6], data[7]]);
    let generation = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    Ok((
        Ext4ExtentHeader {
            magic,
            entries,
            max_entries,
            depth,
            generation,
        },
        HEADER_SIZE,
    ))
}

fn validate_header(header: &Ext4ExtentHeader, max_allowed: u16) -> Result<()> {
    if header.magic != EXT4_EXTENT_MAGIC {
        error!(
            invariant = "header.magic",
            expected = EXT4_EXTENT_MAGIC,
            got = header.magic,
            "extent_invariant_violation"
        );
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!(
                "bad extent magic: expected 0x{EXT4_EXTENT_MAGIC:04X}, got 0x{:04X}",
                header.magic
            ),
        });
    }
    if header.entries > header.max_entries {
        error!(
            invariant = "header.entries<=header.max_entries",
            entries = header.entries,
            max_entries = header.max_entries,
            "extent_invariant_violation"
        );
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!(
                "extent entries {} > max {}",
                header.entries, header.max_entries
            ),
        });
    }
    if header.max_entries > max_allowed {
        error!(
            invariant = "header.max_entries<=max_allowed",
            header_max_entries = header.max_entries,
            max_allowed,
            "extent_invariant_violation"
        );
        return Err(FfsError::Corruption {
            block: 0,
            detail: format!(
                "extent max_entries {} > allowed {}",
                header.max_entries, max_allowed
            ),
        });
    }
    Ok(())
}

fn parse_leaf_entries(data: &[u8], header: &Ext4ExtentHeader) -> Result<Vec<Ext4Extent>> {
    let count = usize::from(header.entries);
    let mut extents = Vec::with_capacity(count);
    for i in 0..count {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        if off + ENTRY_SIZE > data.len() {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "leaf entry out of bounds".into(),
            });
        }
        let logical_block =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let raw_len = u16::from_le_bytes([data[off + 4], data[off + 5]]);
        let start_hi = u16::from_le_bytes([data[off + 6], data[off + 7]]);
        let start_lo =
            u32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);
        let physical_start = u64::from(start_lo) | (u64::from(start_hi) << 32);

        if actual_len(raw_len) == 0 {
            return Err(FfsError::Corruption {
                block: 0,
                detail: format!(
                    "zero-length extent traversal at leaf entry {i} (logical_block {logical_block})"
                ),
            });
        }

        extents.push(Ext4Extent {
            logical_block,
            raw_len,
            physical_start,
        });
    }
    Ok(extents)
}

fn parse_index_entries(data: &[u8], header: &Ext4ExtentHeader) -> Result<Vec<Ext4ExtentIndex>> {
    let count = usize::from(header.entries);
    let mut indexes = Vec::with_capacity(count);
    for i in 0..count {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        if off + ENTRY_SIZE > data.len() {
            return Err(FfsError::Corruption {
                block: 0,
                detail: "index entry out of bounds".into(),
            });
        }
        let logical_block =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let leaf_lo =
            u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let leaf_hi = u16::from_le_bytes([data[off + 8], data[off + 9]]);
        let leaf_block = u64::from(leaf_lo) | (u64::from(leaf_hi) << 32);

        indexes.push(Ext4ExtentIndex {
            logical_block,
            leaf_block,
        });
    }
    Ok(indexes)
}

/// Max entries for an external (non-root) node at the given block size.
/// `(block_size - header_size - tail_checksum_size) / entry_size`.
fn max_entries_external(block_size: u32) -> u16 {
    let usable = (block_size as usize)
        .saturating_sub(HEADER_SIZE)
        .saturating_sub(TAIL_SIZE);
    // Safe: max block size is 64K, so result fits in u16 easily.
    #[expect(clippy::cast_possible_truncation)]
    let max = (usable / ENTRY_SIZE) as u16;
    max
}

/// Actual extent length, masking off the unwritten bit.
fn actual_len(raw_len: u16) -> u16 {
    if raw_len <= EXT_INIT_MAX_LEN {
        raw_len
    } else {
        raw_len - EXT_INIT_MAX_LEN
    }
}

/// Encode length with optional unwritten flag.
fn encode_len(len: u16, unwritten: bool) -> u16 {
    if unwritten {
        len + EXT_INIT_MAX_LEN
    } else {
        len
    }
}

/// Find the index position to descend into for a given logical block.
fn find_index_pos(indexes: &[Ext4ExtentIndex], target: u32) -> usize {
    let pos = indexes.partition_point(|idx| idx.logical_block <= target);
    if pos > 0 { pos - 1 } else { 0 }
}

// ── Serialization helpers ───────────────────────────────────────────────────

/// Write a leaf root (depth 0) back to the 60-byte i_block area.
#[expect(clippy::cast_possible_truncation)]
fn write_leaf_root(root_bytes: &mut [u8; 60], header: &Ext4ExtentHeader, extents: &[Ext4Extent]) {
    root_bytes.fill(0);
    let new_header = Ext4ExtentHeader {
        magic: EXT4_EXTENT_MAGIC,
        entries: extents.len() as u16,
        max_entries: header.max_entries,
        depth: 0,
        generation: header.generation,
    };
    write_header(root_bytes, &new_header);
    for (i, ext) in extents.iter().enumerate() {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        write_leaf_entry(&mut root_bytes[off..off + ENTRY_SIZE], ext);
    }
}

/// Write an index root back to the 60-byte i_block area.
#[expect(clippy::cast_possible_truncation)]
fn write_index_root(
    root_bytes: &mut [u8; 60],
    header: &Ext4ExtentHeader,
    indexes: &[Ext4ExtentIndex],
) {
    root_bytes.fill(0);
    let new_header = Ext4ExtentHeader {
        magic: header.magic,
        entries: indexes.len() as u16,
        max_entries: header.max_entries,
        depth: header.depth,
        generation: header.generation,
    };
    write_header(root_bytes, &new_header);
    for (i, idx) in indexes.iter().enumerate() {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        write_index_entry(&mut root_bytes[off..off + ENTRY_SIZE], idx);
    }
}

/// Serialize a full leaf block (depth 0).
#[expect(clippy::cast_possible_truncation)]
fn serialize_leaf_block(block_size: u32, extents: &[Ext4Extent]) -> Vec<u8> {
    let bs = block_size as usize;
    let mut data = vec![0u8; bs];
    let max = max_entries_external(block_size);
    let header = Ext4ExtentHeader {
        magic: EXT4_EXTENT_MAGIC,
        entries: extents.len() as u16,
        max_entries: max,
        depth: 0,
        generation: 0,
    };
    write_header(&mut data, &header);
    for (i, ext) in extents.iter().enumerate() {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        write_leaf_entry(&mut data[off..off + ENTRY_SIZE], ext);
    }
    data
}

/// Serialize a full index block.
#[expect(clippy::cast_possible_truncation)]
fn serialize_index_block(block_size: u32, depth: u16, indexes: &[Ext4ExtentIndex]) -> Vec<u8> {
    let bs = block_size as usize;
    let mut data = vec![0u8; bs];
    let max = max_entries_external(block_size);
    let header = Ext4ExtentHeader {
        magic: EXT4_EXTENT_MAGIC,
        entries: indexes.len() as u16,
        max_entries: max,
        depth,
        generation: 0,
    };
    write_header(&mut data, &header);
    for (i, idx) in indexes.iter().enumerate() {
        let off = HEADER_SIZE + i * ENTRY_SIZE;
        write_index_entry(&mut data[off..off + ENTRY_SIZE], idx);
    }
    data
}

fn write_header(buf: &mut [u8], h: &Ext4ExtentHeader) {
    buf[0..2].copy_from_slice(&h.magic.to_le_bytes());
    buf[2..4].copy_from_slice(&h.entries.to_le_bytes());
    buf[4..6].copy_from_slice(&h.max_entries.to_le_bytes());
    buf[6..8].copy_from_slice(&h.depth.to_le_bytes());
    buf[8..12].copy_from_slice(&h.generation.to_le_bytes());
}

#[expect(clippy::cast_possible_truncation)]
fn write_leaf_entry(buf: &mut [u8], ext: &Ext4Extent) {
    buf[0..4].copy_from_slice(&ext.logical_block.to_le_bytes());
    buf[4..6].copy_from_slice(&ext.raw_len.to_le_bytes());
    let start_hi = (ext.physical_start >> 32) as u16;
    let start_lo = ext.physical_start as u32;
    buf[6..8].copy_from_slice(&start_hi.to_le_bytes());
    buf[8..12].copy_from_slice(&start_lo.to_le_bytes());
}

#[expect(clippy::cast_possible_truncation)]
fn write_index_entry(buf: &mut [u8], idx: &Ext4ExtentIndex) {
    buf[0..4].copy_from_slice(&idx.logical_block.to_le_bytes());
    let leaf_lo = idx.leaf_block as u32;
    let leaf_hi = (idx.leaf_block >> 32) as u16;
    buf[4..8].copy_from_slice(&leaf_lo.to_le_bytes());
    buf[8..10].copy_from_slice(&leaf_hi.to_le_bytes());
    buf[10..12].fill(0); // reserved
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::match_wildcard_for_single_variants,
    clippy::needless_range_loop,
    clippy::option_if_let_else,
    clippy::significant_drop_tightening
)]
mod tests {
    use super::*;
    use ffs_block::BlockBuf;
    use proptest::prelude::*;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Mutex;

    /// In-memory block device for testing.
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
            let mut blocks = self.blocks.lock().unwrap();
            blocks.insert(block.0, data.to_vec());
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

    /// Simple sequential block allocator.
    struct SeqAllocator {
        next: u64,
        freed: Vec<u64>,
    }

    impl SeqAllocator {
        fn new(start: u64) -> Self {
            Self {
                next: start,
                freed: Vec::new(),
            }
        }

        fn freed_blocks(&self) -> &[u64] {
            &self.freed
        }
    }

    impl BlockAllocator for SeqAllocator {
        fn alloc_block(&mut self, _cx: &Cx) -> Result<BlockNumber> {
            let bn = BlockNumber(self.next);
            self.next += 1;
            Ok(bn)
        }

        fn free_block(&mut self, _cx: &Cx, block: BlockNumber) -> Result<()> {
            self.freed.push(block.0);
            Ok(())
        }
    }

    fn make_root() -> [u8; 60] {
        let mut root = [0u8; 60];
        let header = Ext4ExtentHeader {
            magic: EXT4_EXTENT_MAGIC,
            entries: 0,
            max_entries: ROOT_MAX_ENTRIES,
            depth: 0,
            generation: 0,
        };
        write_header(&mut root, &header);
        root
    }

    fn test_cx() -> Cx {
        Cx::for_testing()
    }

    fn assert_sorted_non_overlapping(extents: &[Ext4Extent]) {
        for pair in extents.windows(2) {
            let left = pair[0];
            let right = pair[1];
            let left_end = left
                .logical_block
                .saturating_add(u32::from(left.actual_len()));
            assert!(
                left.logical_block < right.logical_block,
                "extents out of order"
            );
            assert!(left_end <= right.logical_block, "extents overlap");
        }
    }

    fn subtree_first_logical(
        cx: &Cx,
        dev: &MemBlockDevice,
        block: u64,
        expected_depth: u16,
    ) -> Result<Option<u32>> {
        let buf = dev.read_block(cx, BlockNumber(block))?;
        let data = buf.as_slice();
        let max = max_entries_external(dev.block_size());
        let (header, _) = parse_header(data)?;
        validate_header(&header, max)?;

        if header.depth != expected_depth {
            return Err(FfsError::Corruption {
                block,
                detail: format!(
                    "test invariant depth mismatch: expected {expected_depth}, got {}",
                    header.depth
                ),
            });
        }

        if expected_depth == 0 {
            let extents = parse_leaf_entries(data, &header)?;
            assert_sorted_non_overlapping(&extents);
            Ok(extents.first().map(|ext| ext.logical_block))
        } else {
            let indexes = parse_index_entries(data, &header)?;
            let mut previous = None;
            for idx in &indexes {
                if let Some(prev) = previous {
                    assert!(prev < idx.logical_block, "index entries are not sorted");
                }
                let child_first =
                    subtree_first_logical(cx, dev, idx.leaf_block, expected_depth - 1)?;
                assert_eq!(
                    child_first,
                    Some(idx.logical_block),
                    "separator key mismatch at block {}",
                    idx.leaf_block
                );
                previous = Some(idx.logical_block);
            }
            Ok(indexes.first().map(|idx| idx.logical_block))
        }
    }

    fn assert_tree_invariants(cx: &Cx, dev: &MemBlockDevice, root: &[u8; 60]) -> Result<()> {
        let (root_header, _) = parse_header(root)?;
        validate_header(&root_header, ROOT_MAX_ENTRIES)?;
        if root_header.depth == 0 {
            let root_extents = parse_leaf_entries(root, &root_header)?;
            assert_sorted_non_overlapping(&root_extents);
        } else {
            let root_indexes = parse_index_entries(root, &root_header)?;
            let mut previous = None;
            for idx in &root_indexes {
                if let Some(prev) = previous {
                    assert!(
                        prev < idx.logical_block,
                        "root index entries are not sorted"
                    );
                }
                let child_first =
                    subtree_first_logical(cx, dev, idx.leaf_block, root_header.depth - 1)?;
                assert_eq!(
                    child_first,
                    Some(idx.logical_block),
                    "root separator key mismatch at child block {}",
                    idx.leaf_block
                );
                previous = Some(idx.logical_block);
            }
        }

        let mut walked = Vec::new();
        let count = walk(cx, dev, root, &mut |ext| {
            walked.push(*ext);
            Ok(())
        })?;
        assert_eq!(count, walked.len(), "walk count mismatch");
        assert_sorted_non_overlapping(&walked);
        Ok(())
    }

    #[test]
    fn search_empty_tree_returns_hole() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let root = make_root();

        let result = search(&cx, &dev, &root, 0).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: u32::MAX });
    }

    #[test]
    fn insert_single_extent_and_search() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Search for block 0: should find it.
        let result = search(&cx, &dev, &root, 0).unwrap();
        assert_eq!(
            result,
            SearchResult::Found {
                extent: ext,
                offset_in_extent: 0,
            }
        );

        // Search for block 5: inside the extent.
        let result = search(&cx, &dev, &root, 5).unwrap();
        assert_eq!(
            result,
            SearchResult::Found {
                extent: ext,
                offset_in_extent: 5,
            }
        );

        // Search for block 10: hole (extends to end of address space).
        let result = search(&cx, &dev, &root, 10).unwrap();
        assert!(matches!(result, SearchResult::Hole { hole_len } if hole_len > 0));
    }

    #[test]
    fn insert_four_extents_fills_root() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for i in 0..4 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 10,
                physical_start: (i as u64) * 1000 + 500,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Verify all four can be found.
        for i in 0..4 {
            let result = search(&cx, &dev, &root, i * 100).unwrap();
            match result {
                SearchResult::Found {
                    extent,
                    offset_in_extent,
                } => {
                    assert_eq!(extent.logical_block, i * 100);
                    assert_eq!(offset_in_extent, 0);
                }
                _ => panic!("expected Found for block {}", i * 100),
            }
        }
    }

    #[test]
    fn insert_fifth_extent_causes_split() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Fill root (4 extents).
        for i in 0..4 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 10,
                physical_start: (i as u64) * 1000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Fifth insert: should trigger root split (depth 0 -> 1).
        let ext = Ext4Extent {
            logical_block: 400,
            raw_len: 10,
            physical_start: 4000,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Root should now be depth 1.
        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 1);

        // All 5 extents should be searchable.
        for i in 0..5 {
            let result = search(&cx, &dev, &root, i * 100).unwrap();
            match result {
                SearchResult::Found { extent, .. } => {
                    assert_eq!(extent.logical_block, i * 100);
                }
                _ => panic!("expected Found for block {}", i * 100),
            }
        }
    }

    #[test]
    fn walk_visits_all_extents_in_order() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert extents in reverse order.
        for i in (0..6).rev() {
            let ext = Ext4Extent {
                logical_block: i * 50,
                raw_len: 5,
                physical_start: (i as u64) * 100,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let mut visited = Vec::new();
        let count = walk(&cx, &dev, &root, &mut |ext| {
            visited.push(ext.logical_block);
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 6);
        // Should be in sorted order.
        for i in 0..6 {
            assert_eq!(visited[i], (i as u32) * 50);
        }
    }

    #[test]
    fn delete_range_removes_full_extent() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 10,
            raw_len: 5,
            physical_start: 200,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        let freed = delete_range(&cx, &dev, &mut root, 10, 5, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].physical_start, 200);
        assert_eq!(freed[0].count, 5);

        // Should be a hole now.
        let result = search(&cx, &dev, &root, 10).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn delete_range_trims_extent_left() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete blocks 5-9 (trim right portion).
        let freed = delete_range(&cx, &dev, &mut root, 5, 5, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].physical_start, 505);
        assert_eq!(freed[0].count, 5);

        // Block 0 should still be found.
        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert_eq!(extent.logical_block, 0);
                assert_eq!(actual_len(extent.raw_len), 5);
            }
            _ => panic!("expected Found"),
        }

        // Block 5 should be a hole.
        let result = search(&cx, &dev, &root, 5).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn delete_range_trims_extent_right() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 10,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete blocks 10-14 (trim left portion).
        let freed = delete_range(&cx, &dev, &mut root, 10, 5, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].physical_start, 500);
        assert_eq!(freed[0].count, 5);

        // Block 15 should still be found.
        let result = search(&cx, &dev, &root, 15).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert_eq!(extent.logical_block, 15);
                assert_eq!(actual_len(extent.raw_len), 5);
                assert_eq!(extent.physical_start, 505);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn search_with_hole_between_extents() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext1 = Ext4Extent {
            logical_block: 0,
            raw_len: 5,
            physical_start: 100,
        };
        let ext2 = Ext4Extent {
            logical_block: 20,
            raw_len: 5,
            physical_start: 200,
        };
        insert(&cx, &dev, &mut root, ext1, &mut alloc).unwrap();
        insert(&cx, &dev, &mut root, ext2, &mut alloc).unwrap();

        // Block 10 is in the hole between extents.
        let result = search(&cx, &dev, &root, 10).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: 10 });
    }

    #[test]
    fn unwritten_extent_flag_preserved() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10 + EXT_INIT_MAX_LEN, // unwritten
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert!(extent.is_unwritten());
                assert_eq!(extent.actual_len(), 10);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn max_entries_external_calculation() {
        // 4K block: (4096 - 12 - 4) / 12 = 340
        assert_eq!(max_entries_external(4096), 340);
        // 1K block: (1024 - 12 - 4) / 12 = 84
        assert_eq!(max_entries_external(1024), 84);
    }

    #[test]
    fn many_inserts_cause_multi_level_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(1000);

        // Insert enough extents to force depth > 1.
        // Root holds 4 index entries -> 4 children.
        // Each child holds 340 leaf extents.
        // So 4 × 340 = 1360 extents fill depth 1.
        // After 1361, we need depth 2.
        // But let's start with 20 to force at least depth 1.
        for i in 0..20 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 50_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert!(header.depth >= 1, "tree should have grown in depth");

        // All 20 extents should be searchable.
        for i in 0..20 {
            let result = search(&cx, &dev, &root, i * 100).unwrap();
            match result {
                SearchResult::Found { extent, .. } => {
                    assert_eq!(extent.logical_block, i * 100);
                }
                _ => panic!("expected Found for block {}", i * 100),
            }
        }

        // Walk should yield 20 in order.
        let mut visited = Vec::new();
        let count = walk(&cx, &dev, &root, &mut |ext| {
            visited.push(ext.logical_block);
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 20);
        for i in 0..20 {
            assert_eq!(visited[i], (i as u32) * 100);
        }
    }

    #[test]
    fn bad_magic_returns_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = [0u8; 60];
        // Write bad magic.
        root[0..2].copy_from_slice(&0xDEADu16.to_le_bytes());

        let result = search(&cx, &dev, &root, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("bad extent magic"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn delete_range_in_middle_splits_extent() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // One extent covering blocks 0-19.
        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 20,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete blocks 5-14 (hole punch in middle).
        let freed = delete_range(&cx, &dev, &mut root, 5, 10, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].physical_start, 505);
        assert_eq!(freed[0].count, 10);

        // Left portion: blocks 0-4.
        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert_eq!(extent.logical_block, 0);
                assert_eq!(actual_len(extent.raw_len), 5);
                assert_eq!(extent.physical_start, 500);
            }
            _ => panic!("expected left portion"),
        }

        // Hole: blocks 5-14.
        let result = search(&cx, &dev, &root, 5).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: 10 });

        // Right portion: blocks 15-19.
        let result = search(&cx, &dev, &root, 15).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert_eq!(extent.logical_block, 15);
                assert_eq!(actual_len(extent.raw_len), 5);
                assert_eq!(extent.physical_start, 515);
            }
            _ => panic!("expected right portion"),
        }
    }

    #[test]
    fn delete_range_updates_parent_separator_when_child_front_changes() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for logical in [0_u32, 10, 20, 30, 40] {
            let ext = Ext4Extent {
                logical_block: logical,
                raw_len: 5,
                physical_start: 10_000 + u64::from(logical),
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (before_header, _) = parse_header(&root).unwrap();
        assert_eq!(before_header.depth, 1);
        let before_indexes = parse_index_entries(&root, &before_header).unwrap();
        assert_eq!(before_indexes.len(), 2);
        assert_eq!(before_indexes[1].logical_block, 20);

        // Remove the first extent in the second leaf; separator should advance to 30.
        delete_range(&cx, &dev, &mut root, 20, 5, &mut alloc).unwrap();

        let (after_header, _) = parse_header(&root).unwrap();
        assert_eq!(after_header.depth, 1);
        let after_indexes = parse_index_entries(&root, &after_header).unwrap();
        assert_eq!(after_indexes.len(), 2);
        assert_eq!(after_indexes[1].logical_block, 30);
        assert!(alloc.freed_blocks().is_empty());

        let removed = search(&cx, &dev, &root, 20).unwrap();
        assert!(matches!(removed, SearchResult::Hole { .. }));
        let kept = search(&cx, &dev, &root, 30).unwrap();
        assert!(matches!(kept, SearchResult::Found { .. }));
        assert_tree_invariants(&cx, &dev, &root).unwrap();
    }

    #[test]
    fn delete_range_empties_leaf_and_shrinks_root_freeing_metadata_blocks() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for logical in [0_u32, 10, 20, 30, 40] {
            let ext = Ext4Extent {
                logical_block: logical,
                raw_len: 5,
                physical_start: 20_000 + u64::from(logical),
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        delete_range(&cx, &dev, &mut root, 20, 30, &mut alloc).unwrap();

        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 0);
        let root_extents = parse_leaf_entries(&root, &header).unwrap();
        assert_eq!(root_extents.len(), 2);
        assert_eq!(root_extents[0].logical_block, 0);
        assert_eq!(root_extents[1].logical_block, 10);
        assert_eq!(alloc.freed_blocks().len(), 2);
        assert!(alloc.freed_blocks().contains(&100));
        assert!(alloc.freed_blocks().contains(&101));

        let deleted = search(&cx, &dev, &root, 20).unwrap();
        assert!(matches!(deleted, SearchResult::Hole { .. }));
        assert_tree_invariants(&cx, &dev, &root).unwrap();
    }

    #[test]
    fn walk_empty_tree_visits_nothing() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let root = make_root();

        let mut visited = Vec::new();
        let count = walk(&cx, &dev, &root, &mut |ext| {
            visited.push(*ext);
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 0);
        assert!(visited.is_empty());
    }

    #[test]
    fn delete_range_no_overlap_is_noop() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 10,
            raw_len: 5,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete range entirely before the extent.
        let freed = delete_range(&cx, &dev, &mut root, 0, 5, &mut alloc).unwrap();
        assert!(freed.is_empty());

        // Delete range entirely after the extent.
        let freed = delete_range(&cx, &dev, &mut root, 20, 10, &mut alloc).unwrap();
        assert!(freed.is_empty());

        // Original extent still intact.
        let result = search(&cx, &dev, &root, 10).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert_eq!(extent.logical_block, 10);
                assert_eq!(actual_len(extent.raw_len), 5);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn search_at_exact_extent_boundary() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Last block of the extent (block 9).
        let result = search(&cx, &dev, &root, 9).unwrap();
        assert_eq!(
            result,
            SearchResult::Found {
                extent: ext,
                offset_in_extent: 9,
            }
        );

        // One past the end (block 10) should be a hole.
        let result = search(&cx, &dev, &root, 10).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn delete_all_extents_yields_empty_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for i in 0..3 {
            let ext = Ext4Extent {
                logical_block: i * 20,
                raw_len: 5,
                physical_start: (i as u64) * 100 + 500,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Delete everything.
        let freed = delete_range(&cx, &dev, &mut root, 0, 100, &mut alloc).unwrap();
        assert_eq!(freed.len(), 3);

        // Tree should be empty.
        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 0);
        assert_eq!(header.entries, 0);

        let result = search(&cx, &dev, &root, 0).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: u32::MAX });

        let count = walk(&cx, &dev, &root, &mut |_| Ok(())).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn header_entries_exceeds_max_returns_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = [0u8; 60];
        // Valid magic, but entries (5) > max_entries (4).
        let header = Ext4ExtentHeader {
            magic: EXT4_EXTENT_MAGIC,
            entries: 5,
            max_entries: ROOT_MAX_ENTRIES,
            depth: 0,
            generation: 0,
        };
        write_header(&mut root, &header);

        let result = search(&cx, &dev, &root, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("entries"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn header_too_small_returns_corruption() {
        let data = [0u8; 6]; // Less than HEADER_SIZE (12).
        let result = parse_header(&data);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("too small"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn walk_visitor_error_propagates() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for i in 0..3 {
            let ext = Ext4Extent {
                logical_block: i * 20,
                raw_len: 1,
                physical_start: (i as u64) * 100,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let mut visit_count = 0;
        let result = walk(&cx, &dev, &root, &mut |_ext| {
            visit_count += 1;
            if visit_count == 2 {
                return Err(FfsError::Corruption {
                    block: 0,
                    detail: "test abort".into(),
                });
            }
            Ok(())
        });

        assert!(result.is_err());
        assert_eq!(visit_count, 2);
    }

    #[test]
    fn delete_range_spans_multiple_extents() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Three non-contiguous extents.
        for i in 0..3 {
            let ext = Ext4Extent {
                logical_block: i * 10,
                raw_len: 5,
                physical_start: (i as u64) * 100 + 1000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Delete range that covers all three (0..30).
        let freed = delete_range(&cx, &dev, &mut root, 0, 30, &mut alloc).unwrap();
        assert_eq!(freed.len(), 3);

        // Verify all freed ranges.
        let mut physical_starts: Vec<u64> = freed.iter().map(|f| f.physical_start).collect();
        physical_starts.sort_unstable();
        assert_eq!(physical_starts, vec![1000, 1100, 1200]);

        // Tree should be empty now.
        let result = search(&cx, &dev, &root, 0).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn delete_range_preserves_unwritten_flag_on_trim() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 20 + EXT_INIT_MAX_LEN, // unwritten
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete blocks 10-19, keeping 0-9.
        let freed = delete_range(&cx, &dev, &mut root, 10, 10, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].count, 10);

        // Remaining extent should still be unwritten.
        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Found { extent, .. } => {
                assert!(extent.is_unwritten());
                assert_eq!(extent.actual_len(), 10);
                assert_eq!(extent.physical_start, 500);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn search_hole_in_multi_level_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(1000);

        // Insert 10 extents with large gaps (forces depth >= 1).
        for i in 0..10 {
            let ext = Ext4Extent {
                logical_block: i * 1000,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 50_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert!(header.depth >= 1);

        // Search in a gap between extents.
        let result = search(&cx, &dev, &root, 500).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: 500 });

        // Search past all extents.
        let result = search(&cx, &dev, &root, 9500).unwrap();
        assert!(matches!(result, SearchResult::Hole { hole_len } if hole_len > 0));
    }

    #[test]
    fn insert_reverse_order_walk_yields_sorted() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert 10 extents in strictly reverse order.
        for i in (0..10).rev() {
            let ext = Ext4Extent {
                logical_block: i * 50,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 2000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let mut visited = Vec::new();
        let count = walk(&cx, &dev, &root, &mut |ext| {
            visited.push(ext.logical_block);
            Ok(())
        })
        .unwrap();

        assert_eq!(count, 10);
        for i in 0..10 {
            assert_eq!(visited[i], (i as u32) * 50);
        }

        // Verify all extents are searchable.
        for i in 0..10 {
            let result = search(&cx, &dev, &root, i * 50).unwrap();
            match result {
                SearchResult::Found { extent, .. } => {
                    assert_eq!(extent.logical_block, i * 50);
                }
                _ => panic!("expected Found for block {}", i * 50),
            }
        }
    }

    // ── Error-path and edge-case hardening tests ─────────────────────────

    struct FailAfter {
        remaining: usize,
    }
    impl BlockAllocator for FailAfter {
        fn alloc_block(&mut self, _cx: &Cx) -> Result<BlockNumber> {
            if self.remaining == 0 {
                return Err(FfsError::NoSpace);
            }
            self.remaining -= 1;
            // Return a unique block each time.
            Ok(BlockNumber(500 + self.remaining as u64))
        }
        fn free_block(&mut self, _cx: &Cx, _block: BlockNumber) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn alloc_failure_during_root_split_propagates_error() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();

        // Fill root with 4 extents using a working allocator.
        let mut alloc = SeqAllocator::new(100);
        for i in 0..4 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // 5th insert triggers root split which needs 2 blocks; give it 0.
        let mut fail_alloc = FailAfter { remaining: 0 };
        let ext5 = Ext4Extent {
            logical_block: 400,
            raw_len: 1,
            physical_start: 4000,
        };
        let result = insert(&cx, &dev, &mut root, ext5, &mut fail_alloc);
        assert!(result.is_err(), "alloc failure should propagate");
    }

    #[test]
    fn depth_mismatch_in_child_returns_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert 5 extents to create a depth-1 tree.
        for i in 0..5 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 1);

        // Corrupt a child block: overwrite its header with wrong depth.
        let indexes = parse_index_entries(&root, &header).unwrap();
        let child_block = indexes[0].leaf_block;
        let buf = dev.read_block(&cx, BlockNumber(child_block)).unwrap();
        let mut child_data = buf.as_slice().to_vec();
        // Set depth to 5 (should be 0).
        child_data[6..8].copy_from_slice(&5_u16.to_le_bytes());
        dev.write_block(&cx, BlockNumber(child_block), &child_data)
            .unwrap();

        // Search should detect depth mismatch.
        let result = search(&cx, &dev, &root, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("depth mismatch"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn walk_depth_mismatch_in_child_returns_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for i in 0..5 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        let indexes = parse_index_entries(&root, &header).unwrap();
        let child_block = indexes[0].leaf_block;
        let buf = dev.read_block(&cx, BlockNumber(child_block)).unwrap();
        let mut child_data = buf.as_slice().to_vec();
        child_data[6..8].copy_from_slice(&3_u16.to_le_bytes());
        dev.write_block(&cx, BlockNumber(child_block), &child_data)
            .unwrap();

        let result = walk(&cx, &dev, &root, &mut |_| Ok(()));
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("depth mismatch"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn delete_range_on_empty_tree_is_noop() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let freed = delete_range(&cx, &dev, &mut root, 0, 100, &mut alloc).unwrap();
        assert!(freed.is_empty());

        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.entries, 0);
        assert_eq!(header.depth, 0);
    }

    #[test]
    fn insert_at_u32_max_logical_block() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: u32::MAX - 1,
            raw_len: 1,
            physical_start: 999,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        let result = search(&cx, &dev, &root, u32::MAX - 1).unwrap();
        assert_eq!(
            result,
            SearchResult::Found {
                extent: ext,
                offset_in_extent: 0,
            }
        );
    }

    #[test]
    fn search_u32_max_in_empty_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let root = make_root();

        let result = search(&cx, &dev, &root, u32::MAX).unwrap();
        assert_eq!(result, SearchResult::Hole { hole_len: u32::MAX });
    }

    #[test]
    fn insert_and_delete_all_then_reinsert() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert 10 extents forcing multi-level tree.
        for i in 0..10 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000 + 50_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Delete everything.
        delete_range(&cx, &dev, &mut root, 0, 10000, &mut alloc).unwrap();

        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 0);
        assert_eq!(header.entries, 0);

        // Reinsert — tree should work from scratch.
        for i in 0..3 {
            let ext = Ext4Extent {
                logical_block: i * 10,
                raw_len: 5,
                physical_start: (i as u64) * 100 + 80_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        for i in 0..3 {
            let result = search(&cx, &dev, &root, i * 10).unwrap();
            match result {
                SearchResult::Found { extent, .. } => {
                    assert_eq!(extent.logical_block, i * 10);
                }
                _ => panic!("expected Found after reinsert for block {}", i * 10),
            }
        }
    }

    #[test]
    fn max_entries_external_1k_block() {
        // 1K block: (1024 - 12 - 4) / 12 = 84
        assert_eq!(max_entries_external(1024), 84);
    }

    #[test]
    fn max_entries_external_2k_block() {
        // 2K: (2048 - 12 - 4) / 12 = 169
        assert_eq!(max_entries_external(2048), 169);
    }

    #[test]
    fn max_entries_external_tiny_block_saturates() {
        // Block size smaller than header+tail: should return 0.
        assert_eq!(max_entries_external(12), 0);
        assert_eq!(max_entries_external(0), 0);
    }

    #[test]
    fn header_max_entries_exceeds_allowed_returns_corruption() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = [0u8; 60];
        // Valid magic, but max_entries (100) > ROOT_MAX_ENTRIES (4).
        let header = Ext4ExtentHeader {
            magic: EXT4_EXTENT_MAGIC,
            entries: 1,
            max_entries: 100,
            depth: 0,
            generation: 0,
        };
        write_header(&mut root, &header);

        let result = search(&cx, &dev, &root, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            FfsError::Corruption { detail, .. } => {
                assert!(detail.contains("max_entries"));
            }
            other => panic!("expected Corruption, got {other:?}"),
        }
    }

    #[test]
    fn encode_decode_len_roundtrip() {
        for len in [1, 10, 100, 32767] {
            assert_eq!(actual_len(encode_len(len, false)), len);
            assert_eq!(actual_len(encode_len(len, true)), len);

            let written = encode_len(len, true);
            assert!(written > EXT_INIT_MAX_LEN);
            let normal = encode_len(len, false);
            assert!(normal <= EXT_INIT_MAX_LEN);
        }
    }

    #[test]
    fn delete_range_with_count_zero_is_noop() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 10,
            raw_len: 5,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // count=0 means empty range, should delete nothing.
        let freed = delete_range(&cx, &dev, &mut root, 10, 0, &mut alloc).unwrap();
        assert!(freed.is_empty());

        let result = search(&cx, &dev, &root, 10).unwrap();
        assert!(matches!(result, SearchResult::Found { .. }));
    }

    #[test]
    fn delete_range_saturating_end_does_not_panic() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: u32::MAX - 10,
            raw_len: 5,
            physical_start: 777,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete starting near u32::MAX with large count — should saturate, not overflow.
        let freed =
            delete_range(&cx, &dev, &mut root, u32::MAX - 10, u32::MAX, &mut alloc).unwrap();
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].physical_start, 777);
        assert_eq!(freed[0].count, 5);
    }

    #[test]
    fn insert_many_then_delete_individually() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(1000);

        // Insert 15 extents (forces depth >= 1).
        for i in 0..15 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000 + 50_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Delete them one by one in reverse order.
        for i in (0..15).rev() {
            let freed = delete_range(&cx, &dev, &mut root, i * 100, 1, &mut alloc).unwrap();
            assert_eq!(
                freed.len(),
                1,
                "should free exactly 1 range for block {}",
                i * 100
            );
            assert_tree_invariants(&cx, &dev, &root).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert_eq!(header.depth, 0);
        assert_eq!(header.entries, 0);
    }

    #[test]
    fn insert_interleaved_with_deletes_preserves_invariants() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(1000);

        // Insert 8 extents.
        for i in 0..8 {
            let ext = Ext4Extent {
                logical_block: i * 50,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 10_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }
        assert_tree_invariants(&cx, &dev, &root).unwrap();

        // Delete extents at positions 100, 200.
        delete_range(&cx, &dev, &mut root, 100, 1, &mut alloc).unwrap();
        assert_tree_invariants(&cx, &dev, &root).unwrap();
        delete_range(&cx, &dev, &mut root, 200, 1, &mut alloc).unwrap();
        assert_tree_invariants(&cx, &dev, &root).unwrap();

        // Insert new extents at those positions.
        let ext_a = Ext4Extent {
            logical_block: 100,
            raw_len: 1,
            physical_start: 99_000,
        };
        insert(&cx, &dev, &mut root, ext_a, &mut alloc).unwrap();
        assert_tree_invariants(&cx, &dev, &root).unwrap();

        let ext_b = Ext4Extent {
            logical_block: 200,
            raw_len: 1,
            physical_start: 99_100,
        };
        insert(&cx, &dev, &mut root, ext_b, &mut alloc).unwrap();
        assert_tree_invariants(&cx, &dev, &root).unwrap();

        // Verify the new extents are searchable.
        let r = search(&cx, &dev, &root, 100).unwrap();
        assert!(matches!(r, SearchResult::Found { extent, .. } if extent.physical_start == 99_000));
        let r = search(&cx, &dev, &root, 200).unwrap();
        assert!(matches!(r, SearchResult::Found { extent, .. } if extent.physical_start == 99_100));
    }

    #[test]
    fn small_block_size_tree_operations() {
        // Use 1K block size: max_entries_external(1024) = 84.
        let cx = test_cx();
        let dev = MemBlockDevice::new(1024);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert 8 extents (forces split with smaller blocks).
        for i in 0..8 {
            let ext = Ext4Extent {
                logical_block: i * 50,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 5_000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert!(header.depth >= 1, "1K block tree should also grow");

        for i in 0..8 {
            let result = search(&cx, &dev, &root, i * 50).unwrap();
            assert!(matches!(result, SearchResult::Found { .. }));
        }

        assert_tree_invariants(&cx, &dev, &root).unwrap();
    }

    // ── Edge-case and boundary tests (bd-36j1) ────────────────────────

    #[test]
    fn search_target_u32_max_returns_hole() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert an extent at block 0.
        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Searching at u32::MAX should return a hole.
        let result = search(&cx, &dev, &root, u32::MAX).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn search_block_zero_in_populated_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert extents starting at block 5 and 20.
        insert(
            &cx,
            &dev,
            &mut root,
            Ext4Extent {
                logical_block: 5,
                raw_len: 3,
                physical_start: 500,
            },
            &mut alloc,
        )
        .unwrap();
        insert(
            &cx,
            &dev,
            &mut root,
            Ext4Extent {
                logical_block: 20,
                raw_len: 2,
                physical_start: 600,
            },
            &mut alloc,
        )
        .unwrap();

        // Block 0 should be a hole with length 5 (until the first extent).
        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Hole { hole_len } => {
                assert_eq!(hole_len, 5, "hole should extend to block 5");
            }
            _ => panic!("expected Hole, got {result:?}"),
        }
    }

    #[test]
    fn insert_max_raw_len_extent() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert extent with max valid length (32767 blocks = 0x7FFF).
        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 0x7FFF,
            physical_start: 1000,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        let result = search(&cx, &dev, &root, 0).unwrap();
        match result {
            SearchResult::Found {
                extent,
                offset_in_extent,
            } => {
                assert_eq!(offset_in_extent, 0);
                assert_eq!(extent.actual_len(), 0x7FFF);
            }
            _ => panic!("expected Found"),
        }

        // Search at block 32766 (last block of extent) should hit.
        let result = search(&cx, &dev, &root, 32766).unwrap();
        assert!(matches!(result, SearchResult::Found { .. }));

        // Block 32767 should be a hole.
        let result = search(&cx, &dev, &root, 32767).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn delete_range_u32_max_boundary() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: 10,
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Delete with logical_start near u32::MAX — should not panic from overflow.
        let freed = delete_range(&cx, &dev, &mut root, u32::MAX - 5, 10, &mut alloc).unwrap();
        assert!(freed.is_empty(), "nothing to delete at u32::MAX region");

        // Original extent should still be intact.
        let result = search(&cx, &dev, &root, 0).unwrap();
        assert!(matches!(result, SearchResult::Found { .. }));
    }

    #[test]
    fn freed_range_debug_clone_eq() {
        let a = FreedRange {
            physical_start: 100,
            count: 10,
        };
        let b = a;
        assert_eq!(a, b);
        let _ = format!("{a:?}");
    }

    #[test]
    fn search_result_debug_clone_eq() {
        let found = SearchResult::Found {
            extent: Ext4Extent {
                logical_block: 0,
                raw_len: 1,
                physical_start: 100,
            },
            offset_in_extent: 0,
        };
        let cloned = found.clone();
        assert_eq!(found, cloned);
        let _ = format!("{found:?}");

        let hole = SearchResult::Hole { hole_len: 42 };
        let _ = format!("{hole:?}");
        assert_ne!(found, hole);
    }

    #[test]
    fn walk_after_deleting_all_from_multi_level_tree() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert enough extents to force a multi-level tree.
        for i in 0..10 {
            let ext = Ext4Extent {
                logical_block: i * 100,
                raw_len: 1,
                physical_start: (i as u64) * 1000 + 5000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let (header, _) = parse_header(&root).unwrap();
        assert!(header.depth >= 1, "should be multi-level");

        // Delete everything.
        delete_range(&cx, &dev, &mut root, 0, u32::MAX, &mut alloc).unwrap();

        // Walk should visit 0 extents.
        let mut count = 0;
        walk(&cx, &dev, &root, &mut |_: &Ext4Extent| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 0, "tree should be empty after deleting all");

        // Search anything should return Hole.
        let result = search(&cx, &dev, &root, 0).unwrap();
        assert!(matches!(result, SearchResult::Hole { .. }));
    }

    #[test]
    fn insert_adjacent_extents_stay_separate() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert two adjacent extents at [0..5) and [5..10).
        insert(
            &cx,
            &dev,
            &mut root,
            Ext4Extent {
                logical_block: 0,
                raw_len: 5,
                physical_start: 1000,
            },
            &mut alloc,
        )
        .unwrap();
        insert(
            &cx,
            &dev,
            &mut root,
            Ext4Extent {
                logical_block: 5,
                raw_len: 5,
                physical_start: 2000,
            },
            &mut alloc,
        )
        .unwrap();

        // Both extents should exist as separate entries.
        let mut walked = Vec::new();
        walk(&cx, &dev, &root, &mut |ext| {
            walked.push(*ext);
            Ok(())
        })
        .unwrap();
        assert_eq!(walked.len(), 2);
        assert_eq!(walked[0].logical_block, 0);
        assert_eq!(walked[0].physical_start, 1000);
        assert_eq!(walked[1].logical_block, 5);
        assert_eq!(walked[1].physical_start, 2000);

        // Search at block 4 hits first, block 5 hits second.
        match search(&cx, &dev, &root, 4).unwrap() {
            SearchResult::Found {
                extent,
                offset_in_extent,
            } => {
                assert_eq!(extent.physical_start, 1000);
                assert_eq!(offset_in_extent, 4);
            }
            _ => panic!("expected Found for block 4"),
        }
        match search(&cx, &dev, &root, 5).unwrap() {
            SearchResult::Found {
                extent,
                offset_in_extent,
            } => {
                assert_eq!(extent.physical_start, 2000);
                assert_eq!(offset_in_extent, 0);
            }
            _ => panic!("expected Found for block 5"),
        }
    }

    #[test]
    fn delete_partial_from_multi_level_preserves_remainder() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert 8 single-block extents to force multi-level tree.
        for i in 0..8 {
            let ext = Ext4Extent {
                logical_block: i * 10,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 5000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        // Delete the middle range [20..50) — should remove blocks at 20, 30, 40.
        let freed = delete_range(&cx, &dev, &mut root, 20, 30, &mut alloc).unwrap();
        let freed_blocks: u16 = freed.iter().map(|f| f.count).sum();
        assert_eq!(freed_blocks, 3, "should free blocks at logical 20, 30, 40");

        // Remaining should be [0, 10, 50, 60, 70].
        let mut remaining = Vec::new();
        walk(&cx, &dev, &root, &mut |ext| {
            remaining.push(ext.logical_block);
            Ok(())
        })
        .unwrap();
        assert_eq!(remaining, vec![0, 10, 50, 60, 70]);
        assert_tree_invariants(&cx, &dev, &root).unwrap();
    }

    #[test]
    fn insert_unwritten_then_search_preserves_flag() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        // Insert an unwritten extent (bit 15 set in raw_len).
        let ext = Ext4Extent {
            logical_block: 0,
            raw_len: EXT_INIT_MAX_LEN | 0x0A, // 0x8000 | 0x0A = unwritten, 10 blocks
            physical_start: 500,
        };
        insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();

        // Searching should find the extent with unwritten flag.
        match search(&cx, &dev, &root, 5).unwrap() {
            SearchResult::Found {
                extent,
                offset_in_extent,
            } => {
                assert_eq!(offset_in_extent, 5);
                assert!(
                    extent.raw_len & EXT_INIT_MAX_LEN != 0,
                    "unwritten flag should be preserved"
                );
                assert_eq!(extent.actual_len(), 10);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn walk_count_matches_inserted_count() {
        let cx = test_cx();
        let dev = MemBlockDevice::new(4096);
        let mut root = make_root();
        let mut alloc = SeqAllocator::new(100);

        for i in 0..20 {
            let ext = Ext4Extent {
                logical_block: i * 50,
                raw_len: 1,
                physical_start: (i as u64) * 100 + 5000,
            };
            insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
        }

        let count = walk(&cx, &dev, &root, &mut |_: &Ext4Extent| Ok(())).unwrap();
        assert_eq!(count, 20, "walk should visit exactly 20 extents");
        assert_tree_invariants(&cx, &dev, &root).unwrap();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn mutation_sequences_preserve_sorted_non_overlapping_tree(
            insert_keys in proptest::collection::vec(0_u16..300_u16, 1..80),
            delete_ranges in proptest::collection::vec((0_u16..600_u16, 1_u16..25_u16), 0..80),
        ) {
            let cx = test_cx();
            let dev = MemBlockDevice::new(4096);
            let mut root = make_root();
            let mut alloc = SeqAllocator::new(10_000);
            let mut model = BTreeMap::<u32, u64>::new();

            for key in insert_keys {
                let logical = u32::from(key) * 2;
                if model.contains_key(&logical) {
                    continue;
                }
                let physical = 1_000_000 + u64::from(logical);
                let ext = Ext4Extent {
                    logical_block: logical,
                    raw_len: 1,
                    physical_start: physical,
                };
                insert(&cx, &dev, &mut root, ext, &mut alloc).unwrap();
                model.insert(logical, physical);
            }

            for (start, count) in delete_ranges {
                let delete_start = u32::from(start);
                let delete_end = delete_start.saturating_add(u32::from(count));
                delete_range(&cx, &dev, &mut root, delete_start, u32::from(count), &mut alloc).unwrap();
                model.retain(|logical, _| *logical < delete_start || *logical >= delete_end);
                assert_tree_invariants(&cx, &dev, &root).unwrap();
            }

            let mut walked = Vec::new();
            walk(&cx, &dev, &root, &mut |ext| {
                walked.push(*ext);
                Ok(())
            })
            .unwrap();
            prop_assert_eq!(walked.len(), model.len());
            for extent in walked {
                prop_assert_eq!(extent.actual_len(), 1);
                prop_assert_eq!(
                    model.get(&extent.logical_block).copied(),
                    Some(extent.physical_start)
                );
            }

            for probe in 0_u32..600 {
                let result = search(&cx, &dev, &root, probe).unwrap();
                if let Some(expected_physical) = model.get(&probe).copied() {
                    match result {
                        SearchResult::Found {
                            extent,
                            offset_in_extent,
                        } => {
                            prop_assert_eq!(offset_in_extent, 0);
                            prop_assert_eq!(extent.logical_block, probe);
                            prop_assert_eq!(extent.actual_len(), 1);
                            prop_assert_eq!(extent.physical_start, expected_physical);
                        }
                        SearchResult::Hole { .. } => {
                            prop_assert!(false, "reference model expected mapped block at {probe}");
                        }
                    }
                } else if !matches!(result, SearchResult::Hole { .. }) {
                    prop_assert!(false, "reference model expected hole at {probe}");
                }
            }
        }
    }
}
