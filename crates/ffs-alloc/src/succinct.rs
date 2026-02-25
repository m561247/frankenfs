//! Succinct rank/select bitmap for O(1) free-space queries.
//!
//! Provides [`SuccinctBitmap`], an immutable acceleration index built over a
//! raw `&[u8]` bitmap. The index enables:
//!
//! - **`rank0(i)`**: count zero (free) bits in `[0, i)` — O(1)
//! - **`rank1(i)`**: count one (allocated) bits in `[0, i)` — O(1)
//! - **`select0(k)`**: find position of the k-th zero bit — O(log n)
//! - **`select1(k)`**: find position of the k-th one bit — O(log n)
//!
//! # Design
//!
//! Two-level structure following Jacobson (1989) / Clark (1996):
//!
//! - **Level 1 (superblocks)**: every [`SUPERBLOCK_BITS`] bits, store cumulative
//!   popcount of *all* preceding bits.
//! - **Level 2 (blocks)**: every [`BLOCK_BITS`] bits within a superblock, store
//!   the local popcount since the start of that superblock.
//!
//! ## Space overhead
//!
//! For an n-bit (n/8 byte) bitmap:
//! - Superblocks: `ceil(n / 2048) * 4` bytes (~1.7% of bitmap)
//! - Blocks: `ceil(n / 256) * 2` bytes (~6.3% of bitmap)
//! - **Total overhead: ~7.9%** of bitmap storage (under the 10% target)
//!
//! ## Complexity
//!
//! | Operation | Time | Space |
//! |-----------|------|-------|
//! | `build()` | O(n) | n + o(n) bits |
//! | `rank0/1` | O(1) | — |
//! | `select0/1` | O(log(n/S)) | — |
//! | `find_free` | O(log n) | — |
//! | `find_contiguous` | O(n) worst case, fast skip | — |
//!
//! # `unsafe_code = "forbid"` Compliance
//!
//! All operations are safe Rust, using standard arithmetic and indexing.

/// Bits per superblock (level-1 chunk).
///
/// Chosen so that the two-level index stays under 10% of bitmap size.
/// For a 32 768-bit ext4 bitmap (4 096 bytes): 17 superblocks × 4 B = 68 B.
const SUPERBLOCK_BITS: u32 = 2048;

/// Bits per block (level-2 chunk).
///
/// For a 32 768-bit bitmap: 128 blocks × 2 B = 256 B.
/// Total index: 324 B ≈ 7.9% of 4 096 B bitmap.
const BLOCK_BITS: u32 = 256;

/// Blocks per superblock.
const BLOCKS_PER_SUPER: u32 = SUPERBLOCK_BITS / BLOCK_BITS;

/// Succinct rank/select index over a raw bitmap.
///
/// Built from a `&[u8]` bitmap slice, this structure caches popcount
/// information at two levels to enable O(1) rank and O(log n) select.
///
/// The original bitmap data is NOT stored — callers retain ownership.
/// The index only stores the acceleration tables.
#[derive(Debug, Clone)]
pub struct SuccinctBitmap {
    /// Total number of valid bits (may be less than `bytes.len() * 8`).
    len: u32,
    /// Total number of set (one) bits.
    ones: u32,
    /// Level-1: cumulative popcount at each superblock boundary.
    /// `superblocks[i]` = number of 1-bits in positions `[0, i * SUPERBLOCK_BITS)`.
    superblocks: Vec<u32>,
    /// Level-2: local popcount within each superblock.
    /// `blocks[j]` = number of 1-bits from the start of this block's superblock
    /// to position `j * BLOCK_BITS` (relative to superblock start).
    blocks: Vec<u16>,
    /// Cache of the raw bitmap bytes for rank/select within a block.
    /// We store this to avoid requiring the caller to pass the bitmap
    /// on every query. For a 4096-byte ext4 bitmap, this is 4 KiB.
    data: Vec<u8>,
}

impl SuccinctBitmap {
    /// Build a succinct index over `bitmap` considering the first `len` bits.
    ///
    /// # Panics
    ///
    /// Panics if `len > bitmap.len() * 8`.
    #[must_use]
    pub fn build(bitmap: &[u8], len: u32) -> Self {
        assert!(
            len as usize <= bitmap.len().saturating_mul(8),
            "len ({len}) exceeds bitmap capacity ({})",
            bitmap.len() * 8
        );

        let num_superblocks = len.div_ceil(SUPERBLOCK_BITS);
        let num_blocks = len.div_ceil(BLOCK_BITS);

        let mut superblocks = Vec::with_capacity(num_superblocks as usize + 1);
        let mut blocks = Vec::with_capacity(num_blocks as usize);

        let mut cumulative = 0_u32;
        let mut super_local = 0_u32;

        for block_idx in 0..num_blocks {
            let bit_start = block_idx * BLOCK_BITS;
            let block_within_super = block_idx % BLOCKS_PER_SUPER;

            // Start of a new superblock?
            if block_within_super == 0 {
                superblocks.push(cumulative);
                super_local = 0;
            }

            // Store local popcount within this superblock.
            #[expect(clippy::cast_possible_truncation)]
            let local = super_local as u16;
            blocks.push(local);

            // Count ones in this 64-bit block.
            let popcount = popcount_range(bitmap, bit_start, BLOCK_BITS.min(len - bit_start));
            cumulative += popcount;
            super_local += popcount;
        }

        // Sentinel superblock for the end.
        superblocks.push(cumulative);

        Self {
            len,
            ones: cumulative,
            superblocks,
            blocks,
            data: bitmap[..len.div_ceil(8) as usize].to_vec(),
        }
    }

    /// Total number of bits in the bitmap.
    #[must_use]
    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether the bitmap is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of set (one) bits.
    #[must_use]
    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.ones
    }

    /// Total number of zero (free) bits.
    #[must_use]
    #[inline]
    pub fn count_zeros(&self) -> u32 {
        self.len - self.ones
    }

    /// Count the number of 1-bits in positions `[0, i)`.
    ///
    /// Returns 0 if `i == 0`. Saturates at `self.len()`.
    #[must_use]
    pub fn rank1(&self, i: u32) -> u32 {
        let i = i.min(self.len);
        if i == 0 {
            return 0;
        }
        if i == self.len {
            // `blocks` stores one entry per logical block start. When `i`
            // lands exactly on `len` and `len` is block-aligned, `i / BLOCK_BITS`
            // points one past the final block entry.
            return self.ones;
        }

        let super_idx = (i / SUPERBLOCK_BITS) as usize;
        let block_idx = (i / BLOCK_BITS) as usize;
        let bit_offset = i % BLOCK_BITS;

        let mut rank = self.superblocks[super_idx];
        rank += u32::from(self.blocks[block_idx]);

        // Count remaining bits within the partial block.
        if bit_offset > 0 {
            let block_start = i - bit_offset; // == block_idx * BLOCK_BITS
            rank += popcount_range(&self.data, block_start, bit_offset);
        }

        rank
    }

    /// Count the number of 0-bits in positions `[0, i)`.
    #[must_use]
    #[inline]
    pub fn rank0(&self, i: u32) -> u32 {
        let i = i.min(self.len);
        i - self.rank1(i)
    }

    /// Find the position of the k-th 1-bit (0-indexed).
    ///
    /// Returns `None` if there are fewer than `k + 1` one-bits.
    #[must_use]
    #[expect(clippy::cast_possible_truncation)] // indices bounded by self.len (u32)
    pub fn select1(&self, k: u32) -> Option<u32> {
        if k >= self.ones {
            return None;
        }

        // Binary search over superblocks for the containing superblock.
        let mut lo = 0_usize;
        let mut hi = self.superblocks.len() - 1;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.superblocks[mid + 1] <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let super_idx = lo;
        let mut remaining = k - self.superblocks[super_idx];

        // Linear search within the superblock's blocks.
        let block_start = super_idx * BLOCKS_PER_SUPER as usize;
        let block_end = (block_start + BLOCKS_PER_SUPER as usize).min(self.blocks.len());

        let mut block_idx = block_start;
        for bi in block_start..block_end {
            let next_bi = bi + 1;
            let next_local = if next_bi < block_end {
                u32::from(self.blocks[next_bi])
            } else {
                // End of superblock: use next superblock cumulative.
                self.superblocks[super_idx + 1] - self.superblocks[super_idx]
            };
            if next_local > remaining {
                block_idx = bi;
                remaining -= u32::from(self.blocks[bi]);
                break;
            }
        }

        // Scan within the 64-bit block.
        let bit_base = (block_idx as u32) * BLOCK_BITS;
        let bits_in_block = BLOCK_BITS.min(self.len - bit_base);

        for bit in 0..bits_in_block {
            let pos = bit_base + bit;
            if self.get_bit(pos) {
                if remaining == 0 {
                    return Some(pos);
                }
                remaining -= 1;
            }
        }

        None // Should not reach here if k < ones.
    }

    /// Find the position of the k-th 0-bit (0-indexed).
    ///
    /// Returns `None` if there are fewer than `k + 1` zero-bits.
    #[must_use]
    #[expect(clippy::cast_possible_truncation)] // indices bounded by self.len (u32)
    pub fn select0(&self, k: u32) -> Option<u32> {
        let zeros = self.count_zeros();
        if k >= zeros {
            return None;
        }

        // Binary search over superblocks for the containing superblock.
        // zeros_before_super[i] = i * SUPERBLOCK_BITS - superblocks[i]
        let mut lo = 0_usize;
        let mut hi = self.superblocks.len() - 1;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let zeros_before_next =
                ((mid + 1) as u32) * SUPERBLOCK_BITS - self.superblocks[mid + 1];
            let zeros_before_next = zeros_before_next.min(zeros);
            if zeros_before_next <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let super_idx = lo;
        let zeros_before_super = (super_idx as u32) * SUPERBLOCK_BITS - self.superblocks[super_idx];
        let mut remaining = k - zeros_before_super;

        // Linear search within blocks.
        let block_start = super_idx * BLOCKS_PER_SUPER as usize;
        let block_end = (block_start + BLOCKS_PER_SUPER as usize).min(self.blocks.len());

        let mut block_idx = block_start;
        for bi in block_start..block_end {
            let block_bit_start = (bi as u32) * BLOCK_BITS;
            let local_ones = u32::from(self.blocks[bi]);
            let local_zeros = block_bit_start - (super_idx as u32) * SUPERBLOCK_BITS - local_ones;

            let next_bi = bi + 1;
            let next_zeros = if next_bi < block_end {
                let next_bit_start = (next_bi as u32) * BLOCK_BITS;
                let next_local_ones = u32::from(self.blocks[next_bi]);
                next_bit_start - (super_idx as u32) * SUPERBLOCK_BITS - next_local_ones
            } else {
                let end_bit = (self.len).min(((super_idx + 1) as u32) * SUPERBLOCK_BITS);
                let total_ones = self.superblocks[super_idx + 1] - self.superblocks[super_idx];
                end_bit - (super_idx as u32) * SUPERBLOCK_BITS - total_ones
            };

            if next_zeros > remaining {
                block_idx = bi;
                remaining -= local_zeros;
                break;
            }
        }

        // Scan within the block.
        let bit_base = (block_idx as u32) * BLOCK_BITS;
        let bits_in_block = BLOCK_BITS.min(self.len - bit_base);

        for bit in 0..bits_in_block {
            let pos = bit_base + bit;
            if !self.get_bit(pos) {
                if remaining == 0 {
                    return Some(pos);
                }
                remaining -= 1;
            }
        }

        None
    }

    /// Find the first zero bit at or after `start`, wrapping around.
    ///
    /// Equivalent to the allocator's "find first free" operation but
    /// accelerated via select0 with a starting position hint.
    #[must_use]
    pub fn find_free(&self, start: u32) -> Option<u32> {
        if self.count_zeros() == 0 {
            return None;
        }

        // Count zeros before `start` to determine which zero-bit index to seek.
        let zeros_before = self.rank0(start);
        let total_zeros = self.count_zeros();

        // First try: find the `zeros_before`-th zero bit (0-indexed).
        if zeros_before < total_zeros {
            let pos = self.select0(zeros_before)?;
            if pos >= start {
                return Some(pos);
            }
        }

        // Wrap around: find the 0th zero bit.
        self.select0(0)
    }

    /// Find `n` contiguous zero bits in the bitmap.
    ///
    /// Uses word-level skip acceleration: if a 64-bit word is all-ones,
    /// skip it entirely.
    #[must_use]
    pub fn find_contiguous(&self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        if n > self.count_zeros() {
            return None;
        }

        let mut run_start = 0_u32;
        let mut run_len = 0_u32;

        // Process in 64-bit chunks for fast skipping.
        let full_words = self.len / 64;

        for word_idx in 0..full_words {
            let word = self.read_word(word_idx);
            if word == u64::MAX {
                // All ones — no free bits in this word.
                run_start = (word_idx + 1) * 64;
                run_len = 0;
                continue;
            }

            // Scan individual bits in partial word.
            let base = word_idx * 64;
            for bit in 0..64 {
                let pos = base + bit;
                if (word >> bit) & 1 == 1 {
                    run_start = pos + 1;
                    run_len = 0;
                } else {
                    run_len += 1;
                    if run_len >= n {
                        return Some(run_start);
                    }
                }
            }
        }

        // Handle remaining bits.
        let remaining_start = full_words * 64;
        for pos in remaining_start..self.len {
            if self.get_bit(pos) {
                run_start = pos + 1;
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

    /// Read a single bit.
    #[must_use]
    #[inline]
    fn get_bit(&self, pos: u32) -> bool {
        let byte_idx = (pos / 8) as usize;
        let bit_idx = pos % 8;
        (self.data[byte_idx] >> bit_idx) & 1 == 1
    }

    /// Read a 64-bit word from the bitmap (little-endian bit order).
    #[must_use]
    fn read_word(&self, word_idx: u32) -> u64 {
        let byte_start = (word_idx * 8) as usize;
        if byte_start + 8 <= self.data.len() {
            let bytes: [u8; 8] = self.data[byte_start..byte_start + 8]
                .try_into()
                .expect("8 bytes");
            u64::from_le_bytes(bytes)
        } else {
            // Partial word at end.
            let mut word = 0_u64;
            for (i, &byte) in self.data[byte_start..].iter().enumerate() {
                word |= u64::from(byte) << (i * 8);
            }
            word
        }
    }
}

/// Count set bits in `bitmap` starting at `bit_start` for `count` bits.
fn popcount_range(bitmap: &[u8], bit_start: u32, count: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let mut total = 0_u32;
    let end = bit_start + count;

    // Process byte-aligned portions.
    let first_byte = (bit_start / 8) as usize;
    let first_bit_in_byte = bit_start % 8;

    if first_bit_in_byte != 0 {
        // Partial first byte.
        let byte = bitmap[first_byte];
        let bits_in_first = (8 - first_bit_in_byte).min(count);
        let mask = ((1_u16 << bits_in_first) - 1) << first_bit_in_byte;
        #[expect(clippy::cast_possible_truncation)]
        let masked = (byte & mask as u8).count_ones();
        total += masked;

        let next_start = bit_start + bits_in_first;
        if next_start >= end {
            return total;
        }

        // Full bytes in the middle.
        let full_start = (next_start / 8) as usize;
        let full_end = (end / 8) as usize;
        for &byte in &bitmap[full_start..full_end] {
            total += byte.count_ones();
        }

        // Partial last byte.
        let remainder = end % 8;
        if remainder > 0 && full_end < bitmap.len() {
            let byte = bitmap[full_end];
            let mask = (1_u16 << remainder) - 1;
            #[expect(clippy::cast_possible_truncation)]
            let masked = (byte & mask as u8).count_ones();
            total += masked;
        }
    } else {
        // Aligned start.
        let full_end = (end / 8) as usize;
        for &byte in &bitmap[first_byte..full_end] {
            total += byte.count_ones();
        }

        let remainder = end % 8;
        if remainder > 0 && full_end < bitmap.len() {
            let byte = bitmap[full_end];
            let mask = (1_u16 << remainder) - 1;
            #[expect(clippy::cast_possible_truncation)]
            let masked = (byte & mask as u8).count_ones();
            total += masked;
        }
    }

    total
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a bitmap from a bit pattern string ("10110...").
    fn bitmap_from_pattern(pattern: &str) -> Vec<u8> {
        let bits: Vec<bool> = pattern.chars().map(|c| c == '1').collect();
        let len = bits.len();
        let bytes = len.div_ceil(8);
        let mut bitmap = vec![0_u8; bytes];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bitmap[byte_idx] |= 1 << bit_idx;
            }
        }
        bitmap
    }

    #[test]
    fn empty_bitmap() {
        let bitmap = vec![0_u8; 0];
        let sb = SuccinctBitmap::build(&bitmap, 0);
        assert_eq!(sb.len(), 0);
        assert_eq!(sb.count_ones(), 0);
        assert_eq!(sb.count_zeros(), 0);
        assert_eq!(sb.rank0(0), 0);
        assert_eq!(sb.rank1(0), 0);
        assert_eq!(sb.select0(0), None);
        assert_eq!(sb.select1(0), None);
    }

    #[test]
    fn single_byte_all_zeros() {
        let bitmap = vec![0_u8; 1];
        let sb = SuccinctBitmap::build(&bitmap, 8);
        assert_eq!(sb.count_zeros(), 8);
        assert_eq!(sb.count_ones(), 0);
        assert_eq!(sb.rank0(4), 4);
        assert_eq!(sb.rank1(4), 0);
        assert_eq!(sb.select0(0), Some(0));
        assert_eq!(sb.select0(7), Some(7));
        assert_eq!(sb.select0(8), None);
        assert_eq!(sb.select1(0), None);
    }

    #[test]
    fn single_byte_all_ones() {
        let bitmap = vec![0xFF_u8; 1];
        let sb = SuccinctBitmap::build(&bitmap, 8);
        assert_eq!(sb.count_zeros(), 0);
        assert_eq!(sb.count_ones(), 8);
        assert_eq!(sb.rank1(4), 4);
        assert_eq!(sb.rank0(4), 0);
        assert_eq!(sb.select1(0), Some(0));
        assert_eq!(sb.select1(7), Some(7));
        assert_eq!(sb.select1(8), None);
        assert_eq!(sb.select0(0), None);
    }

    #[test]
    fn alternating_bits() {
        // 0b01010101 = 0x55: bits 0,2,4,6 are set; 1,3,5,7 are clear
        // But in our encoding, bit 0 is the LSB of byte 0.
        let bitmap = vec![0x55_u8; 1];
        let sb = SuccinctBitmap::build(&bitmap, 8);
        assert_eq!(sb.count_ones(), 4);
        assert_eq!(sb.count_zeros(), 4);

        // rank1 at positions 0..8
        assert_eq!(sb.rank1(0), 0);
        assert_eq!(sb.rank1(1), 1); // bit 0 is set
        assert_eq!(sb.rank1(2), 1); // bit 1 is clear
        assert_eq!(sb.rank1(3), 2); // bit 2 is set
        assert_eq!(sb.rank1(4), 2);
        assert_eq!(sb.rank1(8), 4);

        // select1
        assert_eq!(sb.select1(0), Some(0));
        assert_eq!(sb.select1(1), Some(2));
        assert_eq!(sb.select1(2), Some(4));
        assert_eq!(sb.select1(3), Some(6));

        // select0
        assert_eq!(sb.select0(0), Some(1));
        assert_eq!(sb.select0(1), Some(3));
        assert_eq!(sb.select0(2), Some(5));
        assert_eq!(sb.select0(3), Some(7));
    }

    #[test]
    fn pattern_bitmap() {
        // "11001010" — bits 0,1 set; 2,3 clear; 4 set; 5 clear; 6 set; 7 clear
        let bitmap = bitmap_from_pattern("11001010");
        let sb = SuccinctBitmap::build(&bitmap, 8);

        assert_eq!(sb.count_ones(), 4);
        assert_eq!(sb.rank1(2), 2); // bits 0,1 are set
        assert_eq!(sb.rank1(5), 3); // bits 0,1,4 set
        assert_eq!(sb.rank0(4), 2); // bits 2,3 clear

        assert_eq!(sb.select1(0), Some(0));
        assert_eq!(sb.select1(1), Some(1));
        assert_eq!(sb.select1(2), Some(4));
        assert_eq!(sb.select1(3), Some(6));

        assert_eq!(sb.select0(0), Some(2));
        assert_eq!(sb.select0(1), Some(3));
    }

    #[test]
    fn large_bitmap_4096_bytes() {
        // Simulate an ext4 block bitmap (4096 bytes = 32768 bits).
        let mut bitmap = vec![0xFF_u8; 4096];
        // Free blocks 1000..1100 (100 free blocks).
        for i in 1000..1100 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bitmap[byte_idx] &= !(1 << bit_idx);
        }
        // Free blocks 30000..30050 (50 free blocks).
        for i in 30000..30050 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bitmap[byte_idx] &= !(1 << bit_idx);
        }

        let sb = SuccinctBitmap::build(&bitmap, 32768);
        assert_eq!(sb.count_zeros(), 150);
        assert_eq!(sb.count_ones(), 32768 - 150);

        // rank0 before, at, and after free regions.
        assert_eq!(sb.rank0(1000), 0);
        assert_eq!(sb.rank0(1050), 50);
        assert_eq!(sb.rank0(1100), 100);
        assert_eq!(sb.rank0(30000), 100);
        assert_eq!(sb.rank0(30050), 150);

        // select0 finds free blocks in order.
        assert_eq!(sb.select0(0), Some(1000));
        assert_eq!(sb.select0(99), Some(1099));
        assert_eq!(sb.select0(100), Some(30000));
        assert_eq!(sb.select0(149), Some(30049));
        assert_eq!(sb.select0(150), None);
    }

    #[test]
    fn find_free_with_start() {
        let mut bitmap = vec![0xFF_u8; 128]; // 1024 bits, all allocated.
        // Free bit 500 and bit 100.
        bitmap[500 / 8] &= !(1 << (500 % 8));
        bitmap[100 / 8] &= !(1 << (100 % 8));

        let sb = SuccinctBitmap::build(&bitmap, 1024);
        assert_eq!(sb.count_zeros(), 2);

        // Start at 0: find bit 100 first.
        assert_eq!(sb.find_free(0), Some(100));

        // Start at 200: find bit 500.
        assert_eq!(sb.find_free(200), Some(500));

        // Start at 600: wrap around, find bit 100.
        assert_eq!(sb.find_free(600), Some(100));
    }

    #[test]
    fn find_contiguous_basic() {
        let mut bitmap = vec![0xFF_u8; 16]; // 128 bits, all ones.
        // Free bits 40..50 (10 contiguous zeros).
        for i in 40..50 {
            bitmap[i / 8] &= !(1 << (i % 8));
        }

        let sb = SuccinctBitmap::build(&bitmap, 128);
        assert_eq!(sb.find_contiguous(5), Some(40));
        assert_eq!(sb.find_contiguous(10), Some(40));
        assert_eq!(sb.find_contiguous(11), None);
    }

    #[test]
    fn find_contiguous_word_skip() {
        // 256 bits: first 64 all ones, next 64 all zeros, rest ones.
        let mut bitmap = vec![0xFF_u8; 32];
        for byte in &mut bitmap[8..16] {
            *byte = 0;
        }
        let sb = SuccinctBitmap::build(&bitmap, 256);
        assert_eq!(sb.find_contiguous(32), Some(64));
        assert_eq!(sb.find_contiguous(64), Some(64));
        assert_eq!(sb.find_contiguous(65), None);
    }

    #[test]
    fn partial_byte_len() {
        // Only 5 valid bits in 1 byte.
        let bitmap = vec![0b0001_0110_u8]; // bits: 0=0, 1=1, 2=1, 3=0, 4=1
        let sb = SuccinctBitmap::build(&bitmap, 5);
        assert_eq!(sb.count_ones(), 3);
        assert_eq!(sb.count_zeros(), 2);
        assert_eq!(sb.rank1(3), 2);
        assert_eq!(sb.select0(0), Some(0));
        assert_eq!(sb.select0(1), Some(3));
        assert_eq!(sb.select0(2), None); // only 2 zeros in 5 bits
    }

    #[test]
    fn rank_select_consistency() {
        // Property: for all valid k, rank0(select0(k)) == k
        //           for all valid k, rank1(select1(k)) == k
        let mut bitmap = vec![0_u8; 64]; // 512 bits
        // Set every 3rd bit.
        for i in (0..512).step_by(3) {
            bitmap[i / 8] |= 1 << (i % 8);
        }

        let sb = SuccinctBitmap::build(&bitmap, 512);

        // Check select1 → rank1 roundtrip.
        for k in 0..sb.count_ones() {
            let pos = sb.select1(k).expect("select1 should succeed");
            let rank = sb.rank1(pos);
            assert_eq!(rank, k, "rank1(select1({k})) = {rank} != {k}");
        }

        // Check select0 → rank0 roundtrip.
        for k in 0..sb.count_zeros() {
            let pos = sb.select0(k).expect("select0 should succeed");
            let rank = sb.rank0(pos);
            assert_eq!(rank, k, "rank0(select0({k})) = {rank} != {k}");
        }
    }

    #[test]
    fn space_overhead_under_10_percent() {
        // For a 4096-byte bitmap (ext4 standard), measure index size.
        let bitmap = vec![0_u8; 4096];
        let sb = SuccinctBitmap::build(&bitmap, 32768);

        let bitmap_size = 4096;
        let index_overhead = sb.superblocks.len() * 4 + sb.blocks.len() * 2;
        let overhead_pct = (index_overhead as f64 / f64::from(bitmap_size)) * 100.0;

        // Must be under 10% as per acceptance criteria.
        assert!(
            overhead_pct < 10.0,
            "overhead {overhead_pct:.1}% exceeds 10% limit"
        );
    }

    #[test]
    fn cross_superblock_boundary() {
        // Create a bitmap that spans multiple superblocks (>512 bits).
        let mut bitmap = vec![0_u8; 256]; // 2048 bits = 4 superblocks.
        // Set bits 510, 511, 512, 513 (crossing superblock boundary).
        for i in [510, 511, 512, 513] {
            bitmap[i / 8] |= 1 << (i % 8);
        }

        let sb = SuccinctBitmap::build(&bitmap, 2048);
        assert_eq!(sb.rank1(510), 0);
        assert_eq!(sb.rank1(511), 1);
        assert_eq!(sb.rank1(512), 2);
        assert_eq!(sb.rank1(513), 3);
        assert_eq!(sb.rank1(514), 4);

        assert_eq!(sb.select1(0), Some(510));
        assert_eq!(sb.select1(1), Some(511));
        assert_eq!(sb.select1(2), Some(512));
        assert_eq!(sb.select1(3), Some(513));
    }

    #[test]
    fn rank1_len_on_block_boundary_is_safe() {
        let mut bitmap = vec![0_u8; 128]; // 1024 bits, exact multiple of 256.
        for i in [0_u32, 255, 256, 511, 700, 1023] {
            let byte_idx = usize::try_from(i / 8).expect("test index fits usize");
            let bit_idx = i % 8;
            bitmap[byte_idx] |= 1 << bit_idx;
        }

        let sb = SuccinctBitmap::build(&bitmap, 1024);
        assert_eq!(sb.count_ones(), 6);
        assert_eq!(sb.rank1(1024), 6);
        assert_eq!(sb.rank0(1024), 1018);
    }

    /// Exhaustive monotonicity check across block and superblock boundaries.
    #[test]
    fn rank1_monotonicity_at_boundaries() {
        let test_lens: Vec<u32> = vec![
            1, 7, 8, 9, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049, 2303, 2304, 2305, 4096,
        ];

        for &len in &test_lens {
            let byte_len = len.div_ceil(8) as usize;
            for pattern in [0x00_u8, 0xFF, 0x55, 0xAA] {
                let bm = vec![pattern; byte_len];
                verify_rank1_monotonic(&bm, len);
            }

            // Single set bit at boundary positions.
            for &boundary_bit in &[0_u32, len / 2, len.saturating_sub(1)] {
                if boundary_bit < len {
                    let mut bm = vec![0_u8; byte_len];
                    bm[(boundary_bit / 8) as usize] |= 1 << (boundary_bit % 8);
                    verify_rank1_monotonic(&bm, len);
                }
            }
        }
    }

    fn verify_rank1_monotonic(bitmap: &[u8], len: u32) {
        let sb = SuccinctBitmap::build(bitmap, len);
        let mut prev = 0_u32;
        for i in 1..=len {
            let r = sb.rank1(i);
            assert!(
                r >= prev,
                "rank1({i}) = {r} < rank1({}) = {prev} (len={len})",
                i - 1
            );
            assert!(
                r <= prev + 1,
                "rank1 jumped by more than 1 at position {i} (len={len}): {prev} -> {r}",
            );
            prev = r;
        }
        assert_eq!(
            sb.rank1(len),
            sb.count_ones(),
            "rank1(len) != count_ones for len={len}"
        );
    }

    // ── Property-based tests (proptest) ────────────────────────────────

    use proptest::prelude::*;

    /// Strategy: generate an arbitrary bitmap of 1..512 bytes with a valid bit length.
    fn bitmap_strategy() -> impl Strategy<Value = (Vec<u8>, u32)> {
        (1_usize..512).prop_flat_map(|byte_len| {
            let max_bits =
                u32::try_from(byte_len * 8).expect("byte_len bound keeps bit length within u32");
            (prop::collection::vec(any::<u8>(), byte_len), 1..=max_bits)
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn proptest_rank1_select1_roundtrip((ref bitmap, len) in bitmap_strategy()) {
            let sb = SuccinctBitmap::build(bitmap, len);
            for k in 0..sb.count_ones() {
                let pos = sb.select1(k);
                prop_assert!(pos.is_some(), "select1({}) returned None but ones={}", k, sb.count_ones());
                let pos = pos.unwrap();
                let rank = sb.rank1(pos);
                prop_assert_eq!(rank, k, "rank1(select1({})) = {} != {}", k, rank, k);
            }
            // Out-of-range select1 should return None.
            prop_assert!(sb.select1(sb.count_ones()).is_none());
        }

        #[test]
        fn proptest_rank0_select0_roundtrip((ref bitmap, len) in bitmap_strategy()) {
            let sb = SuccinctBitmap::build(bitmap, len);
            for k in 0..sb.count_zeros() {
                let pos = sb.select0(k);
                prop_assert!(pos.is_some(), "select0({}) returned None but zeros={}", k, sb.count_zeros());
                let pos = pos.unwrap();
                let rank = sb.rank0(pos);
                prop_assert_eq!(rank, k, "rank0(select0({})) = {} != {}", k, rank, k);
            }
            // Out-of-range select0 should return None.
            prop_assert!(sb.select0(sb.count_zeros()).is_none());
        }

        #[test]
        fn proptest_rank1_monotonic((ref bitmap, len) in bitmap_strategy()) {
            let sb = SuccinctBitmap::build(bitmap, len);
            let mut prev = 0_u32;
            for i in 1..=len {
                let r = sb.rank1(i);
                prop_assert!(r >= prev, "rank1({}) = {} < rank1({}) = {}", i, r, i - 1, prev);
                prop_assert!(r <= prev + 1, "rank1 jumped by more than 1 at position {}", i);
                prev = r;
            }
            prop_assert_eq!(sb.rank1(len), sb.count_ones());
        }

        #[test]
        fn proptest_count_ones_plus_zeros_equals_len((ref bitmap, len) in bitmap_strategy()) {
            let sb = SuccinctBitmap::build(bitmap, len);
            prop_assert_eq!(sb.count_ones() + sb.count_zeros(), sb.len());
        }

        #[test]
        fn proptest_find_free_returns_zero_bit((ref bitmap, len) in bitmap_strategy(), start in 0_u32..4096) {
            let sb = SuccinctBitmap::build(bitmap, len);
            let start = start % (len + 1); // Clamp to valid range.
            if let Some(pos) = sb.find_free(start) {
                prop_assert!(pos < len, "find_free returned {} >= len {}", pos, len);
                prop_assert!(!sb.get_bit(pos), "find_free({}) returned {} which is a 1-bit", start, pos);
            } else {
                prop_assert_eq!(sb.count_zeros(), 0, "find_free returned None but there are free bits");
            }
        }

        #[test]
        fn proptest_find_contiguous_returns_valid_run(
            (ref bitmap, len) in bitmap_strategy(),
            n in 1_u32..64,
        ) {
            let sb = SuccinctBitmap::build(bitmap, len);
            if let Some(start) = sb.find_contiguous(n) {
                prop_assert!(start + n <= len, "contiguous run [{}, {}) exceeds len {}", start, start + n, len);
                for i in start..start + n {
                    prop_assert!(!sb.get_bit(i), "bit {} in contiguous run [{}, {}) is set", i, start, start + n);
                }
            }
        }
    }
}
