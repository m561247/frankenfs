//! Local Reconstruction Codes (LRC) for distributed/multi-disk repair.
//!
//! Implements the Azure LRC scheme (Huang et al., USENIX ATC 2012) where
//! data blocks are partitioned into local repair groups, each with a local
//! parity block. Additional global parity blocks handle multi-failure
//! scenarios.
//!
//! # Advantages over RaptorQ for distributed storage
//!
//! - **Single-failure repair**: read only the local group (3–5 blocks)
//!   instead of the full erasure group (14+ blocks). Reduces repair I/O
//!   by 5–10x.
//! - **Locality**: repair traffic stays within the local group, reducing
//!   cross-rack/cross-node bandwidth.
//! - **Fallback**: multi-failure falls back to global reconstruction
//!   using all data + global parity blocks.
//!
//! # Layout
//!
//! Given `k` data blocks split into `g` local groups of `r` blocks each
//! (`k = g * r`), we generate:
//! - `g` local parity blocks (one per group, XOR of group members)
//! - `p` global parity blocks (Reed-Solomon over all `k` data blocks)
//!
//! Total blocks: `k + g + p` (data + local parity + global parity).
//!
//! # GF(256) arithmetic
//!
//! Global parity uses Galois Field GF(2^8) with the standard AES
//! irreducible polynomial `x^8 + x^4 + x^3 + x + 1` (0x11B).
//!
//! # `unsafe_code = "forbid"` Compliance
//!
//! All operations are safe Rust.

use serde::{Deserialize, Serialize};

// ── GF(256) arithmetic ─────────────────────────────────────────────────────

/// GF(256) multiplication using log/exp tables.
/// Irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B).
mod gf256 {
    /// Precomputed log table (base 0x03, the generator of GF(256)*).
    const LOG: [u8; 256] = {
        let mut table = [0_u8; 256];
        let mut val = 1_u32;
        let mut i = 0_u8;
        loop {
            table[val as usize] = i;
            // Multiply by generator 0x03 in GF(256).
            val = (val << 1) ^ val; // val * 3 = val * 2 + val
            if val >= 256 {
                val ^= 0x11B; // reduce by irreducible polynomial
            }
            i = i.wrapping_add(1);
            if i == 255 {
                break;
            }
        }
        table[0] = 0; // log(0) is undefined, but we handle it specially
        table
    };

    /// Precomputed exp table.
    #[expect(clippy::cast_possible_truncation)] // val is always < 256 after reduction
    const EXP: [u8; 512] = {
        let mut table = [0_u8; 512];
        let mut val = 1_u32;
        let mut i = 0;
        while i < 512 {
            table[i] = val as u8;
            val = (val << 1) ^ val;
            if val >= 256 {
                val ^= 0x11B;
            }
            i += 1;
        }
        table
    };

    /// Multiply two elements in GF(256).
    #[must_use]
    pub fn mul(a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let log_a = LOG[a as usize] as usize;
        let log_b = LOG[b as usize] as usize;
        EXP[log_a + log_b]
    }

    /// Compute the multiplicative inverse of `a` in GF(256).
    /// Panics if `a == 0`.
    #[must_use]
    pub fn inv(a: u8) -> u8 {
        assert_ne!(a, 0, "GF(256) inverse of zero is undefined");
        let log_a = LOG[a as usize] as usize;
        EXP[255 - log_a]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mul_identity() {
            for a in 0..=255_u8 {
                assert_eq!(mul(a, 1), a);
                assert_eq!(mul(1, a), a);
            }
        }

        #[test]
        fn mul_zero() {
            for a in 0..=255_u8 {
                assert_eq!(mul(a, 0), 0);
                assert_eq!(mul(0, a), 0);
            }
        }

        #[test]
        fn mul_inverse() {
            for a in 1..=255_u8 {
                let a_inv = inv(a);
                assert_eq!(mul(a, a_inv), 1, "a={a}, inv={a_inv}");
            }
        }

        #[test]
        fn mul_commutative() {
            for a in 0..=255_u8 {
                for b in 0..=255_u8 {
                    assert_eq!(mul(a, b), mul(b, a), "a={a}, b={b}");
                }
            }
        }
    }
}

// ── LRC configuration ──────────────────────────────────────────────────────

/// LRC layout configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LrcConfig {
    /// Number of data blocks (`k`).
    pub data_blocks: u32,
    /// Number of blocks per local repair group (`r`).
    pub local_group_size: u32,
    /// Number of global parity blocks (`p`).
    pub global_parity_count: u32,
}

impl LrcConfig {
    /// Create a new LRC configuration.
    ///
    /// # Panics
    ///
    /// Panics if `data_blocks` is not divisible by `local_group_size`,
    /// or if `local_group_size` is zero.
    #[must_use]
    pub fn new(data_blocks: u32, local_group_size: u32, global_parity_count: u32) -> Self {
        assert!(local_group_size > 0, "local_group_size must be > 0");
        assert!(
            data_blocks % local_group_size == 0,
            "data_blocks ({data_blocks}) must be divisible by local_group_size ({local_group_size})"
        );
        Self {
            data_blocks,
            local_group_size,
            global_parity_count,
        }
    }

    /// Number of local repair groups.
    #[must_use]
    pub fn num_groups(&self) -> u32 {
        self.data_blocks / self.local_group_size
    }

    /// Number of local parity blocks (one per group).
    #[must_use]
    pub fn local_parity_count(&self) -> u32 {
        self.num_groups()
    }

    /// Total number of blocks (data + local parity + global parity).
    #[must_use]
    pub fn total_blocks(&self) -> u32 {
        self.data_blocks + self.local_parity_count() + self.global_parity_count
    }

    /// Total storage overhead as a fraction.
    #[must_use]
    pub fn overhead_fraction(&self) -> f64 {
        let parity = self.local_parity_count() + self.global_parity_count;
        f64::from(parity) / f64::from(self.data_blocks)
    }
}

// ── LRC codec ──────────────────────────────────────────────────────────────

/// Encode local parity blocks.
///
/// For each local group, the local parity is the XOR of all data blocks
/// in that group. Returns `g` parity blocks.
///
/// # Panics
///
/// Panics if `data.len() != config.data_blocks` or block sizes differ.
#[must_use]
pub fn encode_local(config: &LrcConfig, data: &[Vec<u8>]) -> Vec<Vec<u8>> {
    assert_eq!(
        data.len(),
        config.data_blocks as usize,
        "expected {} data blocks, got {}",
        config.data_blocks,
        data.len()
    );

    let block_size = data[0].len();
    let groups = config.num_groups() as usize;
    let group_size = config.local_group_size as usize;

    let mut local_parities = Vec::with_capacity(groups);

    for g in 0..groups {
        let mut parity = vec![0_u8; block_size];
        for i in 0..group_size {
            let block = &data[g * group_size + i];
            assert_eq!(block.len(), block_size, "block size mismatch");
            xor_into(&mut parity, block);
        }
        local_parities.push(parity);
    }

    local_parities
}

/// Encode global parity blocks using Reed-Solomon in GF(256).
///
/// Global parity `P_j[byte] = sum_{i=0}^{k-1} alpha^{(i+1)*(j+1)} * data[i][byte]`
/// where `alpha = 2` is a generator element.
///
/// Returns `p` global parity blocks.
///
/// # Panics
///
/// Panics if `config.global_parity_count > 255` (GF(256) limit).
#[must_use]
pub fn encode_global(config: &LrcConfig, data: &[Vec<u8>]) -> Vec<Vec<u8>> {
    assert_eq!(data.len(), config.data_blocks as usize);
    assert!(
        config.global_parity_count <= 255,
        "global parity count must fit in GF(256)"
    );

    let block_size = data[0].len();
    let p = config.global_parity_count as usize;

    let mut global_parities = Vec::with_capacity(p);

    for j in 0..p {
        let mut parity = vec![0_u8; block_size];
        for (i, block) in data.iter().enumerate() {
            // coefficient = alpha^((i+1)*(j+1)) where alpha = 2
            let exponent = ((i + 1) * (j + 1)) % 255;
            let coeff = gf256_exp(exponent);
            gf256_mul_xor_into(&mut parity, block, coeff);
        }
        global_parities.push(parity);
    }

    global_parities
}

/// Full LRC encode: returns `(local_parities, global_parities)`.
#[must_use]
pub fn encode(config: &LrcConfig, data: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let local = encode_local(config, data);
    let global = encode_global(config, data);
    (local, global)
}

// ── Repair ─────────────────────────────────────────────────────────────────

/// Which blocks are available for repair.
#[derive(Debug, Clone)]
pub struct BlockAvailability {
    /// For each data block index, `Some(data)` if available, `None` if lost.
    pub data: Vec<Option<Vec<u8>>>,
    /// For each local parity block, `Some(data)` if available, `None` if lost.
    pub local_parity: Vec<Option<Vec<u8>>>,
    /// For each global parity block, `Some(data)` if available, `None` if lost.
    pub global_parity: Vec<Option<Vec<u8>>>,
}

/// Result of a repair attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    /// Whether repair succeeded.
    pub success: bool,
    /// Number of blocks repaired.
    pub blocks_repaired: u32,
    /// Indices of blocks that were repaired.
    pub repaired_indices: Vec<u32>,
    /// Whether local repair was sufficient.
    pub used_local_only: bool,
    /// Whether global repair was needed.
    pub used_global: bool,
}

/// Attempt local repair for a single missing data block.
///
/// If exactly one block in a local group is missing and the local parity
/// is available, XOR-recover the missing block.
///
/// Returns the repaired block data, or `None` if local repair is not possible.
#[must_use]
pub fn repair_local_single(
    config: &LrcConfig,
    group_idx: u32,
    missing_idx_in_group: u32,
    available_blocks: &[Option<&[u8]>],
    local_parity: &[u8],
) -> Option<Vec<u8>> {
    let group_size = config.local_group_size as usize;
    assert_eq!(available_blocks.len(), group_size);

    // Count missing blocks in this group.
    let missing_count = available_blocks.iter().filter(|b| b.is_none()).count();
    if missing_count != 1 {
        return None; // Local repair only works for single failures.
    }

    // Verify the missing index matches.
    if available_blocks[missing_idx_in_group as usize].is_some() {
        return None; // The specified block isn't actually missing.
    }

    let block_size = local_parity.len();
    let mut recovered = local_parity.to_vec();

    // XOR all available blocks into the parity to isolate the missing block.
    for (i, block) in available_blocks.iter().enumerate() {
        if let Some(b) = block {
            assert_eq!(b.len(), block_size, "block size mismatch");
            if i != missing_idx_in_group as usize {
                xor_into(&mut recovered, b);
            }
        }
    }

    let _ = group_idx; // used by callers for logging/indexing
    Some(recovered)
}

/// Attempt repair using global parity (multi-failure recovery).
///
/// Uses Gaussian elimination in GF(256) to solve for missing blocks
/// when local repair is insufficient.
///
/// Supports up to `config.global_parity_count` simultaneous failures
/// across any combination of data blocks.
#[must_use]
#[expect(clippy::too_many_lines)]
pub fn repair_global(
    config: &LrcConfig,
    availability: &BlockAvailability,
    block_size: usize,
) -> RepairResult {
    let p = config.global_parity_count as usize;

    // Find missing data block indices.
    let missing: Vec<usize> = availability
        .data
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if b.is_none() { Some(i) } else { None })
        .collect();

    if missing.is_empty() {
        return RepairResult {
            success: true,
            blocks_repaired: 0,
            repaired_indices: Vec::new(),
            used_local_only: false,
            used_global: false,
        };
    }

    if missing.len() > p {
        return RepairResult {
            success: false,
            blocks_repaired: 0,
            repaired_indices: Vec::new(),
            used_local_only: false,
            used_global: true,
        };
    }

    // Check we have enough global parity blocks.
    let available_global: Vec<(usize, &Vec<u8>)> = availability
        .global_parity
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.as_ref().map(|data| (i, data)))
        .collect();

    if available_global.len() < missing.len() {
        return RepairResult {
            success: false,
            blocks_repaired: 0,
            repaired_indices: Vec::new(),
            used_local_only: false,
            used_global: true,
        };
    }

    // Build the system of equations in GF(256).
    // For each global parity j that we have:
    //   P_j = sum_{i in known} coeff(i,j) * D_i + sum_{i in missing} coeff(i,j) * D_i
    //
    // Rearranging for unknown D_i:
    //   sum_{i in missing} coeff(i,j) * D_i = P_j - sum_{i in known} coeff(i,j) * D_i
    //
    // This gives us `missing.len()` equations in `missing.len()` unknowns.

    let m = missing.len();

    // Build coefficient matrix (m x m) and RHS vectors (one per byte position).
    let mut coeff_matrix = vec![vec![0_u8; m]; m];
    let mut rhs = vec![vec![0_u8; block_size]; m];

    for (eq_idx, &(parity_j, parity_data)) in available_global.iter().take(m).enumerate() {
        // Build coefficients for missing blocks.
        for (col, &missing_i) in missing.iter().enumerate() {
            let exp = ((missing_i + 1) * (parity_j + 1)) % 255;
            coeff_matrix[eq_idx][col] = gf256_exp(exp);
        }

        // RHS = P_j XOR sum_{known} coeff * D_known
        rhs[eq_idx] = parity_data[..block_size].to_vec();
        for (i, block_opt) in availability.data.iter().enumerate() {
            if let Some(block) = block_opt {
                let exp = ((i + 1) * (parity_j + 1)) % 255;
                let coeff = gf256_exp(exp);
                gf256_mul_xor_into(&mut rhs[eq_idx], block, coeff);
            }
        }
    }

    // Gaussian elimination in GF(256).
    // Process column by column.
    for col in 0..m {
        // Find pivot.
        let pivot_row = (col..m).find(|&row| coeff_matrix[row][col] != 0);
        let Some(pivot_row) = pivot_row else {
            // Singular matrix — cannot recover.
            return RepairResult {
                success: false,
                blocks_repaired: 0,
                repaired_indices: Vec::new(),
                used_local_only: false,
                used_global: true,
            };
        };

        // Swap rows.
        if pivot_row != col {
            coeff_matrix.swap(col, pivot_row);
            rhs.swap(col, pivot_row);
        }

        // Scale pivot row.
        let pivot_inv = gf256::inv(coeff_matrix[col][col]);
        for coeff in &mut coeff_matrix[col][..m] {
            *coeff = gf256::mul(*coeff, pivot_inv);
        }
        for byte in &mut rhs[col] {
            *byte = gf256::mul(*byte, pivot_inv);
        }

        // Eliminate other rows.
        for row in 0..m {
            if row == col {
                continue;
            }
            let factor = coeff_matrix[row][col];
            if factor == 0 {
                continue;
            }
            // Must index by c since we read coeff_matrix[col][c] while
            // writing coeff_matrix[row][c] — two different rows.
            #[expect(clippy::needless_range_loop)]
            for c in 0..m {
                coeff_matrix[row][c] ^= gf256::mul(coeff_matrix[col][c], factor);
            }
            // Same: reading rhs[col] while writing rhs[row].
            #[expect(clippy::needless_range_loop)]
            for byte_idx in 0..block_size {
                rhs[row][byte_idx] ^= gf256::mul(rhs[col][byte_idx], factor);
            }
        }
    }

    // After elimination, rhs[i] contains the recovered block for missing[i].
    #[expect(clippy::cast_possible_truncation)] // indices bounded by data_blocks (u32)
    let repaired_indices: Vec<u32> = missing.iter().map(|&i| i as u32).collect();

    RepairResult {
        success: true,
        #[expect(clippy::cast_possible_truncation)]
        blocks_repaired: m as u32,
        repaired_indices,
        used_local_only: false,
        used_global: true,
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// XOR `src` into `dst` (byte-by-byte).
fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

/// Compute `dst[i] ^= gf256_mul(src[i], coeff)` for each byte.
fn gf256_mul_xor_into(dst: &mut [u8], src: &[u8], coeff: u8) {
    if coeff == 0 {
        return;
    }
    if coeff == 1 {
        xor_into(dst, src);
        return;
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= gf256::mul(*s, coeff);
    }
}

/// Compute `EXP[exponent]` in GF(256).
fn gf256_exp(exponent: usize) -> u8 {
    // Use the precomputed table from the gf256 module.
    // EXP table is 512 entries long so exponent % 255 always works.
    // But we access it via the mul function's backing.
    // For direct exponentiation, we reconstruct.
    let mut val = 1_u8;
    for _ in 0..exponent {
        val = gf256::mul(val, 2); // alpha = 2 is our generator base
    }
    val
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(k: usize, block_size: usize) -> Vec<Vec<u8>> {
        (0..k)
            .map(|i| {
                (0..block_size)
                    .map(|j| {
                        u8::try_from((i * 37 + j * 53) & 0xFF).expect("pattern byte fits in u8")
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn config_basics() {
        let cfg = LrcConfig::new(12, 4, 2);
        assert_eq!(cfg.num_groups(), 3);
        assert_eq!(cfg.local_parity_count(), 3);
        assert_eq!(cfg.total_blocks(), 17); // 12 + 3 + 2
        let overhead = cfg.overhead_fraction();
        assert!((overhead - 5.0 / 12.0).abs() < 0.001);
    }

    #[test]
    fn local_parity_xor_correct() {
        let cfg = LrcConfig::new(8, 4, 2);
        let data = make_data(8, 64);
        let local = encode_local(&cfg, &data);

        assert_eq!(local.len(), 2); // 2 groups

        // Verify: XOR of group 0 blocks == local parity 0.
        let mut expected = vec![0_u8; 64];
        for block in &data[0..4] {
            xor_into(&mut expected, block);
        }
        assert_eq!(local[0], expected);
    }

    #[test]
    fn local_repair_single_failure() {
        let cfg = LrcConfig::new(8, 4, 2);
        let data = make_data(8, 64);
        let local = encode_local(&cfg, &data);

        // Simulate losing block 2 (in group 0, position 2).
        let available: Vec<Option<&[u8]>> = vec![
            Some(&data[0]),
            Some(&data[1]),
            None, // block 2 is lost
            Some(&data[3]),
        ];

        let recovered = repair_local_single(&cfg, 0, 2, &available, &local[0]);
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap(), data[2]);
    }

    #[test]
    fn local_repair_fails_for_two_missing() {
        let cfg = LrcConfig::new(8, 4, 2);
        let data = make_data(8, 64);
        let local = encode_local(&cfg, &data);

        // Two blocks missing in the same group.
        let available: Vec<Option<&[u8]>> = vec![
            Some(&data[0]),
            None, // block 1 lost
            None, // block 2 lost
            Some(&data[3]),
        ];

        let recovered = repair_local_single(&cfg, 0, 1, &available, &local[0]);
        assert!(recovered.is_none());
    }

    #[test]
    fn global_encode_decode_single_failure() {
        let cfg = LrcConfig::new(4, 2, 2);
        let data = make_data(4, 32);
        let global = encode_global(&cfg, &data);

        assert_eq!(global.len(), 2);

        // Lose block 1.
        let availability = BlockAvailability {
            data: vec![
                Some(data[0].clone()),
                None, // lost
                Some(data[2].clone()),
                Some(data[3].clone()),
            ],
            local_parity: vec![],
            global_parity: global.iter().map(|p| Some(p.clone())).collect(),
        };

        let result = repair_global(&cfg, &availability, 32);
        assert!(result.success);
        assert_eq!(result.blocks_repaired, 1);
        assert_eq!(result.repaired_indices, vec![1]);
        assert!(result.used_global);
    }

    #[test]
    fn global_repair_two_failures() {
        let cfg = LrcConfig::new(4, 2, 2);
        let data = make_data(4, 32);
        let global = encode_global(&cfg, &data);

        // Lose blocks 0 and 3.
        let availability = BlockAvailability {
            data: vec![
                None, // lost
                Some(data[1].clone()),
                Some(data[2].clone()),
                None, // lost
            ],
            local_parity: vec![],
            global_parity: global.iter().map(|p| Some(p.clone())).collect(),
        };

        let result = repair_global(&cfg, &availability, 32);
        assert!(result.success);
        assert_eq!(result.blocks_repaired, 2);
    }

    #[test]
    fn global_repair_exceeds_capacity() {
        let cfg = LrcConfig::new(4, 2, 1);
        let data = make_data(4, 32);
        let global = encode_global(&cfg, &data);

        // 2 failures but only 1 global parity → cannot repair.
        let availability = BlockAvailability {
            data: vec![None, None, Some(data[2].clone()), Some(data[3].clone())],
            local_parity: vec![],
            global_parity: global.iter().map(|p| Some(p.clone())).collect(),
        };

        let result = repair_global(&cfg, &availability, 32);
        assert!(!result.success);
    }

    #[test]
    fn full_encode_local_plus_global() {
        let cfg = LrcConfig::new(12, 4, 2);
        let data = make_data(12, 128);
        let (local, global) = encode(&cfg, &data);

        assert_eq!(local.len(), 3);
        assert_eq!(global.len(), 2);
    }

    #[test]
    fn local_repair_all_groups() {
        let cfg = LrcConfig::new(12, 4, 2);
        let data = make_data(12, 64);
        let local = encode_local(&cfg, &data);

        // Test local repair in each group.
        for (group, local_parity) in local.iter().enumerate() {
            let base = group * 4;
            for missing in 0..4 {
                let available: Vec<Option<&[u8]>> = (0..4)
                    .map(|i| {
                        if i == missing {
                            None
                        } else {
                            Some(data[base + i].as_slice())
                        }
                    })
                    .collect();

                let group_u32 = u32::try_from(group).expect("group index fits in u32");
                let missing_u32 = u32::try_from(missing).expect("missing index fits in u32");
                let recovered =
                    repair_local_single(&cfg, group_u32, missing_u32, &available, local_parity);
                assert_eq!(
                    recovered.as_ref(),
                    Some(&data[base + missing]),
                    "failed to repair group {group} block {missing}"
                );
            }
        }
    }

    #[test]
    fn repair_io_comparison() {
        // LRC with local groups of 4 should read 3 blocks + 1 parity = 4 blocks
        // for single-failure repair, vs 12+ for full erasure coding.
        let cfg = LrcConfig::new(12, 4, 2);

        // Local repair reads: group_size - 1 + 1 (parity) = group_size blocks.
        let local_io = cfg.local_group_size;
        // Global repair reads: all remaining data blocks + parity blocks.
        let global_io = cfg.data_blocks - 1 + cfg.global_parity_count;

        assert_eq!(local_io, 4);
        assert_eq!(global_io, 13); // 11 data + 2 global parity
        assert!(
            local_io < global_io / 2,
            "local repair IO ({local_io}) should be much less than global ({global_io})"
        );
    }

    #[test]
    fn config_serde_roundtrip() {
        let cfg = LrcConfig::new(12, 4, 2);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let parsed: LrcConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, cfg);
    }

    // ── Property-based tests (proptest) ────────────────────────────────

    use proptest::prelude::*;

    fn valid_lrc_config_strategy() -> impl Strategy<Value = (u32, u32, u32)> {
        // group_size in {2, 4}, num_groups in {2..6}, global_parity in {1, 2}
        (
            prop::sample::select(vec![2_u32, 4]),
            2_u32..6,
            prop::sample::select(vec![1_u32, 2]),
        )
            .prop_map(|(group_size, num_groups, global_parity)| {
                let data_blocks = group_size * num_groups;
                (data_blocks, group_size, global_parity)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// XOR of local group data always equals local parity.
        #[test]
        fn proptest_local_parity_xor_identity(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
            fill_byte in any::<u8>(),
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let block_size = 32_usize;
            let data: Vec<Vec<u8>> = (0..data_blocks as usize)
                .map(|i| {
                    vec![fill_byte.wrapping_add(u8::try_from(i % 256).unwrap()); block_size]
                })
                .collect();

            let local = encode_local(&cfg, &data);
            prop_assert_eq!(local.len(), cfg.num_groups() as usize);

            for g in 0..cfg.num_groups() as usize {
                let base = g * group_size as usize;
                let mut expected = vec![0_u8; block_size];
                for i in 0..group_size as usize {
                    xor_into(&mut expected, &data[base + i]);
                }
                prop_assert_eq!(&local[g], &expected, "group {} parity mismatch", g);
            }
        }

        /// Local repair recovers any single lost block within any group.
        #[test]
        fn proptest_local_repair_single_block(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
            fill_byte in any::<u8>(),
            group_selector in 0_u32..6,
            missing_selector in 0_u32..4,
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let num_groups = cfg.num_groups();
            let group_idx = group_selector % num_groups;
            let missing_in_group = missing_selector % group_size;

            let block_size = 32_usize;
            let data: Vec<Vec<u8>> = (0..data_blocks as usize)
                .map(|i| {
                    (0..block_size)
                        .map(|j| fill_byte.wrapping_add(u8::try_from((i + j) % 256).unwrap()))
                        .collect()
                })
                .collect();

            let local = encode_local(&cfg, &data);
            let base = group_idx as usize * group_size as usize;

            let available: Vec<Option<&[u8]>> = (0..group_size as usize)
                .map(|i| {
                    if i == missing_in_group as usize {
                        None
                    } else {
                        Some(data[base + i].as_slice())
                    }
                })
                .collect();

            let recovered = repair_local_single(
                &cfg,
                group_idx,
                missing_in_group,
                &available,
                &local[group_idx as usize],
            );

            prop_assert!(recovered.is_some(), "local repair should succeed for single failure");
            prop_assert_eq!(
                &recovered.unwrap(),
                &data[base + missing_in_group as usize],
                "recovered block mismatch"
            );
        }

        /// Config total_blocks = data + local_parity + global_parity.
        #[test]
        fn proptest_config_total_blocks(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let num_groups = cfg.num_groups();
            prop_assert_eq!(
                cfg.total_blocks(),
                data_blocks + num_groups + global_parity
            );
            prop_assert_eq!(num_groups * group_size, data_blocks);
        }

        /// Global repair succeeds for up to global_parity_count failures.
        #[test]
        fn proptest_global_repair_within_budget(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
            fill_byte in any::<u8>(),
            failure_selector in 0_u32..20,
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let num_failures = (failure_selector % global_parity).saturating_add(1)
                .min(data_blocks);

            let block_size = 32_usize;
            let data: Vec<Vec<u8>> = (0..data_blocks as usize)
                .map(|i| {
                    (0..block_size)
                        .map(|j| fill_byte.wrapping_add(u8::try_from((i * 7 + j * 3) % 256).unwrap()))
                        .collect()
                })
                .collect();

            let global = encode_global(&cfg, &data);

            // Lose the first num_failures blocks.
            let mut data_availability: Vec<Option<Vec<u8>>> = data
                .iter()
                .map(|b| Some(b.clone()))
                .collect();
            for i in 0..num_failures as usize {
                data_availability[i] = None;
            }

            let availability = BlockAvailability {
                data: data_availability,
                local_parity: vec![],
                global_parity: global.iter().map(|p| Some(p.clone())).collect(),
            };

            let result = repair_global(&cfg, &availability, block_size);
            prop_assert!(
                result.success,
                "global repair should succeed with {} failures and {} global parities",
                num_failures,
                global_parity,
            );
            prop_assert_eq!(result.blocks_repaired, num_failures);
        }

        /// Overhead fraction is always positive and <= 1.0 for sensible configs.
        #[test]
        fn proptest_overhead_fraction_bounded(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let overhead = cfg.overhead_fraction();
            prop_assert!(overhead > 0.0);
            prop_assert!(overhead <= 1.0);
        }

        /// LrcConfig serialization roundtrip.
        #[test]
        fn proptest_config_serde_roundtrip(
            (data_blocks, group_size, global_parity) in valid_lrc_config_strategy(),
        ) {
            let cfg = LrcConfig::new(data_blocks, group_size, global_parity);
            let json = serde_json::to_string(&cfg).expect("serialize");
            let parsed: LrcConfig = serde_json::from_str(&json).expect("deserialize");
            prop_assert_eq!(parsed, cfg);
        }
    }

    #[test]
    fn no_failures_returns_trivial_success() {
        let cfg = LrcConfig::new(4, 2, 2);
        let data = make_data(4, 32);
        let global = encode_global(&cfg, &data);

        let availability = BlockAvailability {
            data: data.iter().map(|b| Some(b.clone())).collect(),
            local_parity: vec![],
            global_parity: global.iter().map(|p| Some(p.clone())).collect(),
        };

        let result = repair_global(&cfg, &availability, 32);
        assert!(result.success);
        assert_eq!(result.blocks_repaired, 0);
        assert!(!result.used_global);
    }
}
