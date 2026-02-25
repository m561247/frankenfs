//! Bayesian adaptive overhead selection for repair symbol refresh.
//!
//! This module maintains a Beta posterior over per-block corruption
//! probability and chooses repair overhead by minimizing expected loss:
//!
//! `E[loss] = P(unrecoverable) * data_loss_cost + overhead * storage_cost`
//!
//! Unrecoverable risk is estimated with a Beta-Binomial tail probability.

use serde::{Deserialize, Serialize};

const CANDIDATE_STEP: f64 = 0.001;
const MIN_POSITIVE_PARAM: f64 = 1e-9;

/// Adaptive overhead decision for a block group.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OverheadDecision {
    /// Selected overhead fraction (0.05 = 5% repair symbols).
    pub overhead_ratio: f64,
    /// Posterior mean corruption probability.
    pub corruption_posterior: f64,
    /// Beta posterior alpha parameter.
    pub posterior_alpha: f64,
    /// Beta posterior beta parameter.
    pub posterior_beta: f64,
    /// Probability of corruption exceeding decodable symbols.
    pub risk_bound: f64,
    /// Expected loss at this overhead.
    pub expected_loss: f64,
    /// Number of repair symbols implied by `overhead_ratio`.
    pub symbols_selected: u32,
    /// Whether metadata multiplier was applied.
    pub metadata_group: bool,
}

/// Bayesian durability autopilot used by repair policy selection.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DurabilityAutopilot {
    /// Beta posterior alpha parameter ("corrupt" evidence count).
    pub alpha: f64,
    /// Beta posterior beta parameter ("clean" evidence count).
    pub beta: f64,
    /// Cost multiplier for unrecoverable corruption.
    pub data_loss_cost: f64,
    /// Cost multiplier for repair-symbol overhead.
    pub storage_cost: f64,
    /// Minimum overhead fraction considered by the optimizer.
    pub min_overhead: f64,
    /// Maximum overhead fraction considered by the optimizer.
    pub max_overhead: f64,
    /// Multiplier for metadata-critical groups.
    pub metadata_multiplier: f64,
}

impl Default for DurabilityAutopilot {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 100.0,
            data_loss_cost: 1_000_000.0,
            storage_cost: 1.0,
            min_overhead: 0.03,
            max_overhead: 0.10,
            metadata_multiplier: 2.0,
        }
    }
}

impl DurabilityAutopilot {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bayesian conjugate update from one scrub cycle observation.
    pub fn update_posterior(&mut self, corrupted: u64, checked: u64) {
        if checked == 0 {
            return;
        }
        let clamped_corrupted = corrupted.min(checked);
        let clean = checked.saturating_sub(clamped_corrupted);

        self.alpha = self.alpha.max(MIN_POSITIVE_PARAM) + clamped_corrupted as f64;
        self.beta = self.beta.max(MIN_POSITIVE_PARAM) + clean as f64;
    }

    #[must_use]
    pub fn posterior_mean(&self) -> f64 {
        let alpha = self.alpha.max(MIN_POSITIVE_PARAM);
        let beta = self.beta.max(MIN_POSITIVE_PARAM);
        let denom = alpha + beta;
        if denom <= 0.0 {
            return 0.0;
        }
        (alpha / denom).clamp(0.0, 1.0)
    }

    #[must_use]
    pub fn posterior_params(&self) -> (f64, f64) {
        (
            self.alpha.max(MIN_POSITIVE_PARAM),
            self.beta.max(MIN_POSITIVE_PARAM),
        )
    }

    /// Compute unrecoverable risk bound for overhead fraction and group size.
    #[must_use]
    pub fn risk_bound(&self, overhead: f64, source_block_count: u32) -> f64 {
        if source_block_count == 0 {
            return 0.0;
        }

        let capped_overhead = overhead.clamp(0.0, 1.0);
        let decodable = (f64::from(source_block_count) * capped_overhead)
            .floor()
            .clamp(0.0, f64::from(source_block_count));
        let cutoff = floor_to_u32(decodable, source_block_count);
        if cutoff >= source_block_count {
            return 0.0;
        }

        let (alpha, beta) = self.posterior_params();
        beta_binomial_tail(alpha, beta, source_block_count, cutoff)
    }

    /// Expected-loss objective at a specific overhead fraction.
    #[must_use]
    pub fn expected_loss(&self, overhead: f64, source_block_count: u32) -> f64 {
        let bounded_overhead = overhead.clamp(0.0, 1.0);
        let risk = self.risk_bound(bounded_overhead, source_block_count);
        let redundancy_cost = self.storage_cost * bounded_overhead;
        let corruption_cost = self.data_loss_cost * risk;
        redundancy_cost + corruption_cost
    }

    /// Choose optimal overhead in `[min_overhead, max_overhead]`.
    #[must_use]
    pub fn optimal_overhead(&self, source_block_count: u32) -> f64 {
        let (min_overhead, max_overhead) = self.normalized_bounds();
        if self.posterior_mean() >= max_overhead {
            return max_overhead;
        }

        let mut best_overhead = min_overhead;
        let mut best_loss = self.expected_loss(min_overhead, source_block_count);
        let mut candidate = min_overhead;

        loop {
            let loss = self.expected_loss(candidate, source_block_count);
            if loss < best_loss - f64::EPSILON {
                best_loss = loss;
                best_overhead = candidate;
            }

            if (candidate - max_overhead).abs() <= f64::EPSILON {
                break;
            }
            let next_candidate = (candidate + CANDIDATE_STEP).min(max_overhead);
            if (next_candidate - candidate).abs() <= f64::EPSILON {
                break;
            }
            candidate = next_candidate;
        }

        best_overhead
    }

    /// Choose optimal overhead for metadata-critical groups.
    #[must_use]
    pub fn optimal_overhead_metadata(&self, source_block_count: u32) -> f64 {
        let scaled = self.optimal_overhead(source_block_count) * self.metadata_multiplier.max(0.0);
        scaled.clamp(0.0, 1.0)
    }

    /// Compute a full decision summary for one group.
    #[must_use]
    pub fn decision_for_group(
        &self,
        source_block_count: u32,
        metadata_group: bool,
    ) -> OverheadDecision {
        let selected_overhead = if metadata_group {
            self.optimal_overhead_metadata(source_block_count)
        } else {
            self.optimal_overhead(source_block_count)
        };
        let risk_bound = self.risk_bound(selected_overhead, source_block_count);
        let expected_loss = self.expected_loss(selected_overhead, source_block_count);

        OverheadDecision {
            overhead_ratio: selected_overhead,
            corruption_posterior: self.posterior_mean(),
            posterior_alpha: self.alpha.max(MIN_POSITIVE_PARAM),
            posterior_beta: self.beta.max(MIN_POSITIVE_PARAM),
            risk_bound,
            expected_loss,
            symbols_selected: Self::symbol_count_for_overhead(
                source_block_count,
                selected_overhead,
            ),
            metadata_group,
        }
    }

    /// Convert overhead fraction to repair symbol count.
    #[must_use]
    pub fn symbol_count_for_overhead(source_block_count: u32, overhead_ratio: f64) -> u32 {
        if source_block_count == 0 {
            return 0;
        }
        let bounded_overhead = overhead_ratio.clamp(0.0, 1.0);
        let raw_count = f64::from(source_block_count) * bounded_overhead;
        if !raw_count.is_finite() || raw_count <= 0.0 {
            return 0;
        }
        let floor_count = floor_to_u32(raw_count, source_block_count);
        if f64::from(floor_count) >= raw_count {
            floor_count
        } else {
            floor_count.saturating_add(1).min(source_block_count)
        }
    }

    #[must_use]
    fn normalized_bounds(&self) -> (f64, f64) {
        let min_overhead = self.min_overhead.clamp(0.0, 1.0);
        let max_overhead = self.max_overhead.clamp(min_overhead, 1.0);
        (min_overhead, max_overhead)
    }
}

#[must_use]
fn beta_binomial_tail(alpha: f64, beta: f64, draws: u32, cutoff: u32) -> f64 {
    if draws == 0 || cutoff >= draws {
        return 0.0;
    }

    let alpha = alpha.max(MIN_POSITIVE_PARAM);
    let beta = beta.max(MIN_POSITIVE_PARAM);

    let mut log_total = f64::NEG_INFINITY;
    let mut log_tail = f64::NEG_INFINITY;
    let mut log_weight = 0.0;

    for k in 0..=draws {
        log_total = log_add_exp(log_total, log_weight);
        if k > cutoff {
            log_tail = log_add_exp(log_tail, log_weight);
        }

        if k == draws {
            break;
        }

        let remaining = f64::from(draws - k);
        let next_index = f64::from(k + 1);
        let log_comb_ratio = (remaining / next_index).ln();
        let log_beta_ratio = (f64::from(k) + alpha).ln() - (f64::from(draws - k - 1) + beta).ln();
        log_weight += log_comb_ratio + log_beta_ratio;
    }

    if !log_tail.is_finite() {
        return 0.0;
    }

    (log_tail - log_total).exp().clamp(0.0, 1.0)
}

#[must_use]
fn log_add_exp(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() || rhs.is_nan() {
        return f64::NAN;
    }
    if lhs == f64::INFINITY || rhs == f64::INFINITY {
        return f64::INFINITY;
    }
    if lhs == f64::NEG_INFINITY {
        return rhs;
    }
    if rhs == f64::NEG_INFINITY {
        return lhs;
    }
    let hi = lhs.max(rhs);
    hi + ((lhs - hi).exp() + (rhs - hi).exp()).ln()
}

#[must_use]
fn floor_to_u32(value: f64, upper: u32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    if value >= f64::from(upper) {
        return upper;
    }

    let mut low = 0_u32;
    let mut high = upper;
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if f64::from(mid) <= value {
            low = mid;
        } else {
            high = mid.saturating_sub(1);
        }
    }
    low
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn posterior_update_tracks_counts() {
        let mut ap = DurabilityAutopilot::default();
        ap.update_posterior(10, 1000);
        assert!((ap.alpha - 11.0).abs() < 1e-12);
        assert!((ap.beta - 1090.0).abs() < 1e-12);
        assert!((ap.posterior_mean() - (11.0 / 1101.0)).abs() < 1e-12);
    }

    #[test]
    fn optimal_overhead_respects_bounds() {
        let ap = DurabilityAutopilot::default();
        let overhead = ap.optimal_overhead(32_768);
        assert!(overhead >= ap.min_overhead);
        assert!(overhead <= ap.max_overhead);
    }

    #[test]
    fn higher_corruption_selects_higher_overhead() {
        let mut low = DurabilityAutopilot::default();
        for _ in 0..5 {
            low.update_posterior(0, 20_000);
        }

        let mut high = DurabilityAutopilot::default();
        for _ in 0..5 {
            high.update_posterior(1_000, 20_000);
        }

        let low_overhead = low.optimal_overhead(32_768);
        let high_overhead = high.optimal_overhead(32_768);
        assert!(
            high_overhead >= low_overhead,
            "expected high corruption overhead >= low corruption overhead ({high_overhead} >= {low_overhead})"
        );
    }

    #[test]
    fn edge_cases_zero_and_full_corruption() {
        let mut zero = DurabilityAutopilot::default();
        zero.update_posterior(0, 10_000);
        let zero_overhead = zero.optimal_overhead(4_096);
        assert!(zero_overhead <= 0.05);

        let mut full = DurabilityAutopilot::default();
        full.update_posterior(10_000, 10_000);
        let full_overhead = full.optimal_overhead(4_096);
        assert!(full_overhead >= zero_overhead);
        assert!(full_overhead >= 0.09);
    }

    #[test]
    fn single_block_group_decision_is_valid() {
        let mut ap = DurabilityAutopilot::default();
        ap.update_posterior(1, 1);
        let decision = ap.decision_for_group(1, false);
        assert!(decision.overhead_ratio.is_finite());
        assert!(decision.expected_loss.is_finite());
        assert!(decision.risk_bound.is_finite());
        assert!(decision.symbols_selected <= 1);
    }

    #[test]
    fn metadata_multiplier_applies_to_optimal_overhead() {
        let mut ap = DurabilityAutopilot::default();
        ap.update_posterior(50, 10_000);

        let base = ap.optimal_overhead(8_192);
        let metadata = ap.optimal_overhead_metadata(8_192);
        let expected = (base * ap.metadata_multiplier).clamp(0.0, 1.0);
        assert!((metadata - expected).abs() < 1e-12);
        assert!(metadata >= base);
    }

    proptest! {
        #[test]
        fn optimal_overhead_monotonic_with_corruption_rate(
            low_rate in 0_u8..=100,
            delta in 0_u8..=100,
        ) {
            let high_rate = low_rate.saturating_add(delta).min(100);
            let checked = 20_000_u64;
            let low_corrupted = checked * u64::from(low_rate) / 100;
            let high_corrupted = checked * u64::from(high_rate) / 100;

            let mut low = DurabilityAutopilot::default();
            low.update_posterior(low_corrupted, checked);
            let low_overhead = low.optimal_overhead(32_768);

            let mut high = DurabilityAutopilot::default();
            high.update_posterior(high_corrupted, checked);
            let high_overhead = high.optimal_overhead(32_768);

            prop_assert!(
                high_overhead + 1e-12 >= low_overhead
                    || (low_overhead - high_overhead) <= CANDIDATE_STEP
            );
        }

        /// `posterior_mean()` always returns a value in [0, 1].
        #[test]
        fn proptest_posterior_mean_in_unit_interval(
            alpha in -100.0_f64..1_000_000.0,
            beta in -100.0_f64..1_000_000.0,
        ) {
            let ap = DurabilityAutopilot { alpha, beta, ..DurabilityAutopilot::default() };
            let mean = ap.posterior_mean();
            prop_assert!((0.0..=1.0).contains(&mean), "posterior_mean {} out of [0,1]", mean);
        }

        /// `risk_bound()` always returns a value in [0, 1].
        #[test]
        fn proptest_risk_bound_in_unit_interval(
            alpha in 0.001_f64..1000.0,
            beta in 0.001_f64..1000.0,
            overhead in 0.0_f64..1.0,
            source_blocks in 0_u32..512,
        ) {
            let ap = DurabilityAutopilot { alpha, beta, ..DurabilityAutopilot::default() };
            let risk = ap.risk_bound(overhead, source_blocks);
            prop_assert!(
                (0.0..=1.0).contains(&risk),
                "risk_bound {} out of [0,1] for alpha={}, beta={}, overhead={}, blocks={}",
                risk, alpha, beta, overhead, source_blocks
            );
        }

        /// `risk_bound()` returns 0 when overhead >= 1.0 (full redundancy).
        #[test]
        fn proptest_risk_bound_zero_at_full_overhead(
            alpha in 0.001_f64..100.0,
            beta in 0.001_f64..100.0,
            source_blocks in 1_u32..1024,
        ) {
            let ap = DurabilityAutopilot { alpha, beta, ..DurabilityAutopilot::default() };
            let risk = ap.risk_bound(1.0, source_blocks);
            prop_assert!(
                risk.abs() < 1e-12,
                "risk_bound at overhead=1.0 should be 0, got {}",
                risk
            );
        }

        /// `beta_binomial_tail()` always returns a value in [0, 1].
        #[test]
        fn proptest_beta_binomial_tail_in_unit_interval(
            alpha in 0.001_f64..100.0,
            beta in 0.001_f64..100.0,
            draws in 1_u32..128,
            cutoff_frac in 0.0_f64..1.0,
        ) {
            let cutoff = floor_to_u32(f64::from(draws) * cutoff_frac, draws);
            let tail = beta_binomial_tail(alpha, beta, draws, cutoff);
            prop_assert!(
                (0.0..=1.0).contains(&tail),
                "beta_binomial_tail {} out of [0,1] for alpha={}, beta={}, draws={}, cutoff={}",
                tail, alpha, beta, draws, cutoff
            );
        }

        /// `beta_binomial_tail()` is monotonically non-increasing in cutoff.
        #[test]
        fn proptest_beta_binomial_tail_monotonic_in_cutoff(
            alpha in 0.1_f64..50.0,
            beta in 0.1_f64..50.0,
            draws in 2_u32..64,
            lo in 0_u32..32,
            delta in 1_u32..32,
        ) {
            let lo = lo.min(draws.saturating_sub(1));
            let hi = lo.saturating_add(delta).min(draws.saturating_sub(1));
            if lo < hi {
                let tail_lo = beta_binomial_tail(alpha, beta, draws, lo);
                let tail_hi = beta_binomial_tail(alpha, beta, draws, hi);
                prop_assert!(
                    tail_lo + 1e-12 >= tail_hi,
                    "tail not monotonic: cutoff {} → {}, cutoff {} → {}",
                    lo, tail_lo, hi, tail_hi
                );
            }
        }

        /// `log_add_exp()` is commutative.
        #[test]
        fn proptest_log_add_exp_commutative(
            a in -100.0_f64..100.0,
            b in -100.0_f64..100.0,
        ) {
            let ab = log_add_exp(a, b);
            let ba = log_add_exp(b, a);
            prop_assert!(
                (ab - ba).abs() < 1e-10,
                "log_add_exp not commutative: ({}, {}) → {}, ({}, {}) → {}",
                a, b, ab, b, a, ba
            );
        }

        /// `log_add_exp(a, NEG_INFINITY)` returns `a`.
        #[test]
        fn proptest_log_add_exp_identity(a in -100.0_f64..100.0) {
            let result = log_add_exp(a, f64::NEG_INFINITY);
            prop_assert!(
                (result - a).abs() < 1e-12,
                "log_add_exp({}, -inf) = {}, expected {}",
                a, result, a
            );
        }

        /// `floor_to_u32()` returns a value <= the input and >= input - 1.
        #[test]
        fn proptest_floor_to_u32_accuracy(
            value in 0.0_f64..1_000_000.0,
            upper in 1_u32..=u32::MAX,
        ) {
            let result = floor_to_u32(value, upper);
            let f_result = f64::from(result);
            // result <= value (it's a floor)
            prop_assert!(
                f_result <= value + 1e-9,
                "floor_to_u32({}, {}) = {} exceeds input",
                value, upper, result
            );
            // result >= value - 1 (within one of the true floor)
            if value < f64::from(upper) {
                prop_assert!(
                    f_result >= value - 1.0 - 1e-9,
                    "floor_to_u32({}, {}) = {} too far below input",
                    value, upper, result
                );
            }
            // result <= upper
            prop_assert!(result <= upper);
        }

        /// `optimal_overhead()` always returns a value within bounds.
        #[test]
        fn proptest_optimal_overhead_within_bounds(
            corrupted in 0_u64..10_000,
            checked in 1_u64..20_000,
            source_blocks in 1_u32..4096,
        ) {
            let mut ap = DurabilityAutopilot::default();
            ap.update_posterior(corrupted.min(checked), checked);
            let overhead = ap.optimal_overhead(source_blocks);
            prop_assert!(
                overhead >= ap.min_overhead - 1e-12 && overhead <= ap.max_overhead + 1e-12,
                "optimal_overhead {} not in [{}, {}]",
                overhead, ap.min_overhead, ap.max_overhead
            );
        }

        /// `symbol_count_for_overhead()` is bounded by source_block_count.
        #[test]
        fn proptest_symbol_count_bounded(
            source_blocks in 0_u32..65536,
            overhead in 0.0_f64..2.0,
        ) {
            let count = DurabilityAutopilot::symbol_count_for_overhead(source_blocks, overhead);
            prop_assert!(
                count <= source_blocks,
                "symbol_count {} > source_blocks {}",
                count, source_blocks
            );
        }

        /// `expected_loss()` is always non-negative and finite.
        #[test]
        fn proptest_expected_loss_nonneg_finite(
            alpha in 0.001_f64..1000.0,
            beta in 0.001_f64..1000.0,
            overhead in 0.0_f64..1.0,
            source_blocks in 0_u32..512,
        ) {
            let ap = DurabilityAutopilot { alpha, beta, ..DurabilityAutopilot::default() };
            let loss = ap.expected_loss(overhead, source_blocks);
            prop_assert!(loss >= 0.0, "expected_loss {} < 0", loss);
            prop_assert!(loss.is_finite(), "expected_loss is not finite");
        }

        /// Posterior alpha + beta increases monotonically with observations.
        #[test]
        fn proptest_posterior_params_grow_with_observations(
            corrupted in 0_u64..1000,
            checked in 1_u64..10_000,
        ) {
            let mut ap = DurabilityAutopilot::default();
            let (a0, b0) = ap.posterior_params();
            let sum_before = a0 + b0;

            ap.update_posterior(corrupted.min(checked), checked);
            let (a1, b1) = ap.posterior_params();
            let sum_after = a1 + b1;

            prop_assert!(
                sum_after >= sum_before - 1e-12,
                "posterior sum decreased: {} → {}",
                sum_before, sum_after
            );
        }
    }
}
