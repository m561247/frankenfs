//! Formal decision contract for self-healing repair policy.
//!
//! Defines the states the system can be in, actions available in each
//! state, a loss function for state/action pairs, and a policy engine
//! that selects the optimal action given current state and risk.

use serde::{Deserialize, Serialize};

// ── States ──────────────────────────────────────────────────────────────────

/// Observable system states for repair policy decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepairState {
    /// No corruption detected, all checksums pass.
    Clean,
    /// Less than 1% of blocks corrupted; local repair symbols should suffice.
    MinorCorruption,
    /// 1–5% of blocks corrupted; repair may be insufficient.
    SevereCorruption,
    /// I/O errors prevent reading some blocks (hardware degradation).
    IoStall,
}

impl RepairState {
    /// Classify a corruption observation into a state.
    #[must_use]
    pub fn classify(corrupted_blocks: u64, total_blocks: u64, io_errors: bool) -> Self {
        if io_errors {
            return Self::IoStall;
        }
        if total_blocks == 0 || corrupted_blocks == 0 {
            return Self::Clean;
        }
        let ratio = corrupted_blocks as f64 / total_blocks as f64;
        if ratio < 0.01 {
            Self::MinorCorruption
        } else {
            Self::SevereCorruption
        }
    }
}

// ── Actions ─────────────────────────────────────────────────────────────────

/// Available repair actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepairAction {
    /// Repair using local RaptorQ symbols (low cost, bounded capability).
    RepairLocal,
    /// Full block group re-encode (higher cost, resets redundancy).
    RepairGlobal,
    /// Schedule repair for next scrub cycle (risk of further degradation).
    DeferRepair,
    /// Switch to read-only mode (preserve data, lose writability).
    DegradeReadonly,
    /// Trigger backup to external storage (highest cost, highest safety).
    EmergencyBackup,
}

impl RepairAction {
    /// All available actions.
    pub const ALL: [Self; 5] = [
        Self::RepairLocal,
        Self::RepairGlobal,
        Self::DeferRepair,
        Self::DegradeReadonly,
        Self::EmergencyBackup,
    ];
}

// ── Loss function ───────────────────────────────────────────────────────────

/// Compute the loss (cost) for taking `action` in `state`.
///
/// Lower values are better. The units are abstract "cost units" that
/// weigh data-loss risk against operational overhead.
#[must_use]
#[allow(clippy::enum_glob_use, clippy::match_same_arms)]
pub fn loss(state: RepairState, action: RepairAction) -> f64 {
    use RepairAction::*;
    use RepairState::*;

    // Loss matrix: each cell is an intentional design choice.
    // Coincidental value equality across rows is expected.
    match (state, action) {
        // Clean state: repairs are unnecessary overhead.
        (Clean, RepairLocal) => 5.0,
        (Clean, RepairGlobal) => 20.0,
        (Clean, DeferRepair) => 0.0,      // correct: do nothing
        (Clean, DegradeReadonly) => 50.0, // massive over-reaction
        (Clean, EmergencyBackup) => 100.0,

        // Minor corruption: local repair is the expected case.
        (MinorCorruption, RepairLocal) => 2.0,   // best action
        (MinorCorruption, RepairGlobal) => 10.0, // overkill
        (MinorCorruption, DeferRepair) => 15.0,  // risky delay
        (MinorCorruption, DegradeReadonly) => 40.0, // over-reaction
        (MinorCorruption, EmergencyBackup) => 80.0,

        // Severe corruption: need stronger intervention.
        (SevereCorruption, RepairLocal) => 30.0, // may not suffice
        (SevereCorruption, RepairGlobal) => 8.0, // best action
        (SevereCorruption, DeferRepair) => 60.0, // high data-loss risk
        (SevereCorruption, DegradeReadonly) => 20.0, // safe but costly
        (SevereCorruption, EmergencyBackup) => 25.0,

        // I/O stall: repairs are unreliable, safety first.
        (IoStall, RepairLocal) => 50.0,     // repair may fail
        (IoStall, RepairGlobal) => 50.0,    // repair may fail
        (IoStall, DeferRepair) => 70.0,     // hardware degrading
        (IoStall, DegradeReadonly) => 10.0, // best action
        (IoStall, EmergencyBackup) => 15.0, // also reasonable
    }
}

// ── Safety envelope ─────────────────────────────────────────────────────────

/// Hard safety threshold for corruption posterior.
/// If posterior exceeds this, force immediate repair.
pub const SAFETY_POSTERIOR_THRESHOLD: f64 = 0.05;

/// Check whether the safety envelope forces an override action.
///
/// Returns `Some(action)` if the safety envelope requires an immediate
/// override, `None` if the normal policy engine can decide.
#[must_use]
pub fn safety_override(
    corruption_posterior: f64,
    repair_symbols_sufficient: bool,
) -> Option<RepairAction> {
    if corruption_posterior > SAFETY_POSTERIOR_THRESHOLD {
        if repair_symbols_sufficient {
            Some(RepairAction::RepairLocal)
        } else {
            // Insufficient symbols: go read-only to preserve data.
            Some(RepairAction::DegradeReadonly)
        }
    } else {
        None
    }
}

// ── Policy engine ───────────────────────────────────────────────────────────

/// Decision output from the repair policy engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairDecision {
    /// The state that was observed.
    pub state: RepairState,
    /// The action selected.
    pub action: RepairAction,
    /// The loss of the selected action.
    pub loss: f64,
    /// Whether the safety envelope forced this decision.
    pub safety_override: bool,
    /// Corruption posterior at decision time.
    pub corruption_posterior: f64,
}

/// Select the optimal action for the given state.
///
/// Checks the safety envelope first. If no override is needed, picks
/// the action with the lowest loss.
#[must_use]
pub fn select_action(
    state: RepairState,
    corruption_posterior: f64,
    repair_symbols_sufficient: bool,
) -> RepairDecision {
    // Safety envelope check.
    if let Some(forced) = safety_override(corruption_posterior, repair_symbols_sufficient) {
        return RepairDecision {
            state,
            action: forced,
            loss: loss(state, forced),
            safety_override: true,
            corruption_posterior,
        };
    }

    // Normal policy: minimize loss.
    let mut best_action = RepairAction::DeferRepair;
    let mut best_loss = f64::MAX;

    for action in RepairAction::ALL {
        let l = loss(state, action);
        if l < best_loss {
            best_loss = l;
            best_action = action;
        }
    }

    RepairDecision {
        state,
        action: best_action,
        loss: best_loss,
        safety_override: false,
        corruption_posterior,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn state_classify_clean_when_no_corruption() {
        assert_eq!(RepairState::classify(0, 10_000, false), RepairState::Clean);
        assert_eq!(RepairState::classify(0, 0, false), RepairState::Clean);
    }

    #[test]
    fn state_classify_minor_under_1pct() {
        // 50 out of 10_000 = 0.5% → MinorCorruption
        assert_eq!(
            RepairState::classify(50, 10_000, false),
            RepairState::MinorCorruption
        );
    }

    #[test]
    fn state_classify_severe_above_1pct() {
        // 200 out of 10_000 = 2% → SevereCorruption
        assert_eq!(
            RepairState::classify(200, 10_000, false),
            RepairState::SevereCorruption
        );
    }

    #[test]
    fn state_classify_io_stall_overrides_corruption() {
        // Even with zero corrupted blocks, I/O errors → IoStall.
        assert_eq!(RepairState::classify(0, 10_000, true), RepairState::IoStall);
        assert_eq!(
            RepairState::classify(500, 10_000, true),
            RepairState::IoStall
        );
    }

    #[test]
    fn policy_clean_selects_defer() {
        let decision = select_action(RepairState::Clean, 0.001, true);
        assert_eq!(decision.action, RepairAction::DeferRepair);
        assert!(!decision.safety_override);
        assert!(decision.loss < 1.0); // DeferRepair in Clean has loss 0.0
    }

    #[test]
    fn policy_minor_corruption_selects_repair_local() {
        let decision = select_action(RepairState::MinorCorruption, 0.005, true);
        assert_eq!(decision.action, RepairAction::RepairLocal);
        assert!(!decision.safety_override);
    }

    #[test]
    fn policy_severe_corruption_selects_repair_global() {
        let decision = select_action(RepairState::SevereCorruption, 0.03, true);
        assert_eq!(decision.action, RepairAction::RepairGlobal);
        assert!(!decision.safety_override);
    }

    #[test]
    fn policy_io_stall_selects_degrade_readonly() {
        let decision = select_action(RepairState::IoStall, 0.01, true);
        assert_eq!(decision.action, RepairAction::DegradeReadonly);
        assert!(!decision.safety_override);
    }

    #[test]
    fn safety_override_forces_repair_when_posterior_exceeds_threshold() {
        // Corruption posterior > 0.05 with sufficient symbols → force RepairLocal.
        let decision = select_action(RepairState::Clean, 0.08, true);
        assert_eq!(decision.action, RepairAction::RepairLocal);
        assert!(decision.safety_override);
    }

    #[test]
    fn safety_override_degrades_readonly_when_symbols_insufficient() {
        // Corruption posterior > 0.05 without sufficient symbols → DegradeReadonly.
        let decision = select_action(RepairState::MinorCorruption, 0.10, false);
        assert_eq!(decision.action, RepairAction::DegradeReadonly);
        assert!(decision.safety_override);
    }

    #[test]
    fn loss_monotonicity_deferred_repair_risk_increases_with_severity() {
        // DeferRepair loss should increase as state gets worse.
        let clean = loss(RepairState::Clean, RepairAction::DeferRepair);
        let minor = loss(RepairState::MinorCorruption, RepairAction::DeferRepair);
        let severe = loss(RepairState::SevereCorruption, RepairAction::DeferRepair);
        let stall = loss(RepairState::IoStall, RepairAction::DeferRepair);

        assert!(clean < minor, "clean < minor defer loss");
        assert!(minor < severe, "minor < severe defer loss");
        assert!(severe < stall, "severe < stall defer loss");
    }

    #[test]
    fn decision_json_round_trip() {
        let decision = select_action(RepairState::SevereCorruption, 0.03, true);
        let json = serde_json::to_string(&decision).expect("serialize");
        let parsed: RepairDecision = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.state, decision.state);
        assert_eq!(parsed.action, decision.action);
        assert!((parsed.loss - decision.loss).abs() < f64::EPSILON);
        assert_eq!(parsed.safety_override, decision.safety_override);
    }

    #[test]
    fn every_state_has_a_well_defined_optimal_action() {
        // For every possible state, the policy engine should return a valid decision.
        let states = [
            RepairState::Clean,
            RepairState::MinorCorruption,
            RepairState::SevereCorruption,
            RepairState::IoStall,
        ];
        for state in states {
            let decision = select_action(state, 0.01, true);
            assert!(decision.loss.is_finite(), "finite loss for {state:?}");
            assert!(!decision.safety_override);
        }
    }

    #[test]
    fn safety_threshold_boundary() {
        // At exactly the threshold, no override (>= vs >).
        let at_threshold = select_action(RepairState::Clean, SAFETY_POSTERIOR_THRESHOLD, true);
        assert!(!at_threshold.safety_override);

        // Just above threshold, override kicks in.
        let above = select_action(RepairState::Clean, SAFETY_POSTERIOR_THRESHOLD + 0.001, true);
        assert!(above.safety_override);
    }

    fn arb_state() -> impl Strategy<Value = RepairState> {
        prop_oneof![
            Just(RepairState::Clean),
            Just(RepairState::MinorCorruption),
            Just(RepairState::SevereCorruption),
            Just(RepairState::IoStall),
        ]
    }

    proptest! {
        /// `classify()` always returns IoStall when io_errors is true.
        #[test]
        fn proptest_classify_io_errors_always_stall(
            corrupted in 0_u64..1_000_000,
            total in 0_u64..1_000_000,
        ) {
            prop_assert_eq!(
                RepairState::classify(corrupted, total, true),
                RepairState::IoStall
            );
        }

        /// `classify()` returns Clean when corrupted_blocks == 0 and no io_errors.
        #[test]
        fn proptest_classify_zero_corruption_is_clean(total in 0_u64..1_000_000) {
            prop_assert_eq!(
                RepairState::classify(0, total, false),
                RepairState::Clean
            );
        }

        /// `loss()` is always non-negative and finite for all state/action pairs.
        #[test]
        fn proptest_loss_nonneg_finite(state in arb_state()) {
            for action in RepairAction::ALL {
                let l = loss(state, action);
                prop_assert!(l >= 0.0, "loss({:?}, {:?}) = {} < 0", state, action, l);
                prop_assert!(l.is_finite(), "loss({:?}, {:?}) is not finite", state, action);
            }
        }

        /// `select_action()` always returns finite loss and a valid action.
        #[test]
        fn proptest_select_action_valid(
            state in arb_state(),
            posterior in 0.0_f64..1.0,
            sufficient in any::<bool>(),
        ) {
            let decision = select_action(state, posterior, sufficient);
            prop_assert!(decision.loss.is_finite(), "decision loss not finite");
            prop_assert!(decision.loss >= 0.0, "decision loss < 0");
            prop_assert_eq!(decision.state, state);
        }

        /// Safety override fires iff posterior > SAFETY_POSTERIOR_THRESHOLD.
        #[test]
        fn proptest_safety_override_threshold(
            posterior in 0.0_f64..1.0,
            sufficient in any::<bool>(),
        ) {
            let result = safety_override(posterior, sufficient);
            if posterior > SAFETY_POSTERIOR_THRESHOLD {
                prop_assert!(result.is_some(), "should override at posterior={}", posterior);
            } else {
                prop_assert!(result.is_none(), "should not override at posterior={}", posterior);
            }
        }

        /// DeferRepair loss monotonically increases with severity.
        #[test]
        fn proptest_defer_loss_monotonic_with_severity(
            corrupted in 0_u64..100_000,
            total in 1_u64..100_000,
        ) {
            let corrupted = corrupted.min(total);
            let state = RepairState::classify(corrupted, total, false);
            let defer_loss = loss(state, RepairAction::DeferRepair);

            // Clean should have the lowest DeferRepair loss
            let clean_loss = loss(RepairState::Clean, RepairAction::DeferRepair);
            prop_assert!(defer_loss >= clean_loss - f64::EPSILON);
        }
    }
}
