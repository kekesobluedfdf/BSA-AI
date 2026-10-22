from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

import behavior_sequence as bs  # noqa: E402


def ai_frame(
    codes: list[str],
    starts: list[str] | None = None,
    window_ids: list[int] | None = None,
    sequence_ids: list[str] | None = None,
) -> pd.DataFrame:
    if starts is None:
        starts = [f"00:00:{index * 5:02d}" for index in range(len(codes))]
    if window_ids is None:
        window_ids = list(range(1, len(codes) + 1))
    data = {
        "window_id": window_ids,
        "start_hms": starts,
        "code": codes,
        "text": [f"window {index}" for index in range(len(codes))],
    }
    if sequence_ids is not None:
        data["sequence_id"] = sequence_ids
    return pd.DataFrame(data)


def tables_for(codes: list[str], lag: int = 1, excluded_codes: set[str] | None = None) -> dict:
    return bs.compute_tables(codes, lag=lag, alpha=0.05, excluded_codes=excluded_codes)


class BehaviorSequenceAuditTests(unittest.TestCase):
    def test_frequency_and_transition_counts(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        frequency = tables["frequency"].set_index("code")["frequency"].to_dict()
        self.assertEqual(frequency, {"A": 3, "B": 2})
        observed = tables["transition_counts"]
        self.assertEqual(int(observed.loc["A", "B"]), 2)
        self.assertEqual(int(observed.loc["B", "A"]), 2)
        self.assertEqual(int(observed.loc["A", "A"]), 0)
        self.assertEqual(int(observed.loc["B", "B"]), 0)
        self.assertEqual(int(tables["total_transitions"]), 4)

    def test_expected_frequency_cross_checked_with_scipy(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        observed = tables["transition_counts"].to_numpy()
        _, _, _, expected = chi2_contingency(observed)
        np.testing.assert_allclose(tables["expected"].to_numpy(), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(tables["expected"].to_numpy(), np.ones((2, 2)), rtol=1e-12, atol=1e-12)

    def test_adjusted_residual_formula(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        observed = tables["transition_counts"].to_numpy(dtype=float)
        expected = tables["expected"].to_numpy(dtype=float)
        n = tables["total_transitions"]
        row = observed.sum(axis=1)
        col = observed.sum(axis=0)
        denominator = np.sqrt(expected * (1 - row[:, None] / n) * (1 - col[None, :] / n))
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_z = np.where(denominator > 0, (observed - expected) / denominator, np.nan)
        np.testing.assert_allclose(tables["adjusted_residuals"].to_numpy(), expected_z, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(float(tables["adjusted_residuals"].loc["A", "B"]), 2.0, places=12)
        self.assertAlmostEqual(float(tables["adjusted_residuals"].loc["B", "A"]), 2.0, places=12)
        self.assertAlmostEqual(float(tables["adjusted_residuals"].loc["A", "A"]), -2.0, places=12)
        self.assertAlmostEqual(float(tables["adjusted_residuals"].loc["B", "B"]), -2.0, places=12)

    def test_pearson_residual_is_not_adjusted_residual(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        observed = tables["transition_counts"].to_numpy(dtype=float)
        expected = tables["expected"].to_numpy(dtype=float)
        pearson = (observed - expected) / np.sqrt(expected)
        adjusted = tables["adjusted_residuals"].to_numpy()
        self.assertAlmostEqual(float(pearson[0, 0]), -1.0, places=12)
        self.assertAlmostEqual(float(adjusted[0, 0]), -2.0, places=12)
        self.assertFalse(np.allclose(pearson, adjusted))

    def test_transition_probability_is_conditional_on_row(self):
        tables = tables_for(["A", "A", "B", "C", "A", "B"])
        probability = tables["transition_probability"]
        for value in probability.sum(axis=1):
            self.assertAlmostEqual(float(value), 1.0, places=12)
        self.assertAlmostEqual(float(probability.loc["A", "B"]), 2 / 3, places=12)
        self.assertAlmostEqual(float(probability.loc["C", "A"]), 1.0, places=12)

    def test_lag_two_uses_only_pairs_two_steps_apart(self):
        tables = tables_for(["A", "B", "C", "A"], lag=2)
        observed = tables["transition_counts"]
        self.assertEqual(int(observed.loc["A", "C"]), 1)
        self.assertEqual(int(observed.loc["B", "A"]), 1)
        self.assertEqual(int(tables["total_transitions"]), 2)
        self.assertEqual(int(observed.loc["A", "B"]), 0)

    def test_single_event_and_zero_margins_are_undefined(self):
        tables = tables_for(["A"])
        self.assertEqual(int(tables["total_transitions"]), 0)
        self.assertFalse(bool(tables["testable"].loc["A", "A"]))
        self.assertTrue(math.isnan(float(tables["adjusted_residuals"].loc["A", "A"])))
        self.assertTrue(math.isnan(float(tables["p_values"].loc["A", "A"])))
        self.assertTrue(math.isnan(float(tables["transition_probability"].loc["A", "A"])))
        self.assertEqual(tables["adjusted_residuals_starred"].loc["A", "A"], "NA")

        tables = tables_for(["A", "B"])
        self.assertFalse(bool(tables["testable"].loc["A", "B"]))
        self.assertTrue(math.isnan(float(tables["adjusted_residuals"].loc["A", "B"])))
        self.assertAlmostEqual(float(tables["transition_probability"].loc["A", "B"]), 1.0, places=12)

    def test_low_expected_count_flagged_but_not_silently_dropped(self):
        tables = tables_for(["A", "B", "A", "B", "A", "B", "A", "B"])
        expected = float(tables["expected"].loc["A", "B"])
        z = float(tables["adjusted_residuals"].loc["A", "B"])
        self.assertLess(expected, 5)
        self.assertGreater(abs(z), 1.96)
        self.assertFalse(tables["significant"].empty)
        self.assertIn("A", set(tables["significant"]["from_code"]))
        self.assertIn("B", set(tables["significant"]["to_code"]))
        self.assertTrue(bool(tables["expected_lt_5"].loc["A", "B"]))
        self.assertIn("expected_lt_5", tables["warning_reason"].loc["A", "B"])

    def test_multi_sequence_boundary_prevents_cross_sequence_transition(self):
        ai = ai_frame(
            ["A", "B", "C", "D"],
            sequence_ids=["S1", "S1", "S2", "S2"],
        )
        sequences, metadata = bs.prepare_sequences(
            ai, None, "ai", lag=1, sequence_column="sequence_id", single_sequence=False
        )
        tables = bs.compute_tables_for_sequences(sequences, 1, 0.05)
        self.assertEqual(metadata["sequence_mode"], "explicit_sequence_id")
        self.assertEqual(metadata["sequence_count"], 2)
        self.assertEqual(int(tables["total_transitions"]), 2)
        self.assertEqual(int(tables["transition_counts"].loc["A", "B"]), 1)
        self.assertEqual(int(tables["transition_counts"].loc["C", "D"]), 1)
        if "B" in tables["transition_counts"].index and "C" in tables["transition_counts"].columns:
            self.assertEqual(int(tables["transition_counts"].loc["B", "C"]), 0)

    def test_multi_sequence_transitions_are_summed(self):
        ai = ai_frame(
            ["A", "B", "A", "B", "A", "B"],
            sequence_ids=["S1", "S1", "S1", "S2", "S2", "S2"],
        )
        sequences, metadata = bs.prepare_sequences(
            ai, None, "ai", lag=1, sequence_column="sequence_id", single_sequence=False
        )
        tables = bs.compute_tables_for_sequences(sequences, 1, 0.05)
        self.assertEqual(int(tables["transition_counts"].loc["A", "B"]), 2)
        self.assertEqual(int(tables["transition_counts"].loc["B", "A"]), 2)
        self.assertEqual(int(tables["total_transitions"]), 4)

    def test_single_sequence_requires_explicit_confirmation(self):
        ai = ai_frame(["A", "B", "C"])
        with self.assertRaises(ValueError):
            bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=False)

    def test_order_uses_time_not_window_id_and_records_reordering(self):
        ai = ai_frame(
            ["A", "B", "C"],
            starts=["00:00:20", "00:00:00", "00:00:10"],
            window_ids=[3, 1, 2],
        )
        sequences, metadata = bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)
        self.assertEqual([row["code"] for row in sequences[0]], ["B", "C", "A"])
        self.assertTrue(metadata["input_time_out_of_order"])

    def test_duplicate_window_id_is_rejected(self):
        ai = ai_frame(["A", "B", "C"], window_ids=[1, 1, 2])
        with self.assertRaises(ValueError):
            bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)

    def test_concurrent_timestamps_are_rejected(self):
        ai = ai_frame(["A", "B", "C"], starts=["00:00:00", "00:00:00", "00:00:05"])
        with self.assertRaises(ValueError):
            bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)

    def test_overlapping_windows_are_rejected(self):
        ai = pd.DataFrame(
            {
                "window_id": [1, 2],
                "start_ms": [0, 4000],
                "end_ms": [5000, 9000],
                "code": ["A", "B"],
            }
        )
        with self.assertRaises(ValueError):
            bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)

    def test_consecutive_identical_codes_are_not_merged(self):
        tables = tables_for(["A", "A", "A"])
        self.assertEqual(int(tables["frequency"].set_index("code").loc["A", "frequency"]), 3)
        self.assertEqual(int(tables["transition_counts"].loc["A", "A"]), 2)
        self.assertEqual(int(tables["total_transitions"]), 2)

    def test_excluding_code_does_not_connect_across_it(self):
        tables = tables_for(["A", "N", "B"], excluded_codes={"N"})
        self.assertEqual(int(tables["total_transitions"]), 0)
        self.assertNotIn("N", tables["transition_counts"].index)
        self.assertNotIn("N", tables["transition_counts"].columns)

    def test_significance_direction_and_stars(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        significant = tables["significant"]
        self.assertFalse(significant.empty)
        self.assertEqual(set(significant["direction"]), {"positive"})
        self.assertTrue((significant["raw_p_value"] < 0.05).all())
        self.assertGreater(int(tables["diagnostics"]["negative_suppressed"].sum()), 0)
        for _, row in significant.iterrows():
            expected = "***" if row["raw_p_value"] < 0.001 else "**" if row["raw_p_value"] < 0.01 else "*"
            self.assertEqual(row["significance"], expected)

    def test_graph_positive_only_labels_z_and_probability(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        positive_dot = bs.build_dot(tables, alpha=0.05, edge_mode="positive", max_edges=20, method="raw")
        self.assertIn('"A" -> "B"', positive_dot)
        self.assertIn('"B" -> "A"', positive_dot)
        self.assertNotIn('"A" -> "A"', positive_dot)
        self.assertIn("Z=2.00", positive_dot)
        self.assertIn("P=1.00", positive_dot)
        self.assertIn("style=\"dotted\"", positive_dot)  # E=1 < 5 in this test
        both_dot = bs.build_dot(tables, alpha=0.05, edge_mode="both", max_edges=20, method="raw")
        # Negative significant residuals are retained in diagnostics but are
        # deliberately not drawn as substantive main-result edges.
        self.assertNotIn('"A" -> "A"', both_dot)
        self.assertGreater(int(tables["diagnostics"]["negative_suppressed"].sum()), 0)

    def test_holm_matches_statsmodels(self):
        p_values = np.array([0.001, 0.02, 0.5, 0.9])
        expected = multipletests(p_values, method="holm")[1]
        np.testing.assert_allclose(bs.holm_adjust(p_values), expected, rtol=1e-12, atol=1e-12)

    def test_bh_matches_statsmodels(self):
        p_values = np.array([0.001, 0.02, 0.5, 0.9])
        expected = multipletests(p_values, method="fdr_bh")[1]
        np.testing.assert_allclose(bs.bh_adjust(p_values), expected, rtol=1e-12, atol=1e-12)

    def test_undefined_p_values_are_not_in_correction_family(self):
        tables = tables_for(["A", "B"])
        self.assertEqual(tables["family_size"], 0)
        self.assertTrue(tables["p_holm"].isna().all().all())
        self.assertTrue(tables["q_bh"].isna().all().all())
        self.assertFalse(tables["sig_holm"].to_numpy().any())
        self.assertFalse(tables["sig_bh"].to_numpy().any())

    def test_monte_carlo_does_not_cross_sequence_boundary(self):
        ai = ai_frame(
            ["A", "B", "C", "D"],
            sequence_ids=["S1", "S1", "S2", "S2"],
        )
        sequences, _ = bs.prepare_sequences(
            ai, None, "ai", lag=1, sequence_column="sequence_id", single_sequence=False
        )
        tables = bs.compute_tables_for_sequences(sequences, 1, 0.05)
        labels = tables["labels"]
        simulations = bs.monte_carlo_counts(sequences, labels, 1, 20, 7)
        b_index = labels.index("B")
        c_index = labels.index("C")
        self.assertEqual(int(simulations[:, b_index, c_index].sum()), 0)

    def test_monte_carlo_preserves_each_sequence_frequencies(self):
        ai = ai_frame(["A", "A", "B", "C", "C", "C"], sequence_ids=["S1"] * 3 + ["S2"] * 3)
        sequences, _ = bs.prepare_sequences(
            ai, None, "ai", lag=1, sequence_column="sequence_id", single_sequence=False
        )
        rng = np.random.default_rng(11)
        permuted = bs.permuted_code_lists(sequences, rng)
        for original, shuffled in zip(sequences, permuted):
            original_codes = [record["code"] for record in original]
            self.assertEqual(sorted(original_codes), sorted(shuffled))

    def test_monte_carlo_reproducible_with_seed(self):
        ai = ai_frame(["A", "B", "A", "C", "A", "B"])
        sequences, _ = bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)
        labels = ["A", "B", "C"]
        first = bs.monte_carlo_counts(sequences, labels, 1, 20, 42)
        second = bs.monte_carlo_counts(sequences, labels, 1, 20, 42)
        np.testing.assert_array_equal(first, second)

    def test_monte_carlo_lag_transition_counts(self):
        ai = ai_frame(["A", "B", "C"])
        sequences, _ = bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)
        labels = ["A", "B", "C"]
        lag1 = bs.monte_carlo_counts(sequences, labels, 1, 5, 3)
        lag2 = bs.monte_carlo_counts(sequences, labels, 2, 5, 3)
        self.assertTrue((lag1.sum(axis=(1, 2)) == 2).all())
        self.assertTrue((lag2.sum(axis=(1, 2)) == 1).all())

    def test_empirical_p_values_add_one_correction_and_tails(self):
        result = bs.empirical_p_values(5, [0, 1, 2])
        self.assertAlmostEqual(result["empirical_p_upper"], 0.25)
        self.assertAlmostEqual(result["empirical_p_lower"], 1.0)
        self.assertAlmostEqual(result["empirical_p_two_sided"], 0.5)
        result = bs.empirical_p_values(0, [0, 1, 2])
        self.assertAlmostEqual(result["empirical_p_upper"], 1.0)
        self.assertAlmostEqual(result["empirical_p_lower"], 0.5)
        self.assertAlmostEqual(result["empirical_p_two_sided"], 1.0)

    def test_empirical_p_zero_variance_is_finite(self):
        result = bs.empirical_p_values(1, [0, 0, 0, 0])
        self.assertEqual(result["simulated_mean"], 0.0)
        self.assertEqual(result["simulated_sd"], 0.0)
        self.assertAlmostEqual(result["empirical_p_upper"], 0.2)
        self.assertAlmostEqual(result["empirical_p_lower"], 1.0)
        self.assertAlmostEqual(result["empirical_p_two_sided"], 0.4)
        self.assertTrue(result["simulation_degenerate"])

    def test_collapsed_states_do_not_cross_sequences(self):
        ai = ai_frame(
            ["A", "A", "B", "B", "B", "C"],
            sequence_ids=["S1"] * 3 + ["S2"] * 3,
        )
        sequences, _ = bs.prepare_sequences(
            ai, None, "ai", lag=1, sequence_column="sequence_id", single_sequence=False
        )
        collapsed = bs.collapse_sequences(sequences)
        self.assertEqual([[row["code"] for row in seq] for seq in collapsed], [["A", "B"], ["B", "C"]])
        collapsed_tables = bs.compute_tables_for_sequences(collapsed, 1, 0.05)
        self.assertEqual(int(collapsed_tables["transition_counts"].loc["A", "B"]), 1)
        self.assertEqual(int(collapsed_tables["transition_counts"].loc["B", "C"]), 1)
        self.assertEqual(int(collapsed_tables["total_transitions"]), 2)

    def test_sensitivity_does_not_change_original_tables(self):
        sequences, _ = bs.prepare_sequences(
            ai_frame(["A", "A", "A", "B", "B", "C"]), None, "ai", lag=1, single_sequence=True
        )
        original = bs.compute_tables_for_sequences(sequences, 1, 0.05)
        original_counts = original["transition_counts"].copy()
        collapsed = bs.collapse_sequences(sequences)
        bs.compute_tables_for_sequences(collapsed, 1, 0.05)
        pd.testing.assert_frame_equal(original["transition_counts"], original_counts)
        self.assertEqual(int(original["transition_counts"].loc["A", "A"]), 2)
        collapsed_tables = bs.compute_tables_for_sequences(collapsed, 1, 0.05)
        self.assertEqual(int(collapsed_tables["transition_counts"].loc["A", "B"]), 1)

    def test_dot_uses_requested_method_and_mc_labels(self):
        tables = tables_for(["A", "B", "A", "B", "A"])
        holm_dot = bs.build_dot(tables, 0.05, "positive", 20, method="holm")
        self.assertIn("Holm", holm_dot)
        # Build a minimal MC-like significant table without invoking random simulation.
        mc_table = pd.DataFrame(
            [
                {
                    "from_code": "A",
                    "to_code": "B",
                    "observed": 2,
                    "simulated_mean": 0.5,
                    "simulated_sd": 0.5,
                    "empirical_p_two_sided": 0.01,
                    "empirical_p_holm": 0.02,
                    "empirical_q_bh": 0.02,
                    "expected_lt_5": True,
                    "transition_probability": 1.0,
                    "significance": "**",
                    "direction": "positive",
                }
            ]
        )
        mc_dot = bs.build_dot(tables, 0.05, "positive", 20, method="mc_raw", significant=mc_table)
        self.assertIn("Monte Carlo", mc_dot)
        self.assertIn("p=0.010", mc_dot)

    def test_run_monte_carlo_corrections_and_family(self):
        sequence = ["A", "B", "A", "B", "A"]
        tables = tables_for(sequence)
        records = [[{"code": code} for code in sequence]]
        mc = bs.run_monte_carlo(tables, records, 1, 200, 17, 0.05)
        diagnostics = mc["diagnostics"]
        self.assertEqual(len(diagnostics), 4)
        self.assertEqual(set(mc["significant_by_method"]), {"mc_raw", "mc_holm", "mc_bh"})
        self.assertTrue(diagnostics["empirical_p_two_sided"].notna().all())
        self.assertTrue(diagnostics["empirical_p_holm"].notna().all())
        self.assertTrue(diagnostics["empirical_q_bh"].notna().all())
        self.assertTrue((diagnostics["simulation_count"] == 200).all())
        self.assertTrue((diagnostics["random_seed"] == 17).all())

    def test_real_flower_data_regression(self):
        ai_path = (
            ROOT_DIR
            / "results"
            / "ui_runs"
            / "20261022_073434"
            / "03_iflytek_classroom"
            / "lesson_ai_codes.csv"
        )
        old_dir = ai_path.parent / "behavior_sequence"
        old_counts_path = old_dir / "lesson_transition_frequency.csv"
        old_adjusted_path = old_dir / "lesson_adjusted_residuals.csv"
        if not ai_path.exists() or not old_counts_path.exists() or not old_adjusted_path.exists():
            self.skipTest("真实回归数据或旧结果不存在")
        ai = pd.read_csv(ai_path)
        sequences, metadata = bs.prepare_sequences(ai, None, "ai", lag=1, single_sequence=True)
        tables = bs.compute_tables_for_sequences(sequences, 1, 0.05)
        self.assertEqual(metadata["sequence_mode"], "single_sequence_assumed")
        self.assertEqual(metadata["total_windows"] if "total_windows" in metadata else tables["total_windows"], 437)
        self.assertEqual(tables["total_transitions"], 436)
        old_counts = pd.read_csv(old_counts_path, index_col=0)
        self.assertEqual(tables["transition_counts"].shape, old_counts.shape)
        self.assertTrue(tables["transition_counts"].equals(old_counts))
        old_adjusted = pd.read_csv(old_adjusted_path, index_col=0)
        valid = tables["testable"].to_numpy()
        new_values = tables["adjusted_residuals"].to_numpy()
        old_values = old_adjusted.to_numpy()
        # V0.1 wrote residuals rounded to three decimals; compare at that precision.
        np.testing.assert_allclose(new_values[valid], old_values[valid], rtol=0, atol=5e-4)
        self.assertTrue(np.isnan(new_values[~valid]).all())


if __name__ == "__main__":
    unittest.main()
