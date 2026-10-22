from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import behavior_sequence as bs  # noqa: E402
from coding_dictionary import label_for, load_dictionary  # noqa: E402
from graph_export import render_transition_graph  # noqa: E402

ANALYSIS_DIR = (
    ROOT_DIR
    / "results"
    / "ui_runs"
    / "20261022_073434"
    / "03_iflytek_classroom"
    / "behavior_sequence_v02c"
)
AI_CODES = ANALYSIS_DIR.parent / "lesson_ai_codes.csv"


class LsaV02cTests(unittest.TestCase):
    def test_dictionary_mapping_and_unknown_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dictionary.csv"
            path.write_text("code,label,definition,category\nA,教师情感,接受学生情感,教师\n", encoding="utf-8")
            dictionary = load_dictionary(path)
            self.assertEqual(dictionary["A"]["label"], "教师情感")
            self.assertEqual(label_for("A", dictionary, "both"), "A 教师情感")
            self.assertEqual(label_for("Z", dictionary, "both"), "Z")

    def test_self_transition_filter_only_changes_display(self):
        tables = bs.compute_tables(["A", "B", "A", "B", "A"], lag=1, alpha=0.05)
        counts_before = tables["transition_counts"].copy()
        fake_positive_self = pd.DataFrame(
            [
                {
                    "from_code": "A",
                    "to_code": "A",
                    "observed": 5,
                    "transition_probability": 0.8,
                    "adjusted_residual": 3.0,
                    "raw_p_value": 0.001,
                    "p_holm": 0.01,
                    "q_bh": 0.01,
                    "expected_lt_5": False,
                    "significance": "**",
                    "direction": "positive",
                }
            ]
        )
        dot_with_self = bs.build_dot(
            tables, 0.05, "both", 20, method="raw",
            significant=fake_positive_self, show_self=True,
        )
        dot_without_self = bs.build_dot(
            tables, 0.05, "both", 20, method="raw",
            significant=fake_positive_self, show_self=False,
        )
        self.assertIn('"A" -> "A"', dot_with_self)
        self.assertNotIn('"A" -> "A"', dot_without_self)
        pd.testing.assert_frame_equal(tables["transition_counts"], counts_before)

    def test_graph_filters_do_not_change_statistics(self):
        tables = bs.compute_tables(["A", "B", "A", "B", "A"], lag=1, alpha=0.05)
        counts_before = tables["transition_counts"].copy()
        adjusted_before = tables["adjusted_residuals"].copy()
        bs.build_dot(
            tables, 0.05, "positive", 10, method="raw", show_self=False,
            min_count=5, min_probability=0.9,
        )
        pd.testing.assert_frame_equal(tables["transition_counts"], counts_before)
        pd.testing.assert_frame_equal(tables["adjusted_residuals"], adjusted_before)

    def test_collapsed_duration_and_no_cross_sequence(self):
        sequences = [
            [
                {"code": "A", "start_ms": 0, "end_ms": 5000},
                {"code": "A", "start_ms": 5000, "end_ms": 10000},
                {"code": "B", "start_ms": 10000, "end_ms": 15000},
            ],
            [
                {"code": "B", "start_ms": 0, "end_ms": 5000},
                {"code": "C", "start_ms": 5000, "end_ms": 10000},
            ],
        ]
        collapsed = bs.collapse_sequences(sequences, window_seconds=5)
        self.assertEqual([[r["code"] for r in seq] for seq in collapsed], [["A", "B"], ["B", "C"]])
        self.assertEqual(collapsed[0][0]["window_count"], 2)
        self.assertEqual(collapsed[0][0]["duration_s"], 10.0)
        self.assertEqual(collapsed[0][0]["duration_source"], "time_fields")

    def test_state_segment_view_is_descriptive_only(self):
        metadata = json.loads((ANALYSIS_DIR / "lesson_collapsed_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["analysis_unit"], "behavior_state_segment")
        self.assertEqual(metadata["inference"], "descriptive_only")
        self.assertFalse((ANALYSIS_DIR / "lesson_collapsed_significant_transitions.csv").exists())

    def test_input_hash_and_seed_recorded(self):
        metadata = json.loads((ANALYSIS_DIR / "lesson_behavior_sequence_metadata.json").read_text(encoding="utf-8"))
        expected = hashlib.sha256(AI_CODES.read_bytes()).hexdigest()
        self.assertEqual(metadata["input_sha256"], expected)
        self.assertEqual(metadata["random_seed"], 20261022)
        self.assertEqual(metadata["simulation_count"], 10000)

    def test_svg_png_export_is_valid(self):
        nodes = pd.DataFrame(
            [
                {"code": "A", "display_label": "A", "frequency": 10},
                {"code": "B", "display_label": "B", "frequency": 8},
            ]
        )
        edges = pd.DataFrame(
            [
                {
                    "from_code": "A",
                    "to_code": "B",
                    "observed": 4,
                    "transition_probability": 0.8,
                    "direction": "positive",
                    "expected_lt_5": False,
                    "weight": 2.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_transition_graph(nodes, edges, Path(tmp), "test_graph", "Test")
            svg = Path(paths["svg"]).read_text(encoding="utf-8")
            png = Path(paths["png"]).read_bytes()
            self.assertIn("<svg", svg)
            self.assertTrue(png.startswith(b"\x89PNG"))

    def test_html_report_numeric_and_method_content(self):
        metadata = json.loads((ANALYSIS_DIR / "lesson_behavior_sequence_metadata.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "report_builder.py"),
                    str(ANALYSIS_DIR),
                    "--method",
                    "holm",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("经典 LSA：Holm 校正", text)
            self.assertIn(str(metadata["total_windows"]), text)
            self.assertIn(str(metadata["total_transitions"]), text)
            self.assertIn("输入 SHA256", text)
            self.assertIn(str(metadata["random_seed"]), text)
            self.assertIn("低期望频数", text)
            self.assertIn("显著转移不代表因果", text)

    def test_html_report_uses_selected_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "report_builder.py"),
                    str(ANALYSIS_DIR),
                    "--method",
                    "mc_holm",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("Monte Carlo：Holm 校正", text)
            self.assertIn("序列内随机排列", text)

    def test_v02a_and_v02b_results_not_overwritten(self):
        self.assertTrue((ANALYSIS_DIR.parent / "behavior_sequence_v02a").exists())
        self.assertTrue((ANALYSIS_DIR.parent / "behavior_sequence_v02b").exists())
        self.assertTrue((ANALYSIS_DIR / "lesson_behavior_frequency.csv").exists())
        self.assertTrue((ANALYSIS_DIR.parent / "behavior_sequence_v02b" / "lesson_transition_frequency.csv").exists())

    def test_dictionary_label_in_dot(self):
        tables = bs.compute_tables(["A", "B", "A", "B", "A"], lag=1, alpha=0.05)
        dictionary = {"A": {"label": "教师情感"}, "B": {"label": "鼓励"}}
        dot = bs.build_dot(
            tables, 0.05, "both", 20, method="raw", dictionary=dictionary, label_mode="both"
        )
        self.assertIn("A 教师情感", dot)
        self.assertIn("B 鼓励", dot)

    def test_original_ai_codes_unchanged_by_analysis(self):
        before = hashlib.sha256(AI_CODES.read_bytes()).hexdigest()
        metadata = json.loads((ANALYSIS_DIR / "lesson_behavior_sequence_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["input_sha256"], before)
        # The analysis output is separate from the original coding CSV.
        self.assertNotEqual(AI_CODES.parent, ANALYSIS_DIR)


if __name__ == "__main__":
    unittest.main()
