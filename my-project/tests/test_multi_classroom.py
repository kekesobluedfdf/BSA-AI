from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
ROOT_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import multi_classroom as mc  # noqa: E402


def write_class(root: Path, class_id: str, codes: list[str], seq_ids: list[str] | None = None) -> dict:
    directory = root / "classes" / class_id
    directory.mkdir(parents=True, exist_ok=True)
    seq_ids = seq_ids or ["S1"] * len(codes)
    rows = []
    for index, (code, seq) in enumerate(zip(codes, seq_ids)):
        start = index * 5000
        end = start + 5000
        rows.append(
            {
                "window_id": index + 1,
                "start_ms": start,
                "end_ms": end,
                "start_hms": f"00:{start // 60000:02d}:{(start // 1000) % 60:02d}.000",
                "end_hms": f"00:{end // 60000:02d}:{(end // 1000) % 60:02d}.000",
                "duration_s": 5.0,
                "code": code,
                "sequence_id": seq,
                "text": code,
            }
        )
    path = directory / "lesson_ai_codes.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "class_id": class_id,
        "teacher_id": "T01",
        "source_video": f"synthetic_{class_id}.mp4",
        "coding_scheme_id": "scheme_v1",
        "window_duration_seconds": 5.0,
        "review_status": "reviewed",
        "file": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sequence_id_column": "sequence_id",
        "single_sequence": False,
    }


def make_study(root: Path, specs: list[tuple[str, list[str], list[str] | None]]) -> dict:
    entries = [write_class(root, class_id, codes, seqs) for class_id, codes, seqs in specs]
    manifest = {
        "study_id": "test_study",
        "teacher_id": "T01",
        "coding_scheme_id": "scheme_v1",
        "window_duration_seconds": 5.0,
        "lag": 1,
        "classes": entries,
    }
    (root / "study_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


class MultiClassroomTests(unittest.TestCase):
    def test_no_cross_class_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B"], None), ("c2", ["C", "D"], None)])
            result = mc.analyze(root, manifest)
            self.assertEqual(int(result["pooled_counts"].loc["B", "C"]), 0)
            self.assertEqual(int(result["pooled_counts"].loc["A", "B"]), 1)
            self.assertEqual(int(result["pooled_counts"].loc["C", "D"]), 1)

    def test_no_cross_sequence_transition_within_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root, [("c1", ["A", "B", "C", "D"], ["S1", "S1", "S2", "S2"])]
            )
            result = mc.analyze(root, manifest)
            self.assertEqual(int(result["pooled_counts"].loc["B", "C"]), 0)
            self.assertEqual(int(result["pooled_counts"].loc["A", "B"]), 1)
            self.assertEqual(int(result["pooled_counts"].loc["C", "D"]), 1)

    def test_pooled_counts_equal_sum_of_class_matrices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "A"], None), ("c2", ["A", "B"], None)])
            result = mc.analyze(root, manifest)
            expected = pd.DataFrame(0, index=result["labels"], columns=result["labels"], dtype=int)
            for item in result["classes"]:
                expected += item["tables"]["transition_counts"].reindex(
                    index=result["labels"], columns=result["labels"], fill_value=0
                )
            pd.testing.assert_frame_equal(result["pooled_counts"], expected)

    def test_pooled_probability_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "A"], None), ("c2", ["A", "B"], None)])
            result = mc.analyze(root, manifest)
            counts = result["pooled_counts"]
            expected = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
            pd.testing.assert_frame_equal(result["pooled_probability"], expected)

    def test_equal_weight_and_missing_start_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root,
                [
                    ("c1", ["A", "B", "A"], None),
                    ("c2", ["A", "B", "A", "A"], None),
                    ("c3", ["B", "C"], None),  # no A start
                ],
            )
            result = mc.analyze(root, manifest)
            self.assertAlmostEqual(float(result["equal_probability"].loc["A", "B"]), 0.75, places=12)
            self.assertEqual(int(result["equal_n"].loc["A", "B"]), 2)
            self.assertTrue(math.isnan(float(result["equal_probability"].loc["A", "C"])) is False)
            self.assertEqual(int(result["equal_n"].loc["A", "C"]), 2)

    def test_pooled_and_equal_weight_differ_with_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root,
                [
                    ("short", ["A", "B", "A"], None),
                    ("long", ["A", "C", "A", "C", "A", "C"], None),
                ],
            )
            result = mc.analyze(root, manifest)
            pooled_b = float(result["pooled_probability"].loc["A", "B"])
            pooled_c = float(result["pooled_probability"].loc["A", "C"])
            equal_b = float(result["equal_probability"].loc["A", "B"])
            equal_c = float(result["equal_probability"].loc["A", "C"])
            self.assertGreater(pooled_c, pooled_b)
            self.assertAlmostEqual(equal_b, 0.5, places=12)
            self.assertAlmostEqual(equal_c, 0.5, places=12)
            self.assertNotAlmostEqual(pooled_c, equal_c, places=3)

    def test_duplicate_class_id_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B"], None)])
            manifest["classes"].append(dict(manifest["classes"][0]))
            errors = mc.validate_manifest(root, manifest)
            self.assertTrue(any("重复 class_id" in error for error in errors))

    def test_duplicate_file_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B"], None)])
            duplicate = dict(manifest["classes"][0])
            duplicate["class_id"] = "c2"
            manifest["classes"].append(duplicate)
            errors = mc.validate_manifest(root, manifest)
            self.assertTrue(any("重复文件" in error for error in errors))

    def test_coding_scheme_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B"], None)])
            manifest["classes"][0]["coding_scheme_id"] = "other_scheme"
            errors = mc.validate_manifest(root, manifest)
            self.assertTrue(any("编码体系不一致" in error for error in errors))

    def test_window_duration_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B"], None)])
            manifest["classes"][0]["window_duration_seconds"] = 3.0
            errors = mc.validate_manifest(root, manifest)
            self.assertTrue(any("窗口长度不一致" in error for error in errors))

    def test_time_out_of_order_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "C"], None)])
            path = root / manifest["classes"][0]["file"]
            frame = pd.read_csv(path)
            frame.loc[0, "start_ms"] = 20000
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            manifest["classes"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            result = mc.analyze(root, manifest)
            self.assertTrue(result["classes"][0]["sequence_meta"]["input_time_out_of_order"])
            self.assertFalse(bool(result["classroom_summary"].iloc[0]["time_order_ok"]))

    def test_zero_transition_class_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A"], None), ("c2", ["A", "B"], None)])
            result = mc.analyze(root, manifest)
            summary = result["classroom_summary"].set_index("class_id")
            self.assertEqual(int(summary.loc["c1", "transitions"]), 0)
            self.assertEqual(int(summary.loc["c2", "transitions"]), 1)

    def test_stability_presence_and_descriptives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root,
                [("c1", ["A", "B", "A"], None), ("c2", ["A", "B", "A", "A"], None)],
            )
            result = mc.analyze(root, manifest)
            row = result["stability"].query("from_code == 'A' and to_code == 'B'").iloc[0]
            self.assertEqual(int(row["valid_classes"]), 2)
            self.assertEqual(int(row["present_classes"]), 2)
            self.assertAlmostEqual(float(row["presence_ratio"]), 1.0)
            self.assertAlmostEqual(float(row["mean_probability"]), 0.75, places=12)
            self.assertAlmostEqual(float(row["iqr_probability"]), 0.25, places=12)

    def test_loco_recomputes_without_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root,
                [("c1", ["A", "B", "A"], None), ("c2", ["A", "C", "A", "C"], None)],
            )
            result = mc.analyze(root, manifest)
            loco = result["loco"]
            row = loco[(loco["excluded_class"] == "c1") & (loco["from_code"] == "A") & (loco["to_code"] == "B")].iloc[0]
            self.assertAlmostEqual(float(row["loco_probability"]), 0.0, places=12)

    def test_pooled_outputs_have_no_significance_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "A"], None)])
            result = mc.analyze(root, manifest)
            self.assertFalse(any("p_" in column or "significant" in column for column in result["pooled_probability"].columns))
            self.assertFalse(any("p_" in column or "significant" in column for column in result["stability"].columns))

    def test_single_classroom_regression_unchanged(self):
        from behavior_sequence import compute_tables_for_sequences, prepare_sequences

        ai_path = (
            ROOT_DIR
            / "results"
            / "ui_runs"
            / "20261022_073434"
            / "03_iflytek_classroom"
            / "lesson_ai_codes.csv"
        )
        if not ai_path.exists():
            self.skipTest("真实单课堂数据不存在")
        frame = pd.read_csv(ai_path)
        sequences, _ = prepare_sequences(frame, None, "ai", 1, single_sequence=True)
        tables = compute_tables_for_sequences(sequences, 1, 0.05)
        self.assertEqual(tables["total_windows"], 437)
        self.assertEqual(tables["total_transitions"], 436)

    def test_classroom_comparison_probability_and_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "C", "A", "C"], None)])
            result = mc.analyze(root, manifest)
            row = result["classroom_comparison"].query(
                "class_id == 'c1' and from_code == 'A' and to_code == 'B'"
            ).iloc[0]
            self.assertEqual(int(row["observed"]), 1)
            self.assertEqual(int(row["row_total"]), 2)
            self.assertAlmostEqual(float(row["probability"]), 0.5, places=12)
            self.assertEqual(int(row["rank_in_source"]), 1)

    def test_sequence_count_and_window_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root, [("c1", ["A", "B", "C", "D"], ["S1", "S1", "S2", "S2"])]
            )
            result = mc.analyze(root, manifest)
            row = result["classroom_summary"].iloc[0]
            self.assertEqual(int(row["sequences"]), 2)
            self.assertEqual(int(row["windows"]), 4)
            self.assertEqual(int(row["transitions"]), 2)

    def test_multi_report_contains_limitation_and_class_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(
                root,
                [("c1", ["A", "B"], None), ("c2", ["C", "D"], None), ("c3", ["A", "C"], None)],
            )
            output = mc.run(root, manifest, root / "results", lag=1)
            text = (output / "multi_classroom_report.html").read_text(encoding="utf-8")
            self.assertIn("尚未提供经过课堂层级依赖调整的显著性推断", text)
            self.assertIn("c1", text)
            self.assertIn("c2", text)
            self.assertIn("c3", text)
            self.assertTrue((output / "classroom_summary.csv").exists())

    def test_original_data_not_modified_and_metadata_traceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_study(root, [("c1", ["A", "B", "A"], None)])
            path = root / manifest["classes"][0]["file"]
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            output = mc.run(root, manifest, root / "results", lag=1)
            after = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            metadata = json.loads((output / "analysis_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["classes"][0]["sha256"], before)
            self.assertTrue((output / "multi_classroom_report.html").exists())


if __name__ == "__main__":
    unittest.main()
