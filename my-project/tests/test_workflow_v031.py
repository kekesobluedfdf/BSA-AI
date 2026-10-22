from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import classroom_registry as cr  # noqa: E402
import multi_classroom as mc  # noqa: E402


def make_ai_codes(path: Path, codes: list[str], human: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, code in enumerate(codes):
        start = index * 5000
        rows.append(
            {
                "window_id": index + 1,
                "start_ms": start,
                "end_ms": start + 5000,
                "start_hms": f"00:{start // 60000:02d}:{(start // 1000) % 60:02d}.000",
                "end_hms": f"00:{(start + 5000) // 60000:02d}:{((start + 5000) // 1000) % 60:02d}.000",
                "duration_s": 5.0,
                "speech_s": 5.0,
                "text": code,
                "code": code,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    if human is not None:
        pd.DataFrame(
            {"window_id": [row["window_id"] for row in rows], "human_code": human, "human_note": ""}
        ).to_csv(path.parent / path.name.replace("_ai_codes.csv", "_human_codes.csv"), index=False, encoding="utf-8-sig")


def make_run(root: Path, run_id: str, codes: list[str] | None = None, human: list[str] | None = None) -> Path:
    run = root / "ui_runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "lesson.wav").write_bytes(b"fake audio")
    (run / "settings.json").write_text(
        json.dumps(
            {
                "audio_sha256": "a" * 64,
                "audio_filename": "lesson.wav",
                "window_seconds": 5.0,
            }
        ),
        encoding="utf-8",
    )
    if codes:
        code_dir = run / "03_iflytek_classroom"
        make_ai_codes(code_dir / "lesson_ai_codes.csv", codes, human)
        (code_dir / "lesson_ai_code_metadata.json").write_text(
            json.dumps({"coding_scheme_id": "scheme_v1", "model": "deepseek-chat"}),
            encoding="utf-8",
        )
    return run


def make_study(root: Path, entries: list[dict], study_id: str = "study") -> tuple[Path, Path]:
    study_dir = root / "studies" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": study_id,
        "teacher_id": "T01",
        "coding_scheme_id": "scheme_v1",
        "window_duration_seconds": 5.0,
        "lag": 1,
        "classes": entries,
    }
    path = study_dir / "study_manifest.json"
    mc.save_manifest(path, manifest)
    return study_dir, path


class ClassroomWorkflowV031Tests(unittest.TestCase):
    def test_discover_classrooms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B", "A"])
            records = cr.discover_runs(root / "ui_runs")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["status"], "ready")
            self.assertEqual(records[0]["total_windows"], 3)
            self.assertEqual(records[0]["coding_scheme_id"], "scheme_v1")

    def test_repeated_scan_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            registry_path = root / "registry.json"
            first = cr.scan_and_update(root / "ui_runs", registry_path)
            second = cr.scan_and_update(root / "ui_runs", registry_path)
            self.assertEqual(len(first["classrooms"]), 1)
            self.assertEqual(len(second["classrooms"]), 1)

    def test_one_run_multiple_output_dirs_is_one_classroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root, "20260101_000000", ["A", "B"])
            make_ai_codes(run / "03_iflytek_r10" / "lesson_ai_codes.csv", ["A", "C", "A"])
            records = cr.discover_runs(root / "ui_runs")
            self.assertEqual(len(records), 1)

    def test_missing_metadata_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root, "20260101_000000", ["A", "B"])
            (run / "03_iflytek_classroom" / "lesson_ai_code_metadata.json").unlink()
            records = cr.discover_runs(root / "ui_runs")
            self.assertEqual(records[0]["coding_scheme_id"], "unknown")
            self.assertEqual(records[0]["status"], "ready")

    def test_missing_ai_codes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", None)
            records = cr.discover_runs(root / "ui_runs")
            self.assertEqual(records[0]["status"], "missing_ai_codes")
            self.assertEqual(records[0]["ai_codes_path"], "")

    def test_class_id_stable_and_display_name_editable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            registry_path = root / "registry.json"
            registry = cr.scan_and_update(root / "ui_runs", registry_path)
            run_id = next(iter(registry["classrooms"]))
            old_class_id = registry["classrooms"][run_id]["class_id"]
            cr.update_classroom(registry_path, run_id, {"display_name": "课堂一"})
            registry = cr.scan_and_update(root / "ui_runs", registry_path)
            self.assertEqual(registry["classrooms"][run_id]["class_id"], old_class_id)
            self.assertEqual(registry["classrooms"][run_id]["display_name"], "课堂一")

    def test_human_complete_and_partial_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B", "A"], ["A", "B", "A"])
            records = cr.discover_runs(root / "ui_runs")
            self.assertEqual(records[0]["review_status"], "complete")
            self.assertEqual(records[0]["reviewed_windows"], 3)
            make_run(root, "20260101_000001", ["A", "B"], ["A", ""])
            records = cr.discover_runs(root / "ui_runs")
            partial = next(record for record in records if record["run_id"] == "20260101_000001")
            self.assertEqual(partial["review_status"], "partial")

    def test_manifest_entry_marks_human_and_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root, "20260101_000000", ["A", "B"], ["A", "B"])
            record = cr.discover_runs(root / "ui_runs")[0]
            entry = cr.record_to_manifest_entry(record, "human")
            self.assertEqual(entry["code_source"], "human")
            self.assertEqual(entry["source_run_dir"], str(run))
            self.assertTrue(entry["human_file"])
            self.assertEqual(entry["human_sha256"], hashlib.sha256(Path(entry["human_file"]).read_bytes()).hexdigest())

    def test_append_and_remove_classroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            make_run(root, "20260101_000001", ["B", "C"])
            records = cr.discover_runs(root / "ui_runs")
            entries = [cr.record_to_manifest_entry(record) for record in records]
            study_dir, manifest_path = make_study(root, [entries[0]])
            mc.append_classes(study_dir, manifest_path, [entries[1]])
            manifest = mc.load_manifest(manifest_path)
            self.assertEqual(len(manifest["classes"]), 2)
            mc.remove_classes(study_dir, manifest_path, [entries[1]["class_id"]])
            manifest = mc.load_manifest(manifest_path)
            self.assertEqual(len(manifest["classes"]), 1)

    def test_append_duplicate_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0])
            study_dir, manifest_path = make_study(root, [entry])
            with self.assertRaises(ValueError):
                mc.append_classes(study_dir, manifest_path, [entry])

    def test_display_name_update_keeps_class_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0])
            study_dir, manifest_path = make_study(root, [entry])
            mc.update_display_name(manifest_path, entry["class_id"], "新名称")
            manifest = mc.load_manifest(manifest_path)
            self.assertEqual(manifest["classes"][0]["display_name"], "新名称")
            self.assertEqual(manifest["classes"][0]["class_id"], entry["class_id"])

    def test_external_field_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "external.csv"
            pd.DataFrame(
                {
                    "behavior": ["A", "B", "A"],
                    "timestamp": ["00:00:00", "00:00:05", "00:00:10"],
                    "segment": ["S1", "S1", "S1"],
                    "length": [5, 5, 5],
                    "idx": [1, 2, 3],
                }
            ).to_csv(raw, index=False)
            target = root / "normalized.csv"
            mc.normalize_external_csv(
                raw,
                {
                    "code": "behavior",
                    "time": "timestamp",
                    "time_kind": "start_hms",
                    "chain": "ignored",
                    "window_id": "idx",
                    "sequence_id": "segment",
                    "duration": "length",
                },
                target,
            )
            frame = pd.read_csv(target)
            self.assertEqual(frame["code"].tolist(), ["A", "B", "A"])
            self.assertEqual(frame["sequence_id"].tolist(), ["S1", "S1", "S1"])
            self.assertEqual(frame["window_id"].tolist(), [1, 2, 3])

    def test_external_mapping_rejects_missing_code_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "external.csv"
            pd.DataFrame({"x": ["A"]}).to_csv(raw, index=False)
            with self.assertRaises(ValueError):
                mc.normalize_external_csv(raw, {"code": "missing", "time": "x"}, root / "out.csv")

    def test_manifest_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "study_manifest.json"
            mc.save_manifest(path, {"study_id": "x", "classes": []})
            self.assertTrue(path.exists())
            self.assertFalse((Path(tmp) / "study_manifest.json.tmp").exists())

    def test_human_source_requires_complete_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root, "20260101_000000", ["A", "B"], ["A", ""])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0], "human")
            study_dir, manifest_path = make_study(root, [entry])
            errors = mc.validate_manifest(study_dir, mc.load_manifest(manifest_path))
            self.assertTrue(any("人工复核不完整" in error for error in errors))

    def test_human_source_analysis_uses_human_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"], ["C", "C"])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0], "human")
            study_dir, manifest_path = make_study(root, [entry])
            result = mc.analyze(study_dir, mc.load_manifest(manifest_path))
            self.assertEqual(result["classroom_summary"].iloc[0]["code_source"], "human")
            self.assertEqual(int(result["pooled_counts"].loc["C", "C"]), 1)

    def test_external_class_has_no_source_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = {
                "class_id": "external",
                "file": "external.csv",
                "coding_scheme_id": "scheme_v1",
                "window_duration_seconds": 5.0,
                "single_sequence": True,
                "sequence_id_column": "sequence_id",
                "code_source": "ai",
                "source_run_dir": "",
            }
            self.assertEqual(entry["source_run_dir"], "")

    def test_old_analysis_results_preserved_on_maintenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0])
            study_dir, manifest_path = make_study(root, [entry])
            old_output = root / "old_analysis"
            old_output.mkdir()
            before = (old_output / "result.txt")
            before.write_text("old", encoding="utf-8")
            mc.update_display_name(manifest_path, entry["class_id"], "new")
            self.assertEqual(before.read_text(encoding="utf-8"), "old")

    def test_original_csv_not_modified_by_registry_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = make_run(root, "20260101_000000", ["A", "B"])
            codes = run / "03_iflytek_classroom" / "lesson_ai_codes.csv"
            before = hashlib.sha256(codes.read_bytes()).hexdigest()
            cr.scan_and_update(root / "ui_runs", root / "registry.json")
            self.assertEqual(hashlib.sha256(codes.read_bytes()).hexdigest(), before)

    def test_research_project_reopen_preserves_class_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "20260101_000000", ["A", "B"])
            entry = cr.record_to_manifest_entry(cr.discover_runs(root / "ui_runs")[0])
            study_dir, manifest_path = make_study(root, [entry])
            reopened = mc.load_manifest(manifest_path)
            self.assertEqual(reopened["classes"][0]["class_id"], entry["class_id"])
            self.assertEqual(reopened["classes"][0]["source_run_dir"], entry["source_run_dir"])


if __name__ == "__main__":
    unittest.main()
