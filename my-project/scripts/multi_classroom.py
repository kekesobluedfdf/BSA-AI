from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_sequence import (
    CODE_GROUPS,
    CODE_NAMES,
    VALID_CODES,
    compute_tables_for_sequences,
    prepare_sequences,
)
from graph_export import render_transition_graph

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_STUDY_ROOT = ROOT_DIR / "studies"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def resolve_class_file(study_dir: Path, entry: dict) -> Path:
    path = Path(entry["file"])
    return path if path.is_absolute() else study_dir / path


def append_classes(study_dir: Path, manifest_path: Path, new_entries: list[dict]) -> dict:
    manifest = load_manifest(manifest_path)
    proposed = json.loads(json.dumps(manifest))
    proposed.setdefault("classes", []).extend(new_entries)
    errors = validate_manifest(study_dir, proposed)
    if errors:
        raise ValueError("追加课堂校验失败：\n" + "\n".join(errors))
    save_manifest(manifest_path, proposed)
    return proposed


def remove_classes(study_dir: Path, manifest_path: Path, class_ids: list[str]) -> dict:
    manifest = load_manifest(manifest_path)
    remove = set(class_ids)
    proposed = json.loads(json.dumps(manifest))
    proposed["classes"] = [
        entry for entry in proposed.get("classes", []) if entry.get("class_id") not in remove
    ]
    if not proposed["classes"]:
        raise ValueError("研究项目至少需要保留一节课堂")
    save_manifest(manifest_path, proposed)
    return proposed


def update_display_name(manifest_path: Path, class_id: str, display_name: str) -> dict:
    manifest = load_manifest(manifest_path)
    found = False
    for entry in manifest.get("classes", []):
        if entry.get("class_id") == class_id:
            entry["display_name"] = display_name
            found = True
    if not found:
        raise KeyError(class_id)
    save_manifest(manifest_path, manifest)
    return manifest


def normalize_external_csv(
    source_path: Path,
    mapping: dict,
    target_path: Path,
    default_sequence: str = "S1",
) -> Path:
    frame = pd.read_csv(source_path)
    code_column = mapping.get("code")
    time_column = mapping.get("time")
    if not code_column or code_column not in frame.columns:
        raise ValueError("外部 CSV 未映射行为编码列")
    if not time_column or time_column not in frame.columns:
        raise ValueError("外部 CSV 未映射时间列")
    normalized = pd.DataFrame()
    normalized["code"] = frame[code_column].astype(str).str.strip().str.upper()
    time_kind = mapping.get("time_kind", "start_ms")
    normalized[time_kind] = frame[time_column]
    window_column = mapping.get("window_id")
    normalized["window_id"] = (
        pd.to_numeric(frame[window_column], errors="coerce")
        if window_column and window_column in frame.columns
        else pd.Series(range(1, len(frame) + 1))
    )
    sequence_column = mapping.get("sequence_id")
    normalized["sequence_id"] = (
        frame[sequence_column].fillna(default_sequence).astype(str)
        if sequence_column and sequence_column in frame.columns
        else default_sequence
    )
    duration_column = mapping.get("duration")
    if duration_column and duration_column in frame.columns:
        normalized["duration_s"] = pd.to_numeric(frame[duration_column], errors="coerce")
    else:
        normalized["duration_s"] = np.nan
    if normalized["window_id"].isna().any():
        raise ValueError("窗口编号存在非数值")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(target_path, index=False, encoding="utf-8-sig")
    return target_path


def validate_manifest(study_dir: Path, manifest: dict) -> list[dict]:
    errors = []
    classes = manifest.get("classes", [])
    if not classes:
        errors.append("研究项目没有课堂数据")
    seen_ids = set()
    seen_hashes = {}
    scheme = manifest.get("coding_scheme_id")
    window_seconds = manifest.get("window_duration_seconds")
    for entry in classes:
        class_id = str(entry.get("class_id", "")).strip()
        if not class_id:
            errors.append("存在 class_id 为空的课堂")
            continue
        if class_id in seen_ids:
            errors.append(f"重复 class_id：{class_id}")
        seen_ids.add(class_id)
        path = resolve_class_file(study_dir, entry)
        if not path.exists():
            errors.append(f"{class_id} 文件不存在：{path}")
            continue
        file_hash = sha256_file(path)
        if file_hash in seen_hashes:
            errors.append(f"重复文件：{class_id} 与 {seen_hashes[file_hash]} 哈希相同")
        seen_hashes[file_hash] = class_id
        if entry.get("coding_scheme_id", scheme) != scheme:
            errors.append(f"{class_id} 编码体系不一致：{entry.get('coding_scheme_id')} != {scheme}")
        if abs(float(entry.get("window_duration_seconds", window_seconds)) - float(window_seconds)) > 1e-6:
            errors.append(
                f"{class_id} 窗口长度不一致：{entry.get('window_duration_seconds')} != {window_seconds}"
            )
        try:
            frame = pd.read_csv(path)
            if "code" not in frame.columns:
                errors.append(f"{class_id} 缺少 code 列")
            if "duration_s" in frame.columns and window_seconds:
                actual_duration = float(pd.to_numeric(frame["duration_s"], errors="coerce").median())
                if abs(actual_duration - float(window_seconds)) > 0.01:
                    errors.append(
                        f"{class_id} 实际窗口长度 {actual_duration} 与声明 {window_seconds} 不一致"
                    )
            source = entry.get("code_source", "ai")
            human_frame = None
            if source in {"human", "human_or_ai"}:
                human_file = entry.get("human_file")
                human_path = Path(human_file) if human_file else None
                if not human_path or not human_path.exists():
                    errors.append(f"{class_id} 选择了人工编码，但 human_file 不存在")
                else:
                    human_frame = pd.read_csv(human_path)
                    valid = (
                        set(
                            human_frame.loc[
                                human_frame["human_code"].astype(str).str.upper().isin(set(VALID_CODES)),
                                "window_id",
                            ]
                        )
                        if "human_code" in human_frame.columns and "window_id" in human_frame.columns
                        else set()
                    )
                    if source == "human" and not set(frame["window_id"]).issubset(valid):
                        errors.append(f"{class_id} 人工复核不完整，不能直接作为 human 编码来源")
            # Reuse V0.2a/V0.2b boundary and time validation.
            prepare_sequences(
                frame,
                human_frame,
                source,
                int(manifest.get("lag", 1)),
                sequence_column=entry.get("sequence_id_column", "sequence_id"),
                single_sequence=bool(entry.get("single_sequence", False)),
                time_column=entry.get("time_column", "auto"),
            )
        except Exception as exc:
            errors.append(f"{class_id} 数据结构错误：{exc}")
    return errors


def analyze(study_dir: Path, manifest: dict, lag: int = 1) -> dict:
    errors = validate_manifest(study_dir, manifest)
    if errors:
        raise ValueError("数据质量检查失败：\n" + "\n".join(errors))

    classes = manifest["classes"]
    class_tables = []
    all_labels = set()
    for entry in classes:
        file_path = resolve_class_file(study_dir, entry)
        frame = pd.read_csv(file_path)
        source = entry.get("code_source", "ai")
        human_path = Path(entry["human_file"]) if entry.get("human_file") else None
        human_frame = pd.read_csv(human_path) if human_path and human_path.exists() else None
        sequences, sequence_meta = prepare_sequences(
            frame,
            human_frame,
            source,
            lag,
            sequence_column=entry.get("sequence_id_column", "sequence_id"),
            single_sequence=bool(entry.get("single_sequence", False)),
            time_column=entry.get("time_column", "auto"),
        )
        tables = compute_tables_for_sequences(sequences, lag, 0.05, set())
        all_labels.update(tables["labels"])
        class_tables.append(
            {
                "entry": entry,
                "path": file_path,
                "sha256": sha256_file(file_path),
                "sequences": sequences,
                "sequence_meta": sequence_meta,
                "tables": tables,
            }
        )
    labels = [code for code in VALID_CODES if code in all_labels]
    if not labels:
        raise ValueError("没有可分析的行为编码")

    pooled_counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    class_counts = {}
    classroom_rows = []
    comparison_rows = []
    for item in class_tables:
        entry = item["entry"]
        table = item["tables"]
        counts = table["transition_counts"].reindex(index=labels, columns=labels, fill_value=0)
        class_counts[entry["class_id"]] = counts
        pooled_counts += counts
        diagonal = int(np.diag(counts.to_numpy()).sum())
        total = int(counts.to_numpy().sum())
        classroom_rows.append(
            {
                "class_id": entry["class_id"],
                "teacher_id": entry.get("teacher_id", manifest.get("teacher_id", "")),
                "sequences": item["sequence_meta"]["sequence_count"],
                "windows": table["total_windows"],
                "transitions": total,
                "self_transitions": diagonal,
                "non_self_transitions": total - diagonal,
                "self_ratio": round(diagonal / total, 4) if total else math.nan,
                "non_self_ratio": round((total - diagonal) / total, 4) if total else math.nan,
                "coding_scheme_id": entry.get("coding_scheme_id", manifest.get("coding_scheme_id")),
                "window_duration_seconds": entry.get(
                    "window_duration_seconds", manifest.get("window_duration_seconds")
                ),
                "review_status": entry.get("review_status", ""),
                "code_source": entry.get("code_source", "ai"),
                "human_review_status": entry.get("review_status", "absent"),
                "human_sha256": sha256_file(human_path) if human_path and human_path.exists() else "",
                "time_source": item["sequence_meta"]["time_source"],
                "time_order_ok": not bool(item["sequence_meta"]["input_time_out_of_order"]),
                "file": str(item["path"]),
                "sha256": item["sha256"],
            }
        )
        probability = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        for source in labels:
            row_total = int(counts.loc[source].sum())
            if row_total == 0:
                continue
            ranked = probability.loc[source].dropna().sort_values(ascending=False)
            for target in labels:
                observed = int(counts.loc[source, target])
                comparison_rows.append(
                    {
                        "class_id": entry["class_id"],
                        "from_code": source,
                        "to_code": target,
                        "observed": observed,
                        "row_total": row_total,
                        "probability": float(probability.loc[source, target]),
                        "rank_in_source": int(list(ranked.index).index(target) + 1),
                    }
                )

    pooled_row_totals = pooled_counts.sum(axis=1)
    pooled_probability = pooled_counts.div(pooled_row_totals.replace(0, np.nan), axis=0)
    equal_probability = pd.DataFrame(index=labels, columns=labels, dtype=float)
    equal_n = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for source in labels:
        for target in labels:
            values = []
            for item in class_tables:
                counts = class_counts[item["entry"]["class_id"]]
                row_total = counts.loc[source].sum()
                if row_total > 0:
                    values.append(float(counts.loc[source, target] / row_total))
            equal_n.loc[source, target] = len(values)
            equal_probability.loc[source, target] = float(np.mean(values)) if values else math.nan

    frequency = pd.DataFrame(
        {
            "code": labels,
            "code_name": [CODE_NAMES[code] for code in labels],
            "code_group": [CODE_GROUPS[code] for code in labels],
            "frequency": [
                sum(
                    int(item["tables"]["frequency"].set_index("code")["frequency"].get(code, 0))
                    for item in class_tables
                )
                for code in labels
            ],
        }
    )
    frequency["percentage"] = (
        frequency["frequency"] / frequency["frequency"].sum() * 100
    ).round(2)
    frequency = frequency.sort_values(["frequency", "code"], ascending=[False, True]).reset_index(drop=True)

    stability_rows = []
    for source in labels:
        for target in labels:
            values = []
            present = 0
            valid = 0
            total_observed = 0
            for item in class_tables:
                counts = class_counts[item["entry"]["class_id"]]
                row_total = int(counts.loc[source].sum())
                observed = int(counts.loc[source, target])
                if row_total > 0:
                    valid += 1
                    values.append(observed / row_total)
                    total_observed += observed
                    if observed > 0:
                        present += 1
            if valid:
                array = np.asarray(values, dtype=float)
                stability_rows.append(
                    {
                        "from_code": source,
                        "to_code": target,
                        "valid_classes": valid,
                        "present_classes": present,
                        "presence_ratio": round(present / valid, 4),
                        "mean_probability": round(float(array.mean()), 4),
                        "median_probability": round(float(np.median(array)), 4),
                        "sd_probability": round(float(array.std(ddof=1)), 4) if valid > 1 else math.nan,
                        "min_probability": round(float(array.min()), 4),
                        "max_probability": round(float(array.max()), 4),
                        "q1_probability": round(float(np.quantile(array, 0.25)), 4),
                        "q3_probability": round(float(np.quantile(array, 0.75)), 4),
                        "iqr_probability": round(float(np.quantile(array, 0.75) - np.quantile(array, 0.25)), 4),
                        "total_observed": total_observed,
                        "warning": "valid_classes_lt_2" if valid < 2 else ("observed_lt_5" if total_observed < 5 else ""),
                    }
                )
    stability = pd.DataFrame(stability_rows)

    full_probability = pooled_probability.copy()
    loco_rows = []
    for excluded in class_tables:
        excluded_id = excluded["entry"]["class_id"]
        leave_counts = pooled_counts.copy()
        leave_counts -= class_counts[excluded_id]
        leave_row = leave_counts.sum(axis=1)
        leave_probability = leave_counts.div(leave_row.replace(0, np.nan), axis=0)
        for source in labels:
            for target in labels:
                loco = leave_probability.loc[source, target]
                full = full_probability.loc[source, target]
                if pd.isna(full) and pd.isna(loco):
                    continue
                loco_rows.append(
                    {
                        "excluded_class": excluded_id,
                        "from_code": source,
                        "to_code": target,
                        "full_probability": full,
                        "loco_probability": loco,
                        "delta": loco - full if pd.notna(loco) and pd.notna(full) else math.nan,
                    }
                )
    loco = pd.DataFrame(loco_rows)

    return {
        "study_id": manifest.get("study_id"),
        "teacher_id": manifest.get("teacher_id"),
        "labels": labels,
        "classes": class_tables,
        "classroom_summary": pd.DataFrame(classroom_rows),
        "frequency": frequency,
        "pooled_counts": pooled_counts,
        "pooled_probability": pooled_probability,
        "equal_probability": equal_probability,
        "equal_n": equal_n,
        "classroom_comparison": pd.DataFrame(comparison_rows),
        "stability": stability,
        "loco": loco,
        "full_probability": full_probability,
    }


def write_graph(result: dict, output_dir: Path, prefix: str) -> dict[str, str]:
    frequency = result["frequency"]
    nodes = frequency[["code", "frequency"]].copy()
    nodes["display_label"] = nodes["code"]
    edges_rows = []
    for source in result["labels"]:
        for target in result["labels"]:
            probability = result["pooled_probability"].loc[source, target]
            if pd.isna(probability):
                continue
            edges_rows.append(
                {
                    "from_code": source,
                    "to_code": target,
                    "observed": int(result["pooled_counts"].loc[source, target]),
                    "transition_probability": float(probability),
                    "direction": "positive",
                    "expected_lt_5": False,
                    "weight": max(0.5, float(probability) * 6),
                }
            )
    edges = pd.DataFrame(edges_rows).sort_values("transition_probability", ascending=False).head(60)
    return render_transition_graph(nodes, edges, output_dir, prefix, "多课堂联合转移图")


def build_html(result: dict, metadata: dict, graph_paths: dict[str, str]) -> str:
    def table(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, classes="data", border=0, escape=True, na_rep="NA")

    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>多课堂 LSA 报告</title>",
        "<style>body{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:32px;color:#222} table{border-collapse:collapse;font-size:12px;margin:8px 0 18px} th,td{border:1px solid #ddd;padding:4px 6px} .warning{background:#fff4e5;padding:8px 12px;border-left:4px solid #b9770e}</style></head><body>",
        "<h1>多课堂 LSA 报告</h1>",
        f"<p>研究项目：{html.escape(str(metadata.get('study_id')))}；教师：{html.escape(str(metadata.get('teacher_id')))}；生成时间：{html.escape(metadata.get('created_at', ''))}</p>",
        "<div class='warning'>多课堂联合结果为描述性汇总，尚未提供经过课堂层级依赖调整的显著性推断。不得把单课堂显著性数量相加解释为联合显著性。</div>",
        "<h2>1. 数据与课堂</h2>",
        table(result["classroom_summary"]),
        "<h2>2. 联合行为频次</h2>",
        table(result["frequency"]),
        "<h2>3. 联合转移频数</h2>",
        result["pooled_counts"].to_html(classes="data", border=0, na_rep="NA"),
        "<h2>4. 联合条件转移概率</h2>",
        result["pooled_probability"].round(4).to_html(classes="data", border=0, na_rep="NA"),
        "<h2>5. 课堂等权转移概率</h2>",
        result["equal_probability"].round(4).to_html(classes="data", border=0, na_rep="NA"),
        "<h2>6. 跨课堂稳定性</h2>",
        table(result["stability"]),
        "<h2>7. Leave-One-Class-Out</h2>",
        table(result["loco"].head(200)),
        "<h2>8. 转换图</h2>",
    ]
    if graph_paths.get("svg"):
        parts.append(f"<img src='{html.escape(Path(graph_paths['svg']).name)}' style='max-width:100%;border:1px solid #ddd'>")
    parts += [
        "<h2>9. 统计限制</h2>",
        "<ul><li>课堂间结果按课堂分别计算，联合矩阵为描述性汇总。</li>",
        "<li>课堂长度不同，不能仅凭原始频数比较课堂差异。</li>",
        "<li>多课堂联合未实现课堂层级显著性推断。</li>",
        "<li>显著不等于因果；单课堂结果不能外推。</li></ul>",
        "</body></html>",
    ]
    return "\n".join(parts)


def run(study_dir: Path, manifest: dict, output_root: Path, lag: int = 1) -> Path:
    result = analyze(study_dir, manifest, lag)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / str(manifest["study_id"]) / "lsa_v03" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(output_dir / "study_manifest.json", manifest)

    result["classroom_summary"].to_csv(output_dir / "classroom_summary.csv", index=False, encoding="utf-8-sig")
    result["frequency"].to_csv(output_dir / "pooled_frequency.csv", index=False, encoding="utf-8-sig")
    result["pooled_counts"].to_csv(output_dir / "pooled_transition_counts.csv", encoding="utf-8-sig")
    result["pooled_probability"].round(4).to_csv(output_dir / "pooled_transition_probabilities.csv", encoding="utf-8-sig")
    result["equal_probability"].round(4).to_csv(output_dir / "equal_weight_transition_probabilities.csv", encoding="utf-8-sig")
    result["equal_n"].to_csv(output_dir / "equal_weight_class_counts.csv", encoding="utf-8-sig")
    result["classroom_comparison"].to_csv(output_dir / "classroom_transition_comparison.csv", index=False, encoding="utf-8-sig")
    result["stability"].to_csv(output_dir / "transition_stability.csv", index=False, encoding="utf-8-sig")
    result["loco"].to_csv(output_dir / "leave_one_class_out.csv", index=False, encoding="utf-8-sig")

    graph_paths = write_graph(result, output_dir, "pooled_transition_graph")
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "program_version": "LSA V0.3",
        "study_id": manifest.get("study_id"),
        "teacher_id": manifest.get("teacher_id"),
        "coding_scheme_id": manifest.get("coding_scheme_id"),
        "window_duration_seconds": manifest.get("window_duration_seconds"),
        "analysis_unit": "fixed_time_window_state",
        "lag": lag,
        "class_count": len(result["classes"]),
        "sequence_count": int(sum(item["sequence_meta"]["sequence_count"] for item in result["classes"])),
        "total_windows": int(sum(item["tables"]["total_windows"] for item in result["classes"])),
        "total_transitions": int(result["pooled_counts"].to_numpy().sum()),
        "pooled_weight": "transition-opportunity weighted (sum O_cij / sum R_ci)",
        "equal_weight": "mean of P_c(j|i) over classes with R_ci > 0",
        "classes": [
            {
                "class_id": item["entry"]["class_id"],
                "display_name": item["entry"].get("display_name", ""),
                "file": str(item["path"]),
                "sha256": item["sha256"],
                "human_file": item["entry"].get("human_file", ""),
                "human_sha256": sha256_file(Path(item["entry"]["human_file"])) if item["entry"].get("human_file") and Path(item["entry"]["human_file"]).exists() else "",
                "review_status": item["entry"].get("review_status", ""),
                "code_source": item["entry"].get("code_source", "ai"),
                "source_run_dir": item["entry"].get("source_run_dir", ""),
                "source_final_dir": item["entry"].get("source_final_dir", ""),
                "source_audio": item["entry"].get("source_audio", ""),
                "audio_sha256": item["entry"].get("audio_sha256", ""),
            }
            for item in result["classes"]
        ],
        "files": {
            "classroom_summary": "classroom_summary.csv",
            "pooled_frequency": "pooled_frequency.csv",
            "pooled_transition_counts": "pooled_transition_counts.csv",
            "pooled_transition_probabilities": "pooled_transition_probabilities.csv",
            "equal_weight_transition_probabilities": "equal_weight_transition_probabilities.csv",
            "equal_weight_class_counts": "equal_weight_class_counts.csv",
            "classroom_transition_comparison": "classroom_transition_comparison.csv",
            "transition_stability": "transition_stability.csv",
            "leave_one_class_out": "leave_one_class_out.csv",
            "graph_svg": graph_paths.get("svg"),
            "graph_png": graph_paths.get("png"),
        },
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_html(result, metadata, graph_paths)
    (output_dir / "multi_classroom_report.html").write_text(report, encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output-root", default=str(DEFAULT_STUDY_ROOT))
    parser.add_argument("--lag", type=int, default=1)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    study_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    output_dir = run(study_dir, manifest, Path(args.output_root), args.lag)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
