from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ms_to_hms(value: int) -> str:
    value = max(0, int(round(value)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, _ = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to *_segments.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--teacher-id", default=None)
    parser.add_argument("--bin-min", type=int, default=5)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = csv_path.stem.removesuffix("_segments")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No segments found.")
        return 1

    teacher_id = args.teacher_id
    if not teacher_id:
        teacher_id = df.groupby("speaker_id")["duration_s"].sum().idxmax()
    df["role"] = np.where(df.speaker_id == teacher_id, "teacher", "student")

    total_s = float(df["duration_s"].sum())
    grouped = df.groupby("speaker_id")
    summary = grouped.agg(
        role=("role", "first"),
        segments=("segment_id", "count"),
        speech_s=("duration_s", "sum"),
        first_start_ms=("start_ms", "min"),
        last_end_ms=("end_ms", "max"),
        median_segment_s=("duration_s", "median"),
    ).reset_index()
    summary["speech_pct"] = (summary["speech_s"] / total_s * 100).round(1)
    summary["first_hms"] = summary["first_start_ms"].apply(ms_to_hms)
    summary["last_hms"] = summary["last_end_ms"].apply(ms_to_hms)
    summary["median_segment_s"] = summary["median_segment_s"].round(2)
    summary = summary[
        [
            "speaker_id",
            "role",
            "segments",
            "speech_s",
            "speech_pct",
            "first_hms",
            "last_hms",
            "median_segment_s",
        ]
    ]
    summary_path = output_dir / f"{prefix}_speaker_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    role_columns = [
        "recording_id",
        "segment_id",
        "start_ms",
        "end_ms",
        "start_hms",
        "end_hms",
        "duration_s",
        "speaker_id",
        "role",
        "text",
        "raw_text",
    ]
    role_path = output_dir / f"{prefix}_segments_with_role.csv"
    df[role_columns].to_csv(role_path, index=False, encoding="utf-8-sig")

    bin_ms = args.bin_min * 60_000
    df["bin_ms"] = (df["start_ms"] // bin_ms) * bin_ms
    df["bin_start_hms"] = df["bin_ms"].apply(ms_to_hms)
    pivot = df.pivot_table(
        index=["bin_ms", "bin_start_hms"],
        columns="speaker_id",
        values="duration_s",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    roles = df.pivot_table(
        index="bin_ms", columns="role", values="duration_s", aggfunc="sum", fill_value=0
    )
    counts = df.pivot_table(
        index="bin_ms", columns="role", values="segment_id", aggfunc="count", fill_value=0
    )
    timeline = pivot.merge(roles, on="bin_ms", how="left").merge(
        counts, on="bin_ms", how="left", suffixes=("", "_turns")
    )
    timeline = timeline.rename(
        columns={
            "teacher": "teacher_s",
            "student": "student_s",
            "teacher_turns": "teacher_segments",
            "student_turns": "student_segments",
        }
    )
    speaker_columns = [c for c in timeline.columns if c.startswith("S") and c[1:].isdigit()]
    timeline["total_s"] = timeline[speaker_columns].sum(axis=1).round(1)
    timeline = timeline[
        ["bin_start_hms"]
        + speaker_columns
        + ["teacher_s", "student_s", "teacher_segments", "student_segments", "total_s"]
    ].round(1)
    timeline_path = output_dir / f"{prefix}_timeline_{args.bin_min}min.csv"
    timeline.to_csv(timeline_path, index=False, encoding="utf-8-sig")

    print(f"Teacher speaker: {teacher_id}")
    print(summary.to_string(index=False))
    print(f"\nSegments with role: {role_path}")
    print(f"Speaker summary: {summary_path}")
    print(f"Timeline: {timeline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
