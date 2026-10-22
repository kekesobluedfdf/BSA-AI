from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def speaker_label(speaker_id: str) -> str:
    if str(speaker_id).upper() == "CHORUS":
        return "齐读"
    digits = re.sub(r"\D", "", str(speaker_id))
    return f"说话人{int(digits)}" if digits else f"说话人{speaker_id}"


def ms_to_clock(value: int) -> str:
    total_seconds = max(0, int(round(value / 1000)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def ms_to_hms(value: int) -> str:
    value = max(0, int(round(value)))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def infer_teacher(rows: list[dict[str, Any]], requested: str | None) -> str:
    speaker_ids = {str(row["speaker_id"]) for row in rows}
    if requested and requested != "auto" and requested in speaker_ids:
        return requested
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        totals[str(row["speaker_id"])] += float(row.get("duration_s", 0))
    return max(totals, key=totals.get) if totals else (requested or "S01")


def add_roles(rows: list[dict[str, Any]], teacher_id: str) -> None:
    for row in rows:
        speaker_id = str(row["speaker_id"])
        if speaker_id.upper() == "CHORUS":
            row["role"] = "chorus"
        elif speaker_id == teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"


def _join_parts(parts: list[str], starts: list[int], ends: list[int]) -> str:
    text = ""
    previous_end = None
    for part, start, end in zip(parts, starts, ends):
        part = str(part).strip()
        if not part:
            continue
        gap = start - previous_end if previous_end is not None else 0
        if (
            text
            and gap >= 250
            and not re.search(r"[，。！？；：,.!?;:]$", text)
            and not re.match(r"^[，。！？；：,.!?;:]", part)
        ):
            text += "，"
        text += part
        previous_end = end
    return text


def _build_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["start_ms"]), int(row["segment_id"])))
    turns: list[dict[str, Any]] = []
    for row in ordered:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        speaker_id = str(row["speaker_id"])
        if not turns or turns[-1]["speaker_id"] != speaker_id:
            turns.append(
                {
                    "speaker_id": speaker_id,
                    "start_ms": int(row["start_ms"]),
                    "end_ms": int(row["end_ms"]),
                    "parts": [text],
                    "part_starts": [int(row["start_ms"])],
                    "part_ends": [int(row["end_ms"])],
                    "segment_ids": [int(row["segment_id"])],
                }
            )
        else:
            turns[-1]["parts"].append(text)
            turns[-1]["part_starts"].append(int(row["start_ms"]))
            turns[-1]["part_ends"].append(int(row["end_ms"]))
            turns[-1]["segment_ids"].append(int(row["segment_id"]))
            turns[-1]["end_ms"] = max(turns[-1]["end_ms"], int(row["end_ms"]))

    for index, turn in enumerate(turns):
        turn["turn_id"] = index + 1
        turn["text"] = _join_parts(turn["parts"], turn["part_starts"], turn["part_ends"])
    return turns


def write_verbatim(rows: list[dict[str, Any]], path: Path) -> None:
    blocks = []
    for turn in _build_turns(rows):
        blocks.append(
            f"{speaker_label(turn['speaker_id'])} {ms_to_clock(turn['start_ms'])}\n{turn['text']}"
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8-sig")


def write_role_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
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
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def _timestamp_pair(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        start = int(float(value.get("start_time", 0)) * 1000)
        end = int(float(value.get("end_time", 0)) * 1000)
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start, end = int(float(value[0])), int(float(value[1]))
    else:
        return None
    return (start, end) if end > start else None


def _timestamp_pairs(row: dict[str, Any]) -> list[tuple[int, int]]:
    values = row.get("segment_timestamp") or []
    if not isinstance(values, list):
        return []
    pairs = [_timestamp_pair(value) for value in values]
    return [pair for pair in pairs if pair is not None]


def _char_times(row: dict[str, Any], text: str) -> list[float]:
    length = len(text)
    if length == 0:
        return []
    pairs = _timestamp_pairs(row)
    if pairs:
        mids = np.array([(start + end) / 2 for start, end in pairs], dtype=float)
        if len(mids) == 1:
            return [float(mids[0])] * length
        indices = np.rint(np.linspace(0, len(mids) - 1, length)).astype(int)
        return [float(mids[index]) for index in indices]
    start = float(row["start_ms"])
    end = max(start + 1, float(row["end_ms"]))
    return list(np.linspace(start, end, length, endpoint=False))


def build_fixed_windows(
    rows: list[dict[str, Any]], window_ms: int, role_map: dict[str, str]
) -> list[dict[str, Any]]:
    turns = _build_turns(rows)
    if not turns:
        return []

    max_end = max(int(turn["end_ms"]) for turn in turns)
    count = max(1, math.ceil(max_end / window_ms))

    entries_by_window: dict[int, list[tuple[int, str, str, int]]] = defaultdict(list)
    speaker_seconds: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    segments_by_window: dict[int, set[int]] = defaultdict(set)
    speaker_order: dict[str, int] = {}

    for turn in turns:
        speaker_id = str(turn["speaker_id"])
        if speaker_id not in speaker_order:
            speaker_order[speaker_id] = len(speaker_order)

        start = max(0, int(turn["start_ms"]))
        end = max(start + 1, int(turn["end_ms"]))
        text = str(turn["text"])
        first = min(count - 1, max(0, start // window_ms))
        last = min(count - 1, max(0, (end - 1) // window_ms))
        overlaps = []
        for window in range(first, last + 1):
            window_start = window * window_ms
            window_end = window_start + window_ms
            overlap = max(0, min(end, window_end) - max(start, window_start)) / 1000
            if overlap > 0:
                overlaps.append((window, overlap))

        if overlaps:
            if text and len(text) >= len(overlaps):
                total_overlap = sum(overlap for _, overlap in overlaps)
                cumulative = 0.0
                for window, overlap in overlaps:
                    start_char = int(round(len(text) * cumulative / total_overlap))
                    cumulative += overlap
                    end_char = int(round(len(text) * cumulative / total_overlap))
                    chunk = text[start_char:end_char]
                    if chunk:
                        entries_by_window[window].append((start, speaker_id, chunk, int(turn["turn_id"])))
            elif text:
                window, _ = max(overlaps, key=lambda item: item[1])
                entries_by_window[window].append((start, speaker_id, text, int(turn["turn_id"])))

        for window, overlap in overlaps:
            speaker_seconds[window][speaker_id] += overlap
            segments_by_window[window].update(turn["segment_ids"])

    windows = []
    for window in range(count):
        start_ms = window * window_ms
        end_ms = (window + 1) * window_ms

        turns_in_window: list[list[str]] = []
        for _, sid, content, _ in sorted(
            entries_by_window.get(window, []), key=lambda item: (item[0], item[3])
        ):
            if not turns_in_window or turns_in_window[-1][0] != sid:
                turns_in_window.append([sid, content])
            else:
                turns_in_window[-1][1] += content

        text_speakers = list(dict.fromkeys(turn[0] for turn in turns_in_window))
        if len(text_speakers) > 1:
            text = " ".join(
                f"[{speaker_label(sid)}] {content.strip()}"
                for sid, content in turns_in_window
                if content.strip()
            )
        else:
            text = turns_in_window[0][1].strip() if turns_in_window else ""

        per_speaker = dict(speaker_seconds.get(window, {}))
        speakers = sorted(per_speaker, key=lambda sid: speaker_order.get(sid, 999))
        dominant = (
            max(per_speaker, key=lambda sid: (per_speaker[sid], -speaker_order.get(sid, 999)))
            if per_speaker
            else ""
        )
        segment_ids = sorted(segments_by_window.get(window, set()))
        windows.append(
            {
                "window_id": window + 1,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_hms": ms_to_hms(start_ms),
                "end_hms": ms_to_hms(end_ms),
                "duration_s": round(window_ms / 1000, 3),
                "dominant_speaker": dominant,
                "dominant_role": role_map.get(dominant, ""),
                "speakers": "|".join(speakers),
                "speaker_count": len(speakers),
                "segment_count": len(segment_ids),
                "segment_ids": "|".join(str(segment_id) for segment_id in segment_ids),
                "speech_s": round(sum(per_speaker.values()), 3),
                "text": text,
            }
        )
    return windows


def _block_record(
    block: list[dict[str, Any]], window_id: int, role_map: dict[str, str], speaker_order: dict[str, int]
) -> dict[str, Any]:
    start_ms = min(int(row["start_ms"]) for row in block)
    end_ms = max(int(row["end_ms"]) for row in block)
    per_speaker: dict[str, float] = defaultdict(float)
    for row in block:
        per_speaker[str(row["speaker_id"])] += float(row.get("duration_s", 0))

    turns: list[list[str]] = []
    for row in block:
        speaker_id = str(row["speaker_id"])
        content = str(row.get("text") or "").strip()
        if not content:
            continue
        if not turns or turns[-1][0] != speaker_id:
            turns.append([speaker_id, content])
        else:
            turns[-1][1] += content

    if len(per_speaker) > 1:
        text = " ".join(
            f"[{speaker_label(sid)}] {content.strip()}" for sid, content in turns if content.strip()
        )
    else:
        text = turns[0][1].strip() if turns else ""

    speakers = sorted(per_speaker, key=lambda sid: speaker_order.get(sid, 999))
    dominant = (
        max(per_speaker, key=lambda sid: (per_speaker[sid], -speaker_order.get(sid, 999)))
        if per_speaker
        else ""
    )
    return {
        "window_id": window_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_hms": ms_to_hms(start_ms),
        "end_hms": ms_to_hms(end_ms),
        "duration_s": round((end_ms - start_ms) / 1000, 3),
        "dominant_speaker": dominant,
        "dominant_role": role_map.get(dominant, ""),
        "speakers": "|".join(speakers),
        "speaker_count": len(speakers),
        "segment_count": len(block),
        "segment_ids": "|".join(str(int(row["segment_id"])) for row in block),
        "speech_s": round(sum(float(row.get("duration_s", 0)) for row in block), 3),
        "text": text,
    }


def build_merged_windows(
    rows: list[dict[str, Any]], target_ms: int, role_map: dict[str, str]
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (int(row["start_ms"]), int(row["segment_id"])))
    speaker_order: dict[str, int] = {}
    for row in ordered:
        speaker_id = str(row["speaker_id"])
        if speaker_id not in speaker_order:
            speaker_order[speaker_id] = len(speaker_order)

    max_gap_ms = min(1500, max(500, target_ms // 2))
    blocks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    block_start = 0
    block_end = 0

    for row in ordered:
        start = max(0, int(row["start_ms"]))
        end = max(start + 1, int(row["end_ms"]))
        if current and start - block_end > max_gap_ms:
            blocks.append(current)
            current = []
        if not current:
            current = [row]
            block_start = start
            block_end = end
        else:
            current.append(row)
            block_end = max(block_end, end)
        if block_end - block_start >= target_ms:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    return [
        _block_record(block, index + 1, role_map, speaker_order)
        for index, block in enumerate(blocks)
    ]


def write_window_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    window_seconds: float,
    role_map: dict[str, str],
    window_mode: str = "fixed",
) -> tuple[Path, Path] | None:
    if window_seconds <= 0:
        return None
    window_ms = max(1, int(window_seconds * 1000))
    if window_mode == "fixed":
        windows = build_fixed_windows(rows, window_ms, role_map)
    else:
        windows = build_merged_windows(rows, window_ms, role_map)

    csv_path = output_dir / f"{prefix}_window_{window_seconds:g}s_{window_mode}.csv"
    pd.DataFrame(windows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    blocks = []
    for window in windows:
        if not window["text"]:
            continue
        speaker_text = (
            "/".join(speaker_label(sid) for sid in window["speakers"].split("|") if sid)
            if window["speakers"]
            else "静音"
        )
        blocks.append(
            f"{ms_to_clock(window['start_ms'])}-{ms_to_clock(window['end_ms'])} {speaker_text}\n{window['text']}"
        )
    txt_path = output_dir / f"{prefix}_window_{window_seconds:g}s_{window_mode}.txt"
    txt_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8-sig")
    return csv_path, txt_path


def build_stats(
    rows: list[dict[str, Any]], stats_ms: int, teacher_id: str, role_map: dict[str, str]
) -> list[dict[str, Any]]:
    max_end = max(int(row["end_ms"]) for row in rows)
    count = max(1, math.ceil(max_end / stats_ms))
    speaker_order: dict[str, int] = {}
    for row in rows:
        speaker_id = str(row["speaker_id"])
        if speaker_id not in speaker_order:
            speaker_order[speaker_id] = len(speaker_order)

    stats = []
    for window in range(count):
        window_start = window * stats_ms
        window_end = min((window + 1) * stats_ms, max_end)
        per_speaker: dict[str, float] = defaultdict(float)
        teacher_segments: set[int] = set()
        student_segments: set[int] = set()
        chorus_segments: set[int] = set()

        for row in rows:
            start = int(row["start_ms"])
            end = int(row["end_ms"])
            overlap = max(0, min(end, window_end) - max(start, window_start)) / 1000
            if overlap <= 0:
                continue
            speaker_id = str(row["speaker_id"])
            per_speaker[speaker_id] += overlap
            role = role_map.get(speaker_id)
            if role == "teacher":
                teacher_segments.add(int(row["segment_id"]))
            elif role == "chorus":
                chorus_segments.add(int(row["segment_id"]))
            else:
                student_segments.add(int(row["segment_id"]))

        teacher_s = per_speaker.get(teacher_id, 0.0)
        chorus_s = per_speaker.get("CHORUS", 0.0)
        total_s = sum(per_speaker.values())
        student_s = total_s - teacher_s - chorus_s
        non_teacher_s = student_s + chorus_s
        dominant = (
            max(per_speaker, key=lambda sid: (per_speaker[sid], -speaker_order.get(sid, 999)))
            if per_speaker
            else ""
        )
        record: dict[str, Any] = {
            "window_id": window + 1,
            "start_ms": window_start,
            "end_ms": window_end,
            "start_hms": ms_to_hms(window_start),
            "end_hms": ms_to_hms(window_end),
        }
        for speaker_id in speaker_order:
            record[speaker_id] = round(per_speaker.get(speaker_id, 0.0), 3)
        record.update(
            {
                "dominant_speaker": dominant,
                "teacher_s": round(teacher_s, 3),
                "student_s": round(student_s, 3),
                "chorus_s": round(chorus_s, 3),
                "non_teacher_s": round(non_teacher_s, 3),
                "total_s": round(total_s, 3),
                "student_ratio": round(non_teacher_s / total_s, 4) if total_s else 0.0,
                "chorus_ratio": round(chorus_s / total_s, 4) if total_s else 0.0,
                "teacher_segments": len(teacher_segments),
                "student_segments": len(student_segments),
                "chorus_segments": len(chorus_segments),
            }
        )
        stats.append(record)
    return stats


def write_speaker_summary(rows: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(rows)
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
    summary["speech_pct"] = (summary["speech_s"] / total_s * 100).round(1) if total_s else 0.0
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
    summary.to_csv(path, index=False, encoding="utf-8-sig")


def export_all(
    jsonl_path: Path,
    output_dir: Path | None = None,
    window_seconds: float = 3,
    stats_window_min: float = 5,
    teacher_id: str | None = "S01",
    window_mode: str = "fixed",
) -> dict[str, str]:
    rows = load_jsonl(jsonl_path)
    if not rows:
        return {}

    output_dir = output_dir or jsonl_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    teacher_id = infer_teacher(rows, teacher_id)
    add_roles(rows, teacher_id)
    role_map = {str(row["speaker_id"]): str(row["role"]) for row in rows}

    paths: dict[str, str] = {}
    verbatim_path = output_dir / f"{prefix}_verbatim.txt"
    write_verbatim(rows, verbatim_path)
    paths["verbatim"] = str(verbatim_path)

    role_path = output_dir / f"{prefix}_segments_with_role.csv"
    write_role_csv(rows, role_path)
    paths["segments_with_role"] = str(role_path)

    window_paths = write_window_outputs(
        rows, output_dir, prefix, window_seconds, role_map, window_mode
    )
    if window_paths:
        paths["window_csv"], paths["window_txt"] = map(str, window_paths)

    stats_path = output_dir / f"{prefix}_stats_{stats_window_min:g}min.csv"
    pd.DataFrame(build_stats(rows, max(1, int(stats_window_min * 60_000)), teacher_id, role_map)).to_csv(
        stats_path, index=False, encoding="utf-8-sig"
    )
    paths["stats"] = str(stats_path)

    summary_path = output_dir / f"{prefix}_speaker_summary.csv"
    write_speaker_summary(rows, summary_path)
    paths["speaker_summary"] = str(summary_path)
    paths["teacher_id"] = teacher_id
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to *_segments.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window-seconds", type=float, default=3)
    parser.add_argument("--window-mode", choices=["merge", "fixed"], default="fixed")
    parser.add_argument("--stats-window-min", type=float, default=5)
    parser.add_argument("--teacher-id", default="S01")
    args = parser.parse_args()

    paths = export_all(
        Path(args.jsonl),
        Path(args.output_dir) if args.output_dir else None,
        args.window_seconds,
        args.stats_window_min,
        args.teacher_id,
        args.window_mode,
    )
    if not paths:
        print("No segments found.")
        return 1
    print(f"Teacher speaker: {paths['teacher_id']}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
