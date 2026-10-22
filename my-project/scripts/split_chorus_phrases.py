from __future__ import annotations

import argparse
import bisect
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from funasr import AutoModel

from transcript_exports import export_all, ms_to_hms

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_MODEL_ROOT = ROOT_DIR / "models"

os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT_DIR / "models" / ".cache"))
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

CUE_PATTERN = re.compile(r"齐读课题|齐读|一起.*读|预备齐|预备起|全班.*读|一起朗读")


def context_cue(rows: list[dict], index: int, window_ms: int = 8000) -> bool:
    start = int(rows[index]["start_ms"])
    for position in range(index - 1, max(-1, index - 3), -1):
        row = rows[position]
        if int(row["end_ms"]) < start - window_ms:
            break
        if CUE_PATTERN.search(str(row.get("text") or "")):
            return True
    return False


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    return audio, sample_rate


def embed_batch(model: AutoModel, waveforms: list[np.ndarray], sample_rate: int) -> np.ndarray:
    result = model.generate(input=waveforms, fs=sample_rate, disable_pbar=True)
    matrix = np.stack([item["spk_embedding"].detach().cpu().numpy().reshape(-1) for item in result])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def make_waveform(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_sample = max(0, int((start - 0.05) * sample_rate))
    end_sample = min(len(audio), int((end + 0.05) * sample_rate))
    segment = audio[start_sample:end_sample]
    minimum = int(0.3 * sample_rate)
    if len(segment) < minimum:
        segment = np.pad(segment, (0, minimum - len(segment)))
    return segment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("audio")
    parser.add_argument("--nano-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--gap-seconds", type=float, default=0.8)
    parser.add_argument("--teacher-sim", type=float, default=0.35)
    parser.add_argument("--chorus-teacher-sim", type=float, default=0.28)
    parser.add_argument("--chorus-max-student-sim", type=float, default=0.5)
    parser.add_argument("--min-chorus-duration", type=float, default=1.0)
    parser.add_argument("--context-teacher-sim", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--subwindow-seconds", type=float, default=0.4)
    parser.add_argument("--subwindow-hop", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = load_rows(jsonl_path)
    audio, sample_rate = load_audio(Path(args.audio))
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModel(
        model=str(Path(args.model_root) / "campplus"),
        device=device,
        disable_update=True,
    )

    if args.embeddings and Path(args.embeddings).exists():
        cached = np.load(args.embeddings, allow_pickle=True)
        full_embeddings = cached["embeddings"] if len(cached["embeddings"]) == len(rows) else None
    else:
        full_embeddings = None
    if full_embeddings is None:
        minimum = int(0.3 * sample_rate)
        waveforms = []
        for row in rows:
            start = max(0, int(float(row["start_ms"]) * sample_rate / 1000))
            end = min(len(audio), int(float(row["end_ms"]) * sample_rate / 1000))
            if end <= start:
                end = min(len(audio), start + minimum)
            segment = audio[start:end]
            if len(segment) < minimum:
                segment = np.pad(segment, (0, minimum - len(segment)))
            waveforms.append(segment)
        batches = []
        for begin in range(0, len(waveforms), args.batch_size):
            batches.append(embed_batch(model, waveforms[begin : begin + args.batch_size], sample_rate))
        full_embeddings = np.concatenate(batches, axis=0)

    centroids = {}
    for speaker_id in sorted({str(row["speaker_id"]) for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["speaker_id"]) == speaker_id and float(row["duration_s"]) >= 1.5
        ]
        if not indices:
            indices = [index for index, row in enumerate(rows) if str(row["speaker_id"]) == speaker_id]
        centroid = full_embeddings[indices].mean(axis=0)
        centroids[speaker_id] = centroid / np.linalg.norm(centroid)
    speaker_ids = list(centroids)
    centroid_matrix = np.stack([centroids[sid] for sid in speaker_ids])
    teacher_position = speaker_ids.index(args.teacher_id)
    student_ids = [sid for sid in speaker_ids if sid != args.teacher_id]

    nano_data = json.loads(Path(args.nano_result).read_text(encoding="utf-8"))
    tokens = nano_data["tokens"] if isinstance(nano_data, dict) and "tokens" in nano_data else nano_data[0]["timestamps"]
    tokens = sorted(tokens, key=lambda token: float(token.get("start_time", 0)))

    tokens_by_row: dict[int, list[dict]] = {index: [] for index in range(len(rows))}
    starts = [float(row["start_ms"]) / 1000 for row in rows]
    for token in tokens:
        token_start = float(token.get("start_time", 0))
        token_end = float(token.get("end_time", token_start))
        middle = (token_start + token_end) / 2
        index = bisect.bisect_right(starts, middle) - 1
        candidates = [position for position in (index, index + 1) if 0 <= position < len(rows)]
        target = None
        for position in candidates:
            if float(rows[position]["start_ms"]) / 1000 <= middle <= float(rows[position]["end_ms"]) / 1000:
                target = position
                break
        if target is None and candidates:
            distances = []
            for position in candidates:
                row_start = float(rows[position]["start_ms"]) / 1000
                row_end = float(rows[position]["end_ms"]) / 1000
                distance = min(abs(middle - row_start), abs(middle - row_end))
                distances.append((distance, position))
            distance, position = min(distances)
            if distance <= 0.6:
                target = position
        if target is not None:
            tokens_by_row[target].append(token)

    group_specs = []
    for row_index, row in enumerate(rows):
        if str(row["speaker_id"]) != args.teacher_id:
            continue
        row_tokens = tokens_by_row.get(row_index, [])
        if len(row_tokens) < 3:
            continue
        cue = context_cue(rows, row_index)
        groups = []
        current = [row_tokens[0]]
        for token in row_tokens[1:]:
            previous_end = float(current[-1].get("end_time", 0))
            start = float(token.get("start_time", 0))
            if start - previous_end > args.gap_seconds:
                groups.append(current)
                current = [token]
            else:
                current.append(token)
        groups.append(current)
        for group in groups:
            start = float(group[0]["start_time"])
            end = float(group[-1]["end_time"])
            group_specs.append((row_index, group, start, end, cue))

    group_waveforms = [make_waveform(audio, sample_rate, start, end) for _, _, start, end, _ in group_specs]
    group_embeddings = []
    for begin in range(0, len(group_waveforms), args.batch_size):
        group_embeddings.append(
            embed_batch(model, group_waveforms[begin : begin + args.batch_size], sample_rate)
        )
        print(
            f"embedded groups {min(begin + args.batch_size, len(group_waveforms))}/{len(group_waveforms)}",
            flush=True,
        )
    group_embeddings = np.concatenate(group_embeddings, axis=0)
    group_sims = group_embeddings @ centroid_matrix.T

    group_results = {}
    for index, (row_index, group, start, end, cue) in enumerate(group_specs):
        sims = group_sims[index]
        teacher_sim = float(sims[teacher_position])
        student_positions = [speaker_ids.index(sid) for sid in student_ids]
        best_student_position = max(student_positions, key=lambda position: sims[position])
        best_student_sim = float(sims[best_student_position])
        best_student = speaker_ids[best_student_position]
        is_teacher = teacher_sim >= args.teacher_sim and teacher_sim >= best_student_sim - args.margin

        sub_labels = []
        if not is_teacher and end - start >= 0.7:
            sub_waveforms = []
            position = start
            while position + args.subwindow_seconds <= end:
                sub_waveforms.append(
                    make_waveform(audio, sample_rate, position, position + args.subwindow_seconds)
                )
                position += args.subwindow_hop
            if len(sub_waveforms) >= 2:
                sub_embeddings = embed_batch(model, sub_waveforms, sample_rate)
                sub_sims = sub_embeddings @ centroid_matrix.T
                for sub_sim in sub_sims:
                    sub_teacher = float(sub_sim[teacher_position])
                    sub_best_position = max(student_positions, key=lambda pos: sub_sim[pos])
                    sub_best = speaker_ids[sub_best_position]
                    if sub_teacher < args.teacher_sim or float(sub_sim[sub_best_position]) > sub_teacher + args.margin:
                        sub_labels.append(sub_best)

        counts = Counter(sub_labels)
        if not is_teacher and counts:
            dominant, count = counts.most_common(1)[0]
            share = count / len(sub_labels)
            teacher_limit = args.context_teacher_sim if cue else args.chorus_teacher_sim
            is_chorus = (
                (len(counts) >= 2 or share < 0.75)
                and teacher_sim <= teacher_limit
                and best_student_sim <= args.chorus_max_student_sim
                and end - start >= args.min_chorus_duration
            )
        else:
            dominant = ""
            share = 1.0
            is_chorus = False

        if is_chorus:
            group_results.setdefault(row_index, []).append(
                {
                    "start_ms": int(start * 1000),
                    "end_ms": int(end * 1000),
                    "text": "".join(str(token.get("token", "")) for token in group).strip(),
                    "dominant": dominant,
                    "share": round(share, 3),
                    "teacher_sim": round(teacher_sim, 4),
                    "best_student": best_student,
                    "best_student_sim": round(best_student_sim, 4),
                    "duration_s": round(end - start, 3),
                    "context_cue": cue,
                    "labels": counts,
                }
            )

    output_rows = []
    review = []
    for row_index, row in enumerate(rows):
        chorus_groups = group_results.get(row_index, [])
        if not chorus_groups:
            output_rows.append(row)
            continue
        row_tokens = tokens_by_row.get(row_index, [])
        token_end_ms = max(
            (int(float(token["end_time"]) * 1000) for token in row_tokens),
            default=int(row["end_ms"]),
        )
        row_end_ms = max(int(row["end_ms"]), token_end_ms)
        boundaries = [int(row["start_ms"])]
        for group in chorus_groups:
            boundaries.extend([group["start_ms"], group["end_ms"]])
        boundaries.append(row_end_ms)
        boundaries = sorted(set(boundaries))

        for index in range(len(boundaries) - 1):
            start = boundaries[index]
            end = boundaries[index + 1]
            if end <= start:
                continue
            chorus = next(
                (group for group in chorus_groups if group["start_ms"] <= start < group["end_ms"]),
                None,
            )
            if chorus:
                text = chorus["text"]
                speaker = "CHORUS"
            else:
                selected = [
                    token
                    for token in row_tokens
                    if start <= (float(token["start_time"]) + float(token["end_time"])) * 500 < end
                ]
                text = "".join(str(token.get("token", "")) for token in selected).strip()
                speaker = str(row["speaker_id"])
            if not text:
                continue
            new_row = dict(row)
            new_row["start_ms"] = start
            new_row["end_ms"] = end
            new_row["start_hms"] = ms_to_hms(start)
            new_row["end_hms"] = ms_to_hms(end)
            new_row["duration_s"] = round((end - start) / 1000, 3)
            new_row["speaker_id"] = speaker
            new_row["speaker_original"] = row.get("speaker_original", row["speaker_id"])
            new_row["speaker_refined"] = speaker != row["speaker_id"]
            new_row["text"] = text
            new_row["split_from_segment_id"] = int(row["segment_id"])
            new_row["split_index"] = index + 1
            if speaker == "CHORUS":
                new_row["role"] = "chorus"
                new_row["chorus_refined"] = True
                new_row["chorus_source"] = "phrase_voiceprint"
            output_rows.append(new_row)
            if speaker == "CHORUS":
                review.append(
                    {
                        "source_segment_id": int(row["segment_id"]),
                        "start_hms": new_row["start_hms"],
                        "end_hms": new_row["end_hms"],
                        "text": text,
                        "dominant_student": chorus["dominant"],
                        "dominant_share": chorus["share"],
                        "teacher_sim": chorus["teacher_sim"],
                        "best_student": chorus["best_student"],
                        "best_student_sim": chorus["best_student_sim"],
                        "duration_s": chorus["duration_s"],
                        "context_cue": chorus["context_cue"],
                        "student_labels": json.dumps(chorus["labels"], ensure_ascii=False),
                    }
                )

    output_rows.sort(key=lambda item: (int(item["start_ms"]), int(item.get("split_from_segment_id", 0))))
    for index, row in enumerate(output_rows, start=1):
        row["segment_id"] = index
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"

    output_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

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
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in output_rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    review_path = output_dir / f"{prefix}_phrase_chorus_review.csv"
    review_columns = [
        "source_segment_id",
        "start_hms",
        "end_hms",
        "text",
        "dominant_student",
        "dominant_share",
        "teacher_sim",
        "best_student",
        "best_student_sim",
        "duration_s",
        "context_cue",
        "student_labels",
    ]
    pd.DataFrame(review, columns=review_columns).to_csv(
        review_path, index=False, encoding="utf-8-sig"
    )

    metadata_path = output_dir / f"{prefix}_phrase_split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "audio": str(args.audio),
                "nano_result": args.nano_result,
                "teacher_id": args.teacher_id,
                "gap_seconds": args.gap_seconds,
                "chorus_regions": len(review),
                "output_segments": len(output_rows),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    paths = export_all(
        output_jsonl,
        output_dir,
        window_seconds=args.output_window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Chorus regions: {len(review)}")
    print(f"Output segments: {len(output_rows)}")
    print(f"Review: {review_path}")
    print(f"Metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
