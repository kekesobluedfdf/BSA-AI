from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
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
    embeddings = [item["spk_embedding"].detach().cpu().numpy().reshape(-1) for item in result]
    matrix = np.stack(embeddings)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def build_centroids(
    rows: list[dict], embeddings: np.ndarray, min_duration: float
) -> dict[str, np.ndarray]:
    centroids = {}
    for speaker_id in sorted({str(row["speaker_id"]) for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row["speaker_id"]) == speaker_id and float(row["duration_s"]) >= min_duration
        ]
        if not indices:
            indices = [index for index, row in enumerate(rows) if str(row["speaker_id"]) == speaker_id]
        centroid = embeddings[indices].mean(axis=0)
        centroids[speaker_id] = centroid / np.linalg.norm(centroid)
    return centroids


def moving_average(values: list[float], radius: int) -> list[float]:
    result = []
    for index in range(len(values)):
        window = values[max(0, index - radius) : index + radius + 1]
        result.append(float(np.mean(window)))
    return result


def majority_filter(values: list[bool], radius: int) -> list[bool]:
    result = []
    for index in range(len(values)):
        window = values[max(0, index - radius) : index + radius + 1]
        result.append(sum(window) * 2 > len(window))
    return result


def run_lengths(flags: list[bool]) -> list[tuple[int, int]]:
    runs = []
    start = 0
    for index in range(1, len(flags)):
        if flags[index] != flags[start]:
            runs.append((start, index - 1))
            start = index
    runs.append((start, len(flags) - 1))
    return runs


def adjust_boundary(text: str, index: int, minimum: int, maximum: int) -> int:
    if len(text) <= 1 or maximum < minimum:
        return min(max(index, 0), len(text))
    index = min(max(index, minimum), maximum)
    search = max(1, int(len(text) * 0.1))
    candidates = [
        position + 1
        for position in range(max(minimum, index - search), min(maximum, index + search))
        if text[position] in "，。！？；：,.!?;:"
    ]
    if candidates:
        return min(candidates, key=lambda position: abs(position - index))
    return index


def text_chunk(text: str, parent_start: int, parent_end: int, start: int, end: int) -> str:
    if not text or parent_end <= parent_start:
        return ""
    start_char = adjust_boundary(
        text,
        int(round(len(text) * (start - parent_start) / (parent_end - parent_start))),
        0,
        len(text),
    )
    end_char = adjust_boundary(
        text,
        int(round(len(text) * (end - parent_start) / (parent_end - parent_start))),
        start_char,
        len(text),
    )
    return text[start_char:end_char].strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to refined *_segments.jsonl")
    parser.add_argument("audio", help="Source audio file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--min-segment-seconds", type=float, default=3.0)
    parser.add_argument("--min-region-seconds", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--min-student-sim", type=float, default=0.35)
    parser.add_argument("--consistency", type=float, default=0.6)
    parser.add_argument("--min-slice-seconds", type=float, default=0.5)
    parser.add_argument("--min-centroid-duration", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accept-ids", default=None)
    parser.add_argument("--output-window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()
    accepted_ids = (
        {int(value) for value in args.accept_ids.split(",") if value.strip()}
        if args.accept_ids
        else None
    )

    jsonl_path = Path(args.jsonl)
    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = load_rows(jsonl_path)
    audio, sample_rate = load_audio(audio_path)
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
        waveforms = []
        minimum = int(0.3 * sample_rate)
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

    centroids = build_centroids(rows, full_embeddings, args.min_centroid_duration)
    speaker_ids = list(centroids)
    centroid_matrix = np.stack([centroids[speaker_id] for speaker_id in speaker_ids])
    teacher_position = speaker_ids.index(args.teacher_id)
    student_ids = [speaker_id for speaker_id in speaker_ids if speaker_id != args.teacher_id]

    window_samples = int(args.window_seconds * sample_rate)
    hop_samples = int(args.hop_seconds * sample_rate)

    windows_by_row: dict[int, list[dict]] = defaultdict(list)
    window_waveforms = []
    for index, row in enumerate(rows):
        if str(row["speaker_id"]) != args.teacher_id:
            continue
        if float(row["duration_s"]) < args.min_segment_seconds:
            continue
        start_sample = max(0, int(float(row["start_ms"]) * sample_rate / 1000))
        end_sample = min(len(audio), int(float(row["end_ms"]) * sample_rate / 1000))
        position = start_sample
        while position + window_samples <= end_sample:
            window_waveforms.append(audio[position : position + window_samples])
            windows_by_row[index].append(
                {
                    "start_ms": int(position * 1000 / sample_rate),
                    "end_ms": int((position + window_samples) * 1000 / sample_rate),
                }
            )
            position += hop_samples

    window_embeddings = []
    for begin in range(0, len(window_waveforms), args.batch_size):
        window_embeddings.append(
            embed_batch(model, window_waveforms[begin : begin + args.batch_size], sample_rate)
        )
        print(
            f"embedded windows {min(begin + args.batch_size, len(window_waveforms))}/{len(window_waveforms)}",
            flush=True,
        )
    window_embeddings = np.concatenate(window_embeddings, axis=0)
    window_sims = window_embeddings @ centroid_matrix.T

    review = []
    candidates_by_row: dict[int, list[dict]] = defaultdict(list)
    cursor = 0
    for row_index, windows in windows_by_row.items():
        row = rows[row_index]
        count = len(windows)
        if count < 3:
            continue
        similarities = window_sims[cursor : cursor + count]
        cursor += count

        teacher_sims = similarities[:, teacher_position]
        best_student_positions = [speaker_ids.index(speaker_id) for speaker_id in student_ids]
        student_sims = similarities[:, best_student_positions]
        best_student_indices = student_sims.argmax(axis=1)
        best_student_sims = student_sims[np.arange(count), best_student_indices]
        best_student_labels = [student_ids[index] for index in best_student_indices]
        scores = [float(best_student_sims[index] - teacher_sims[index]) for index in range(count)]

        smoothed_scores = moving_average(scores, radius=1)
        flags = [
            smoothed_scores[index] >= args.margin and best_student_sims[index] >= args.min_student_sim
            for index in range(count)
        ]
        flags = majority_filter(flags, radius=1)
        runs = run_lengths(flags)

        parent_start = int(row["start_ms"])
        parent_end = int(row["end_ms"])
        for start_index, end_index in runs:
            if not flags[start_index]:
                continue
            region_start = windows[start_index]["start_ms"]
            region_end = windows[end_index]["end_ms"]
            duration = (region_end - region_start) / 1000
            if duration < args.min_region_seconds:
                continue
            labels = best_student_labels[start_index : end_index + 1]
            proposed, label_count = Counter(labels).most_common(1)[0]
            consistency = label_count / len(labels)
            average_margin = float(np.mean(smoothed_scores[start_index : end_index + 1]))
            chunk = text_chunk(str(row["text"]), parent_start, parent_end, region_start, region_end)
            candidate_id = len(review) + 1
            if accepted_ids is not None:
                should_apply = candidate_id in accepted_ids and len(chunk) >= 2
            else:
                should_apply = bool(
                    args.apply
                    and average_margin >= args.margin
                    and consistency >= args.consistency
                    and len(chunk) >= 2
                )
            candidate = {
                "candidate_id": candidate_id,
                "source_segment_id": int(row["segment_id"]),
                "parent_speaker": args.teacher_id,
                "region_start_ms": region_start,
                "region_end_ms": region_end,
                "region_duration_s": round(duration, 3),
                "proposed_speaker": proposed,
                "average_margin": round(average_margin, 4),
                "consistency": round(consistency, 3),
                "window_count": end_index - start_index + 1,
                "applied": should_apply,
                "region_text": chunk,
                "parent_text": row["text"],
            }
            candidates_by_row[row_index].append(candidate)
            review.append(candidate)

    applied_by_row = {
        row_index: [candidate for candidate in candidates if candidate["applied"]]
        for row_index, candidates in candidates_by_row.items()
    }

    output_rows = []
    for row_index, row in enumerate(rows):
        candidates = sorted(applied_by_row.get(row_index, []), key=lambda item: item["region_start_ms"])
        if not candidates:
            output_rows.append(row)
            continue

        parent_start = int(row["start_ms"])
        parent_end = int(row["end_ms"])
        text = str(row["text"])
        boundaries = [parent_start]
        for candidate in candidates:
            boundaries.extend([candidate["region_start_ms"], candidate["region_end_ms"]])
        boundaries.append(parent_end)
        boundaries = sorted(set(boundaries))
        slices = []
        for index in range(len(boundaries) - 1):
            slice_start = boundaries[index]
            slice_end = boundaries[index + 1]
            slice_speaker = args.teacher_id
            for candidate in candidates:
                if candidate["region_start_ms"] <= slice_start < candidate["region_end_ms"]:
                    slice_speaker = candidate["proposed_speaker"]
                    break
            slices.append(
                {"start_ms": slice_start, "end_ms": slice_end, "speaker_id": slice_speaker}
            )

        min_slice_ms = int(args.min_slice_seconds * 1000)
        changed = True
        while changed and len(slices) > 1:
            changed = False
            for index, item in enumerate(slices):
                if item["end_ms"] - item["start_ms"] >= min_slice_ms:
                    continue
                if index > 0:
                    slices[index - 1]["end_ms"] = item["end_ms"]
                    del slices[index]
                else:
                    slices[index + 1]["start_ms"] = item["start_ms"]
                    del slices[index]
                changed = True
                break

        merged_slices = []
        for item in slices:
            if merged_slices and merged_slices[-1]["speaker_id"] == item["speaker_id"]:
                merged_slices[-1]["end_ms"] = item["end_ms"]
            else:
                merged_slices.append(item)

        for index, item in enumerate(merged_slices):
            slice_start = item["start_ms"]
            slice_end = item["end_ms"]
            new_row = dict(row)
            new_row["start_ms"] = slice_start
            new_row["end_ms"] = slice_end
            new_row["start_hms"] = ms_to_hms(slice_start)
            new_row["end_hms"] = ms_to_hms(slice_end)
            new_row["duration_s"] = round((slice_end - slice_start) / 1000, 3)
            new_row["speaker_id"] = item["speaker_id"]
            new_row["speaker_original"] = row.get("speaker_original", args.teacher_id)
            new_row["speaker_refined"] = True
            new_row["split_from_segment_id"] = int(row["segment_id"])
            new_row["split_index"] = index + 1
            new_row["text"] = text_chunk(text, parent_start, parent_end, slice_start, slice_end)
            output_rows.append(new_row)

    output_rows.sort(key=lambda item: (int(item["start_ms"]), int(item.get("split_from_segment_id", 0)), int(item.get("split_index", 0))))
    for index, row in enumerate(output_rows, start=1):
        row["segment_id"] = index
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"

    review_path = output_dir / f"{prefix}_intra_review.csv"
    pd.DataFrame(review).to_csv(review_path, index=False, encoding="utf-8-sig")

    split_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with split_jsonl.open("w", encoding="utf-8") as handle:
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
        "speaker_original",
        "speaker_refined",
        "split_from_segment_id",
        "split_index",
        "text",
        "raw_text",
    ]
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in output_rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    clip_dir = output_dir / "split_samples"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_rows = []
    for candidate in review:
        start = max(0.0, candidate["region_start_ms"] / 1000 - 0.2)
        end = min(len(audio) / sample_rate, candidate["region_end_ms"] / 1000 + 0.2)
        clip = audio[int(start * sample_rate) : int(end * sample_rate)]
        filename = (
            f"cand{candidate['candidate_id']:03d}_"
            f"seg{candidate['source_segment_id']:04d}_"
            f"{candidate['region_start_ms']}_{candidate['region_end_ms']}.wav"
        )
        path = clip_dir / filename
        sf.write(str(path), clip, sample_rate)
        candidate = dict(candidate)
        candidate["clip"] = str(path)
        clip_rows.append(candidate)
    if clip_rows:
        pd.DataFrame(clip_rows).to_csv(clip_dir / "split_samples.csv", index=False, encoding="utf-8-sig")

    metadata_path = output_dir / f"{prefix}_intra_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "audio": str(audio_path),
                "teacher_id": args.teacher_id,
                "window_seconds": args.window_seconds,
                "hop_seconds": args.hop_seconds,
                "min_segment_seconds": args.min_segment_seconds,
                "min_region_seconds": args.min_region_seconds,
                "margin": args.margin,
                "min_student_sim": args.min_student_sim,
                "consistency": args.consistency,
                "candidates": len(review),
                "applied": sum(1 for candidate in review if candidate["applied"]),
                "source_segments": len(rows),
                "output_segments": len(output_rows),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    paths = export_all(
        split_jsonl,
        output_dir,
        window_seconds=args.output_window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Source segments: {len(rows)}")
    print(f"Output segments: {len(output_rows)}")
    print(f"Candidates: {len(review)}, applied: {sum(1 for candidate in review if candidate['applied'])}")
    print(f"Review CSV: {review_path}")
    print(f"Metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
