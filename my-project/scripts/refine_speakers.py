from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from funasr import AutoModel

from transcript_exports import export_all

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


def build_segments(rows: list[dict], audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    minimum = int(0.3 * sample_rate)
    segments = []
    for row in rows:
        start = max(0, int(float(row["start_ms"]) * sample_rate / 1000))
        end = min(len(audio), int(float(row["end_ms"]) * sample_rate / 1000))
        if end <= start:
            end = min(len(audio), start + minimum)
        segment = audio[start:end]
        if len(segment) < minimum:
            segment = np.pad(segment, (0, minimum - len(segment)))
        segments.append(segment)
    return segments


def compute_embeddings(
    rows: list[dict],
    audio: np.ndarray,
    sample_rate: int,
    model: AutoModel,
    batch_size: int,
) -> np.ndarray:
    segments = build_segments(rows, audio, sample_rate)
    embeddings = []
    for begin in range(0, len(segments), batch_size):
        batch = segments[begin : begin + batch_size]
        result = model.generate(input=batch, fs=sample_rate, disable_pbar=True)
        for item in result:
            embeddings.append(item["spk_embedding"].detach().cpu().numpy().reshape(-1))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to *_segments.jsonl")
    parser.add_argument("audio", help="Source audio file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--short-threshold", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-student-sim", type=float, default=0.2)
    parser.add_argument("--min-centroid-duration", type=float, default=1.0)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    args = parser.parse_args()

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

    embeddings = None
    if args.embeddings and Path(args.embeddings).exists():
        cached = np.load(args.embeddings, allow_pickle=True)
        if len(cached["embeddings"]) == len(rows):
            embeddings = cached["embeddings"]
            print(f"Loaded embeddings: {args.embeddings}")

    if embeddings is None:
        model = AutoModel(
            model=str(Path(args.model_root) / "campplus"),
            device=device,
            disable_update=True,
        )
        embeddings = compute_embeddings(rows, audio, sample_rate, model, args.batch_size)
        cache_path = output_dir / f"{prefix}_embeddings.npz"
        np.savez_compressed(
            cache_path,
            segment_ids=np.array([int(row["segment_id"]) for row in rows]),
            speaker_ids=np.array([str(row["speaker_id"]) for row in rows]),
            durations=np.array([float(row["duration_s"]) for row in rows]),
            embeddings=embeddings,
        )
        print(f"Saved embeddings: {cache_path}")

    centroids = build_centroids(rows, embeddings, args.min_centroid_duration)
    speaker_ids = list(centroids)
    centroid_matrix = np.stack([centroids[speaker_id] for speaker_id in speaker_ids])
    similarities = embeddings @ centroid_matrix.T
    teacher_index = speaker_ids.index(args.teacher_id)
    student_ids = [speaker_id for speaker_id in speaker_ids if speaker_id != args.teacher_id]

    for row in rows:
        row["speaker_original"] = str(row["speaker_id"])
        row["speaker_refined"] = False

    review = []
    changed_ids = []
    applied = 0
    for index, row in enumerate(rows):
        if str(row["speaker_id"]) != args.teacher_id:
            continue
        if float(row["duration_s"]) >= args.short_threshold:
            continue

        teacher_sim = float(similarities[index, teacher_index])
        student_sims = {
            speaker_id: float(similarities[index, speaker_ids.index(speaker_id)])
            for speaker_id in student_ids
        }
        best_student = max(student_sims, key=student_sims.get)
        best_student_sim = student_sims[best_student]
        margin = best_student_sim - teacher_sim
        should_apply = bool(
            args.apply and margin >= args.margin and best_student_sim >= args.min_student_sim
        )
        if should_apply:
            row["speaker_id"] = best_student
            row["speaker_refined"] = True
            changed_ids.append(int(row["segment_id"]))
            applied += 1

        review.append(
            {
                "segment_id": int(row["segment_id"]),
                "start_hms": row["start_hms"],
                "end_hms": row["end_hms"],
                "duration_s": float(row["duration_s"]),
                "old_speaker": args.teacher_id,
                "proposed_speaker": best_student,
                "teacher_sim": round(teacher_sim, 4),
                "student_sim": round(best_student_sim, 4),
                "margin": round(margin, 4),
                "applied": should_apply,
                "text": row["text"],
            }
        )

    review_path = output_dir / f"{prefix}_speaker_review.csv"
    pd.DataFrame(review).to_csv(review_path, index=False, encoding="utf-8-sig")

    metadata_path = output_dir / f"{prefix}_refinement.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "audio": str(audio_path),
                "teacher_id": args.teacher_id,
                "short_threshold_s": args.short_threshold,
                "margin": args.margin,
                "min_student_sim": args.min_student_sim,
                "min_centroid_duration_s": args.min_centroid_duration,
                "reviewed_segments": len(review),
                "applied_changes": applied,
                "changed_segment_ids": changed_ids,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    for row in rows:
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"

    refined_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with refined_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
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
        "text",
        "raw_text",
    ]
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    paths = export_all(
        refined_jsonl,
        output_dir,
        window_seconds=args.window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )

    print(f"Short teacher segments reviewed: {len(review)}")
    print(f"Applied speaker changes: {applied}")
    print(f"Review CSV: {review_path}")
    print(f"Refinement metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
