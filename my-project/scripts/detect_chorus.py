from __future__ import annotations

import argparse
import json
import os
import re
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

CUE_PATTERN = re.compile(r"一起读|齐读|预备起|全班|大家一起|我们一起来|读一遍|跟我读")


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


def compute_embeddings(
    rows: list[dict], audio: np.ndarray, sample_rate: int, model: AutoModel, batch_size: int
) -> np.ndarray:
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
    for begin in range(0, len(waveforms), batch_size):
        batches.append(embed_batch(model, waveforms[begin : begin + batch_size], sample_rate))
    return np.concatenate(batches, axis=0)


def context_text(rows: list[dict], index: int, window_ms: int = 10000) -> str:
    current_start = int(rows[index]["start_ms"])
    parts = []
    for position in range(index - 1, max(-1, index - 6), -1):
        row = rows[position]
        if int(row["end_ms"]) < current_start - window_ms:
            break
        parts.append(str(row.get("text") or ""))
    return " ".join(reversed(parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="Path to final *_segments.jsonl")
    parser.add_argument("audio", help="Source audio file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--energy-ratio", type=float, default=1.6)
    parser.add_argument("--teacher-sim", type=float, default=0.35)
    parser.add_argument("--output-window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto-energy-ratio", type=float, default=2.0)
    parser.add_argument("--auto-teacher-sim", type=float, default=0.4)
    parser.add_argument("--accept-ids", default=None)
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

    embeddings = None
    if args.embeddings and Path(args.embeddings).exists():
        cached = np.load(args.embeddings, allow_pickle=True)
        if len(cached["embeddings"]) == len(rows):
            embeddings = cached["embeddings"]
    if embeddings is None:
        embeddings = compute_embeddings(rows, audio, sample_rate, model, args.batch_size)
        cache_path = output_dir / f"{prefix}_embeddings.npz"
        np.savez_compressed(
            cache_path,
            segment_ids=np.array([int(row["segment_id"]) for row in rows]),
            speaker_ids=np.array([str(row["speaker_id"]) for row in rows]),
            durations=np.array([float(row["duration_s"]) for row in rows]),
            embeddings=embeddings,
        )

    teacher_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["speaker_id"]) == args.teacher_id and float(row["duration_s"]) >= 1.0
    ]
    if not teacher_indices:
        raise RuntimeError("No sufficiently long teacher segments found")
    teacher_centroid = embeddings[teacher_indices].mean(axis=0)
    teacher_centroid = teacher_centroid / np.linalg.norm(teacher_centroid)
    teacher_similarity = embeddings @ teacher_centroid
    teacher_median_sim = float(np.median(teacher_similarity[teacher_indices]))

    rms_values = []
    for row in rows:
        start = max(0, int(float(row["start_ms"]) * sample_rate / 1000))
        end = min(len(audio), int(float(row["end_ms"]) * sample_rate / 1000))
        segment = audio[start:end]
        rms_values.append(float(np.sqrt(np.mean(segment * segment))) if len(segment) else 0.0)
    rms_values = np.array(rms_values)
    teacher_median_rms = float(np.median(rms_values[teacher_indices]))

    review = []
    applied_rows = set()
    for index, row in enumerate(rows):
        duration = float(row["duration_s"])
        if duration < args.min_duration:
            continue
        if str(row["speaker_id"]).upper() == "CHORUS":
            continue

        teacher_sim = float(teacher_similarity[index])
        energy_ratio = float(rms_values[index] / teacher_median_rms) if teacher_median_rms else 0.0
        cue = bool(CUE_PATTERN.search(context_text(rows, index)))
        score = energy_ratio * 0.5 + max(0.0, teacher_median_sim - teacher_sim) + (0.4 if cue else 0.0)
        candidate = (
            energy_ratio >= args.energy_ratio and teacher_sim <= args.teacher_sim
        ) or (cue and energy_ratio >= 1.6 and teacher_sim <= 0.6)
        if not candidate:
            continue

        candidate_id = len(review) + 1
        auto_apply = bool(
            energy_ratio >= args.auto_energy_ratio
            and teacher_sim <= args.auto_teacher_sim
        )
        if accepted_ids is not None:
            should_apply = candidate_id in accepted_ids
        elif args.auto:
            should_apply = auto_apply
        else:
            should_apply = args.apply
        if should_apply:
            applied_rows.add(index)
        review.append(
            {
                "candidate_id": candidate_id,
                "segment_id": int(row["segment_id"]),
                "start_ms": int(row["start_ms"]),
                "end_ms": int(row["end_ms"]),
                "start_hms": row["start_hms"],
                "end_hms": row["end_hms"],
                "duration_s": round(duration, 3),
                "current_speaker": str(row["speaker_id"]),
                "teacher_sim": round(teacher_sim, 4),
                "rms": round(float(rms_values[index]), 5),
                "energy_ratio": round(energy_ratio, 3),
                "context_cue": cue,
                "score": round(score, 4),
                "auto_applied": auto_apply,
                "applied": should_apply,
                "text": str(row.get("text") or ""),
            }
        )

    for index in applied_rows:
        row = rows[index]
        row["speaker_original"] = row.get("speaker_original", row["speaker_id"])
        row["speaker_id"] = "CHORUS"
        row["role"] = "chorus"
        row["chorus_refined"] = True

    review_path = output_dir / f"{prefix}_chorus_review.csv"
    pd.DataFrame(review).to_csv(review_path, index=False, encoding="utf-8-sig")

    output_jsonl = output_dir / f"{prefix}_segments.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
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
        "text",
        "raw_text",
    ]
    for row in rows:
        if str(row["speaker_id"]).upper() == "CHORUS":
            row["role"] = "chorus"
        elif str(row["speaker_id"]) == args.teacher_id:
            row["role"] = "teacher"
        else:
            row["role"] = "student"
    pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows]).to_csv(
        output_dir / f"{prefix}_segments.csv", index=False, encoding="utf-8-sig"
    )

    clip_dir = output_dir / "chorus_samples"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_rows = []
    for item in review:
        start = max(0.0, item["start_ms"] / 1000 - 0.2)
        end = min(len(audio) / sample_rate, item["end_ms"] / 1000 + 0.2)
        clip = audio[int(start * sample_rate) : int(end * sample_rate)]
        filename = f"chorus_{item['candidate_id']:03d}_seg{item['segment_id']:04d}.wav"
        path = clip_dir / filename
        sf.write(str(path), clip, sample_rate)
        item = dict(item)
        item["clip"] = str(path)
        clip_rows.append(item)
    if clip_rows:
        pd.DataFrame(clip_rows).to_csv(clip_dir / "chorus_samples.csv", index=False, encoding="utf-8-sig")

    metadata_path = output_dir / f"{prefix}_chorus_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "audio": str(audio_path),
                "teacher_id": args.teacher_id,
                "teacher_median_sim": round(teacher_median_sim, 4),
                "teacher_median_rms": round(teacher_median_rms, 5),
                "min_duration": args.min_duration,
                "energy_ratio": args.energy_ratio,
                "teacher_sim": args.teacher_sim,
                "auto_energy_ratio": args.auto_energy_ratio,
                "auto_teacher_sim": args.auto_teacher_sim,
                "auto_mode": args.auto,
                "candidates": len(review),
                "auto_candidates": sum(1 for item in review if item["auto_applied"]),
                "applied": sum(1 for item in review if item["applied"]),
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

    print(f"Candidates: {len(review)}")
    print(f"Auto candidates: {sum(1 for item in review if item['auto_applied'])}")
    print(f"Applied: {sum(1 for item in review if item['applied'])}")
    print(f"Review CSV: {review_path}")
    print(f"Metadata: {metadata_path}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
