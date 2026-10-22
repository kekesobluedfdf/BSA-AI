from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to *_segments.csv")
    parser.add_argument("audio", help="Source audio file")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "speaker_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg not found in PATH")

    df = pd.read_csv(csv_path)
    rows = []
    for speaker_id, group in df.groupby("speaker_id"):
        segment = group.loc[group["duration_s"].idxmax()]
        start_s = max(0.0, float(segment["start_ms"]) / 1000 - 0.2)
        duration_s = min(float(segment["duration_s"]), args.max_seconds)
        filename = f"{speaker_id}_{segment['start_hms'].replace(':', '-').replace('.', '_')}.wav"
        clip_path = output_dir / filename
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start_s:.3f}",
                "-t",
                f"{duration_s:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(clip_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rows.append(
            {
                "speaker_id": speaker_id,
                "start_hms": segment["start_hms"],
                "end_hms": segment["end_hms"],
                "duration_s": round(duration_s, 2),
                "text": segment["text"],
                "clip": str(clip_path),
            }
        )

    samples = pd.DataFrame(rows)
    samples_path = output_dir / "speaker_samples.csv"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    print(samples[["speaker_id", "start_hms", "duration_s", "text"]].to_string(index=False))
    print(f"\nSamples: {samples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
