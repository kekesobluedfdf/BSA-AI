from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from funasr import AutoModel

from transcript_exports import ms_to_clock, speaker_label

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_MODEL_ROOT = ROOT_DIR / "models"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_turns(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row["start_ms"]), int(row["segment_id"])))
    turns = []
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
                    "parts": [text],
                }
            )
        else:
            turns[-1]["parts"].append(text)
    return turns


def chunk_parts(parts: list[str], limit: int = 220) -> list[str]:
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if current and len(current) + len(part) > limit:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


def clean_punctuation(text: str) -> str:
    text = re.sub(r"([，。！？；：])\1+", r"\1", text)
    text = text.replace("。，", "。").replace("，。", "。")
    text = text.replace("？，", "？").replace("！，", "！")
    text = text.replace("、，", "、").replace("，、", "、")
    text = text.replace("？。", "？").replace("！。", "！")
    text = text.replace("。？", "？").replace("。！", "！")
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = jsonl_path.stem.removesuffix("_segments")

    rows = load_rows(jsonl_path)
    turns = build_turns(rows)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = AutoModel(
        model=str(Path(args.model_root) / "ct-punc"),
        device=device,
        disable_update=True,
    )

    punctuated_turns = []
    for turn in turns:
        chunks = chunk_parts(turn["parts"])
        results = model.generate(input=chunks, disable_pbar=True)
        text = "".join(str(item.get("text", "")) for item in results)
        punctuated_turns.append(
            {
                "speaker_id": turn["speaker_id"],
                "start_ms": turn["start_ms"],
                "text": clean_punctuation(text),
            }
        )

    output_path = output_dir / f"{prefix}_verbatim.txt"
    blocks = [
        f"{speaker_label(turn['speaker_id'])} {ms_to_clock(turn['start_ms'])}\n{turn['text']}"
        for turn in punctuated_turns
    ]
    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8-sig")

    raw_path = output_dir / f"{prefix}_verbatim_raw.txt"
    raw_blocks = [
        f"{speaker_label(turn['speaker_id'])} {ms_to_clock(turn['start_ms'])}\n"
        f"{''.join(turn['parts'])}"
        for turn in turns
    ]
    raw_path.write_text("\n\n".join(raw_blocks) + ("\n" if raw_blocks else ""), encoding="utf-8-sig")

    metadata_path = output_dir / f"{prefix}_punctuation_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_jsonl": str(jsonl_path),
                "turns": len(turns),
                "punctuated_file": str(output_path),
                "raw_file": str(raw_path),
                "device": device,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Turns: {len(turns)}")
    print(f"Verbatim: {output_path}")
    print(f"Raw: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
