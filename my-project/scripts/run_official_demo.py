from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
MODEL_ROOT = ROOT_DIR / "models"

os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_ROOT / ".cache"))
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--merge-vad", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = AutoModel(
        model=str(MODEL_ROOT / "SenseVoiceSmall"),
        vad_model=str(MODEL_ROOT / "fsmn-vad"),
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model=str(MODEL_ROOT / "campplus"),
        device=device,
        disable_update=True,
    )

    audio = Path(args.audio) if args.audio else MODEL_ROOT / "SenseVoiceSmall" / "example" / "zh.mp3"
    result = model.generate(
        input=str(audio),
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=args.merge_vad,
        merge_length_s=15,
        return_spk_res=True,
    )

    print(f"Audio: {audio}")
    print(f"Device: {device}")
    print("Text:", rich_transcription_postprocess(result[0].get("text", "")))
    print("\nSpeaker Diarization Results:")
    sentences = result[0].get("sentence_info", [])
    for sentence in sentences:
        text = rich_transcription_postprocess(sentence.get("text") or sentence.get("sentence", ""))
        print(f"  Speaker {sentence.get('spk')}: [{sentence.get('start')}ms - {sentence.get('end')}ms] {text}")

    print("\nRaw sentence_info:")
    print(json.dumps(sentences, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
