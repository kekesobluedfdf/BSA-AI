from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
MODEL_ROOT = ROOT_DIR / "models"

os.environ.setdefault("MODELSCOPE_CACHE", str(MODEL_ROOT / ".cache"))
os.environ.setdefault("MODELSCOPE_ENDPOINT", "https://www.modelscope.cn")

MODEL_FILES = {
    "SenseVoiceSmall": ["model.pt", "config.yaml", "tokens.json"],
    "fsmn-vad": ["model.pt", "config.yaml"],
    "campplus": ["campplus_cn_common.bin", "config.yaml"],
    "ct-punc": ["model.pt", "config.yaml", "tokens.json"],
}


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_gb() -> tuple[float, float]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0, 0.0
    return status.ullTotalPhys / 1024**3, status.ullAvailPhys / 1024**3


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        return "NOT FOUND"
    try:
        out = subprocess.check_output([exe, "-version"], stderr=subprocess.STDOUT, text=True)
        return out.splitlines()[0]
    except Exception as exc:
        return f"FOUND but failed: {exc}"


def check_models() -> bool:
    print("\n[Models]")
    all_ok = True
    for name, files in MODEL_FILES.items():
        model_dir = MODEL_ROOT / name
        missing = [f for f in files if not (model_dir / f).exists()]
        if not model_dir.is_dir():
            print(f"  {name:<15} MISSING directory: {model_dir}")
            all_ok = False
            continue
        size_gb = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) / 1024**3
        if missing:
            print(f"  {name:<15} INCOMPLETE ({size_gb:.2f} GB), missing: {', '.join(missing)}")
            all_ok = False
        else:
            print(f"  {name:<15} OK ({size_gb:.2f} GB)")
    return all_ok


def main() -> int:
    print("[System]")
    print(f"  Python          {sys.version.split()[0]}")
    print(f"  Executable      {sys.executable}")
    print(f"  Platform        {platform.platform()}")
    print(f"  CPU threads     {os.cpu_count()}")

    total_ram, free_ram = memory_gb()
    print(f"  RAM             {total_ram:.1f} GB total / {free_ram:.1f} GB free")

    disk = shutil.disk_usage(ROOT_DIR)
    print(f"  Disk free       {disk.free / 1024**3:.1f} GB")

    print("\n[FFmpeg]")
    print(f"  {check_ffmpeg()}")

    print("\n[Python packages]")
    try:
        import funasr
        import modelscope
        import numpy
        import pandas
        import torch
        import torchaudio

        print(f"  torch           {torch.__version__}")
        print(f"  torchaudio      {torchaudio.__version__}")
        print(f"  funasr          {funasr.__version__}")
        print(f"  modelscope      {modelscope.__version__}")
        print(f"  numpy           {numpy.__version__}")
        print(f"  pandas          {pandas.__version__}")
        print(f"  cuda available  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  GPU             {props.name}")
            print(f"  VRAM            {props.total_memory / 1024**3:.1f} GB")
            print(f"  CUDA runtime    {torch.version.cuda}")
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 1

    models_ok = check_models()

    print("\n[Result]")
    if torch.cuda.is_available() and models_ok:
        print("  Environment is ready for GPU transcription.")
        return 0
    if not torch.cuda.is_available():
        print("  WARNING: CUDA is unavailable; transcription will fall back to CPU.")
    if not models_ok:
        print("  WARNING: some local models are missing; ModelScope download will be used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
