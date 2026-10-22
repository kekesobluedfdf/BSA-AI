from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import string
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import pandas as pd
import requests

from transcript_exports import export_all, ms_to_hms

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
UPLOAD_URL = "https://office-api-ist-dx.iflyaisol.com/v2/upload"
RESULT_URL = "https://office-api-ist-dx.iflyaisol.com/v2/getResult"

CSV_COLUMNS = [
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


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


for env_path in (PROJECT_DIR / ".env", ROOT_DIR / ".env"):
    load_env_file(env_path)


def credential(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请检查 {PROJECT_DIR / '.env'}")
    return value


def sign_params(secret: str, params: dict[str, str]) -> str:
    filtered = {
        str(key): str(value)
        for key, value in params.items()
        if key != "signature" and value is not None and str(value) != ""
    }
    base_string = "&".join(
        f"{quote(str(key), safe='')}={quote(str(value), safe='')}"
        for key, value in sorted(filtered.items())
    )
    digest = hmac.new(secret.encode("utf-8"), base_string.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("utf-8")


def build_request_url(base_url: str, params: dict[str, str], signature: str) -> str:
    query = urlencode(params, quote_via=quote, safe="")
    return f"{base_url}?{query}"


def safe_request_error(exc: Exception) -> str:
    text = str(exc)
    for key in ("accessKeyId", "signature", "appId"):
        text = re.sub(rf"({key}=)[^&\s]+", r"\1***", text)
    return text


def now_text() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def random_text() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(16))


def audio_duration_ms(path: Path) -> int | None:
    try:
        import soundfile as sf

        return int(round(float(sf.info(str(path)).duration) * 1000))
    except Exception:
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(round(float(result.stdout.strip()) * 1000))
    except Exception:
        pass
    return None


def normalize_audio(input_path: Path, output_dir: Path) -> Path:
    try:
        import soundfile as sf

        info = sf.info(str(input_path))
        if info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
            return input_path
    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(f"需要将音频转为 16kHz/16bit/单声道，但未找到 ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "lesson_iflytek_upload.wav"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"ffmpeg 转码失败：{result.stderr[-800:]}")
    return target


def upload_audio(audio_path: Path, app_id: str, api_key: str, api_secret: str, args) -> dict:
    file_size = audio_path.stat().st_size
    if file_size > 500 * 1024 * 1024:
        raise RuntimeError("音频超过讯飞 500MB 限制")
    duration_ms = audio_duration_ms(audio_path)
    if duration_ms and duration_ms > 5 * 3600 * 1000:
        raise RuntimeError("音频超过讯飞 5 小时限制")
    signature_random = random_text()
    params = {
        "appId": app_id,
        "accessKeyId": api_key,
        "dateTime": now_text(),
        "signatureRandom": signature_random,
        "fileSize": str(file_size),
        "fileName": audio_path.name,
        "language": args.language,
        "roleType": str(args.role_type),
        "roleNum": str(args.role_num),
        "eng_smoothproc": "true",
        "eng_colloqproc": "false",
        "eng_vad_mdn": "1",
        "audioMode": "fileStream",
    }
    if args.pd:
        params["pd"] = args.pd
    if args.feature_ids:
        params["featureIds"] = args.feature_ids
    if duration_ms:
        params["duration"] = str(duration_ms)
    else:
        params["durationCheckDisable"] = "true"

    signature = sign_params(api_secret, params)
    url = build_request_url(UPLOAD_URL, params, signature)
    print(f"上传音频：{audio_path.name}，{file_size / 1024 / 1024:.1f} MB，时长 {duration_ms or '未知'} ms")
    try:
        with audio_path.open("rb") as handle:
            response = requests.post(
                url,
                data=handle,
                headers={"Content-Type": "application/octet-stream", "signature": signature},
                timeout=300,
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"讯飞上传请求失败：{safe_request_error(exc)}") from None
    except ValueError as exc:
        raise RuntimeError(f"讯飞上传响应不是 JSON：{safe_request_error(exc)}") from None
    if str(payload.get("code")) != "000000":
        raise RuntimeError(f"讯飞上传失败：{payload.get('code')} {payload.get('descInfo')}")
    content = payload.get("content") or {}
    order_id = content.get("orderId")
    if not order_id:
        raise RuntimeError(f"讯飞上传响应缺少 orderId：{payload}")
    print(f"订单已创建：{order_id}，预计耗时 {content.get('taskEstimateTime', '未知')} ms")
    return {
        "order_id": order_id,
        "signature_random": signature_random,
        "duration_ms": duration_ms,
        "file_size": file_size,
        "upload_response": payload,
    }


def query_result(order_id: str, signature_random: str, api_key: str, api_secret: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        params = {
            "accessKeyId": api_key,
            "dateTime": now_text(),
            "signatureRandom": signature_random,
            "orderId": order_id,
            "resultType": "transfer",
        }
        signature = sign_params(api_secret, params)
        url = build_request_url(RESULT_URL, params, signature)
        try:
            response = requests.post(
                url,
                json={},
                headers={"Content-Type": "application/json", "signature": signature},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"讯飞查询请求失败：{safe_request_error(exc)}") from None
        except ValueError as exc:
            raise RuntimeError(f"讯飞查询响应不是 JSON：{safe_request_error(exc)}") from None
        if str(payload.get("code")) != "000000":
            raise RuntimeError(f"讯飞查询失败：{payload.get('code')} {payload.get('descInfo')}")
        content = payload.get("content") or {}
        order_info = content.get("orderInfo") or {}
        status = order_info.get("status")
        fail_type = order_info.get("failType", 0)
        if status != last_status:
            print(f"转写状态：{status}，failType={fail_type}")
            last_status = status
        if status == 4:
            if int(fail_type or 0) != 0:
                raise RuntimeError(f"讯飞转写失败，failType={fail_type}")
            return payload
        if status == -1:
            raise RuntimeError(f"讯飞转写失败，failType={fail_type}")
        time.sleep(3)
    raise TimeoutError(f"讯飞转写超时（{timeout_s}s），订单 {order_id}")


def parse_json_value(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def decode_sentence(item: dict) -> dict | None:
    best = parse_json_value(item.get("json_1best"))
    if not isinstance(best, dict):
        return None
    st = best.get("st", best)
    if not isinstance(st, dict):
        return None
    start_ms = int(float(st.get("bg", 0)))
    end_ms = int(float(st.get("ed", start_ms)))
    speaker = str(st.get("rl", "0"))

    words = []
    text_parts = []
    for rt in st.get("rt", []) or []:
        if not isinstance(rt, dict):
            continue
        ws_list = rt.get("ws")
        if ws_list is None:
            ws_list = [rt] if "cw" in rt else []
        for ws in ws_list:
            candidates = ws.get("cw") or []
            if not candidates:
                continue
            chosen = candidates[0]
            token = str(chosen.get("w", ""))
            if not token:
                continue
            wb = int(float(ws.get("wb", 0)))
            we = int(float(ws.get("we", wb)))
            token_start = start_ms + wb * 10
            token_end = start_ms + we * 10
            if token_end <= token_start:
                token_end = token_start + 10
            words.append(
                {
                    "text": token,
                    "start_ms": token_start,
                    "end_ms": token_end,
                    "wp": str(chosen.get("wp", "n")),
                }
            )
            text_parts.append(token)

    text = re.sub(r"[ \t\u3000]+", " ", "".join(text_parts)).strip()
    if not text:
        return None
    if words:
        end_ms = max(end_ms, words[-1]["end_ms"])
    return {
        "start_ms": start_ms,
        "end_ms": max(end_ms, start_ms + 1),
        "speaker_raw": speaker,
        "text": text,
        "words": words,
        "segment_timestamp": [[word["start_ms"], word["end_ms"]] for word in words],
    }


def parse_segments(result: dict) -> tuple[list[dict], dict]:
    content = result.get("content") or {}
    order_result = parse_json_value(content.get("orderResult") or {})
    lattice = order_result.get("lattice") or []
    sentences = []
    for item in lattice:
        if not isinstance(item, dict):
            continue
        sentence = decode_sentence(item)
        if sentence:
            sentences.append(sentence)

    totals: dict[str, int] = {}
    for sentence in sentences:
        speaker = sentence["speaker_raw"]
        totals[speaker] = totals.get(speaker, 0) + (sentence["end_ms"] - sentence["start_ms"])

    return sentences, {
        "orderInfo": content.get("orderInfo"),
        "taskEstimateTime": content.get("taskEstimateTime"),
        "speaker_totals": totals,
        "lattice_count": len(lattice),
    }


def map_speakers(sentences: list[dict], teacher_id: str, teacher_speaker: str) -> tuple[list[dict], dict]:
    totals: dict[str, int] = {}
    for sentence in sentences:
        speaker = sentence["speaker_raw"]
        totals[speaker] = totals.get(speaker, 0) + (sentence["end_ms"] - sentence["start_ms"])

    teacher_raw = teacher_speaker.strip()
    if teacher_raw == "auto" or teacher_raw not in totals:
        teacher_raw = max(totals, key=totals.get) if totals else "0"

    mapping = {}
    for raw_speaker in totals:
        mapping[raw_speaker] = teacher_id if raw_speaker == teacher_raw else "S02"

    rows = []
    for sentence in sentences:
        raw_speaker = sentence["speaker_raw"]
        speaker_id = mapping.get(raw_speaker, "S02")
        rows.append(
            {
                "start_ms": sentence["start_ms"],
                "end_ms": sentence["end_ms"],
                "speaker_id": speaker_id,
                "role": "teacher" if speaker_id == teacher_id else "student",
                "speaker_original": f"iflytek:{raw_speaker}",
                "speaker_refined": speaker_id,
                "speaker_raw": raw_speaker,
                "text": sentence["text"],
                "raw_text": sentence["text"],
                "segment_timestamp": sentence["segment_timestamp"],
                "asr_engine": "iflytek",
                "iflytek_speaker": raw_speaker,
            }
        )
    return rows, {"teacher_raw": teacher_raw, "mapping": mapping, "totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-id", default="S01")
    parser.add_argument("--teacher-speaker", default="auto", help="auto or iFlytek raw speaker id")
    parser.add_argument("--role-type", type=int, default=1, choices=[0, 1, 3])
    parser.add_argument("--role-num", type=int, default=2)
    parser.add_argument("--feature-ids", default="")
    parser.add_argument("--pd", default="edu", help="iFlytek domain, e.g. edu; empty to disable")
    parser.add_argument("--language", default="autodialect")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--stats-window-min", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    app_id = credential("IFLYTEK_APP_ID")
    api_key = credential("IFLYTEK_API_KEY")
    api_secret = credential("IFLYTEK_API_SECRET")

    audio_path = Path(args.audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    normalized = normalize_audio(audio_path, output_dir)
    duration_ms = audio_duration_ms(normalized)
    try:
        upload = upload_audio(normalized, app_id, api_key, api_secret, args)
        result = query_result(
            upload["order_id"],
            upload["signature_random"],
            api_key,
            api_secret,
            args.timeout_seconds,
        )
    finally:
        if normalized != audio_path and normalized.exists():
            normalized.unlink(missing_ok=True)

    sentences, raw_info = parse_segments(result)
    if not sentences:
        raise RuntimeError("讯飞返回了空转写结果")
    rows, speaker_info = map_speakers(sentences, args.teacher_id, args.teacher_speaker)
    rows.sort(key=lambda row: (int(row["start_ms"]), int(row["end_ms"])))
    recording_id = audio_path.stem
    for index, row in enumerate(rows, start=1):
        row["recording_id"] = recording_id
        row["segment_id"] = index
        row["start_hms"] = ms_to_hms(int(row["start_ms"]))
        row["end_hms"] = ms_to_hms(int(row["end_ms"]))
        row["duration_s"] = round((int(row["end_ms"]) - int(row["start_ms"])) / 1000, 3)

    jsonl_path = output_dir / "lesson_segments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    pd.DataFrame([{column: row.get(column, "") for column in CSV_COLUMNS} for row in rows]).to_csv(
        output_dir / "lesson_segments.csv", index=False, encoding="utf-8-sig"
    )

    meta = {
        "engine": "iflytek",
        "model": "recording_file_llm",
        "pd": args.pd,
        "language": args.language,
        "roleType": args.role_type,
        "roleNum": args.role_num,
        "teacher_id": args.teacher_id,
        "teacher_speaker": args.teacher_speaker,
        "duration_ms": duration_ms,
        "segments": len(rows),
        "speaker_info": speaker_info,
        "raw_info": raw_info,
        "upload_info": {
            key: value for key, value in upload.items() if key not in {"upload_response"}
        },
    }
    (output_dir / "lesson_iflytek_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "lesson_iflytek_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    paths = export_all(
        jsonl_path,
        output_dir,
        window_seconds=args.window_seconds,
        stats_window_min=args.stats_window_min,
        teacher_id=args.teacher_id,
        window_mode="fixed",
    )
    print(f"讯飞转写完成：{len(rows)} 个句子片段")
    print(f"教师原始说话人：{speaker_info['teacher_raw']}，映射：{speaker_info['mapping']}")
    for name, path in paths.items():
        if name != "teacher_id":
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
