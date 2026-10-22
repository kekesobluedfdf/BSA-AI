from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from coding_config import active_paths, copy_sources, read_document

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_API_BASE = "https://api.deepseek.com"

CODE_NAMES = {
    "A": "教师接受情感",
    "B": "教师鼓励表扬",
    "C": "采纳意见",
    "D": "提问开放性问题",
    "E": "提问封闭性问题",
    "F": "讲授",
    "G": "指示",
    "H": "批评",
    "I": "示范与引导",
    "J": "应答（被动反应）",
    "K": "应答（主动反应）",
    "L": "主动提问",
    "M": "与同伴交流",
    "N": "停顿或沉默",
    "O": "思考问题",
    "P": "做练习",
}
CODE_GROUPS = {
    "A": "教师语言-间接影响",
    "B": "教师语言-间接影响",
    "C": "教师语言-间接影响",
    "D": "教师语言-间接影响",
    "E": "教师语言-间接影响",
    "F": "教师语言-直接影响",
    "G": "教师语言-直接影响",
    "H": "教师语言-直接影响",
    "I": "教师语言-直接影响",
    "J": "学生语言",
    "K": "学生语言",
    "L": "学生语言",
    "M": "学生语言",
    "N": "课堂沉寂或学习活动",
    "O": "课堂沉寂或学习活动",
    "P": "课堂沉寂或学习活动",
}
VALID_CODES = set(CODE_NAMES)
VALID_CONFIDENCE = {"High", "Medium", "Low"}
TEACHER_CODES = set("ABCDEFGHI")
STUDENT_CODES = set("JKLM")


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


def deepseek_key() -> str:
    for name in ("deepseek_api_key", "DEEPSEEK_API_KEY", "DEEPSEEK_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"缺少 DeepSeek API Key，请在 {PROJECT_DIR / '.env'} 中设置 deepseek_api_key")


def safe_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-]+", r"\1***", text)
    text = re.sub(r"(api[_-]?key[=:]\s*)[^\s,;]+", r"\1***", text, flags=re.IGNORECASE)
    return text


def build_system_prompt(prompt_text: str, coding_text: str) -> str:
    adapter = """
【本次实际任务：固定时间窗口编码】

研究数据已经按固定时间窗口切分，每个窗口都有 window_id、开始/结束时间、
有效语音秒数 speech_s、主要说话人和原始文本 text。

请以每个 window_id 作为一个编码单元，输出一个主编码。规则如下：
1. 每个目标窗口必须返回且只返回一个主编码，编码只能是 A—P。
2. 一个窗口内如果有多个明显行为，选择占据主导地位、对教学推进最关键的行为作为主编码；
   其他明显行为写入 secondary_codes，不要漏掉整个窗口。
3. 必须结合 context_before 和 context_after 判断，但只对 target_windows 输出结果。
4. 窗口 text 为空且 speech_s 很小（例如小于 0.2 秒）时，可以编码为 N（停顿或沉默）。
5. 窗口 text 为空但 speech_s 明显大于 0 时，不要臆测 O 或 P；优先 N，置信度 Low，
   理由说明“有语音但无可用文本”。
6. 只有存在明确文本证据时才编码 O 或 P；教师说“开始写”本身是 G，不是 P。
7. 不得使用 A—P 之外的编码，不得创造新编码，不得评价教学质量。
8. 如果 dominant_role 与文本内容/行为主体明显冲突，以文本和上下文为准，
   但置信度不得为 High，至少降为 Medium，并在 reason 中说明“角色标签与文本冲突”。
9. 如果窗口跨两个以上明显行为，主编码置信度一般不高于 Medium，并填写 secondary_codes。
10. 输出必须是严格 JSON，不要 Markdown，不要代码块，不要额外解释。

输出 JSON 结构：
{
  "items": [
    {
      "window_id": 1,
      "code": "F",
      "confidence": "High",
      "alternative_code": "-",
      "secondary_codes": ["G"],
      "reason": "一句话说明主要行为功能及判断依据"
    }
  ]
}
"""
    return (
        "你将严格按照以下课堂行为编码手册工作。\n\n"
        + prompt_text
        + "\n\n【编码体系原文】\n"
        + coding_text
        + "\n\n"
        + adapter
    )


def json_records(df: pd.DataFrame, indices: list[int]) -> list[dict]:
    records = []
    for index in indices:
        row = df.loc[index]
        text = row.get("text", "")
        if pd.isna(text):
            text = ""
        records.append(
            {
                "window_id": int(row["window_id"]),
                "start": str(row.get("start_hms", "")),
                "end": str(row.get("end_hms", "")),
                "duration_s": float(row.get("duration_s", 0) or 0),
                "speech_s": float(row.get("speech_s", 0) or 0),
                "dominant_speaker": str(row.get("dominant_speaker", "") or ""),
                "dominant_role": str(row.get("dominant_role", "") or ""),
                "speakers": str(row.get("speakers", "") or ""),
                "speaker_count": int(row.get("speaker_count", 0) or 0),
                "segment_count": int(row.get("segment_count", 0) or 0),
                "text": str(text),
            }
        )
    return records


def parse_json_content(content: str) -> dict:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if isinstance(value, list):
        return {"items": value}
    if isinstance(value, dict) and "items" not in value:
        for key in ("results", "data", "codings"):
            if isinstance(value.get(key), list):
                return {"items": value[key]}
    return value


def normalize_item(item: dict) -> dict:
    window_id = int(item.get("window_id"))
    code = str(item.get("code", "")).strip().upper()
    if code not in VALID_CODES:
        raise ValueError(f"非法编码 {code!r}")
    confidence = str(item.get("confidence", "Low")).strip().title()
    if confidence not in VALID_CONFIDENCE:
        confidence = "Low"
    alternative = str(item.get("alternative_code", "-") or "-").strip().upper()
    if alternative not in VALID_CODES:
        alternative = "-"
    secondary = item.get("secondary_codes", [])
    if isinstance(secondary, str):
        secondary = [part.strip().upper() for part in re.split(r"[,，;/\s]+", secondary) if part.strip()]
    if not isinstance(secondary, list):
        secondary = []
    secondary = [str(value).strip().upper() for value in secondary]
    secondary = [value for value in secondary if value in VALID_CODES and value != code]
    return {
        "window_id": window_id,
        "code": code,
        "confidence": confidence,
        "alternative_code": alternative,
        "secondary_codes": sorted(set(secondary)),
        "reason": str(item.get("reason", "") or "").strip(),
    }


def request_batch(
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    payload: dict,
    timeout: int,
    max_tokens: int,
) -> tuple[dict[int, dict], dict]:
    url = f"{api_base.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(1, 5):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
            parsed = parse_json_content(content)
            items = parsed.get("items", [])
            if not isinstance(items, list):
                raise ValueError("返回 JSON 缺少 items 数组")
            normalized = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized_item = normalize_item(item)
                normalized[normalized_item["window_id"]] = normalized_item
            if not normalized:
                raise ValueError("返回结果没有有效编码")
            return normalized, data.get("usage", {})
        except Exception as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(2 * attempt)
    raise RuntimeError(f"DeepSeek 请求失败：{safe_error(last_error)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("window_csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--coding-system", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--context-windows", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=0, help="debug only: code first N windows")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    api_key = deepseek_key()
    window_path = Path(args.window_csv)
    output_dir = Path(args.output_dir) if args.output_dir else window_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    active_coding, active_prompt = active_paths()
    coding_path = Path(args.coding_system) if args.coding_system else active_coding
    prompt_path = Path(args.prompt) if args.prompt else active_prompt
    if not coding_path.exists():
        raise FileNotFoundError(coding_path)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)

    coding_text = read_document(coding_path)
    prompt_text = read_document(prompt_path)
    system_prompt = build_system_prompt(prompt_text, coding_text)
    copied_sources = copy_sources(coding_path, prompt_path, output_dir)

    df = pd.read_csv(window_path)
    if "window_id" not in df.columns:
        df["window_id"] = range(1, len(df) + 1)
    df["window_id"] = df["window_id"].astype(int)
    df = df.sort_values("window_id").reset_index(drop=True)
    total_available = len(df)
    if args.max_windows and args.max_windows > 0:
        df = df.head(args.max_windows).copy()

    match = re.match(r"(.+)_window_\d+(?:\.\d+)?s_fixed\.csv$", window_path.name)
    prefix = match.group(1) if match else "lesson"
    results: dict[int, dict] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    batch_log = []

    batch_size = max(1, args.batch_size)
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        target_indices = list(range(start, end))
        before_start = max(0, start - max(0, args.context_windows))
        after_end = min(total, end + max(0, args.context_windows))
        payload = {
            "task": "固定时间窗口课堂行为编码",
            "coding_unit_seconds": "window duration_s",
            "target_windows": json_records(df, target_indices),
            "context_before": json_records(df, list(range(before_start, start))),
            "context_after": json_records(df, list(range(end, after_end))),
        }
        batch_no = start // batch_size + 1
        print(f"批次 {batch_no}: 窗口 {df.iloc[start]['window_id']}–{df.iloc[end - 1]['window_id']}")
        batch_results, usage = request_batch(
            args.api_base,
            api_key,
            args.model,
            system_prompt,
            payload,
            args.timeout,
            args.max_tokens,
        )
        for key, value in usage.items():
            if key in usage_totals:
                usage_totals[key] += int(value or 0)
        missing = [
            int(df.loc[index, "window_id"])
            for index in target_indices
            if int(df.loc[index, "window_id"]) not in batch_results
        ]
        if missing:
            print(f"  补编码缺失窗口：{missing}")
            retry_indices = [
                index for index in target_indices if int(df.loc[index, "window_id"]) in missing
            ]
            retry_payload = {
                "task": "补编码缺失窗口，只输出下列窗口",
                "target_windows": json_records(df, retry_indices),
                "context_before": payload["context_before"],
                "context_after": payload["context_after"],
            }
            retry_results, retry_usage = request_batch(
                args.api_base,
                api_key,
                args.model,
                system_prompt,
                retry_payload,
                args.timeout,
                args.max_tokens,
            )
            for key, value in retry_usage.items():
                if key in usage_totals:
                    usage_totals[key] += int(value or 0)
            batch_results.update(retry_results)
        for window_id, item in batch_results.items():
            results[window_id] = item
        batch_log.append(
            {
                "batch": batch_no,
                "window_ids": [int(df.loc[index, "window_id"]) for index in target_indices],
                "returned": len(batch_results),
                "usage": usage,
            }
        )
        print(f"  返回 {len(batch_results)}/{len(target_indices)}")

    rows = []
    for _, source in df.iterrows():
        window_id = int(source["window_id"])
        item = results.get(window_id)
        if not item:
            item = {
                "window_id": window_id,
                "code": "N",
                "confidence": "Low",
                "alternative_code": "-",
                "secondary_codes": [],
                "reason": "模型未返回该窗口编码",
            }
        text = source.get("text", "")
        if pd.isna(text):
            text = ""
        role = str(source.get("dominant_role", "") or "").strip().lower()
        code = item["code"]
        role_conflict = (code in TEACHER_CODES and role == "student") or (
            code in STUDENT_CODES and role == "teacher"
        )
        confidence = item["confidence"]
        if role_conflict and confidence == "High":
            confidence = "Medium"
        reason = item["reason"]
        if role_conflict and "角色标签与文本冲突" not in reason:
            reason = (reason + "；角色标签与文本冲突").strip("；")
        rows.append(
            {
                "window_id": window_id,
                "start_hms": source.get("start_hms", ""),
                "end_hms": source.get("end_hms", ""),
                "duration_s": source.get("duration_s", ""),
                "speech_s": source.get("speech_s", ""),
                "dominant_speaker": source.get("dominant_speaker", ""),
                "dominant_role": source.get("dominant_role", ""),
                "speakers": source.get("speakers", ""),
                "speaker_count": source.get("speaker_count", ""),
                "segment_count": source.get("segment_count", ""),
                "text": text,
                "code": code,
                "code_name": CODE_NAMES[code],
                "code_group": CODE_GROUPS[code],
                "confidence": confidence,
                "role_conflict": role_conflict,
                "alternative_code": item["alternative_code"],
                "secondary_codes": "|".join(item["secondary_codes"]),
                "reason": reason,
            }
        )

    output = pd.DataFrame(rows)
    codes_path = output_dir / f"{prefix}_ai_codes.csv"
    output.to_csv(codes_path, index=False, encoding="utf-8-sig")

    jsonl_path = output_dir / f"{prefix}_ai_codes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = (
        output.groupby(["code", "code_name", "code_group"], dropna=False)
        .agg(
            frequency=("window_id", "count"),
            high=("confidence", lambda values: int((values == "High").sum())),
            medium=("confidence", lambda values: int((values == "Medium").sum())),
            low=("confidence", lambda values: int((values == "Low").sum())),
        )
        .reset_index()
        .sort_values(["code"])
    )
    summary["percentage"] = (summary["frequency"] / max(1, len(output)) * 100).round(2)
    summary_path = output_dir / f"{prefix}_ai_code_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    low_path = output_dir / f"{prefix}_ai_code_low_confidence.csv"
    output[output["confidence"] == "Low"].to_csv(low_path, index=False, encoding="utf-8-sig")
    review_path = output_dir / f"{prefix}_ai_code_review.csv"
    output[(output["confidence"] == "Low") | output["role_conflict"]].to_csv(
        review_path, index=False, encoding="utf-8-sig"
    )

    sequence_path = output_dir / f"{prefix}_ai_code_sequence.txt"
    sequence_path.write_text(
        "\n".join(f"W{row['window_id']:03d}\t{row['start_hms']}\t{row['code']}" for row in rows) + "\n",
        encoding="utf-8-sig",
    )

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window_csv": str(window_path),
        "coding_system": str(coding_path),
        "prompt": str(prompt_path),
        "coding_sources": copied_sources,
        "model": args.model,
        "api_base": args.api_base,
        "batch_size": batch_size,
        "context_windows": args.context_windows,
        "total_available_windows": total_available,
        "coded_windows": len(output),
        "role_conflicts": int(output["role_conflict"].sum()),
        "max_windows": args.max_windows,
        "usage": usage_totals,
        "batches": batch_log,
        "files": {
            "codes_csv": str(codes_path),
            "summary_csv": str(summary_path),
            "low_confidence_csv": str(low_path),
            "review_csv": str(review_path),
            "sequence_txt": str(sequence_path),
            "jsonl": str(jsonl_path),
        },
    }
    metadata_path = output_dir / f"{prefix}_ai_code_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成：{len(output)} 个窗口")
    print(f"编码结果：{codes_path}")
    print(f"编码统计：{summary_path}")
    print(f"低置信度：{low_path}")
    print(f"人工复核：{review_path}")
    print(f"行为序列：{sequence_path}")
    print(f"元数据：{metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
