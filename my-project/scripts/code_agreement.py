from __future__ import annotations

from collections import Counter

import pandas as pd

VALID_CODES = list("ABCDEFGHIJKLMNOP")


def interpret_kappa(value: float) -> str:
    if value < 0:
        return "差（Poor）"
    if value <= 0.20:
        return "轻微（Slight）"
    if value <= 0.40:
        return "一般（Fair）"
    if value <= 0.60:
        return "中等（Moderate）"
    if value <= 0.80:
        return "较强（Substantial）"
    return "几乎完全一致（Almost perfect）"


def agreement_stats(ai_codes: list[str], human_codes: list[str]) -> dict:
    pairs = [
        (str(ai).strip().upper(), str(human).strip().upper())
        for ai, human in zip(ai_codes, human_codes)
        if str(ai).strip().upper() in VALID_CODES and str(human).strip().upper() in VALID_CODES
    ]
    n = len(pairs)
    if n == 0:
        return {
            "reviewed": 0,
            "matches": 0,
            "observed_agreement": None,
            "expected_agreement": None,
            "kappa": None,
            "interpretation": "无已复核窗口",
        }
    matches = sum(1 for ai, human in pairs if ai == human)
    po = matches / n
    ai_counts = Counter(ai for ai, _ in pairs)
    human_counts = Counter(human for _, human in pairs)
    pe = sum((ai_counts[code] / n) * (human_counts[code] / n) for code in VALID_CODES)
    if pe >= 1:
        kappa = 1.0 if po >= 1 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return {
        "reviewed": n,
        "matches": matches,
        "observed_agreement": round(po, 6),
        "expected_agreement": round(pe, 6),
        "kappa": round(kappa, 6),
        "interpretation": interpret_kappa(kappa),
    }


def confusion_matrix(ai_codes: list[str], human_codes: list[str]) -> pd.DataFrame:
    pairs = [
        (str(ai).strip().upper(), str(human).strip().upper())
        for ai, human in zip(ai_codes, human_codes)
        if str(ai).strip().upper() in VALID_CODES and str(human).strip().upper() in VALID_CODES
    ]
    data = pd.crosstab(
        pd.Series([ai for ai, _ in pairs], name="AI"),
        pd.Series([human for _, human in pairs], name="人工"),
        dropna=False,
    )
    return data.reindex(index=VALID_CODES, columns=VALID_CODES, fill_value=0)


def merge_review(ai: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    if "window_id" not in human.columns:
        human = human.copy()
        human["window_id"] = []
    merged = ai.merge(human, on="window_id", how="left", suffixes=("", "_human"))
    if "human_code" not in merged.columns:
        merged["human_code"] = ""
    if "human_note" not in merged.columns:
        merged["human_note"] = ""
    merged["human_code"] = merged["human_code"].fillna("").astype(str).str.strip().str.upper()
    merged["human_note"] = merged["human_note"].fillna("").astype(str)
    return merged


def build_agreement(ai: pd.DataFrame, human: pd.DataFrame) -> dict:
    merged = merge_review(ai, human)
    reviewed = merged[merged["human_code"].isin(VALID_CODES)].copy()
    stats = agreement_stats(reviewed["code"].tolist(), reviewed["human_code"].tolist())
    matrix = confusion_matrix(reviewed["code"].tolist(), reviewed["human_code"].tolist())
    disagreements = reviewed[reviewed["code"] != reviewed["human_code"]].copy()
    high = reviewed[reviewed["confidence"] == "High"] if "confidence" in reviewed.columns else reviewed.iloc[0:0]
    high_stats = agreement_stats(high["code"].tolist(), high["human_code"].tolist()) if len(high) else None
    return {
        "stats": stats,
        "high_confidence_stats": high_stats,
        "confusion": matrix,
        "disagreements": disagreements,
        "reviewed": reviewed,
    }
