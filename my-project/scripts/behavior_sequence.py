from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VALID_CODES = list("ABCDEFGHIJKLMNOP")
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
    **{code: "教师语言-间接影响" for code in "ABCDE"},
    **{code: "教师语言-直接影响" for code in "FGHI"},
    **{code: "学生语言" for code in "JKLM"},
    **{code: "课堂沉寂或学习活动" for code in "NOP"},
}


def stars_for_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def z_critical(alpha: float) -> float:
    return statistics.NormalDist().inv_cdf(1 - alpha / 2)


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def bh_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        value = min(1.0, p[index] * n / (rank + 1))
        running = min(running, value)
        adjusted[index] = running
    return adjusted


def empirical_p_values(observed: int, simulations: np.ndarray) -> dict:
    simulations = np.asarray(simulations, dtype=float)
    b = len(simulations)
    if b == 0:
        return {
            "simulated_mean": math.nan,
            "simulated_sd": math.nan,
            "empirical_p_upper": math.nan,
            "empirical_p_lower": math.nan,
            "empirical_p_two_sided": math.nan,
            "simulation_degenerate": True,
        }
    upper = (int(np.sum(simulations >= observed)) + 1) / (b + 1)
    lower = (int(np.sum(simulations <= observed)) + 1) / (b + 1)
    two_sided = min(1.0, 2 * min(upper, lower))
    return {
        "simulated_mean": float(np.mean(simulations)),
        "simulated_sd": float(np.std(simulations, ddof=0)),
        "empirical_p_upper": float(upper),
        "empirical_p_lower": float(lower),
        "empirical_p_two_sided": float(two_sided),
        "simulation_degenerate": bool(np.ptp(simulations) == 0),
    }


def parse_hms_ms(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            seconds = int(parts[0]) * 60 + float(parts[1])
        else:
            seconds = float(parts[0])
    except (TypeError, ValueError):
        return None
    return int(round(seconds * 1000))


def choose_time_column(df: pd.DataFrame) -> tuple[pd.Series, str, bool]:
    if "start_ms" in df.columns:
        numeric = pd.to_numeric(df["start_ms"], errors="coerce")
        if numeric.notna().all():
            return numeric.astype("int64"), "start_ms", True
    if "start_hms" in df.columns:
        parsed = df["start_hms"].map(parse_hms_ms)
        if parsed.notna().all():
            return parsed.astype("int64"), "start_hms", True
    if "window_id" in df.columns:
        numeric = pd.to_numeric(df["window_id"], errors="coerce")
        if numeric.notna().all():
            return numeric.astype("int64"), "window_id", False
    raise ValueError("找不到可用的时间字段：需要 start_ms、start_hms 或 window_id")


def choose_end_column(df: pd.DataFrame, start_ms: pd.Series) -> tuple[pd.Series | None, str]:
    if "end_ms" in df.columns:
        numeric = pd.to_numeric(df["end_ms"], errors="coerce")
        if numeric.notna().all():
            return numeric.astype("int64"), "end_ms"
    if "duration_s" in df.columns:
        duration = pd.to_numeric(df["duration_s"], errors="coerce")
        if duration.notna().all():
            return (start_ms + (duration * 1000).round().astype("int64")), "duration_s"
    return None, "missing"


def prepare_sequences(
    ai: pd.DataFrame,
    human: pd.DataFrame | None,
    source: str,
    lag: int,
    sequence_column: str = "sequence_id",
    single_sequence: bool = False,
    time_column: str = "auto",
) -> tuple[list[list[dict]], dict]:
    if lag < 1:
        raise ValueError("lag 必须大于等于 1")

    merged = ai.copy()
    if human is not None and not human.empty:
        human = human.copy()
        human["human_code"] = human["human_code"].fillna("").astype(str).str.strip().str.upper()
        merged = merged.merge(human[["window_id", "human_code"]], on="window_id", how="left")
    else:
        merged["human_code"] = ""
    merged["human_code"] = merged["human_code"].fillna("").astype(str).str.upper()

    has_sequence_column = bool(sequence_column) and sequence_column in merged.columns
    if has_sequence_column and merged[sequence_column].notna().any():
        sequence_values = merged[sequence_column].fillna("").astype(str).str.strip()
        if (sequence_values == "").any():
            raise ValueError(f"序列列 {sequence_column!r} 存在空值，无法确定序列边界")
        sequence_mode = "explicit_sequence_id"
        boundary_assumption = f"使用 {sequence_column!r} 作为独立观察序列边界"
    elif single_sequence:
        sequence_values = pd.Series(["__single_sequence__"] * len(merged), index=merged.index)
        sequence_mode = "single_sequence_assumed"
        boundary_assumption = (
            "CSV 没有 sequence_id；用户显式确认整个文件是一个连续观察序列"
        )
    else:
        raise ValueError(
            "CSV 没有有效的 sequence_id。若整个文件确实是一个连续观察序列，"
            "请显式传入 --single-sequence；否则请先提供 sequence_id 列。"
        )

    if time_column == "auto":
        times, time_source, _ = choose_time_column(merged)
    elif time_column in merged.columns:
        if time_column == "start_ms":
            times = pd.to_numeric(merged[time_column], errors="coerce")
            if times.isna().any():
                raise ValueError("start_ms 存在缺失或非数值")
            times = times.astype("int64")
            time_source = "start_ms"
        elif time_column == "start_hms":
            times = merged[time_column].map(parse_hms_ms)
            if times.isna().any():
                raise ValueError("start_hms 存在缺失或非法时间")
            times = times.astype("int64")
            time_source = "start_hms"
        elif time_column == "window_id":
            times = pd.to_numeric(merged[time_column], errors="coerce")
            if times.isna().any():
                raise ValueError("window_id 存在缺失或非数值")
            times = times.astype("int64")
            time_source = "window_id"
        else:
            raise ValueError(f"不支持的 time_column: {time_column}")
    else:
        raise ValueError(f"时间列不存在：{time_column}")

    end_times, end_source = choose_end_column(merged, times)
    if end_times is None:
        end_times = pd.Series([pd.NA] * len(merged), index=merged.index, dtype="Int64")

    merged["_sequence_id"] = sequence_values
    merged["_time_ms"] = times
    merged["_end_ms"] = end_times
    merged["_input_order"] = range(len(merged))

    records_by_sequence: dict[str, list[dict]] = {}
    invalid_code_count = 0
    dropped_blank_human = 0
    input_time_out_of_order = False
    duplicate_window_count = 0
    concurrent_rows = []
    overlap_rows = []
    short_sequences = []

    for sequence_id, group in merged.groupby("_sequence_id", sort=True):
        original_times = group.sort_values("_input_order")["_time_ms"].tolist()
        if any(current < previous for previous, current in zip(original_times, original_times[1:])):
            input_time_out_of_order = True
        group = group.sort_values(["_time_ms", "_input_order"], kind="mergesort")
        if group["window_id"].duplicated().any():
            duplicate_window_count += int(group["window_id"].duplicated().sum())
            raise ValueError(
                f"序列 {sequence_id!r} 存在重复 window_id；同一窗口有多个编码，"
                "当前不支持并发行为，请先拆分或明确多标签处理规则。"
            )
        if group["_time_ms"].duplicated().any():
            duplicate_times = group.loc[group["_time_ms"].duplicated(keep=False), "_time_ms"].unique()
            concurrent_rows.extend([int(value) for value in duplicate_times])
            raise ValueError(
                f"序列 {sequence_id!r} 存在同一时间戳多条记录（{list(duplicate_times)} ms）；"
                "当前不支持并发行为，请先提供多标签处理规则。"
            )
        if end_times is not None and group["_end_ms"].notna().all():
            previous_end = group["_end_ms"].shift(1)
            overlap_mask = (previous_end.notna()) & (group["_time_ms"] < previous_end)
            if overlap_mask.any():
                overlap_rows.extend(group.loc[overlap_mask, "window_id"].astype(int).tolist())
                raise ValueError(
                    f"序列 {sequence_id!r} 存在时间窗口重叠（window_id={overlap_rows[:5]}）；"
                    "当前不支持重叠窗口，请先调整窗口或提供并发处理规则。"
                )

        records = []
        previous_time = None
        for _, row in group.iterrows():
            ai_code = str(row.get("code", "")).strip().upper()
            human_code = str(row.get("human_code", "")).strip().upper()
            if source == "human":
                code = human_code
                if not code:
                    dropped_blank_human += 1
                    continue
            elif source == "human_or_ai":
                code = human_code if human_code in VALID_CODES else ai_code
            else:
                code = ai_code
            if code not in VALID_CODES:
                invalid_code_count += 1
                continue
            start = int(row["_time_ms"])
            end_value = row["_end_ms"]
            end = int(end_value) if pd.notna(end_value) else None
            records.append(
                {
                    "sequence_id": str(sequence_id),
                    "window_id": int(row["window_id"]),
                    "start_ms": start,
                    "end_ms": end,
                    "code": code,
                    "text": str(row.get("text", "") or ""),
                }
            )
            previous_time = start
        if len(records) <= lag:
            short_sequences.append(str(sequence_id))
        if records:
            records_by_sequence[str(sequence_id)] = records

    sequences = list(records_by_sequence.values())
    if not sequences:
        raise ValueError("过滤后没有有效行为序列")
    metadata = {
        "analysis_unit": "fixed_time_window_primary_code",
        "event_level": False,
        "sequence_mode": sequence_mode,
        "sequence_column": sequence_column if sequence_mode == "explicit_sequence_id" else None,
        "boundary_assumption": boundary_assumption,
        "sequence_count": len(sequences),
        "sequence_lengths": {str(key): len(value) for key, value in records_by_sequence.items()},
        "time_source": time_source,
        "end_time_source": end_source,
        "time_is_ms": time_source in {"start_ms", "start_hms"},
        "window_seconds": (
            float(pd.to_numeric(merged["duration_s"], errors="coerce").median())
            if "duration_s" in merged.columns
            and pd.to_numeric(merged["duration_s"], errors="coerce").notna().any()
            else None
        ),
        "input_time_out_of_order": input_time_out_of_order,
        "duplicate_window_count": duplicate_window_count,
        "concurrent_timestamps": sorted(set(concurrent_rows)),
        "overlap_window_ids": sorted(set(overlap_rows)),
        "short_sequences": short_sequences,
        "invalid_code_count": invalid_code_count,
        "dropped_blank_human_count": dropped_blank_human,
        "lag": lag,
    }
    return sequences, metadata


def compute_tables_for_sequences(
    sequences: list[list[dict]], lag: int, alpha: float, excluded_codes: set[str] | None = None
) -> dict:
    if lag < 1:
        raise ValueError("lag 必须大于等于 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha 必须在 0 和 1 之间")
    excluded_codes = set(excluded_codes or set())
    all_codes = {record["code"] for sequence in sequences for record in sequence}
    labels = [code for code in VALID_CODES if code in all_codes and code not in excluded_codes]
    if not labels:
        raise ValueError("没有可分析的行为编码")

    frequency_counts = {code: 0 for code in labels}
    transition_counts = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    total_windows = 0
    for sequence in sequences:
        codes = [record["code"] for record in sequence]
        total_windows += sum(1 for code in codes if code in labels)
        for code in codes:
            if code in frequency_counts:
                frequency_counts[code] += 1
        for index in range(len(codes) - lag):
            source, target = codes[index], codes[index + lag]
            if source in labels and target in labels:
                transition_counts.loc[source, target] += 1

    n_transitions = int(transition_counts.to_numpy().sum())
    row_totals = transition_counts.sum(axis=1)
    col_totals = transition_counts.sum(axis=0)
    expected = pd.DataFrame(index=labels, columns=labels, dtype=float)
    residuals = pd.DataFrame(index=labels, columns=labels, dtype=float)
    adjusted = pd.DataFrame(index=labels, columns=labels, dtype=float)
    p_values = pd.DataFrame(index=labels, columns=labels, dtype=float)
    starred = pd.DataFrame(index=labels, columns=labels, dtype=str)
    status = pd.DataFrame(index=labels, columns=labels, dtype=str)
    warning_reason = pd.DataFrame(index=labels, columns=labels, dtype=str)
    testable = pd.DataFrame(index=labels, columns=labels, dtype=bool)
    expected_lt_5 = pd.DataFrame(index=labels, columns=labels, dtype=bool)
    expected_lt_1 = pd.DataFrame(index=labels, columns=labels, dtype=bool)
    probability = pd.DataFrame(index=labels, columns=labels, dtype=float)

    critical = z_critical(alpha)
    for source in labels:
        for target in labels:
            observed = int(transition_counts.loc[source, target])
            row_total = int(row_totals.loc[source])
            col_total = int(col_totals.loc[target])
            reasons = []
            if n_transitions == 0:
                reasons.append("total_transitions=0")
            expected_value = (row_total * col_total / n_transitions) if n_transitions else math.nan
            expected.loc[source, target] = expected_value
            residuals.loc[source, target] = observed - expected_value if n_transitions else math.nan
            probability.loc[source, target] = observed / row_total if row_total else math.nan

            if row_total == 0:
                reasons.append("row_margin=0")
            if col_total == 0:
                reasons.append("col_margin=0")
            if n_transitions and expected_value == 0:
                reasons.append("expected=0")
            denominator = math.nan
            if n_transitions and expected_value > 0:
                denominator = expected_value * (1 - row_total / n_transitions) * (1 - col_total / n_transitions)
                if denominator <= 0:
                    reasons.append("zero_variance")
            is_testable = not reasons
            testable.loc[source, target] = is_testable
            expected_lt5 = bool(n_transitions and expected_value < 5)
            expected_lt1 = bool(n_transitions and expected_value < 1)
            expected_lt_5.loc[source, target] = expected_lt5
            expected_lt_1.loc[source, target] = expected_lt1
            if is_testable:
                z = (observed - expected_value) / math.sqrt(denominator)
                p = math.erfc(abs(z) / math.sqrt(2))
                adjusted.loc[source, target] = z
                p_values.loc[source, target] = p
                starred.loc[source, target] = f"{z:.2f}{stars_for_p(p)}"
                status.loc[source, target] = "testable"
                if expected_lt5:
                    reasons.append("expected_lt_5")
                if expected_lt1:
                    reasons.append("expected_lt_1")
            else:
                adjusted.loc[source, target] = math.nan
                p_values.loc[source, target] = math.nan
                starred.loc[source, target] = "NA"
                status.loc[source, target] = "undefined"
            warning_reason.loc[source, target] = "; ".join(dict.fromkeys(reasons))

    frequency_table = pd.DataFrame(
        {
            "code": labels,
            "code_name": [CODE_NAMES[code] for code in labels],
            "code_group": [CODE_GROUPS[code] for code in labels],
            "frequency": [int(frequency_counts[code]) for code in labels],
            "percentage": [round(frequency_counts[code] / total_windows * 100, 2) if total_windows else math.nan for code in labels],
        }
    ).sort_values(["frequency", "code"], ascending=[False, True])

    diagnostics_rows = []
    for source in labels:
        for target in labels:
            z = float(adjusted.loc[source, target])
            p = float(p_values.loc[source, target])
            diagnostics_rows.append(
                {
                    "from_code": source,
                    "from_name": CODE_NAMES[source],
                    "to_code": target,
                    "to_name": CODE_NAMES[target],
                    "observed": int(transition_counts.loc[source, target]),
                    "expected": float(expected.loc[source, target]),
                    "transition_probability": float(probability.loc[source, target]),
                    "adjusted_residual": z,
                    "raw_p_value": p,
                    "expected_lt_5": bool(expected_lt_5.loc[source, target]),
                    "expected_lt_1": bool(expected_lt_1.loc[source, target]),
                    "testable": bool(testable.loc[source, target]),
                    "warning_reason": str(warning_reason.loc[source, target]),
                }
            )
    diagnostics = pd.DataFrame(diagnostics_rows)

    p_holm = pd.DataFrame(math.nan, index=labels, columns=labels, dtype=float)
    q_bh = pd.DataFrame(math.nan, index=labels, columns=labels, dtype=float)
    sig_raw = pd.DataFrame(False, index=labels, columns=labels, dtype=bool)
    sig_holm = pd.DataFrame(False, index=labels, columns=labels, dtype=bool)
    sig_bh = pd.DataFrame(False, index=labels, columns=labels, dtype=bool)
    family_mask = testable.to_numpy()
    family_p = p_values.to_numpy()[family_mask]
    adjusted_holm = holm_adjust(family_p) if len(family_p) else np.array([], dtype=float)
    adjusted_bh = bh_adjust(family_p) if len(family_p) else np.array([], dtype=float)
    flat_holm = np.full(p_values.shape, math.nan)
    flat_bh = np.full(p_values.shape, math.nan)
    flat_holm[family_mask] = adjusted_holm
    flat_bh[family_mask] = adjusted_bh
    p_holm.iloc[:, :] = flat_holm
    q_bh.iloc[:, :] = flat_bh
    sig_raw.iloc[:, :] = family_mask & (p_values.to_numpy() < alpha)
    sig_holm.iloc[:, :] = family_mask & (flat_holm < alpha)
    sig_bh.iloc[:, :] = family_mask & (flat_bh < alpha)

    diagnostics["p_holm"] = [
        p_holm.loc[row["from_code"], row["to_code"]] for _, row in diagnostics.iterrows()
    ]
    diagnostics["q_bh"] = [
        q_bh.loc[row["from_code"], row["to_code"]] for _, row in diagnostics.iterrows()
    ]
    diagnostics["sig_raw"] = [
        bool(sig_raw.loc[row["from_code"], row["to_code"]]) for _, row in diagnostics.iterrows()
    ]
    diagnostics["sig_holm"] = [
        bool(sig_holm.loc[row["from_code"], row["to_code"]]) for _, row in diagnostics.iterrows()
    ]
    diagnostics["sig_bh"] = [
        bool(sig_bh.loc[row["from_code"], row["to_code"]]) for _, row in diagnostics.iterrows()
    ]
    diagnostics["direction"] = np.where(
        diagnostics["adjusted_residual"] > 0, "positive", "negative"
    )
    diagnostics["negative_suppressed"] = [
        bool(
            testable.loc[row["from_code"], row["to_code"]]
            and p_values.loc[row["from_code"], row["to_code"]] < alpha
            and adjusted.loc[row["from_code"], row["to_code"]] < 0
        )
        for _, row in diagnostics.iterrows()
    ]

    def significant_table(mask_column: str, p_column: str) -> pd.DataFrame:
        selected = diagnostics[diagnostics[mask_column]].copy()
        # Negative significant residuals remain in diagnostics for audit, but
        # are not treated as substantive findings in the main LSA results.
        selected = selected[selected["adjusted_residual"] > 0]
        if selected.empty:
            return pd.DataFrame()
        selected["significance"] = selected[p_column].map(stars_for_p)
        selected["direction"] = "positive"
        selected["adjusted_residual"] = selected["adjusted_residual"].round(3)
        selected["expected"] = selected["expected"].round(3)
        selected["transition_probability"] = selected["transition_probability"].round(4)
        selected = selected.sort_values("adjusted_residual", ascending=False).reset_index(drop=True)
        return selected

    significant_by_method = {
        "raw": significant_table("sig_raw", "raw_p_value"),
        "holm": significant_table("sig_holm", "p_holm"),
        "bh": significant_table("sig_bh", "q_bh"),
    }
    significant = significant_by_method["raw"].copy()
    if not significant.empty:
        significant["significance"] = significant["raw_p_value"].map(stars_for_p)

    return {
        "labels": labels,
        "frequency": frequency_table,
        "transition_counts": transition_counts,
        "transition_probability": probability,
        "expected": expected,
        "residuals": residuals,
        "adjusted_residuals": adjusted,
        "adjusted_residuals_starred": starred,
        "p_values": p_values,
        "p_holm": p_holm,
        "q_bh": q_bh,
        "sig_raw": sig_raw,
        "sig_holm": sig_holm,
        "sig_bh": sig_bh,
        "status": status,
        "warning_reason": warning_reason,
        "testable": testable,
        "expected_lt_5": expected_lt_5,
        "expected_lt_1": expected_lt_1,
        "diagnostics": diagnostics,
        "significant": significant,
        "significant_by_method": significant_by_method,
        "family_size": int(family_mask.sum()),
        "family_excluded": int((~testable.to_numpy()).sum()),
        "total_windows": total_windows,
        "total_transitions": n_transitions,
        "critical": critical,
    }


def compute_tables(sequence: list[str], lag: int, alpha: float, excluded_codes: set[str] | None = None) -> dict:
    records = [{"code": code} for code in sequence]
    return compute_tables_for_sequences([records], lag, alpha, excluded_codes)


def collapse_sequences(
    sequences: list[list[dict]], window_seconds: float | None = None
) -> list[list[dict]]:
    collapsed = []
    for sequence in sequences:
        states = []
        current = None
        for record in sequence:
            if current is None or current["code"] != record["code"]:
                if current is not None:
                    states.append(current)
                current = dict(record)
                current["window_count"] = 1
                current["start_ms"] = record.get("start_ms")
                current["end_ms"] = record.get("end_ms")
            else:
                current["window_count"] += 1
                if record.get("end_ms") is not None:
                    current["end_ms"] = record.get("end_ms")
        if current is not None:
            states.append(current)
        for state in states:
            start = state.get("start_ms")
            end = state.get("end_ms")
            duration = None
            duration_source = "missing"
            if pd.notna(start) and pd.notna(end):
                duration = max(0.0, (float(end) - float(start)) / 1000)
                duration_source = "time_fields"
            elif window_seconds is not None:
                duration = state["window_count"] * float(window_seconds)
                duration_source = "window_count_times_verified_window_length"
            state["duration_s"] = duration
            state["duration_source"] = duration_source
        if states:
            collapsed.append(states)
    return collapsed


def permuted_code_lists(sequences: list[list[dict]], rng: np.random.Generator) -> list[list[str]]:
    return [
        [str(code) for code in rng.permutation([record["code"] for record in sequence])]
        for sequence in sequences
    ]


def monte_carlo_counts(
    sequences: list[list[dict]],
    labels: list[str],
    lag: int,
    simulation_count: int,
    random_seed: int,
) -> np.ndarray:
    if simulation_count < 1:
        raise ValueError("simulation_count 必须大于等于 1")
    code_to_index = {code: index for index, code in enumerate(labels)}
    counts = np.zeros((simulation_count, len(labels), len(labels)), dtype=np.int32)
    rng = np.random.default_rng(random_seed)
    for simulation in range(simulation_count):
        sources = []
        targets = []
        for shuffled in permuted_code_lists(sequences, rng):
            if len(shuffled) <= lag:
                continue
            permuted = np.array(shuffled, dtype=object)
            sources.append(permuted[:-lag])
            targets.append(permuted[lag:])
        if not sources:
            continue
        source_codes = np.concatenate(sources)
        target_codes = np.concatenate(targets)
        source_index = np.array([code_to_index.get(code, -1) for code in source_codes])
        target_index = np.array([code_to_index.get(code, -1) for code in target_codes])
        valid = (source_index >= 0) & (target_index >= 0)
        # np.add.at is required because the same transition can repeat.
        np.add.at(counts[simulation], (source_index[valid], target_index[valid]), 1)
    return counts


def run_monte_carlo(
    tables: dict,
    sequences: list[list[dict]],
    lag: int,
    simulation_count: int,
    random_seed: int,
    alpha: float,
) -> dict:
    labels = tables["labels"]
    simulations = monte_carlo_counts(sequences, labels, lag, simulation_count, random_seed)
    observed = tables["transition_counts"].to_numpy(dtype=int)
    upper = (np.sum(simulations >= observed[None, :, :], axis=0) + 1) / (simulation_count + 1)
    lower = (np.sum(simulations <= observed[None, :, :], axis=0) + 1) / (simulation_count + 1)
    two_sided = np.minimum(1.0, 2 * np.minimum(upper, lower))
    means = simulations.mean(axis=0)
    standard_deviation = simulations.std(axis=0, ddof=0)
    degenerate = np.ptp(simulations, axis=0) == 0
    family_mask = tables["testable"].to_numpy()
    family_p = two_sided[family_mask]
    p_holm = np.full(two_sided.shape, math.nan)
    q_bh = np.full(two_sided.shape, math.nan)
    p_holm[family_mask] = holm_adjust(family_p)
    q_bh[family_mask] = bh_adjust(family_p)
    sig_raw = family_mask & (two_sided < alpha)
    sig_holm = family_mask & (p_holm < alpha)
    sig_bh = family_mask & (q_bh < alpha)

    rows = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            direction = "positive" if observed[i, j] > means[i, j] else "negative"
            rows.append(
                {
                    "from_code": source,
                    "from_name": CODE_NAMES[source],
                    "to_code": target,
                    "to_name": CODE_NAMES[target],
                    "observed": int(observed[i, j]),
                    "simulated_mean": float(means[i, j]),
                    "simulated_sd": float(standard_deviation[i, j]),
                    "empirical_p_upper": float(upper[i, j]),
                    "empirical_p_lower": float(lower[i, j]),
                    "empirical_p_two_sided": float(two_sided[i, j]),
                    "empirical_p_holm": float(p_holm[i, j]) if family_mask[i, j] else math.nan,
                    "empirical_q_bh": float(q_bh[i, j]) if family_mask[i, j] else math.nan,
                    "simulation_significance": bool(sig_raw[i, j]),
                    "simulation_significance_holm": bool(sig_holm[i, j]),
                    "simulation_significance_bh": bool(sig_bh[i, j]),
                    "simulation_degenerate": bool(degenerate[i, j]),
                    "simulation_count": simulation_count,
                    "random_seed": random_seed,
                    "null_model": "within_sequence_permutation_fixed_counts",
                    "expected_lt_5": bool(tables["expected_lt_5"].to_numpy()[i, j]),
                    "expected_lt_1": bool(tables["expected_lt_1"].to_numpy()[i, j]),
                    "testable": bool(family_mask[i, j]),
                    "direction": direction,
                }
            )
    diagnostics = pd.DataFrame(rows)
    diagnostics["negative_suppressed"] = (
        diagnostics["testable"].astype(bool)
        & diagnostics["simulation_significance"].astype(bool)
        & (diagnostics["direction"] == "negative")
    )

    def significant_table(mask_column: str, p_column: str) -> pd.DataFrame:
        selected = diagnostics[diagnostics[mask_column]].copy()
        selected = selected[selected["direction"] == "positive"]
        if selected.empty:
            return pd.DataFrame()
        selected["significance"] = selected[p_column].map(stars_for_p)
        selected["adjusted_residual"] = math.nan
        selected["transition_probability"] = [
            tables["transition_probability"].to_numpy()[labels.index(row["from_code"]), labels.index(row["to_code"])]
            for _, row in selected.iterrows()
        ]
        selected["expected"] = [
            tables["expected"].to_numpy()[labels.index(row["from_code"]), labels.index(row["to_code"])]
            for _, row in selected.iterrows()
        ]
        selected = selected.sort_values("empirical_p_two_sided").reset_index(drop=True)
        return selected

    return {
        "simulations": simulations,
        "upper": upper,
        "lower": lower,
        "two_sided": two_sided,
        "means": means,
        "standard_deviation": standard_deviation,
        "degenerate": degenerate,
        "diagnostics": diagnostics,
        "significant_by_method": {
            "mc_raw": significant_table("simulation_significance", "empirical_p_two_sided"),
            "mc_holm": significant_table("simulation_significance_holm", "empirical_p_holm"),
            "mc_bh": significant_table("simulation_significance_bh", "empirical_q_bh"),
        },
        "simulation_count": simulation_count,
        "random_seed": random_seed,
        "null_model": "within_sequence_permutation_fixed_counts",
    }


def build_dot(
    tables: dict,
    alpha: float,
    edge_mode: str,
    max_edges: int,
    method: str = "holm",
    significant: pd.DataFrame | None = None,
    dictionary: dict[str, dict] | None = None,
    label_mode: str = "code",
    show_self: bool = True,
    min_count: int = 0,
    min_probability: float = 0.0,
    node_frequency: dict[str, int] | None = None,
) -> str:
    labels = tables["labels"]
    if significant is None:
        significant = tables["significant_by_method"].get(method, tables["significant"])
    classic = method in {"raw", "holm", "bh"}
    method_label = {
        "raw": "经典 LSA：原始 p",
        "holm": "经典 LSA：Holm 校正",
        "bh": "经典 LSA：BH-FDR",
        "mc_raw": "Monte Carlo：原始经验 p",
        "mc_holm": "Monte Carlo：Holm 校正",
        "mc_bh": "Monte Carlo：BH-FDR",
    }.get(method, method)
    lines = [
        "digraph behavior_sequence {",
        f'  graph [rankdir="LR", bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.7", labelloc="t", label="{method_label}；蓝色实线=正向；红色虚线=负向；橙色点线=E<5；边标签={"Z, P(j|i)" if classic else "p, P(j|i)"}", fontsize="10"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#F4F6F7", color="#566573", fontname="Microsoft YaHei"];',
        '  edge [fontname="Microsoft YaHei", fontsize="10"];',
    ]
    for code in labels:
        if label_mode == "code" or not dictionary:
            display = code
        else:
            entry = dictionary.get(code, {})
            name = str(entry.get("label", "") or "").strip()
            display = name if label_mode == "label" and name else (f"{code} {name}" if name else code)
        count = (node_frequency or {}).get(code)
        if count is not None:
            display += f"\\n(n={count})"
        lines.append(f'  "{code}" [label="{display}"];')
    if not significant.empty:
        selected = significant.copy()
        if edge_mode == "positive":
            selected = selected[selected["direction"] == "positive"]
        elif edge_mode == "negative":
            selected = selected[selected["direction"] == "negative"]
        if not show_self:
            selected = selected[selected["from_code"] != selected["to_code"]]
        if min_count > 0:
            selected = selected[selected["observed"] >= min_count]
        if min_probability > 0:
            selected = selected[selected["transition_probability"].fillna(0) >= min_probability]
        if classic:
            selected = selected.reindex(
                selected["adjusted_residual"].abs().sort_values(ascending=False).index
            ).head(max_edges)
        else:
            selected = selected.reindex(
                selected["empirical_p_two_sided"].sort_values(ascending=True).index
            ).head(max_edges)
        for _, row in selected.iterrows():
            low_expected = bool(row.get("expected_lt_5", False))
            probability = row.get("transition_probability", math.nan)
            if classic:
                z = float(row["adjusted_residual"])
                label = f"Z={z:.2f}{row.get('significance', '')}"
                strength = abs(z)
                positive = z > 0
            else:
                p = float(row["empirical_p_two_sided"])
                label = f"p={p:.3f}{row.get('significance', '')}"
                strength = max(0.0, -math.log10(max(p, 1e-12)))
                positive = row.get("direction") == "positive"
            if pd.notna(probability):
                label += f"\\nP={float(probability):.2f}"
            if low_expected:
                label += "\\nE<5"
            color = "#B9770E" if low_expected else ("#2471A3" if positive else "#C0392B")
            style = "dotted" if low_expected else ("solid" if positive else "dashed")
            width = min(4.0, 1.0 + strength / 2.5)
            lines.append(
                f'  "{row["from_code"]}" -> "{row["to_code"]}" '
                f'[label="{label}", color="{color}", fontcolor="{color}", '
                f'style="{style}", penwidth="{width:.2f}"];'
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ai_codes")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--human-codes", default=None)
    parser.add_argument("--source", choices=["ai", "human", "human_or_ai"], default="ai")
    parser.add_argument("--sequence-id-column", default="sequence_id")
    parser.add_argument("--single-sequence", action="store_true")
    parser.add_argument("--time-column", default="auto", choices=["auto", "start_ms", "start_hms", "window_id"])
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--exclude-codes", default="")
    parser.add_argument("--edge-mode", choices=["positive", "negative", "both"], default="positive")
    parser.add_argument("--max-edges", type=int, default=40)
    parser.add_argument("--simulation-count", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20261022)
    args = parser.parse_args()

    if args.lag < 1:
        raise ValueError("lag 必须大于等于 1")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha 必须在 0 和 1 之间")
    if args.simulation_count < 1:
        raise ValueError("simulation_count 必须大于等于 1")

    ai_path = Path(args.ai_codes)
    output_dir = Path(args.output_dir) if args.output_dir else ai_path.parent / "behavior_sequence_v02c"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = ai_path.name.removesuffix("_ai_codes.csv")
    ai = pd.read_csv(ai_path)
    human = pd.read_csv(args.human_codes) if args.human_codes and Path(args.human_codes).exists() else None
    exclude = {code.strip().upper() for code in args.exclude_codes.split(",") if code.strip()}

    sequences, sequence_metadata = prepare_sequences(
        ai,
        human,
        args.source,
        args.lag,
        sequence_column=args.sequence_id_column,
        single_sequence=args.single_sequence,
        time_column=args.time_column,
    )
    tables = compute_tables_for_sequences(sequences, args.lag, args.alpha, exclude)
    monte_carlo = run_monte_carlo(
        tables, sequences, args.lag, args.simulation_count, args.random_seed, args.alpha
    )
    collapsed_sequences = collapse_sequences(sequences, sequence_metadata["window_seconds"])
    collapsed_tables = compute_tables_for_sequences(collapsed_sequences, args.lag, args.alpha, exclude)

    frequency_path = output_dir / f"{prefix}_behavior_frequency.csv"
    tables["frequency"].to_csv(frequency_path, index=False, encoding="utf-8-sig")
    transition_path = output_dir / f"{prefix}_transition_frequency.csv"
    tables["transition_counts"].to_csv(transition_path, encoding="utf-8-sig")
    probability_path = output_dir / f"{prefix}_transition_probability.csv"
    tables["transition_probability"].round(4).to_csv(probability_path, encoding="utf-8-sig")
    expected_path = output_dir / f"{prefix}_expected_frequency.csv"
    tables["expected"].round(3).to_csv(expected_path, encoding="utf-8-sig")
    residual_path = output_dir / f"{prefix}_residuals.csv"
    tables["residuals"].round(3).to_csv(residual_path, encoding="utf-8-sig")
    adjusted_path = output_dir / f"{prefix}_adjusted_residuals.csv"
    tables["adjusted_residuals"].round(3).to_csv(adjusted_path, encoding="utf-8-sig")
    p_values_path = output_dir / f"{prefix}_p_values.csv"
    tables["p_values"].to_csv(p_values_path, encoding="utf-8-sig")
    p_holm_path = output_dir / f"{prefix}_p_values_holm.csv"
    tables["p_holm"].to_csv(p_holm_path, encoding="utf-8-sig")
    q_bh_path = output_dir / f"{prefix}_q_values_bh.csv"
    tables["q_bh"].to_csv(q_bh_path, encoding="utf-8-sig")
    starred_path = output_dir / f"{prefix}_adjusted_residuals_starred.csv"
    tables["adjusted_residuals_starred"].to_csv(starred_path, encoding="utf-8-sig")
    status_path = output_dir / f"{prefix}_cell_status.csv"
    tables["status"].to_csv(status_path, encoding="utf-8-sig")
    diagnostics_path = output_dir / f"{prefix}_cell_diagnostics.csv"
    tables["diagnostics"].to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    diagonal = np.diag(tables["transition_counts"].to_numpy())
    self_transitions = int(diagonal.sum())
    total_transitions = int(tables["total_transitions"])
    self_summary = pd.DataFrame(
        [
            {
                "total_transitions": total_transitions,
                "self_transitions": self_transitions,
                "non_self_transitions": total_transitions - self_transitions,
                "self_ratio": round(self_transitions / total_transitions, 4) if total_transitions else math.nan,
                "non_self_ratio": round((total_transitions - self_transitions) / total_transitions, 4) if total_transitions else math.nan,
            }
        ]
    )
    self_summary_path = output_dir / f"{prefix}_self_transition_summary.csv"
    self_summary.to_csv(self_summary_path, index=False, encoding="utf-8-sig")
    persistence = pd.DataFrame(
        [
            {
                "code": code,
                "given_transitions": int(tables["transition_counts"].loc[code].sum()),
                "self_transitions": int(tables["transition_counts"].loc[code, code]),
                "self_ratio": round(
                    tables["transition_counts"].loc[code, code]
                    / tables["transition_counts"].loc[code].sum(),
                    4,
                )
                if tables["transition_counts"].loc[code].sum()
                else math.nan,
            }
            for code in tables["labels"]
        ]
    )
    persistence_path = output_dir / f"{prefix}_persistence_by_code.csv"
    persistence.to_csv(persistence_path, index=False, encoding="utf-8-sig")

    mc_diag = monte_carlo["diagnostics"].rename(
        columns={"direction": "mc_direction", "negative_suppressed": "mc_negative_suppressed"}
    )
    inference = tables["diagnostics"].merge(
        mc_diag[
            [
                "from_code",
                "to_code",
                "simulated_mean",
                "simulated_sd",
                "empirical_p_upper",
                "empirical_p_lower",
                "empirical_p_two_sided",
                "empirical_p_holm",
                "empirical_q_bh",
                "simulation_significance",
                "simulation_significance_holm",
                "simulation_significance_bh",
                "simulation_degenerate",
                "simulation_count",
                "random_seed",
                "null_model",
                "mc_direction",
                "mc_negative_suppressed",
            ]
        ],
        on=["from_code", "to_code"],
        how="left",
    )
    inference_path = output_dir / f"{prefix}_cell_inference.csv"
    inference.to_csv(inference_path, index=False, encoding="utf-8-sig")
    mc_diagnostics_path = output_dir / f"{prefix}_monte_carlo_diagnostics.csv"
    monte_carlo["diagnostics"].to_csv(mc_diagnostics_path, index=False, encoding="utf-8-sig")

    significant_paths = {}
    for method, frame in tables["significant_by_method"].items():
        path = output_dir / f"{prefix}_significant_transitions_{method}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        significant_paths[method] = str(path)
    for method, frame in monte_carlo["significant_by_method"].items():
        path = output_dir / f"{prefix}_significant_transitions_{method}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        significant_paths[method] = str(path)

    graph_paths = {}
    all_methods = ["raw", "holm", "bh", "mc_raw", "mc_holm", "mc_bh"]
    for method in all_methods:
        frame = (
            tables["significant_by_method"].get(method)
            if method in tables["significant_by_method"]
            else monte_carlo["significant_by_method"].get(method)
        )
        dot = build_dot(
            tables,
            args.alpha,
            args.edge_mode,
            args.max_edges,
            method=method,
            significant=frame,
        )
        path = output_dir / f"{prefix}_transition_graph_{method}.dot"
        path.write_text(dot, encoding="utf-8")
        graph_paths[method] = str(path)

    collapsed_frequency_path = output_dir / f"{prefix}_collapsed_frequency.csv"
    collapsed_tables["frequency"].to_csv(collapsed_frequency_path, index=False, encoding="utf-8-sig")
    collapsed_transition_path = output_dir / f"{prefix}_collapsed_transition_frequency.csv"
    collapsed_tables["transition_counts"].to_csv(collapsed_transition_path, encoding="utf-8-sig")
    collapsed_probability_path = output_dir / f"{prefix}_collapsed_transition_probability.csv"
    collapsed_tables["transition_probability"].round(4).to_csv(collapsed_probability_path, encoding="utf-8-sig")
    collapsed_metadata_path = output_dir / f"{prefix}_collapsed_metadata.json"
    collapsed_metadata = {
        "analysis_unit": "behavior_state_segment",
        "source_analysis_unit": "fixed_time_window_state",
        "lag": args.lag,
        "sequence_count": len(collapsed_sequences),
        "sequence_lengths": [len(sequence) for sequence in collapsed_sequences],
        "state_count": collapsed_tables["total_windows"],
        "transition_count": collapsed_tables["total_transitions"],
        "inference": "descriptive_only",
        "warning": "不要将合并状态后的转移概率解释为每 5 秒窗口的转移概率",
    }
    collapsed_metadata_path.write_text(
        json.dumps(collapsed_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    collapsed_rows = []
    for sequence_index, sequence in enumerate(collapsed_sequences, start=1):
        for state_index, state in enumerate(sequence, start=1):
            collapsed_rows.append(
                {
                    "sequence_id": sequence_index,
                    "state_index": state_index,
                    "code": state["code"],
                    "window_count": state["window_count"],
                    "start_ms": state.get("start_ms"),
                    "end_ms": state.get("end_ms"),
                    "duration_s": state.get("duration_s"),
                    "duration_source": state.get("duration_source"),
                    "text": state.get("text", ""),
                }
            )
    collapsed_segments = pd.DataFrame(collapsed_rows)
    collapsed_segments_path = output_dir / f"{prefix}_collapsed_segments.csv"
    collapsed_segments.to_csv(collapsed_segments_path, index=False, encoding="utf-8-sig")
    durations = pd.to_numeric(collapsed_segments["duration_s"], errors="coerce").dropna()
    collapsed_stats = pd.DataFrame(
        [
            {
                "state_count": len(collapsed_segments),
                "duration_available": len(durations),
                "duration_missing": len(collapsed_segments) - len(durations),
                "mean_s": round(float(durations.mean()), 3) if len(durations) else math.nan,
                "median_s": round(float(durations.median()), 3) if len(durations) else math.nan,
                "min_s": round(float(durations.min()), 3) if len(durations) else math.nan,
                "max_s": round(float(durations.max()), 3) if len(durations) else math.nan,
                "q25_s": round(float(durations.quantile(0.25)), 3) if len(durations) else math.nan,
                "q75_s": round(float(durations.quantile(0.75)), 3) if len(durations) else math.nan,
            }
        ]
    )
    collapsed_stats_path = output_dir / f"{prefix}_collapsed_duration_stats.csv"
    collapsed_stats.to_csv(collapsed_stats_path, index=False, encoding="utf-8-sig")

    untestable = int((~tables["testable"]).to_numpy().sum())
    expected_lt5_cells = int(tables["expected_lt_5"].to_numpy().sum())
    expected_lt5_significant = {
        method: int(frame["expected_lt_5"].sum()) if not frame.empty else 0
        for method, frame in {**tables["significant_by_method"], **monte_carlo["significant_by_method"]}.items()
    }

    method_counts = {
        "classical_raw": int(len(tables["significant_by_method"]["raw"])),
        "classical_holm": int(len(tables["significant_by_method"]["holm"])),
        "classical_bh": int(len(tables["significant_by_method"]["bh"])),
        "monte_carlo_raw": int(len(monte_carlo["significant_by_method"]["mc_raw"])),
        "monte_carlo_holm": int(len(monte_carlo["significant_by_method"]["mc_holm"])),
        "monte_carlo_bh": int(len(monte_carlo["significant_by_method"]["mc_bh"])),
    }
    method_summary = pd.DataFrame(
        [
            {
                "method": method,
                "significant_count": len(frame),
                "expected_lt5_significant": int(frame["expected_lt_5"].sum()) if not frame.empty else 0,
            }
            for method, frame in {
                **tables["significant_by_method"],
                **monte_carlo["significant_by_method"],
            }.items()
        ]
    )
    method_summary_path = output_dir / f"{prefix}_method_summary.csv"
    method_summary.to_csv(method_summary_path, index=False, encoding="utf-8-sig")

    classical_diagnostics = tables["diagnostics"]
    mc_diagnostics = monte_carlo["diagnostics"]
    negative_suppressed_counts = {
        "classical_raw": int((classical_diagnostics["sig_raw"] & (classical_diagnostics["direction"] == "negative")).sum()),
        "classical_holm": int((classical_diagnostics["sig_holm"] & (classical_diagnostics["direction"] == "negative")).sum()),
        "classical_bh": int((classical_diagnostics["sig_bh"] & (classical_diagnostics["direction"] == "negative")).sum()),
        "monte_carlo_raw": int((mc_diagnostics["simulation_significance"] & (mc_diagnostics["direction"] == "negative")).sum()),
        "monte_carlo_holm": int((mc_diagnostics["simulation_significance_holm"] & (mc_diagnostics["direction"] == "negative")).sum()),
        "monte_carlo_bh": int((mc_diagnostics["simulation_significance_bh"] & (mc_diagnostics["direction"] == "negative")).sum()),
    }

    edge_sets = {
        method: {
            (row["from_code"], row["to_code"])
            for _, row in frame.iterrows()
        }
        for method, frame in {**tables["significant_by_method"], **monte_carlo["significant_by_method"]}.items()
    }
    all_edges = sorted(set().union(*edge_sets.values())) if edge_sets else []
    overlap_rows = []
    for from_code, to_code in all_edges:
        row = {"from_code": from_code, "to_code": to_code}
        for method in all_methods:
            row[method] = (from_code, to_code) in edge_sets.get(method, set())
        overlap_rows.append(row)
    overlap = pd.DataFrame(overlap_rows)
    overlap_path = output_dir / f"{prefix}_significant_overlap.csv"
    overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")

    warnings = [
        "low_expected_counts_may_invalidate_normal_approximation",
        "adjacent_transitions_share_windows",
        "fixed_window_states_may_be_autocorrelated",
        "monte_carlo_permutation_does_not_preserve_autocorrelation",
        "single_classroom_results_do_not_generalize",
        "significant_transitions_are_not_causal",
    ]
    if sequence_metadata["sequence_mode"] == "single_sequence_assumed":
        warnings.append("no_sequence_id_single_sequence_assumed")

    input_sha256 = hashlib.sha256(ai_path.read_bytes()).hexdigest()
    code_version = "LSA V0.2c"
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version": code_version,
        "input_sha256": input_sha256,
        "ai_codes": str(ai_path),
        "human_codes": str(args.human_codes) if args.human_codes else None,
        "source": args.source,
        "analysis_unit": "fixed_time_window_state",
        "analysis_unit_detail": sequence_metadata["analysis_unit"],
        "event_level": sequence_metadata["event_level"],
        "window_duration_seconds": sequence_metadata["window_seconds"],
        "sequence_mode": sequence_metadata["sequence_mode"],
        "sequence_column": sequence_metadata["sequence_column"],
        "boundary_assumption": sequence_metadata["boundary_assumption"],
        "sequence_count": sequence_metadata["sequence_count"],
        "sequence_lengths": sequence_metadata["sequence_lengths"],
        "time_source": sequence_metadata["time_source"],
        "end_time_source": sequence_metadata["end_time_source"],
        "input_time_out_of_order": sequence_metadata["input_time_out_of_order"],
        "short_sequences": sequence_metadata["short_sequences"],
        "invalid_code_count": sequence_metadata["invalid_code_count"],
        "dropped_blank_human_count": sequence_metadata["dropped_blank_human_count"],
        "lag": args.lag,
        "alpha": args.alpha,
        "exclude_codes": sorted(exclude),
        "multiple_testing_family": "all_testable_directed_transitions",
        "multiple_testing_family_size": tables["family_size"],
        "multiple_testing_family_excluded": tables["family_excluded"],
        "classical_test_method": "adjusted_residual_normal_two_sided",
        "simulation_null_model": "within_sequence_permutation_fixed_counts",
        "simulation_count": args.simulation_count,
        "random_seed": args.random_seed,
        "sparse_expected_count": 5,
        "undefined_cell_count": untestable,
        "total_windows": tables["total_windows"],
        "total_transitions": tables["total_transitions"],
        "testable_cells": int(tables["testable"].to_numpy().sum()),
        "untestable_cells": untestable,
        "expected_lt5_cells": expected_lt5_cells,
        "expected_lt1_cells": int(tables["expected_lt_1"].to_numpy().sum()),
        "expected_lt5_significant": expected_lt5_significant,
        "method_counts": method_counts,
        "negative_suppressed": True,
        "negative_suppressed_rule": "Z <= -1.96; retained in diagnostics only, not substantive findings",
        "negative_suppressed_counts": negative_suppressed_counts,
        "z_critical": tables["critical"],
        "formula": "adjusted residual = (O - E) / sqrt(E * (1 - R/N) * (1 - C/N))",
        "empirical_p_formula": "p = (b + 1) / (B + 1); two-sided = min(1, 2 * min(upper, lower))",
        "warnings": warnings,
        "sensitivity": collapsed_metadata,
        "files": {
            "frequency": str(frequency_path),
            "transition_frequency": str(transition_path),
            "transition_probability": str(probability_path),
            "expected": str(expected_path),
            "residuals": str(residual_path),
            "adjusted_residuals": str(adjusted_path),
            "p_values": str(p_values_path),
            "p_values_holm": str(p_holm_path),
            "q_values_bh": str(q_bh_path),
            "adjusted_residuals_starred": str(starred_path),
            "cell_status": str(status_path),
            "cell_diagnostics": str(diagnostics_path),
            "cell_inference": str(inference_path),
            "monte_carlo_diagnostics": str(mc_diagnostics_path),
            "significant_by_method": significant_paths,
            "graphs": graph_paths,
            "significant_overlap": str(overlap_path),
            "method_summary": str(method_summary_path),
            "self_transition_summary": str(self_summary_path),
            "persistence_by_code": str(persistence_path),
            "collapsed_frequency": str(collapsed_frequency_path),
            "collapsed_transition_frequency": str(collapsed_transition_path),
            "collapsed_transition_probability": str(collapsed_probability_path),
            "collapsed_metadata": str(collapsed_metadata_path),
            "collapsed_segments": str(collapsed_segments_path),
            "collapsed_duration_stats": str(collapsed_stats_path),
        },
    }
    metadata_path = output_dir / f"{prefix}_behavior_sequence_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"分析单位：{metadata['analysis_unit']}（{metadata['analysis_unit_detail']}），Lag {args.lag}")
    print(f"序列数：{metadata['sequence_count']}，窗口数：{metadata['total_windows']}，转换数：{metadata['total_transitions']}")
    print(f"可检验单元：{metadata['testable_cells']}，不可检验：{metadata['untestable_cells']}")
    print("显著转换数：" + json.dumps(method_counts, ensure_ascii=False))
    print(f"方法对照：{overlap_path}")
    print(f"经典残差：{adjusted_path}")
    print(f"推断明细：{inference_path}")
    print(f"敏感性：{collapsed_transition_path}")
    print(f"元数据：{metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
