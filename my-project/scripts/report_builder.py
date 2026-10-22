from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from coding_dictionary import label_for, load_dictionary
from graph_export import render_transition_graph

METHOD_LABELS = {
    "raw": "经典 LSA：原始 p",
    "holm": "经典 LSA：Holm 校正",
    "bh": "经典 LSA：BH-FDR",
    "mc_raw": "Monte Carlo：原始经验 p",
    "mc_holm": "Monte Carlo：Holm 校正",
    "mc_bh": "Monte Carlo：BH-FDR",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def table_html(frame: pd.DataFrame, classes: str = "data") -> str:
    if frame.empty:
        return "<p class='muted'>无数据</p>"
    return frame.to_html(index=False, classes=classes, border=0, escape=True, na_rep="NA")


def matrix_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p class='muted'>无数据</p>"
    return frame.to_html(classes="matrix", border=0, escape=True, na_rep="NA")


def build_interpretations(significant: pd.DataFrame, method: str, dictionary: dict, single_classroom: bool) -> pd.DataFrame:
    rows = []
    for _, row in significant.iterrows():
        source = label_for(row["from_code"], dictionary, "both")
        target = label_for(row["to_code"], dictionary, "both")
        observed = int(row["observed"])
        probability = row.get("transition_probability", float("nan"))
        if method in {"raw", "holm", "bh"}:
            statistic = (
                f"Z={float(row['adjusted_residual']):.2f}; "
                f"原始 p={float(row['raw_p_value']):.4g}; "
                f"Holm p={float(row['p_holm']):.4g}; "
                f"BH q={float(row['q_bh']):.4g}"
            )
        else:
            statistic = (
                f"经验双侧 p={float(row['empirical_p_two_sided']):.4g}; "
                f"Holm p={float(row['empirical_p_holm']):.4g}; "
                f"BH q={float(row['empirical_q_bh']):.4g}; "
                f"模拟均值={float(row['simulated_mean']):.3f}"
            )
        text = (
            f"在本次课堂观察中，行为 {source} 后紧邻出现行为 {target} 的观察频数为 {observed}，"
            f"条件转移概率为 {float(probability):.2f}。基于 {METHOD_LABELS.get(method, method)}，"
            f"该转移的统计结果为：{statistic}。"
        )
        if bool(row.get("expected_lt_5", False)):
            text += " 该单元期望频数小于 5，正态近似或渐近推断的可靠性有限，应谨慎解释。"
        if method.startswith("mc_"):
            text += " 该结果采用序列内随机排列零假设，不保留原始自相关。"
        if single_classroom:
            text += " 当前仅有一条课堂观察序列，不能直接外推到其他课堂。"
        rows.append(
            {
                "from_code": row["from_code"],
                "to_code": row["to_code"],
                "interpretation": text,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir")
    parser.add_argument("--method", default="holm", choices=list(METHOD_LABELS))
    parser.add_argument("--dictionary", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    directory = Path(args.analysis_dir)
    metadata_path = directory / "lesson_behavior_sequence_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dictionary = load_dictionary(Path(args.dictionary)) if args.dictionary else {}
    method = args.method
    method_label = METHOD_LABELS[method]
    significant = read_csv(directory / f"lesson_significant_transitions_{method}.csv")
    frequency = read_csv(directory / "lesson_behavior_frequency.csv")
    transition = pd.read_csv(directory / "lesson_transition_frequency.csv", index_col=0) if (directory / "lesson_transition_frequency.csv").exists() else pd.DataFrame()
    probability = pd.read_csv(directory / "lesson_transition_probability.csv", index_col=0) if (directory / "lesson_transition_probability.csv").exists() else pd.DataFrame()
    method_summary = read_csv(directory / "lesson_method_summary.csv")
    overlap = read_csv(directory / "lesson_significant_overlap.csv")
    self_summary = read_csv(directory / "lesson_self_transition_summary.csv")
    persistence = read_csv(directory / "lesson_persistence_by_code.csv")
    diagnostics = read_csv(directory / "lesson_cell_inference.csv")
    low_expected = diagnostics[diagnostics.get("expected_lt_5", False).astype(bool)] if not diagnostics.empty else pd.DataFrame()
    collapsed_frequency = read_csv(directory / "lesson_collapsed_frequency.csv")
    collapsed_transition = pd.read_csv(directory / "lesson_collapsed_transition_frequency.csv", index_col=0) if (directory / "lesson_collapsed_transition_frequency.csv").exists() else pd.DataFrame()
    collapsed_probability = pd.read_csv(directory / "lesson_collapsed_transition_probability.csv", index_col=0) if (directory / "lesson_collapsed_transition_probability.csv").exists() else pd.DataFrame()
    collapsed_stats = read_csv(directory / "lesson_collapsed_duration_stats.csv")
    collapsed_metadata = json.loads((directory / "lesson_collapsed_metadata.json").read_text(encoding="utf-8")) if (directory / "lesson_collapsed_metadata.json").exists() else {}

    single_classroom = metadata.get("sequence_count", 0) == 1
    interpretations = build_interpretations(significant, method, dictionary, single_classroom)

    nodes = frequency.copy()
    nodes["display_label"] = [label_for(code, dictionary, "both") for code in nodes["code"]] if not nodes.empty else []
    edge_rows = []
    for _, row in significant.iterrows():
        if method in {"raw", "holm", "bh"}:
            direction = "positive" if float(row.get("adjusted_residual", 0)) > 0 else "negative"
            weight = abs(float(row.get("adjusted_residual", 0)))
        else:
            direction = str(row.get("direction", "positive"))
            p_value = row.get("empirical_p_two_sided")
            weight = max(0.5, -math.log10(max(float(p_value), 1e-12))) if pd.notna(p_value) else 0.5
        edge_rows.append(
            {
                "from_code": row["from_code"],
                "to_code": row["to_code"],
                "observed": row.get("observed", 0),
                "transition_probability": row.get("transition_probability", 0),
                "direction": direction,
                "expected_lt_5": bool(row.get("expected_lt_5", False)),
                "weight": weight,
            }
        )
    edges = pd.DataFrame(edge_rows)
    graph_paths = render_transition_graph(
        nodes,
        edges,
        directory,
        "lesson_report_graph",
        f"{method_label} 显著转移图",
    ) if not nodes.empty else {"svg": "", "png": ""}

    title = "ClassroomResearchAI LSA 研究报告"
    html_parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif;margin:32px;color:#222;line-height:1.6;max-width:1400px}",
        "h1,h2{border-bottom:2px solid #eee;padding-bottom:4px} table{border-collapse:collapse;margin:8px 0 18px;font-size:12px}",
        "table.data th,table.data td{border:1px solid #ddd;padding:4px 6px;vertical-align:top} table.matrix th,table.matrix td{border:1px solid #eee;padding:3px 5px;text-align:center}",
        ".muted{color:#777}.warning{background:#fff4e5;border-left:4px solid #b9770e;padding:8px 12px}.note{background:#eef6fb;border-left:4px solid #2471a3;padding:8px 12px}",
        "img{max-width:100%;border:1px solid #ddd;margin:8px 0}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>生成时间：{html.escape(datetime.now().astimezone().isoformat(timespec='seconds'))}</p>",
        f"<p>代码版本：{html.escape(str(metadata.get('code_version', 'LSA V0.2c')))}；"
        f"输入 SHA256：<code>{html.escape(str(metadata.get('input_sha256', 'NA')))}</code>；"
        f"随机种子：<code>{metadata.get('random_seed', 'NA')}</code></p>",
        "<h2>1. 数据来源与分析单位</h2>",
        f"<p>分析单位：<code>{html.escape(str(metadata.get('analysis_unit')))}</code>；"
        f"序列模式：<code>{html.escape(str(metadata.get('sequence_mode')))}</code>；"
        f"边界假设：{html.escape(str(metadata.get('boundary_assumption')))}。</p>",
        f"<p>窗口数：{metadata.get('total_windows')}；转移数：{metadata.get('total_transitions')}；"
        f"独立序列数：{metadata.get('sequence_count')}；Lag：{metadata.get('lag')}；"
        f"窗口长度：{metadata.get('window_duration_seconds')} 秒。</p>",
        "<div class='warning'>本报告是固定时间窗口状态转移分析，不是事件级行为序列分析；显著转移不代表因果关系。</div>",
        "<h2>2. 编码体系</h2>",
    ]
    if dictionary:
        dictionary_rows = pd.DataFrame(
            [
                {"code": key, "label": value.get("label", ""), "definition": value.get("definition", ""), "category": value.get("category", "")}
                for key, value in dictionary.items()
            ]
        )
        html_parts.append(table_html(dictionary_rows))
    else:
        html_parts.append("<p class='muted'>未提供行为编码字典；图中使用原始 A–P 编码。</p>")

    html_parts += [
        "<h2>3. 行为频次与转换</h2>",
        "<h3>行为频次</h3>",
        table_html(frequency),
        "<h3>转移频数矩阵</h3>",
        matrix_html(transition),
        "<h3>条件转移概率 P(j|i)</h3>",
        matrix_html(probability),
        "<h2>4. 统计方法</h2>",
        f"<p>当前报告方法：<strong>{html.escape(method_label)}</strong>；alpha={metadata.get('alpha')}；"
        f"经典方法：{html.escape(str(metadata.get('classical_test_method')))}；"
        f"Monte Carlo 零假设：{html.escape(str(metadata.get('simulation_null_model')))}；"
        f"模拟次数：{metadata.get('simulation_count')}。</p>",
        "<p>多重比较检验族：所有可定义的有向转移单元；不可定义单元不参与校正。</p>",
        "<h3>各方法显著转移数量</h3>",
        table_html(method_summary),
        "<h3>当前方法显著转移</h3>",
        table_html(significant),
        "<h2>5. 研究解释（按当前方法）</h2>",
        table_html(interpretations),
        "<h2>6. 低期望频数诊断</h2>",
        "<div class='warning'>E&lt;5 不等于不显著；它表示正态近似或渐近推断需要谨慎解释。</div>",
        table_html(low_expected),
        "<h2>7. 自转移与非自转移</h2>",
        table_html(self_summary),
        table_html(persistence),
        "<h2>8. 方法对照</h2>",
        table_html(method_summary),
        table_html(overlap),
        "<h2>9. 状态段合并敏感性分析</h2>",
        f"<p>分析单位：<code>{html.escape(str(collapsed_metadata.get('analysis_unit')))}</code>；"
        f"状态段数：{collapsed_metadata.get('state_count')}；"
        f"状态转移数：{collapsed_metadata.get('transition_count')}。仅描述性，不继承显著性。</p>",
        table_html(collapsed_stats),
        table_html(collapsed_frequency),
        "<h3>状态段转换频数</h3>",
        matrix_html(collapsed_transition),
        "<h3>状态段条件转移概率</h3>",
        matrix_html(collapsed_probability),
        "<h2>10. 转换图</h2>",
    ]
    if graph_paths.get("svg"):
        html_parts.append(f"<img src='{html.escape(Path(graph_paths['svg']).name)}' alt='transition graph'>")
    else:
        html_parts.append("<p class='muted'>未生成图形。</p>")

    html_parts += [
        "<h2>11. 统计限制</h2>",
        "<ul>",
        "<li>低期望频数可能使正态近似不可靠。</li>",
        "<li>相邻转移共享观察窗口，不是独立观测。</li>",
        "<li>固定时间窗口可能存在行为持续性与自相关。</li>",
        "<li>Monte Carlo 序列内随机排列不保留原始自相关。</li>",
        "<li>单课堂结果不能直接推广到其他课堂。</li>",
        "<li>负向显著（Z ≤ -1.96）只保留在诊断表中，不进入主结果。</li>",
        "<li>显著转移不代表因果关系。</li>",
        "</ul>",
        "<h2>12. 可复现参数</h2>",
        "<pre>",
        html.escape(json.dumps(
            {
                "code_version": metadata.get("code_version"),
                "input_sha256": metadata.get("input_sha256"),
                "random_seed": metadata.get("random_seed"),
                "simulation_count": metadata.get("simulation_count"),
                "lag": metadata.get("lag"),
                "alpha": metadata.get("alpha"),
                "method": method,
            },
            ensure_ascii=False,
            indent=2,
        )),
        "</pre>",
        "</body></html>",
    ]
    output = Path(args.output) if args.output else directory / f"lesson_report_{method}.html"
    output.write_text("\n".join(html_parts), encoding="utf-8")
    data_path = output.with_suffix(".json")
    data_path.write_text(
        json.dumps(
            {
                "method": method,
                "metadata": metadata,
                "graph": graph_paths,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
