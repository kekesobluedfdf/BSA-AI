from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render_transition_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    title: str,
    label_mode: str = "both",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()
    for _, row in nodes.iterrows():
        code = str(row["code"])
        graph.add_node(code)
        graph.nodes[code]["label"] = str(row.get("display_label", code))
        graph.nodes[code]["frequency"] = int(row.get("frequency", 0))
    for _, row in edges.iterrows():
        source = str(row["from_code"])
        target = str(row["to_code"])
        if source not in graph:
            graph.add_node(source, label=source, frequency=0)
        if target not in graph:
            graph.add_node(target, label=target, frequency=0)
        graph.add_edge(
            source,
            target,
            observed=int(row.get("observed", 0)),
            probability=float(row.get("transition_probability", 0) or 0),
            direction=str(row.get("direction", "positive")),
            low_expected=bool(row.get("expected_lt_5", False)),
            weight=float(row.get("weight", 1)),
        )

    fig, axis = plt.subplots(figsize=(12, 8))
    axis.set_title(title, fontsize=14)
    axis.axis("off")
    if graph.number_of_nodes():
        positions = nx.spring_layout(graph, seed=42, k=1.3, iterations=200)
        max_frequency = max((graph.nodes[node]["frequency"] for node in graph), default=1)
        sizes = [
            800 + 1800 * graph.nodes[node]["frequency"] / max_frequency
            for node in graph
        ]
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_size=sizes,
            node_color="#D6EAF8",
            edgecolors="#2874A6",
            linewidths=1.5,
        )
        labels = {
            node: f"{graph.nodes[node]['label']}\n(n={graph.nodes[node]['frequency']})"
            for node in graph
        }
        nx.draw_networkx_labels(graph, positions, labels=labels, ax=axis, font_size=8)
        for source, target, data in graph.edges(data=True):
            color = "#B9770E" if data["low_expected"] else ("#2471A3" if data["direction"] == "positive" else "#C0392B")
            style = ":" if data["low_expected"] else ("-" if data["direction"] == "positive" else "--")
            width = max(0.8, min(4.0, data["weight"]))
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target)],
                ax=axis,
                edge_color=color,
                style=style,
                width=width,
                arrows=True,
                arrowsize=18,
                connectionstyle="arc3,rad=0.16" if source == target else "arc3,rad=0.05",
            )
            midpoint = (positions[source] + positions[target]) / 2
            axis.text(
                midpoint[0],
                midpoint[1],
                f"n={data['observed']}, P={data['probability']:.2f}",
                fontsize=6,
                color=color,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
            )
    svg_path = output_dir / f"{prefix}.svg"
    png_path = output_dir / f"{prefix}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"svg": str(svg_path), "png": str(png_path)}
