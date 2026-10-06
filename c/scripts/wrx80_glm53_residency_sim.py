#!/usr/bin/env python3
"""Compare GLM-5.3 resident-expert admission policies from .coli_usage history."""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median


def load_usage(path: Path):
    dims = None
    counts = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            a, b, c = (int(x) for x in line.split())
            if a == -1:
                dims = (b, c)
            elif a >= 0 and c > 0:
                counts[a][b] = c
    if dims is None:
        raise SystemExit("missing -1 dimension header")
    return dims, counts


def take_global(layers, budget):
    items = []
    for layer, layer_counts in layers.items():
        for eid, count in layer_counts.items():
            items.append((count, layer, eid))
    items.sort(key=lambda x: (-x[0], x[1], x[2]))
    return {(layer, eid) for _, layer, eid in items[:budget]}


def take_normalized(layers, budget):
    items = []
    for layer, layer_counts in layers.items():
        total = sum(layer_counts.values())
        if not total:
            continue
        for eid, count in layer_counts.items():
            items.append((count / total, count, layer, eid))
    items.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    return {(layer, eid) for _, _, layer, eid in items[:budget]}


def take_round_robin(layers, budget):
    ranked = {}
    for layer, layer_counts in layers.items():
        ranked[layer] = sorted(layer_counts, key=lambda eid: (-layer_counts[eid], eid))
    chosen = set()
    rank = 0
    layer_ids = sorted(ranked)
    while len(chosen) < budget:
        added = 0
        for layer in layer_ids:
            if len(chosen) >= budget:
                break
            ids = ranked[layer]
            if rank < len(ids):
                chosen.add((layer, ids[rank]))
                added += 1
        if not added:
            break
        rank += 1
    return chosen


def summarize(name, layers, chosen):
    total = sum(sum(c.values()) for c in layers.values())
    covered = 0
    per_layer = []
    expert_counts = []
    for layer in sorted(layers):
        lc = layers[layer]
        lt = sum(lc.values())
        lh = sum(count for eid, count in lc.items() if (layer, eid) in chosen)
        ne = sum(1 for eid in lc if (layer, eid) in chosen)
        covered += lh
        if lt:
            per_layer.append((layer, 100.0 * lh / lt))
            expert_counts.append((layer, ne))
    values = [x[1] for x in per_layer]
    min_layer, min_pct = min(per_layer, key=lambda x: x[1])
    max_layer, max_pct = max(per_layer, key=lambda x: x[1])
    min_exp = min(expert_counts, key=lambda x: x[1])
    max_exp = max(expert_counts, key=lambda x: x[1])
    print(
        f"{name:12s} experts={len(chosen):4d} total={100.0*covered/total:6.2f}% "
        f"min=L{min_layer}:{min_pct:6.2f}% median={median(values):6.2f}% "
        f"max=L{max_layer}:{max_pct:6.2f}% experts/layer={min_exp[1]}..{max_exp[1]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usage", type=Path)
    ap.add_argument("--resident-experts", type=int, default=1142)
    ap.add_argument("--first-layer", type=int, default=3)
    args = ap.parse_args()

    (n_layers, n_experts), counts = load_usage(args.usage)
    layers = {
        layer: counts.get(layer, {})
        for layer in range(args.first_layer, n_layers)
        if counts.get(layer)
    }
    if not layers:
        raise SystemExit("no routed layers in requested range")

    print(
        f"usage={args.usage} layers={len(layers)} model_layers={n_layers} "
        f"experts/layer={n_experts} resident_budget={args.resident_experts}"
    )
    summarize("global", layers, take_global(layers, args.resident_experts))
    summarize("normalized", layers, take_normalized(layers, args.resident_experts))
    summarize("round_robin", layers, take_round_robin(layers, args.resident_experts))


if __name__ == "__main__":
    main()
