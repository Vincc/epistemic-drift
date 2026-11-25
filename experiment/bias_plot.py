# Plots for the biased-decision-log runs, from the summaries bias_analyze.py writes into each run file.
# Single-decision runs: gap against p, one panel per attribute and kind of log, one line per model.
# Chain runs: gap by turn (rolling mean), one panel per kind of log; for --memory file also the record's own gap.
# Usage: uv run --with matplotlib python experiment/bias_plot.py RUN.json [RUN.json ...] [--out results/bias/plots]
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="results/bias/plots")
ap.add_argument("--window", type=int, default=5)  # chain plots: turns per rolling mean
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {"claude-opus-5-5": ("Opus 5.5", "#2a78d6", "o"), "gpt-4.1": ("GPT-4.1", "#eb6834", "s"),
          "gpt-6.1-sol": ("GPT-6.1-sol", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ATTRIBUTE = {"gender": "gender (favoured: male)", "gender_female": "gender (favoured: female)",
             "gpa_refs": "GPA vs references (favoured: lower GPA, higher refs)", "age": "age (favoured: 24)"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def style(ax):
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


runs = [json.load(open(f, encoding="utf-8")) | {"_file": Path(f).stem} for f in args.runs]
for r in runs:
    assert r.get("experiment") == "bias-log" and isinstance(r.get("summary"), dict), f"{r['_file']}: run bias_analyze.py first"

# ---- Single decision: gap against p ----
single = [r for r in runs if not r.get("chain")]
if single:
    rows = defaultdict(list)  # (attribute, kind) -> rows; a control row goes in both kinds at p = 0
    n_pairs = defaultdict(set)
    for r in single:
        for row in r["summary"]["rows"]:
            row = row | {"file": r["_file"]}
            n_pairs[r["attribute"]].add(row["pairs"])
            for kind in (["correlational", "causal"] if row["kind"] == "control" else [row["kind"]]):
                rows[(r["attribute"], kind)].append(row)
    attributes = sorted({a for a, _ in rows}, key=list(ATTRIBUTE).index)
    fig, axes = plt.subplots(len(attributes), 2, figsize=(10, 3.4 * len(attributes)), squeeze=False, sharey=True)
    offset = {m: (i - 1) * 0.025 for i, m in enumerate(MODELS)}
    for i, a in enumerate(attributes):
        for j, kind in enumerate(["correlational", "causal"]):
            ax = axes[i][j]
            style(ax)
            for m, (label, colour, marker) in MODELS.items():
                xs_ = [x for x in rows[(a, kind)] if x["model"] == m]
                if not xs_:
                    continue
                # Rows from a file other than the model's main one (an earlier pilot) are drawn hollow, unjoined,
                # without their control.
                files = [x["file"] for x in xs_]
                main = max(files, key=files.count)
                pts = sorted(((x["p"], x) for x in xs_ if x["file"] == main or x["p"] > 0), key=lambda t: t[0])
                xs = [p + offset[m] for p, x in pts if x["file"] == main]
                ax.plot(xs, [x["gap"] for _, x in pts if x["file"] == main], color=colour, lw=2, marker=marker,
                        ms=7, label=label, zorder=3)
                for p, x in pts:
                    lo, hi = x["gap_ci"]
                    ax.vlines(p + offset[m], lo, hi, color=colour, lw=1.5, zorder=2)
                    if x["file"] != main:
                        ax.plot(p + offset[m], x["gap"], marker=marker, ms=7, mfc="#fcfcfb", mec=colour, mew=2,
                                ls="none", zorder=3)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["control", "0.25", "0.5", "0.75", "1"])
            ax.set_xlim(-0.08, 1.08)
            ax.set_title(f"{ATTRIBUTE.get(a, a)}\n{kind} log", fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("gap = admit(favoured) − admit(other)")
            if i == len(attributes) - 1:
                ax.set_xlabel("bias rate p")
    axes[0][0].legend(frameon=False, loc="lower left")
    pairs = ", ".join(f"{ATTRIBUTE.get(a, a).split(' (')[0]} n={'/'.join(map(str, sorted(n_pairs[a])))}" for a in attributes)
    fig.text(0.01, 0.003, f"Bars: 95% bootstrap CI over test profiles ({pairs} pairs). Hollow markers: an earlier pilot "
             "(different profile set).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = OUT / "single_gap_vs_p.png"
    fig.savefig(path, dpi=150)
    print("wrote", path)

# ---- Chains: gap by turn ----
for r in (r for r in runs if r.get("chain")):
    s = r["summary"]
    K = r["chain"]
    W = max(1, min(args.window, K // 4))  # short chains get a shorter window
    kinds = ["control", "correlational", "causal"]
    # The record's own gap after each turn (memory file only): starting log plus the decisions appended so far.
    record = defaultdict(lambda: [[] for _ in range(K + 1)])
    if r.get("memory") == "file":
        for c in r["conversations"]:
            if len(c["turns"]) != K:
                continue
            entries = [(v, d) for _, v, d in c["source"]["log"]]
            fav = r["variants"]["favoured"]
            for t in range(K + 1):
                if t:
                    turn = c["turns"][t - 1]
                    if turn.get("logged"):
                        entries.append((turn["variant"], turn["logged"]))
                f = [d == "admit" for v, d in entries if v == fav]
                o = [d == "admit" for v, d in entries if v != fav]
                record[(c["model"], c["source"]["kind"])][t].append(sum(f) / len(f) - sum(o) / len(o))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, kind in zip(axes, kinds):
        style(ax)
        for row in s["by_position"]:
            if row["kind"] != kind:
                continue
            label, colour, marker = MODELS[row["model"]]
            g = row["gap"]
            xs = list(range(W, len(g) + 1))
            ax.plot(xs, [sum(g[x - W:x]) / W for x in xs], color=colour, lw=2, label=label, zorder=3)
            rec = record.get((row["model"], kind))
            if rec:
                ax.plot(range(K + 1), [sum(v) / len(v) for v in rec], color=colour, lw=1.2, ls="--", zorder=2)
        ax.set_title(f"{kind} log" + ("" if kind == "control" else " (p = 1)"), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("turn")
        ax.set_xlim(0, K)
    axes[0].set_ylabel("gap = admit(favoured) − admit(other)")
    axes[0].legend(frameon=False, loc="upper left")
    memo = ("solid: the model's gap, rolling mean over " f"{W} turns; dashed: the gap in decisions.md itself "
            "(starting log + appended decisions)") if r.get("memory") == "file" else \
        f"rolling mean over {W} turns; earlier turns stay in the conversation"
    fig.suptitle(f"{ATTRIBUTE.get(r['attribute'], r['attribute'])}: {K}-turn chains, {r.get('log_size', 40)}-entry "
                 f"starting log, memory = {r.get('memory', 'history')}", fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.003, memo, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = OUT / f"chain_{r['_file']}.png"
    fig.savefig(path, dpi=150)
    print("wrote", path)
