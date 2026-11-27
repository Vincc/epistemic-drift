# Nonlinear surrogate models of each AI model's admission policy, per log condition, to check what the linear policy
# model (bias_policy.py) may miss: thresholds, saturation and interactions. For each model and condition:
#   - fidelity: 5-fold cross-validated AUC (folds by applicant) of a linear logistic model, an additive model (boosted
#     stumps: one shape function per field, no interactions, monotone increasing in the merit fields) and a full boosted
#     model (depth-3 trees, interactions allowed; the accuracy ceiling);
#   - shape functions of the additive model, centred at each field's median and overlaid across conditions, so a field
#     that starts to matter more shows a steeper curve whatever the overall leniency;
#   - each field's importance as the spread of its shape function between the 5th and 95th percentile of applicants;
#   - a depth-3 decision tree per condition, printed as rules.
# Usage: uv run --with scikit-learn --with pandas --with matplotlib python experiment/bias_surrogate.py RUN.json
#            [--baseline NOLOG.json] [--models M ...]
import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--baseline")
ap.add_argument("--models", nargs="+")
ap.add_argument("--plot", default="results/bias/plots/13_policy_shapes.png")
args = ap.parse_args()

FEATURES = {"gpa": "GPA", "gre_verbal": "GRE V", "gre_quant": "GRE Q", "gre_writing": "GRE AW",
            "toefl_total": "TOEFL", "refs": "references", "age": "age"}
MONOTONE = {"gpa": 1, "gre_verbal": 1, "gre_quant": 1, "gre_writing": 1, "toefl_total": 1, "refs": 1, "age": 0}
DECISION = re.compile(r"(?<![\w-])(admit|reject)(?![\w-])", re.IGNORECASE)
NAMES = {"claude-opus-5-5": "Opus 5.5", "gpt-4.1": "GPT-4.1", "gpt-6.1-sol": "GPT-6.1-sol"}
COND_C = {"nolog": "#1baf7a", "control": "#8f8e88", "correlational": "#6250d6", "causal": "#e34948"}


def load(path, cond_of):
    run = json.load(open(path, encoding="utf-8"))
    out = []
    for c in run["conversations"]:
        s = c["source"]
        cond = cond_of(s)
        if cond is None or (args.models and c["model"] not in args.models):
            continue
        lines = c["turns"][0]["assistant"].strip().splitlines()
        m = DECISION.search(lines[0]) if lines else None
        if m:
            p = run["profiles"][s["profile"]]
            out.append({"model": c["model"], "cond": cond, "applicant": s["profile"], "favoured": int(s["favoured"]),
                        "admit": int(m[1].lower() == "admit"), **{k: p[k] for k in FEATURES}})
    return out, run


rows, run = load(args.run, lambda s: s["kind"] if s["p"] in (0, 1.0) and s["kind"] in COND_C else None)
if args.baseline:
    rows += load(args.baseline, lambda s: "nolog" if s["kind"] == "control" else None)[0]
df = pd.DataFrame(rows)
conds = [c for c in COND_C if c in set(df["cond"])]
cols = list(FEATURES) + ["favoured"]
mono = [MONOTONE[k] for k in FEATURES] + [0]


def models():
    return {"linear": LogisticRegression(max_iter=2000),
            "additive": HistGradientBoostingClassifier(max_depth=1, learning_rate=0.05, max_iter=300, min_samples_leaf=20,
                                                       monotonic_cst=mono, random_state=0),
            "boosted": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
                                                      random_state=0)}


def cv_auc(d, name):
    apps = d["applicant"].unique()
    fold = dict(zip(apps, np.random.default_rng(0).permutation(len(apps)) % 5))
    f = d["applicant"].map(fold).to_numpy()
    pred = np.zeros(len(d))
    for k in range(5):
        tr, te = f != k, f == k
        X = d[cols].to_numpy(float)
        if name == "linear":
            sc = StandardScaler().fit(X[tr])
            m = models()[name].fit(sc.transform(X[tr]), d["admit"].to_numpy()[tr])
            pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m = models()[name].fit(X[tr], d["admit"].to_numpy()[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(d["admit"], pred)


def shape(model, d, k, grid):
    # Additive model: the log-odds as one field moves over its grid, everything else at the applicants' medians
    # (exactly that field's shape function, since the model has no interactions).
    base = d[cols].median().to_frame().T.astype(float)
    X = pd.concat([base] * len(grid), ignore_index=True)
    X[k] = grid
    return model.decision_function(X[cols].to_numpy(float))


apps = df.drop_duplicates("applicant")
grids = {k: np.linspace(apps[k].quantile(0.05), apps[k].quantile(0.95), 25) for k in FEATURES}
report = {}
for model_id, dm in df.groupby("model"):
    print(f"\n== {NAMES.get(model_id, model_id)}")
    fid, imp, shapes, trees = {}, {}, {}, {}
    for cond in conds:
        d = dm[dm["cond"] == cond].reset_index(drop=True)
        if d["admit"].nunique() < 2:
            continue
        fid[cond] = {name: cv_auc(d, name) for name in ("linear", "additive", "boosted")}
        add = models()["additive"].fit(d[cols].to_numpy(float), d["admit"])
        shapes[cond] = {}
        for k in FEATURES:
            s = shape(add, d, k, grids[k])
            shapes[cond][k] = s - np.interp(apps[k].median(), grids[k], s)
        imp[cond] = {k: float(shapes[cond][k].max() - shapes[cond][k].min()) for k in FEATURES}
        g = shape(add, d, "favoured", np.array([0.0, 1.0]))
        imp[cond]["favoured"] = float(g[1] - g[0])
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=0).fit(d[cols], d["admit"])
        trees[cond] = export_text(tree, feature_names=[FEATURES.get(c, "male" if c == "favoured" else c) for c in cols],
                                  show_weights=False, decimals=2)
    report[model_id] = {"fidelity": fid, "importance": imp, "shapes": shapes}
    print("   cross-validated AUC   " + "  ".join(f"{c:>13s}" for c in fid))
    for name in ("linear", "additive", "boosted"):
        print(f"   {name:20s}  " + "  ".join(f"{fid[c][name]:13.3f}" for c in fid))
    print("   field importance (log-odds spread of the additive shape over the 5th-95th percentile; male: step)")
    for k in list(FEATURES) + ["favoured"]:
        label = FEATURES.get(k, "male (favoured)")
        print(f"   {label:20s}  " + "  ".join(f"{imp[c][k]:+13.2f}" if k == "favoured" else f"{imp[c][k]:13.2f}" for c in imp))
    for cond, txt in trees.items():
        print(f"   -- depth-3 tree, {cond}:")
        print("\n".join("      " + line for line in txt.rstrip().splitlines()))

# Plot: shape functions per field (columns) and model (rows), conditions overlaid.
mids = list(report)
fig, axes = plt.subplots(len(mids), len(FEATURES), figsize=(2.6 * len(FEATURES), 2.4 * len(mids)), sharey="row",
                         squeeze=False)
for i, model_id in enumerate(mids):
    for j, k in enumerate(FEATURES):
        ax = axes[i][j]
        ax.axhline(0, color="#d6d5d0", lw=1)
        for cond, sh in report[model_id]["shapes"].items():
            ax.plot(grids[k], sh[k], color=COND_C[cond], lw=2, label=cond)
        if i == 0:
            ax.set_title(FEATURES[k], fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{NAMES.get(model_id, model_id)}\nlog-odds (centred)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.005, f"{Path(args.run).name}: additive surrogate (boosted stumps, monotone in merit fields) per condition; "
         "each curve is centred at the field's median; x: 5th-95th percentile of applicants.", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.02, 1, 1))
Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.plot, dpi=150, bbox_inches="tight")
print("\nwrote", args.plot)
