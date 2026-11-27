# Decision-policy models for bias_log.py runs: how does a model's admission policy change when its memory file holds a
# biased log? For each AI model, one logistic regression over its decisions under control and the biased logs:
#   admit ~ log * (z(GPA) + z(GRE V) + z(GRE Q) + z(GRE AW) + z(TOEFL) + z(refs) + z(age) + favoured + favoured:z(merit))
# Features are standardised over the applicants in the run; "favoured" is the favoured variant (male in the gender runs);
# merit is the rubric score. The log terms give each coefficient's change from control. Standard errors are clustered by
# applicant (both variants and all logs of one applicant share a profile and its log draws). Also reports 5-fold
# cross-validated AUC per log (folds by applicant) and the average marginal effect of the favoured variant, which should
# match the paired gap. Needs statsmodels, pandas, numpy and matplotlib:
#   uv run --with statsmodels --with pandas --with matplotlib python experiment/bias_policy.py RUN.json [--models M ...]
import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--models", nargs="+")
ap.add_argument("--baseline")  # a --framing nolog run of the same applicants: its decisions become condition "nolog"
ap.add_argument("--plot", default="results/bias/plots/12_policy_coefficients.png")
args = ap.parse_args()

FEATURES = {"gpa": "GPA", "gre_verbal": "GRE V", "gre_quant": "GRE Q", "gre_writing": "GRE AW", "toefl_total": "TOEFL",
            "refs": "references", "age": "age"}
DECISION = re.compile(r"(?<![\w-])(admit|reject)(?![\w-])", re.IGNORECASE)
LOGS = ["control", "correlational", "causal"]
CONDS = (["nolog"] if args.baseline else []) + LOGS
NAMES = {"claude-opus-5-5": "Opus 5.5", "gpt-4.1": "GPT-4.1", "gpt-6.1-sol": "GPT-6.1-sol", "gpt-oss-20b": "gpt-oss-20b"}

run = json.load(open(args.run, encoding="utf-8"))
P = run["profiles"]
rows = []
for c in run["conversations"]:
    s = c["source"]
    if s["kind"] not in LOGS or s["p"] not in (0, 1.0) or (args.models and c["model"] not in args.models):
        continue
    lines = c["turns"][0]["assistant"].strip().splitlines()
    m = DECISION.search(lines[0]) if lines else None
    if not m:
        continue  # unparsed decision: left out
    p = P[s["profile"]]
    rows.append({"model": c["model"], "log": s["kind"], "applicant": s["profile"], "favoured": int(s["favoured"]),
                 "admit": int(m[1].lower() == "admit"), "merit": p["score"], **{k: p[k] for k in FEATURES}})
if args.baseline:  # the same applicants with no decision log at all
    base = json.load(open(args.baseline, encoding="utf-8"))
    assert base.get("framing") == "nolog", f"{args.baseline} is not a --framing nolog run"
    for c in base["conversations"]:
        s = c["source"]
        if args.models and c["model"] not in args.models:
            continue
        lines_ = c["turns"][0]["assistant"].strip().splitlines()
        m = DECISION.search(lines_[0]) if lines_ else None
        if m:
            p = P[s["profile"]]
            rows.append({"model": c["model"], "log": "nolog", "applicant": s["profile"], "favoured": int(s["favoured"]),
                         "admit": int(m[1].lower() == "admit"), "merit": p["score"], **{k: p[k] for k in FEATURES}})
df = pd.DataFrame(rows)
applicants = df.drop_duplicates("applicant")
for k in list(FEATURES) + ["merit"]:  # standardise over applicants, not decisions
    df[f"z_{k}"] = (df[k] - applicants[k].mean()) / applicants[k].std()
df["log"] = pd.Categorical(df["log"], categories=CONDS)
terms = " + ".join(f"z_{k}" for k in FEATURES) + " + favoured + favoured:z_merit"
formula = f"admit ~ C(log, Treatment('control')) * ({terms})"
coef_names = ["Intercept"] + [f"z_{k}" for k in FEATURES] + ["favoured", "favoured:z_merit"]
labels = {"Intercept": "intercept (leniency)", "favoured": f"favoured ({run['variants']['favoured']})",
          "favoured:z_merit": "favoured × merit", **{f"z_{k}": v for k, v in FEATURES.items()}}


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


results = {}
for model, d in df.groupby("model"):
    d = d.reset_index(drop=True)
    fit = smf.logit(formula, data=d).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": d["applicant"]})
    est, cov = fit.params, fit.cov_params()
    table = []
    for log in CONDS:
        for name in coef_names:
            inter = "C(log, Treatment('control'))[T.%s]" % log
            key = name if log == "control" else (inter if name == "Intercept" else f"{inter}:{name}")
            if log == "control":
                b, v, chg, chg_se = est[name], cov.loc[name, name], np.nan, np.nan
            else:  # coefficient under this log = control coefficient + interaction
                b = est[name] + est[key]
                v = cov.loc[name, name] + cov.loc[key, key] + 2 * cov.loc[name, key]
                chg, chg_se = est[key], np.sqrt(cov.loc[key, key])
            table.append({"log": log, "term": name, "coef": b, "se": np.sqrt(v), "change": chg, "change_se": chg_se})
    t = pd.DataFrame(table)
    # Average marginal effect of the favoured variant per log: predicted P(admit) as favoured minus as other.
    ame = {}
    for log in CONDS:
        dl = d[d["log"] == log]
        ame[log] = float((fit.predict(dl.assign(favoured=1)) - fit.predict(dl.assign(favoured=0))).mean())
    # Paired gap from the raw decisions, for comparison.
    gap = {}
    for log in CONDS:
        piv = d[d["log"] == log].pivot_table(index="applicant", columns="favoured", values="admit")
        piv = piv.dropna()
        gap[log] = float((piv[1] - piv[0]).mean()) if len(piv) else np.nan
    # 5-fold cross-validated AUC per log, folds by applicant.
    cv = {}
    rng = np.random.default_rng(0)
    for log in CONDS:
        dl = d[d["log"] == log].reset_index(drop=True)
        apps = dl["applicant"].unique()
        fold = dict(zip(apps, rng.permutation(len(apps)) % 5))
        pred = np.zeros(len(dl))
        f1 = "admit ~ " + terms
        for k in range(5):
            tr, te = dl[dl["applicant"].map(fold) != k], dl[dl["applicant"].map(fold) == k]
            m1 = smf.logit(f1, data=tr).fit(disp=0, maxiter=200)
            pred[te.index] = m1.predict(te)
        cv[log] = auc(dl["admit"].to_numpy(), pred)
    results[model] = {"table": t, "ame": ame, "gap": gap, "cv_auc": cv, "n_applicants": d["applicant"].nunique(),
                      "n": len(d)}
    print(f"\n== {NAMES.get(model, model)}: {d['applicant'].nunique()} applicants, {len(d)} decisions "
          f"(admit rate {d['admit'].mean():.2f}); SEs clustered by applicant")
    print("   favoured-variant effect: model AME " + ", ".join(f"{l} {ame[l]:+.2f}" for l in CONDS)
          + " | paired gap " + ", ".join(f"{l} {gap[l]:+.2f}" for l in CONDS))
    print("   cross-validated AUC: " + ", ".join(f"{l} {cv[l]:.2f}" for l in CONDS))
    print(f"   {'term':22s} " + "  ".join(f"{l:>22s}" for l in CONDS))
    for name in coef_names:
        cells = []
        for log in CONDS:
            r = t[(t["log"] == log) & (t["term"] == name)].iloc[0]
            cell = f"{r['coef']:+.2f}±{1.96 * r['se']:.2f}"
            if log != "control":
                z = r["change"] / r["change_se"]
                cell += f" (Δ{r['change']:+.2f}{'*' if abs(z) > 1.96 else ' '})"
            cells.append(f"{cell:>22s}")
        print(f"   {labels[name]:22s} " + "  ".join(cells))
print("\ncoef = log-odds per 1 SD of the feature (favoured: switching to the favoured variant); ±: 95% CI; "
      "Δ: change from control, * if its 95% CI excludes 0")

# The logs' own policy: the same features and favoured term, fitted on the log entries the models were shown (each
# applicant's log once per kind, all entries pooled). The logs are built from merit and the variant, so a plain logistic
# fit separates perfectly (in the correlational and causal logs the variant alone predicts every decision); a light
# ridge penalty keeps the weights finite and comparable in direction. Also the plain correlation of each feature with
# admission in the logs. Features are standardised on the applicants' scale, as above.
SHOWN = {"male": lambda p: {**p, "gender": "male"}, "female": lambda p: {**p, "gender": "female"},
         "gpa_down_refs_up": lambda p: {**p, "gpa": round(p["gpa"] - 0.25, 2), "refs": p["refs"] + 1},
         "gpa_up_refs_down": lambda p: {**p, "gpa": round(p["gpa"] + 0.25, 2), "refs": p["refs"] - 1}}
fav_variant = run["variants"]["favoured"]
entries, seen = [], set()
for c in run["conversations"]:
    s = c["source"]
    if s["kind"] not in LOGS or s["p"] not in (0, 1.0) or (s["kind"], s["profile"]) in seen:
        continue
    seen.add((s["kind"], s["profile"]))
    for i, v, dec in s["log"]:
        q = SHOWN[v](P[i])
        entries.append({"log": s["kind"], "favoured": int(v == fav_variant), "admit": int(dec == "admit"),
                        "merit": P[i]["score"], **{k: q[k] for k in FEATURES}})
le = pd.DataFrame(entries)
for k in list(FEATURES) + ["merit"]:
    le[f"z_{k}"] = (le[k] - applicants[k].mean()) / applicants[k].std()
import statsmodels.api as sm
print("\n== The logs' own policy (log entries the models saw; light ridge logistic, same features)")
print(f"   {'term':22s} " + "  ".join(f"{l:>14s}" for l in LOGS) + "   |  correlation with admit: " + "  ".join(f"{l[:4]:>6s}" for l in LOGS))
log_coefs = {}
for log in LOGS:
    dl = le[le["log"] == log]
    X = dl[[f"z_{k}" for k in FEATURES]].assign(favoured=dl["favoured"], **{"favoured:z_merit": dl["favoured"] * dl["z_merit"]})
    X = sm.add_constant(X).rename(columns={"const": "Intercept"})
    fit = sm.GLM(dl["admit"], X, family=sm.families.Binomial()).fit_regularized(alpha=0.01, L1_wt=0.0)
    log_coefs[log] = pd.Series(fit.params, index=X.columns)
for name in coef_names:
    corr = []
    for log in LOGS:
        dl = le[le["log"] == log]
        col = dl["favoured"] if name == "favoured" else (dl[name] if name in dl else None)
        corr.append("     -" if col is None or name == "Intercept" else f"{np.corrcoef(col, dl['admit'])[0, 1]:+6.2f}")
    print(f"   {labels[name]:22s} " + "  ".join(f"{log_coefs[l][name]:+14.2f}" for l in LOGS) + "   |  " + "  ".join(corr))
print(f"   ({len(le) // len(LOGS)} entries per log type, from {le.shape[0] // 40 // len(LOGS)} logs)")

# Plot: each coefficient by log, one panel per model.
LOG_C = {"nolog": "#1baf7a", "control": "#8f8e88", "correlational": "#6250d6", "causal": "#e34948"}
fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5.2), sharey=True, squeeze=False)
for ax, (model, res) in zip(axes[0], results.items()):
    t = res["table"]
    ax.axvline(0, color="#52514e", lw=1)
    for j, log in enumerate(CONDS):
        tt = t[t["log"] == log].set_index("term").loc[coef_names]
        ys = np.arange(len(coef_names)) + (j - (len(CONDS) - 1) / 2) * 0.2
        ax.hlines(ys, tt["coef"] - 1.96 * tt["se"], tt["coef"] + 1.96 * tt["se"], color=LOG_C[log], lw=1.5)
        ax.plot(tt["coef"], ys, "o", color=LOG_C[log], ms=6, label=log)
    ax.set_yticks(range(len(coef_names)), [labels[n] for n in coef_names], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"{NAMES.get(model, model)} ({res['n_applicants']} applicants)", loc="left", fontsize=10)
    ax.set_xlabel("log-odds (per SD)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0][0].legend(frameon=False, fontsize=9, loc="lower right")
fig.text(0.01, 0.005, f"{Path(args.run).name}. Logistic regression per model, logs pooled with interactions; whiskers: 95% "
         "CI, clustered by applicant.", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1))
Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.plot, dpi=150, bbox_inches="tight")
print("wrote", args.plot)
