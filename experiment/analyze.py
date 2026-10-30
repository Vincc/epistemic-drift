# Add per-turn metrics and a summary to a run file. The summary depends on the run: 1-turn choice counts and
# hypotheses, probe name counts, or 2-turn effects of the turn-1 answer (sampled or injected) on turn 2.
# Usage: uv run python experiment/analyze.py RUN.json
import json
import re
import sys
import time
from collections import Counter
from math import comb
from pathlib import Path

RUN = Path(sys.argv[1])
THEORIES = ["cdt", "edt", "fdt"]
PRICES = {"claude-opus-5-5": (4e-6, 20e-6), "gpt-6.1-sol": (2e-6, 10e-6), "gpt-4.1": (2e-6, 8e-6)}  # $/token in, out
JUDGE_PRICE = (2e-6, 10e-6)  # claude-sonnet-5-5
ALPHA = 0.05

run = json.load(open(RUN, encoding="utf-8"))
problems = run["problems"]
by_name = {p["problem"]: p for p in problems}
convs = run["conversations"]


def theory_answer(p, t):
    # Theories in preferred_dt pick the winning option; the others pick the other option.
    other = [o for o in p["options"] if o != p["winning_option"]][0]
    return p["winning_option"] if t in p["preferred_dt"] else other


def first_option(text, options):
    # Earliest whole-word mention of an option; hyphen may be a space, apostrophe may be curly.
    best = None
    for o in options:
        pat = re.escape(o).replace(r"\-", "[- ]").replace("'", "['’]")
        m = re.search(r"(?<![\w'’-])" + pat + r"(?![\w-])", text, re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), o)
    return best[1] if best else "none"


def fisher(a, b, c, d):
    # Two-sided Fisher exact test p-value for the 2x2 table [[a, b], [c, d]].
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n1 + n2, k)
    observed = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= observed * (1 + 1e-9))


def share(k, n):
    return k / n if n else None


# Per-turn metrics for every judged turn; choice metrics only for turns that answer a problem.
for c in convs:
    for t in c["turns"]:
        if "judge" not in t:
            continue
        price = PRICES[c["model"]]
        cost = {"cost_usd": round(t["tokens"]["input"] * price[0] + t["tokens"]["output"] * price[1], 6),
                "judge_cost_usd": round(t["judge"]["tokens"]["input"] * JUDGE_PRICE[0] + t["judge"]["tokens"]["output"] * JUDGE_PRICE[1], 6)}
        if t.get("probe"):
            t["metrics"] = cost
            continue
        p = by_name[t["problem"]]
        choice = t["judge"]["choice"]
        first = first_option(t["assistant"], p["options"])
        t["metrics"] = {
            "first_option": first,
            "flip": first != choice,  # first option named differs from the judged final choice
            "chose_winning": choice == p["winning_option"],
            "theories_matched": [th for th in THEORIES if theory_answer(p, th) == choice],
            **cost,
        }


def one_turn_summary():
    models = sorted({c["model"] for c in convs})
    by_model_problem = []
    for p in problems:
        for m in models:
            ts = [c["turns"][0] for c in convs if c["model"] == m and c["turns"][0]["problem"] == p["problem"]]
            if not ts:
                continue
            n = len(ts)
            counts = Counter(t["judge"]["choice"] for t in ts)
            firsts = Counter(t["metrics"]["first_option"] for t in ts)
            top = counts.most_common(2)
            by_model_problem.append({
                "model": m,
                "problem": p["problem"],
                "n": n,
                "counts": {o: counts.get(o, 0) for o in p["options"] + ["none"]},
                "majority": "tie" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0],
                "minority": n - top[0][1],
                "winning": counts[p["winning_option"]],
                "hedged": sum(t["judge"]["hedged"] for t in ts),
                "flips": sum(t["metrics"]["flip"] for t in ts),
                "first_counts": {o: firsts.get(o, 0) for o in p["options"] + ["none"]},
            })
    # H1: FDT's answer is the majority on every problem. H2: no single theory matches every majority.
    # H3: problems where the minority answer appears in at least a tenth of samples (2 of 20).
    # H1 and H2 need every problem, so they are null for a partial run such as a top-up.
    by_model = []
    for m in models:
        rs = [r for r in by_model_problem if r["model"] == m]
        ts = [c["turns"][0] for c in convs if c["model"] == m]
        matches = {t: sum(r["majority"] == theory_answer(by_name[r["problem"]], t) for r in rs) for t in THEORIES}
        complete = len(rs) == len(problems)
        by_model.append({
            "model": m,
            "responses": len(ts),
            "temperature": sorted({str(t["settings"]["temperature"]) for t in ts}),
            "effort": sorted({str(t["settings"]["effort"]) for t in ts}),
            "majority_matches": matches,
            "h1_consistent_fdt": matches["fdt"] == len(problems) if complete else None,
            "h2_varies_by_question": max(matches.values()) < len(problems) if complete else None,
            "h3_spread_problems": [r["problem"] for r in rs if r["minority"] * 10 >= r["n"]],
            "chose_winning": sum(r["winning"] for r in rs),
            "hedged": sum(r["hedged"] for r in rs),
            "flips": sum(r["flips"] for r in rs),
            "none": sum(r["counts"]["none"] for r in rs),
            "cost_usd": round(sum(t["metrics"]["cost_usd"] for t in ts), 2),
            "judge_cost_usd": round(sum(t["metrics"]["judge_cost_usd"] for t in ts), 2),
        })
    return {"by_model": by_model, "by_model_problem": by_model_problem}


def named_counts(turns):
    return dict(Counter(t["judge"]["named"].strip().lower() for t in turns).most_common())


def probe_summary():
    models = sorted({c["model"] for c in convs})
    return {"by_model": [{"model": m, "n": sum(c["model"] == m for c in convs),
                          "named": named_counts([c["turns"][0] for c in convs if c["model"] == m]),
                          "hedged": sum(c["turns"][0]["judge"]["hedged"] for c in convs if c["model"] == m)} for m in models]}


def load_baseline():
    return {(r["model"], r["problem"]): r for r in json.load(open(run["baseline"], encoding="utf-8"))["summary"]["by_model_problem"]}


def split_summary():
    # Turn 2 is a split problem: its answers grouped by turn 1 (theory named, or choice), against the 1-turn baseline.
    baseline = load_baseline()
    rows = []
    for m, p1, p2 in sorted({(c["model"], c["turns"][0]["problem"] or "probe", c["turns"][1]["problem"]) for c in convs}):
        win = by_name[p2]["winning_option"]
        g = [c for c in convs if c["model"] == m and (c["turns"][0]["problem"] or "probe") == p1 and c["turns"][1]["problem"] == p2]
        by_turn1 = {}
        for c in g:
            j = c["turns"][0]["judge"]
            d = by_turn1.setdefault(j["named"].strip().lower() if "named" in j else j["choice"], {"n": 0, "winning": 0})
            d["n"] += 1
            d["winning"] += c["turns"][1]["judge"]["choice"] == win
        winning = sum(d["winning"] for d in by_turn1.values())
        base = baseline[(m, p2)]
        rows.append({"model": m, "turn1": p1, "turn2_problem": p2, "turn2_winning_option": win, "n": len(g), "winning": winning,
                     "by_turn1": by_turn1, "baseline": {"winning": base["winning"], "n": base["n"]},
                     "fisher_p_vs_baseline": fisher(winning, len(g) - winning, base["winning"], base["n"] - base["winning"])})
    return {"rows": rows}


def two_turn_summary():
    # Turn-2 answers grouped by the turn-1 choice, next to the 1-turn baseline.
    baseline = load_baseline()
    cells = sorted({(c["model"], c["turns"][0]["problem"]) for c in convs})
    choice_arm, probe_arm = [], []
    for m, p1 in cells:
        opts1 = by_name[p1]["options"]
        mine = [c for c in convs if c["model"] == m and c["turns"][0]["problem"] == p1]
        for p2 in [p for p in problems if p["problem"] != p1]:
            groups = {}
            for o in opts1:
                g = [c for c in mine if c["turns"][1]["problem"] == p2["problem"] and c["turns"][0]["judge"]["choice"] == o]
                groups[o] = {
                    "n": len(g),
                    "winning": sum(c["turns"][1]["judge"]["choice"] == p2["winning_option"] for c in g),
                    # Turn-2 answers matching each theory's answer.
                    "matches": {th: sum(th in c["turns"][1]["metrics"]["theories_matched"] for c in g) for th in THEORIES},
                }
            a, b = (groups[o] for o in opts1)
            base = baseline.get((m, p2["problem"]))
            choice_arm.append({
                "model": m,
                "turn1_problem": p1,
                "turn2_problem": p2["problem"],
                "turn2_winning_option": p2["winning_option"],
                "by_turn1_choice": groups,
                "winning_diff": None if not (a["n"] and b["n"]) else share(a["winning"], a["n"]) - share(b["winning"], b["n"]),  # first option minus second
                "fisher_p": fisher(a["winning"], a["n"] - a["winning"], b["winning"], b["n"] - b["winning"]),
                "baseline": {"winning": base["winning"], "n": base["n"]} if base else None,
            })
        for o in opts1:
            g = [c["turns"][1] for c in mine if c["turns"][1].get("probe") and c["turns"][0]["judge"]["choice"] == o]
            probe_arm.append({"model": m, "turn1_problem": p1, "turn1_choice": o, "n": len(g), "named": named_counts(g),
                              "hedged": sum(t["judge"]["hedged"] for t in g)})
    threshold = ALPHA / len(choice_arm)  # Bonferroni across all turn-1 x turn-2 pairs
    for r in choice_arm:
        r["significant"] = r["fisher_p"] < threshold
    return {"bonferroni_threshold": threshold, "choice_arm": choice_arm, "probe_arm": probe_arm}


def injected_summary():
    # Turn-2 answers after a turn-1 answer we wrote, next to the 1-turn baseline, per model, injected cell and variant.
    baseline = load_baseline()
    arm = lambda c: (c["model"], c["turns"][0]["problem"], c["turns"][0]["injected"]["option"], c["turns"][0]["injected"]["variant"])
    choice_arm, by_arm = [], []
    for m, p1, o1, v in sorted({arm(c) for c in convs}):
        mine = [c for c in convs if arm(c) == (m, p1, o1, v)]
        others = [p for p in problems if p["problem"] != p1]
        for p2 in others:
            g = [c["turns"][1] for c in mine if c["turns"][1]["problem"] == p2["problem"]]
            counts = Counter(t["judge"]["choice"] for t in g)
            win, base = counts[p2["winning_option"]], baseline[(m, p2["problem"])]
            choice_arm.append({"model": m, "turn1_problem": p1, "injected": o1, "variant": v, "turn2_problem": p2["problem"],
                               "n": len(g), "counts": {o: counts[o] for o in p2["options"] + ["none"]}, "baseline_counts": base["counts"],
                               "fisher_p_vs_baseline": fisher(win, len(g) - win, base["winning"], base["n"] - base["winning"])})
        turns2 = [c["turns"][1] for c in mine]
        answers = [t for t in turns2 if not t.get("probe")]
        base_n = sum(baseline[(m, p["problem"])]["n"] for p in others)
        by_arm.append({
            "model": m, "turn1_problem": p1, "injected": o1, "usual": mine[0]["turns"][0]["injected"]["usual"], "variant": v,
            "injected_theories": [th for th in THEORIES if theory_answer(by_name[p1], th) == o1],
            "sources": len({c["source"]["id"] for c in mine}),
            # Share of turn-2 answers to the other problems that match each theory, and the same in the baseline.
            "theory_share": {th: share(sum(th in t["metrics"]["theories_matched"] for t in answers), len(answers)) for th in THEORIES},
            "baseline_theory_share": {th: share(sum(baseline[(m, p["problem"])]["counts"][theory_answer(p, th)] for p in others), base_n)
                                      for th in THEORIES},
            "probe_named": named_counts([t for t in turns2 if t.get("probe")]),
            "probe_hedged": sum(t["judge"]["hedged"] for t in turns2 if t.get("probe")),
            # How turn 2 treats the turn-1 answer, once judge.py --recognition has labeled it.
            "recognition": dict(Counter(t["recognition"]["stance"] for t in turns2 if "recognition" in t)),
        })
    threshold = ALPHA / len(choice_arm)  # Bonferroni across all injected-arm x turn-2 pairs
    for r in choice_arm:
        r["significant"] = r["fisher_p_vs_baseline"] < threshold
    # Injected option against the control (the usual option sent the same way): winning-option counts per model and
    # turn-2 problem, both variants pooled and each variant on its own, with the pooled Fisher test.
    vs_control = []
    for m, p1 in sorted({(c["model"], c["turns"][0]["problem"]) for c in convs}):
        for p2 in [p for p in problems if p["problem"] != p1]:
            cell = [c for c in convs if c["model"] == m and c["turns"][0]["problem"] == p1 and c["turns"][1]["problem"] == p2["problem"]]
            row = {"model": m, "turn1_problem": p1, "turn2_problem": p2["problem"], "turn2_winning_option": p2["winning_option"]}
            for side, pick in [("injected", lambda t: t["injected"]["option"] != t["injected"]["usual"]),
                               ("control", lambda t: t["injected"]["option"] == t["injected"]["usual"])]:
                g = [c for c in cell if pick(c["turns"][0])]
                row[side] = {"n": len(g), "winning": sum(c["turns"][1]["judge"]["choice"] == p2["winning_option"] for c in g),
                             "by_variant": {v: [sum(c["turns"][1]["judge"]["choice"] == p2["winning_option"] for c in g if c["turns"][0]["injected"]["variant"] == v),
                                                sum(c["turns"][0]["injected"]["variant"] == v for c in g)] for v in ["answer", "reasoning"]}}
            a, b = row["injected"], row["control"]
            row["fisher_p"] = fisher(a["winning"], a["n"] - a["winning"], b["winning"], b["n"] - b["winning"])
            vs_control.append(row)
    for r in vs_control:
        r["significant"] = r["fisher_p"] < ALPHA / len(vs_control)  # Bonferroni across these rows
    return {"bonferroni_threshold": threshold, "by_arm": by_arm, "choice_arm": choice_arm, "injected_vs_control": vs_control}


if not all("judge" in t for c in convs for t in c["turns"]):
    print("not every turn is judged yet; wrote metrics only")
elif run["experiment"] == "2-turn-split":
    run["summary"] = split_summary()
elif run["experiment"] == "2-turn-injected":
    run["summary"] = injected_summary()
elif all(len(c["turns"]) == 2 for c in convs):
    run["summary"] = two_turn_summary()
elif all(len(c["turns"]) == 1 and c["turns"][0].get("probe") for c in convs):
    run["summary"] = probe_summary()
elif all(len(c["turns"]) == 1 and not c["turns"][0].get("probe") for c in convs):
    run["summary"] = one_turn_summary()

tmp = RUN.with_suffix(".tmp")
tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
for attempt in range(20):
    try:
        tmp.replace(RUN)
        break
    except PermissionError:  # Windows: another program has the file open for a moment
        if attempt == 19:
            raise
        time.sleep(0.5)
print(json.dumps(run.get("summary", {}), indent=1)[:3000])
