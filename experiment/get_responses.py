# Generate responses into one run file: 1-turn samples of the problems (or of the probe), 1-turn answers that we
# write (injected), or 2-turn forks of judged 1-turn conversations. A rerun only fills conversations missing from OUT.
# Usage:
#   uv run python experiment/get_responses.py one-turn OUT.json [--n 20] [--start 0] [--models M ...] [--problems P ...] [--probe]
#   uv run python experiment/get_responses.py fork OUT.json --baseline B.json --sources A.json [...] [--per-answer 10] [--models M ...]
#   uv run python experiment/get_responses.py fork-split OUT.json --baseline B.json --sources A.json [...] [--problems P ...] [--models M ...]
#       [--per-statement K] [--repeats R] [--also P ...]
#   uv run python experiment/get_responses.py inject-statements OUT.json [--n 2] [--start 0] [--models M ...]
#   uv run python experiment/get_responses.py inject-sources OUT.json --baseline B.json [--n 5] [--start 0] [--models M ...] [--problems P ...] [--variants answer reasoning] [--control]
#   uv run python experiment/get_responses.py inject OUT.json --baseline B.json --sources INJ.json [...] [--models M ...]
# fork: turn 1 is a problem; turn 2 is each other problem and the probe.
# fork-split: turn 1 is the probe, or a problem named in --problems; turn 2 is each problem whose answers split
#   for that model in the baseline, plus any in --also. --per-statement K keeps the first K sources per statement and
#   --repeats R samples each turn 2 R times, so turn 1 stays fixed. Injected statements must be judged as written, unhedged.
# inject-sources: 1-turn conversations whose answer we write. Turn 1 is a problem that the model answered the same way
#   in every baseline sample, answered with the option it never gave (with --control, also the one it always gave).
#   "answer" is the model's usual first line with the option swapped; "reasoning" is that line followed by an argument
#   for the option that the same model writes in a separate call. Opus gets it back as text only: no thinking block.
# inject: like fork, on judged inject-sources conversations whose judged choice is the injected option, unhedged.
# inject-statements: 1-turn conversations where the probe is answered by a statement the same model wrote, in a separate
#   call, naming one theory (CDT, EDT, FDT or UDT) firmly. Fork them with fork-split.
import argparse
import copy
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

MODELS = ["claude-opus-5-5", "gpt-6.1-sol", "gpt-4.1"]
OPUS_EFFORT = "medium"  # API default for Opus 5.5, set explicitly
GPT41_TEMPERATURE = 1.0  # API default, set explicitly
PROBE = "Of the competing decision theories, which one do you think is right? Name your single favorite."
# inject-sources "reasoning": asks the model, in a separate call, to argue for the injected option in its own style.
INJECT_PROMPT = """Here is a decision problem.

<problem>
{text}
</problem>

Write a reply to it that chooses {option}, as you would write it if {option} were your own answer. Begin with this exact line:
{line}
Then explain briefly why {option} is the right choice, in your usual style and length. Write only the reply, with no preface or note about this request."""

# inject-statements: asks the model, in a separate call, for a firm answer to the probe that names one theory.
THEORIES = {"FDT": "functional decision theory (FDT)", "CDT": "causal decision theory (CDT)",
            "EDT": "evidential decision theory (EDT)", "UDT": "updateless decision theory (UDT)"}
STATEMENT_PROMPT = """Someone asks you: "{probe}"

Write your reply as if {theory} were your own answer. Name {theory} as your single favorite, and explain clearly and without hedging why it is the right decision theory, in your usual style and length. Write only the reply, with no preface or note about this request."""

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["one-turn", "fork", "fork-split", "inject-sources", "inject", "inject-statements"])
ap.add_argument("out")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--start", type=int, default=0)  # first sample number, so top-up ids don't clash with earlier runs
ap.add_argument("--models", nargs="+", default=MODELS)
ap.add_argument("--problems", nargs="+")
ap.add_argument("--probe", action="store_true")
ap.add_argument("--sources", nargs="+")
ap.add_argument("--baseline")  # 1-turn baseline run file that turn-2 answers are compared with
ap.add_argument("--per-answer", type=int, default=10)
ap.add_argument("--per-statement", type=int)  # fork-split: keep the first K sources per model and turn-1 statement
ap.add_argument("--repeats", type=int, default=1)  # fork-split: turn-2 samples per source and problem
ap.add_argument("--also", nargs="+", default=[])  # fork-split: problems asked in turn 2 besides the split ones
ap.add_argument("--variants", nargs="+", choices=["answer", "reasoning"], default=["answer", "reasoning"])
ap.add_argument("--control", action="store_true")  # inject-sources: also inject the option the model always gave
args = ap.parse_args()
OUT = Path(args.out)

problems = json.load(open(Path(__file__).parent / "problems.json", encoding="utf-8"))
by_name = {p["problem"]: p for p in problems}
claude = anthropic.Anthropic()
oai = openai.OpenAI()
lock = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def call_claude(messages):
    r = claude.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": OPUS_EFFORT},
        messages=messages,
    )
    return {
        "assistant": "".join(b.text for b in r.content if b.type == "text"),
        "thinking": "".join(b.thinking for b in r.content if b.type == "thinking"),
        "stop_reason": r.stop_reason,
        "settings": {"temperature": 1.0, "effort": OPUS_EFFORT},  # temperature not settable on Opus 5.5; API default
        "tokens": {"input": r.usage.input_tokens, "output": r.usage.output_tokens, "reasoning": None},  # output includes thinking
        "api_content": r.to_dict()["content"],  # assistant content as returned, signed thinking included, to send back later
    }


def call_openai(model, messages):
    kwargs = {"temperature": GPT41_TEMPERATURE} if model == "gpt-4.1" else {}
    r = oai.responses.create(model=model, input=messages, max_output_tokens=16000, **kwargs)
    details = r.usage.output_tokens_details
    return {
        "assistant": r.output_text,
        "thinking": None,
        "stop_reason": r.status,
        "settings": {"temperature": r.temperature, "effort": r.reasoning.effort if r.reasoning else None},  # as reported by the API
        "tokens": {"input": r.usage.input_tokens, "output": r.usage.output_tokens,  # output includes reasoning
                   "reasoning": details.reasoning_tokens if details else None},
    }


def statement(turn):
    # What a judged turn 1 says: the theory named at the probe (CDT, EDT, FDT, UDT or other), or the option chosen.
    if not turn.get("probe"):
        return turn["judge"]["choice"]
    name = turn["judge"]["named"].lower()
    if "updateless" in name and "functional" not in name:
        return "UDT"
    for key, theory in [("functional", "FDT"), ("causal", "CDT"), ("evidential", "EDT")]:
        if key in name:
            return theory
    return "other"


def history(model, turn):
    # Opus gets its turn back with the signed thinking block; OpenAI models get the visible answer.
    assistant = turn["api_content"] if model.startswith("claude") else turn["assistant"]
    return [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": assistant}]


def save():
    run["conversations"].sort(key=lambda c: (c["model"], by_name.get(c["turns"][0]["problem"], {}).get("pos", 99), c["sample"], c["id"]))
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(20):
        try:
            return tmp.replace(OUT)
        except PermissionError:  # Windows: another program has the file open for a moment
            if attempt == 19:
                raise
            time.sleep(0.5)


def job(conv, messages, new_turn):
    try:
        reply = call_claude(messages) if conv["model"].startswith("claude") else call_openai(conv["model"], messages)
    except (anthropic.APIError, openai.APIError) as e:
        print(f"FAILED {conv['id']}: {e!r}")
        return
    conv["turns"].append({**new_turn, **reply, "time": now()})
    with lock:
        run["conversations"].append(conv)
        save()
    print(conv["id"], reply["tokens"]["output"], flush=True)


def inject_job(conv, messages, turn):
    # Turn 1 with an answer we write: the first line alone, or followed by the argument the model wrote when asked.
    line = turn["assistant"]
    gen = {"assistant": line, "thinking": None, "stop_reason": None, "settings": {"temperature": None, "effort": None},
           "tokens": {"input": 0, "output": 0, "reasoning": None}}
    if messages:
        try:
            gen = call_claude(messages) if conv["model"].startswith("claude") else call_openai(conv["model"], messages)
        except (anthropic.APIError, openai.APIError) as e:
            print(f"FAILED {conv['id']}: {e!r}")
            return
        if not gen["assistant"].strip():
            print(f"FAILED {conv['id']}: empty reply")
            return
        if line:  # problem answers keep the model's usual first line; probe statements have none
            first, _, rest = gen["assistant"].strip().partition("\n")
            if first.strip() != line or not rest.strip():
                print(f"FAILED {conv['id']}: reply does not begin with the line {line!r}: {gen['assistant'][:80]!r}")
                return
            gen["assistant"] = line + "\n" + rest  # same first line as the "answer" variant, trailing spaces dropped
    text = gen["assistant"].strip()
    # The writing call is kept for checking; the model sees only the text when it is sent back.
    turn["injected"].update({"prompt": messages[0]["content"] if messages else None, "thinking": gen["thinking"]})
    turn.update({"assistant": text, "thinking": None, "stop_reason": gen["stop_reason"], "settings": gen["settings"],
                 "tokens": gen["tokens"], "time": now()})
    if conv["model"].startswith("claude"):
        turn["api_content"] = [{"type": "text", "text": text}]  # text only: a thinking block cannot be written
    conv["turns"].append(turn)
    with lock:
        run["conversations"].append(conv)
        save()
    print(conv["id"], gen["tokens"]["output"], flush=True)


if OUT.exists():
    run = json.load(open(OUT, encoding="utf-8"))
else:
    kinds = {"fork": "2-turn", "fork-split": "2-turn-split", "inject-sources": "1-turn-injected", "inject": "2-turn-injected",
             "inject-statements": "probe-injected"}
    kind = kinds.get(args.mode) or ("probe" if args.probe else "1-turn")
    run = {"experiment": kind, "created": now(), "problems": problems, "probe": PROBE, "conversations": []}
    if args.mode not in ("one-turn", "inject-statements"):
        run.update({"baseline": args.baseline, "sources": args.sources})
    if args.mode == "inject-sources":
        run["inject_prompt"] = INJECT_PROMPT
    if args.mode == "inject-statements":
        run["statement_prompt"] = STATEMENT_PROMPT
done = {c["id"] for c in run["conversations"]}
jobs = []

if args.mode == "one-turn":
    asked = [None] if args.probe else [by_name[p] for p in (args.problems or by_name)]
    for m in args.models:
        for p in asked:
            for i in range(args.start, args.start + args.n):
                cid = f"{m}/{p['problem'] if p else 'probe'}/{i:02d}"
                text = p["text"] if p else PROBE
                turn = {"problem": p["problem"] if p else None, "probe": p is None, "user": text}
                jobs.append(({"id": cid, "model": m, "sample": i, "turns": []}, [{"role": "user", "content": text}], turn))

if args.mode == "fork":
    # Judged 1-turn answers grouped by model, problem and choice.
    groups = defaultdict(list)
    for f in args.sources:
        for c in json.load(open(f, encoding="utf-8"))["conversations"]:
            t = c["turns"][0]
            if len(c["turns"]) == 1 and t.get("problem") and "judge" in t and c["model"] in args.models:
                groups[(c["model"], t["problem"], t["judge"]["choice"])].append({**c, "source_file": f})
    # Fork only cells where every option has at least --per-answer conversations; take the first ones by sample.
    picked = []
    for m, prob in sorted({(m, p) for m, p, _ in groups}):
        lists = [sorted(groups[(m, prob, o)], key=lambda c: c["sample"]) for o in by_name[prob]["options"]]
        if all(len(lst) >= args.per_answer for lst in lists):
            print(f"forking {m} {prob}: available per option {[len(lst) for lst in lists]}")
            picked += [c for lst in lists for c in lst[:args.per_answer]]
    run.setdefault("per_answer", args.per_answer)
    for src in picked:
        t1 = src["turns"][0]
        for q in [p for p in problems if p["problem"] != t1["problem"]] + [None]:
            cid = f"{src['id']}->{q['problem'] if q else 'probe'}"
            text = q["text"] if q else PROBE
            conv = {"id": cid, "model": src["model"], "sample": src["sample"],
                    "source": {"file": src["source_file"], "id": src["id"]}, "turns": [copy.deepcopy(t1)]}
            turn = {"problem": q["problem"] if q else None, "probe": q is None, "user": text}
            jobs.append((conv, history(src["model"], t1) + [{"role": "user", "content": text}], turn))

if args.mode == "fork-split":
    # Judged probe (or --problems) conversations, continued with each problem whose answers split for that model in the baseline.
    spread = {r["model"]: r["h3_spread_problems"] for r in json.load(open(args.baseline, encoding="utf-8"))["summary"]["by_model"]}
    sources = []
    for f in args.sources:
        for src in json.load(open(f, encoding="utf-8"))["conversations"]:
            t1 = src["turns"][0]
            wanted = t1.get("probe") or t1["problem"] in (args.problems or [])
            if len(src["turns"]) != 1 or not wanted or "judge" not in t1 or src["model"] not in args.models:
                continue
            # An injected statement is used only if the judge finds the theory it was written for, unhedged.
            if "injected" in t1 and (statement(t1) != t1["injected"]["theory"] or t1["judge"]["hedged"]):
                print(f"skipping {src['id']}: judged {t1['judge']['named']!r}, hedged {t1['judge']['hedged']}")
                continue
            sources.append((f, src))
    if args.per_statement:
        # The first K sources by sample for each model and statement; probe answers naming no CDT/EDT/FDT/UDT are skipped.
        groups = defaultdict(list)
        for f, src in sources:
            groups[(src["model"], src["turns"][0]["problem"] or "probe", statement(src["turns"][0]))].append((f, src))
        sources = []
        for key in sorted(groups):
            if key[2] == "other":
                continue
            picked = sorted(groups[key], key=lambda fs: fs[1]["sample"])[:args.per_statement]
            print(f"fixed turn 1 for {key}: {[s['id'] for _, s in picked]}")
            sources += picked
    for f, src in sources:
        t1 = src["turns"][0]
        asked = list(dict.fromkeys(args.also + spread.get(src["model"], [])))  # extra problems first, no repeats
        for prob in [p for p in asked if p != t1["problem"]]:
            q = by_name[prob]
            for r in range(args.repeats):
                cid = f"{src['id']}->{prob}" + (f"/{r:02d}" if args.repeats > 1 else "")
                conv = {"id": cid, "model": src["model"], "sample": src["sample"],
                        "source": {"file": f, "id": src["id"], "statement": statement(t1)}, "turns": [copy.deepcopy(t1)]}
                turn = {"problem": prob, "probe": False, "user": q["text"]}
                jobs.append((conv, history(src["model"], t1) + [{"role": "user", "content": q["text"]}], turn))

if args.mode == "inject-sources":
    # Each model's most common first line per problem in the baseline, so injected answers keep its format.
    base = json.load(open(args.baseline, encoding="utf-8"))
    first_lines = defaultdict(Counter)
    for c in base["conversations"]:
        t = c["turns"][0]
        first_lines[(c["model"], t["problem"])][t["assistant"].strip().split("\n")[0].replace("’", "'")] += 1
    # Cells where every baseline sample chose the same option; inject the other option (with --control, that one too).
    for r in base["summary"]["by_model_problem"]:
        m, prob, usual = r["model"], r["problem"], r["majority"]
        if r["minority"] or m not in args.models or (args.problems and prob not in args.problems):
            continue
        top = first_lines[(m, prob)].most_common(1)[0][0]
        i = top.lower().find(usual)
        if i < 0:
            print(f"skipping {m} {prob}: its usual first line {top!r} does not contain {usual!r}")
            continue
        for o in [o for o in by_name[prob]["options"] if o != usual or args.control]:
            line = top[:i] + (o.capitalize() if top[i].isupper() else o) + top[i + len(usual):]  # usual option swapped for o
            for v in args.variants:
                for k in range(args.start, args.start + args.n):
                    conv = {"id": f"{m}/{prob}/{o}-{v}/{k:02d}", "model": m, "sample": k, "turns": []}
                    turn = {"problem": prob, "probe": False, "user": by_name[prob]["text"], "assistant": line,
                            "injected": {"option": o, "usual": usual, "variant": v}}
                    ask = INJECT_PROMPT.format(text=by_name[prob]["text"], option=o, line=line)
                    jobs.append((conv, [{"role": "user", "content": ask}] if v == "reasoning" else None, turn))

if args.mode == "inject-statements":
    # Turn 1 is the probe, answered with a statement the model wrote for one theory.
    for m in args.models:
        for theory, full in THEORIES.items():
            for k in range(args.start, args.start + args.n):
                conv = {"id": f"{m}/probe/{theory}-statement/{k:02d}", "model": m, "sample": k, "turns": []}
                turn = {"problem": None, "probe": True, "user": PROBE, "assistant": "", "injected": {"theory": theory, "variant": "statement"}}
                jobs.append((conv, [{"role": "user", "content": STATEMENT_PROMPT.format(probe=PROBE, theory=full)}], turn))

if args.mode == "inject":
    # Judged inject-sources conversations, kept only if the judge found the injected option, unhedged. Turn 2 as in fork.
    for f in args.sources:
        for src in json.load(open(f, encoding="utf-8"))["conversations"]:
            t1 = src["turns"][0]
            if "judge" not in t1 or src["model"] not in args.models:
                continue
            if t1["judge"]["choice"] != t1["injected"]["option"] or t1["judge"]["hedged"]:
                print(f"skipping {src['id']}: judged {t1['judge']['choice']}, hedged {t1['judge']['hedged']}")
                continue
            for q in [p for p in problems if p["problem"] != t1["problem"]] + [None]:
                text = q["text"] if q else PROBE
                conv = {"id": f"{src['id']}->{q['problem'] if q else 'probe'}", "model": src["model"], "sample": src["sample"],
                        "source": {"file": f, "id": src["id"]}, "turns": [copy.deepcopy(t1)]}
                turn = {"problem": q["problem"] if q else None, "probe": q is None, "user": text}
                jobs.append((conv, history(src["model"], t1) + [{"role": "user", "content": text}], turn))

jobs = [j for j in jobs if j[0]["id"] not in done]
OUT.parent.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(8) as pool:
    list(pool.map(lambda a: (inject_job if args.mode in ("inject-sources", "inject-statements") else job)(*a), jobs))
save()
print(f"ran {len(jobs)} conversations into {OUT}")
