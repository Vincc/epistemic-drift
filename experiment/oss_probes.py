# Role probes for gpt-oss-20b, following "Prompt Injection as Role Confusion" (Ye, Cui, Hadfield-Menell 2026): train
# linear probes that tell system / user / cot / assistant text apart from the pre-MLP hidden state, then score turn 1 of
# probe conversations (natural or injected) as the model sees it at turn 2: system prompt, the probe question, and the
# turn-1 answer in the final channel (gpt-oss drops earlier turns' reasoning). Runs on a CUDA GPU.
# Usage: python oss_probes.py OUT.json RUN.json [RUN.json ...]
#   RUN files are get_responses.py runs; turn 1 of each conversation (a probe or question, natural, injected or
#   transplanted) is scored once. Also saves the trained probes to OUT.probes.pt.
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai/gpt-oss-20b"
LAYERS = [4, 8, 12, 16, 20]  # every 4th of 24; the paper's demo reads layer 12
ROLES = ["system", "user", "cot", "assistant"]
N_TEXTS, MAX_CONTENT_TOKENS, KEEP_FRACTION = 250, 512, 0.25  # paper config: 250 sequences; we keep 1/4 of their tokens
C = 5e-3  # paper's L2 setting for gpt-oss-20b (sklearn/cuML convention: C * summed log loss + 0.5 * ||W||^2), no scaling
SEED = 123
# The system message gpt-oss sees by default (the paper's test prefix for gpt-oss-20b).
SYSTEM = ("<|startoftext|><|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\nCurrent date: 2025-12-31\n\nReasoning: medium\n\n"
          "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>")
HEADERS = {"system": "<|start|>system<|message|>", "user": "<|start|>user<|message|>",
           "cot": "<|start|>assistant<|channel|>analysis<|message|>", "assistant": "<|start|>assistant<|channel|>final<|message|>"}

random.seed(SEED)
torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="cuda").eval()
if hasattr(model, "set_experts_implementation"):
    model.set_experts_implementation("eager")  # transformers 5: deterministic MoE routing, as in the paper's demo


def ids(text):
    return tok(text, add_special_tokens=False)["input_ids"]


def build(segments):
    # Segments are (text, role or None); returns token ids and, per token, its role if it is content (tags get None).
    out, roles = [], []
    for text, role in segments:
        piece = ids(text)
        out += piece
        roles += [role] * len(piece)
    return out, roles


@torch.no_grad()
def forward(token_ids):
    # Pre-MLP hidden states (post-attention layernorm output) at LAYERS, and each token's log-probability.
    states = {}
    hooks = [model.model.layers[i].post_attention_layernorm.register_forward_hook(
        lambda m, a, o, i=i: states.__setitem__(i, o[0].float().cpu())) for i in LAYERS]
    try:
        x = torch.tensor([token_ids], device="cuda")
        logits = model(input_ids=x, use_cache=False).logits[0].float()
    finally:
        for h in hooks:
            h.remove()
    logp = torch.log_softmax(logits[:-1], -1).gather(1, x[0, 1:, None])[:, 0].cpu()
    return states, torch.cat([torch.zeros(1), logp])  # first token has no prediction


# 1. Probe training data: the same C4 text wrapped in each role, content tokens only.
texts = []
for row in load_dataset("allenai/c4", "en", split="validation", streaming=True).shuffle(seed=SEED, buffer_size=10_000):
    if len(ids(row["text"])) >= 64:
        texts.append(tok.decode(ids(row["text"])[:MAX_CONTENT_TOKENS]))
    if len(texts) == N_TEXTS:
        break
X = {i: [] for i in LAYERS}
y, group = [], []
for k, text in enumerate(texts):
    for r, role in enumerate(ROLES):
        token_ids, roles = build([(HEADERS[role], None), (text, role), ("<|end|>", None)])
        states, _ = forward(token_ids)
        keep = [p for p, rl in enumerate(roles) if rl and random.random() < KEEP_FRACTION]
        for i in LAYERS:
            X[i].append(states[i][keep].half())
        y += [r] * len(keep)
        group += [k] * len(keep)
    if k % 25 == 0:
        print(f"probe data: {k}/{N_TEXTS} texts", flush=True)
y, group = torch.tensor(y), torch.tensor(group)
test_texts = set(random.sample(range(N_TEXTS), N_TEXTS // 10))
is_test = torch.tensor([g in test_texts for g in group.tolist()])


def fit(x, labels):
    # Multinomial logistic regression, full batch L-BFGS on the GPU.
    x, labels = x.cuda().float(), labels.cuda()
    W = torch.zeros(x.shape[1], len(ROLES), device="cuda", requires_grad=True)
    b = torch.zeros(len(ROLES), device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = C * torch.nn.functional.cross_entropy(x @ W + b, labels, reduction="sum") + 0.5 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


probes, accuracy = {}, {}
for i in LAYERS:
    x = torch.cat(X[i])
    W, b = fit(x[~is_test], y[~is_test])
    pred = (x[is_test].cuda().float() @ W + b).argmax(1).cpu()
    accuracy[i] = {"all": round((pred == y[is_test]).float().mean().item(), 4),
                   **{role: round((pred[y[is_test] == r] == r).float().mean().item(), 4) for r, role in enumerate(ROLES)}}
    probes[i] = (W, b)
    print(f"layer {i} held-out accuracy {accuracy[i]}", flush=True)
del X
torch.save({i: (W.cpu(), b.cpu()) for i, (W, b) in probes.items()}, Path(sys.argv[1]).with_suffix(".probes.pt"))


def role_probs(states, positions):
    return {i: torch.softmax(states[i][positions].cuda() @ probes[i][0] + probes[i][1], -1).cpu() for i in LAYERS}


# 2. Score turn 1 of each conversation as it appears in the turn-2 context.
out = {"model": MODEL, "layers": LAYERS, "roles": ROLES, "C": C, "n_texts": N_TEXTS, "probe_accuracy": accuracy,
       "system_prefix": SYSTEM, "conversations": []}
seen = set()
for f in sys.argv[2:]:
    for c in json.load(open(f, encoding="utf-8"))["conversations"]:
        t1 = c["turns"][0]
        sid = c["source"]["id"] if "source" in c else c["id"]  # 2-turn runs: score the turn-1 source once
        if sid in seen or not (t1.get("probe") or t1.get("problem")):
            continue
        seen.add(sid)
        token_ids, roles = build([(SYSTEM, None), (HEADERS["user"], None), (t1["user"], "user"), ("<|end|>", None),
                                  (HEADERS["assistant"], None), (t1["assistant"], "assistant"), ("<|end|>", None)])
        states, logp = forward(token_ids)
        spans = {span: [p for p, rl in enumerate(roles) if rl == span] for span in ("user", "assistant")}
        rec = {"id": sid, "file": f, "model": c["model"], "problem": t1.get("problem"), "injected": t1.get("injected"),
               "judge": {k: t1["judge"].get(k) for k in ("named", "choice", "hedged")} if "judge" in t1 else None, "spans": {}}
        for span, pos in spans.items():
            probs = role_probs(states, pos)
            rec["spans"][span] = {
                "n_tokens": len(pos),
                "mean_logprob": round(logp[pos].mean().item(), 4),
                "mean_role_prob": {i: {role: round(probs[i][:, r].mean().item(), 4) for r, role in enumerate(ROLES)} for i in LAYERS},
            }
        # Per-token detail for the answer at layers 12 and 16, for plots.
        pos = spans["assistant"]
        probs = role_probs(states, pos)
        rec["tokens"] = [{"token": tok.decode([token_ids[p]]), "logprob": round(logp[p].item(), 3),
                          **{f"L{i}": {role: round(probs[i][k, r].item(), 4) for r, role in enumerate(ROLES)} for i in (12, 16)}}
                         for k, p in enumerate(pos)]
        out["conversations"].append(rec)
        a = rec["spans"]["assistant"]
        print(rec["id"], a["n_tokens"], a["mean_logprob"], a["mean_role_prob"][12], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out['conversations'])} conversations to {sys.argv[1]}")
