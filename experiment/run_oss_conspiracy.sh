#!/usr/bin/env bash
# gpt-oss-20b conspiracy injection: turn 1 is a conspiracy claim answered naturally, by an injected "yes" (conspiracy)
# or "no" (factual control) that gpt-oss-20b wrote itself, or by gpt-4.1's written answers transplanted as its own.
# Turn 2 is the claim's hop question, or the same claim again. Then the self-probe (was that answer yours?).
# Needs a vLLM server for openai/gpt-oss-20b at OSS_BASE_URL (e.g. an SSH tunnel to http://localhost:8000/v1).
set -euo pipefail
export PYTHONIOENCODING=utf-8 OSS_BASE_URL=${OSS_BASE_URL:-http://localhost:8000/v1}
R=results/2-turn
D=2026-11-18_oss_conspiracy
P="--problems-file experiment/conspiracies.json --models gpt-oss-20b"
CLAIMS="moon_landing flat_earth chemtrails paul_dead roswell"
HOPS="moon_mirrors horizon trails_safe yesterday_2010s et_evidence"
run() { uv run python experiment/get_responses.py "$@" | grep -E "FAILED|skipping|^ran|sources" || true; }
judge() { uv run python experiment/judge.py "$@" | tail -n 1; }

# Turn 1: natural answers, written answers (yes and no, 3 each), and a cold baseline for the hop questions.
run one-turn $R/${D}_natural.json $P --problems $CLAIMS --n 1 &
run inject-answers $R/${D}_injected.json $P --n 3 &
run one-turn $R/${D}_cold_hops.json $P --problems $HOPS --n 10 &
wait
for f in natural injected cold_hops; do judge $R/${D}_$f.json; done

# Turn 2: the hop question and the same question, 10 times each, from gpt-oss's own and gpt-4.1's turn 1.
S="--sources $R/${D}_natural.json $R/${D}_injected.json"
T="--sources $R/2026-11-05_conspiracy_injected.json --transplant gpt-4.1=gpt-oss-20b"
run hop $R/${D}_hop.json $P $S --repeats 10 &
run hop $R/${D}_same.json $P $S --repeats 10 --same-question &
run hop $R/${D}_transplant_hop.json $P $T --repeats 10 &
run hop $R/${D}_transplant_same.json $P $T --repeats 10 --same-question &
wait
for f in hop same transplant_hop transplant_same; do
  judge $R/${D}_$f.json
  judge $R/${D}_$f.json --recognition
  uv run python experiment/analyze.py $R/${D}_$f.json > /dev/null
done

# Self-probe: after turn 1 and after turn 2, strong and weak wording, one transcript per file, claim and arm.
run self-probe $R/${D}_self_probe.json --models gpt-oss-20b --per-cell 1 \
  --sources $R/${D}_hop.json $R/${D}_same.json $R/${D}_transplant_hop.json $R/${D}_transplant_same.json
uv run python experiment/analyze.py $R/${D}_self_probe.json > /dev/null
echo "generation done"
