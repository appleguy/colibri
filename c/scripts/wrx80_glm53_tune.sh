#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: $0 LABEL [MAX_TOKENS]}"
max_tokens="${2:-8}"
root="${COLI_ENDPOINT_ROOT:-http://127.0.0.1:8000}"
lock=/tmp/wrx80-glm53-tune.lock
exec 9>"$lock"
flock -n 9 || { echo "another WRX80 GLM tune probe is running" >&2; exit 75; }

health="$(curl -fsS --max-time 5 "$root/health")"
python3 - "$health" <<'PY'
import json,sys
h=json.loads(sys.argv[1]); s=h.get("scheduler") or {}
a=int(s.get("active",0) or 0); q=int(s.get("queued",0) or 0)
if a or q:
    raise SystemExit(f"refusing probe: scheduler active={a} queued={q}")
PY

req="$(mktemp /tmp/glm53-tune-request.XXXXXX.json)"
trap 'rm -f "$req"' EXIT
python3 - "$req" "$max_tokens" <<'PY'
import json,sys
unit=("Analyze this deterministic systems-programming trace. "
      "Track ownership, synchronization, memory movement, cache residency, "
      "latency, throughput, and correctness. Explain interactions precisely. ")
prompt=(unit*20)[:3400]
json.dump({"model":"glm-5.3-flash-colibri","prompt":prompt,
           "max_tokens":int(sys.argv[2]),"temperature":0,"stream":False},
          open(sys.argv[1],"w"))
PY

out="/tmp/glm53-tune-${label}.json"
echo "=== ENV ==="
systemctl show colibri-glm53.service -p Environment --no-pager
echo "=== BEFORE PROFILE ==="
curl -fsS --max-time 5 "$root/profile"; echo
echo "=== BEFORE GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits
echo "=== BEFORE MEM ==="
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo

/usr/bin/time -f 'WALL=%e RSS_KB=%M'   curl -fsS --max-time 1800 -H 'Content-Type: application/json'   -d @"$req" "$root/v1/completions" -o "$out"

python3 - "$out" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); c=(d.get("choices") or [{}])[0]
print("=== RESPONSE ===")
print(json.dumps({"usage":d.get("usage"),"finish":c.get("finish_reason"),
                  "text":c.get("text","")[:160]},indent=2))
PY
echo "=== AFTER PROFILE ==="
curl -fsS --max-time 5 "$root/profile"; echo
echo "=== AFTER GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw --format=csv,noheader,nounits
echo "=== AFTER MEM ==="
grep -E 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
