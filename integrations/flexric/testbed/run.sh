#!/usr/bin/env bash
# HandoverLLM on OAI + FlexRIC: build, start, test and stop the testbed (docs/RAN_INTEGRATION.md §9.7).
#
#   ./run.sh up        fetch OAI configs, build images, start everything in the order OAI needs
#   ./run.sh test      wait for the lab drive to finish, then check every hop and print a report
#   ./run.sh all       up + test
#   ./run.sh logs      tail the interesting logs
#   ./run.sh down      stop and remove everything
#
# Environment: DURATION_S (240), SPEED_KMH (30), PLOSS_MAX_DB (45), LIVE (unset = --live; LIVE= for
# shadow mode), OAI_TAG (2026.w39).
set -euo pipefail
cd "$(dirname "$0")"
DC="docker compose"
export DURATION_S="${DURATION_S:-240}"
mkdir -p out

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%T)" "$*"; }

wait_for() {  # wait_for <description> <timeout_s> <command...>
  local what=$1 t=$2; shift 2
  for ((i = 0; i < t; i++)); do
    if "$@" >/dev/null 2>&1; then echo "  ok: $what (${i}s)"; return 0; fi
    sleep 1
  done
  echo "  TIMEOUT: $what after ${t}s"; return 1
}

logs_have() { docker logs "$1" 2>&1 | grep -q -- "$2"; }

preflight() {
  if ! grep -qw sctp /proc/net/protocols 2>/dev/null && ! lsmod 2>/dev/null | grep -qw sctp; then
    echo "SCTP is not available in this kernel. OAI (F1, NGAP) and FlexRIC (E2) need it:"
    echo "  sudo modprobe sctp      (and make sure your kernel has CONFIG_IP_SCTP)"
    exit 1
  fi
  docker compose version >/dev/null || { echo "Docker Compose v2 is required"; exit 1; }
  [[ -f oai-conf/gnb-cu.sa.band78.106prb.conf ]] || ./fetch_oai_conf.sh
}

up() {
  preflight
  log "building local images (FlexRIC + llm_bridge, ai-ran-llm); the first build takes ~15 min"
  $DC build
  $DC pull --ignore-buildable --quiet || true

  log "5G core"
  $DC up -d mysql oai-amf oai-smf oai-upf oai-ext-dn
  wait_for "mysql healthy" 120 bash -c "docker inspect -f '{{.State.Health.Status}}' llmlab-mysql | grep -q healthy"
  sleep 5

  log "FlexRIC nearRT-RIC"
  $DC up -d nearRT-RIC
  wait_for "nearRT-RIC up" 30 logs_have llmlab-nearRT-RIC "nearRT-RIC IP Address"

  log "OAI CU (E2 agent -> nearRT-RIC)"
  $DC up -d oai-cu
  wait_for "CU E2 Setup at the RIC" 60 logs_have llmlab-nearRT-RIC "E2 SETUP-REQUEST rx"

  log "OAI nrUE (RF simulator server), then DU PCI 0, then DU PCI 1 (order fixes the channel models)"
  $DC up -d oai-nr-ue
  sleep 3
  $DC up -d --no-deps oai-du-pci0
  wait_for "UE attached (oaitun_ue1 has an IP)" 120 bash -c "docker exec llmlab-oai-nr-ue ip -4 addr show oaitun_ue1 | grep -q inet"
  $DC up -d --no-deps oai-du-pci1
  wait_for "DU PCI 1 F1 Setup at the CU" 60 bash -c "echo ci fetch_du_by_ue_id | nc -w 2 192.168.71.150 9090 >/dev/null 2>&1 || docker logs llmlab-oai-cu 2>&1 | grep -c 'F1 Setup' | grep -q 2"
  sleep 5

  log "HandoverLLM: ran-xapp (${LIVE-live}), lab drive, llm_bridge"
  $DC up -d ran-xapp
  wait_for "ran-xapp listening" 90 logs_have llmlab-ran-xapp "ran-bridge listening"
  $DC up -d --no-deps lab-drive
  sleep 2
  $DC up -d --no-deps llm-bridge
  wait_for "llm_bridge sees the CU" 60 logs_have llmlab-llm-bridge "using E2 node"
  wait_for "llm_bridge sees the UE" 60 logs_have llmlab-llm-bridge "attached"
  log "running; the lab drive stops after ${DURATION_S}s ('./run.sh test' waits and checks)"
}

check() {  # check <name> <command...>
  local name=$1; shift
  if "$@" >/dev/null 2>&1; then printf '  \033[32mPASS\033[0m  %s\n' "$name"; else printf '  \033[31mFAIL\033[0m  %s\n' "$name"; FAILS=$((FAILS + 1)); fi
}

test_run() {
  log "waiting for the lab drive to finish (${DURATION_S}s drive)"
  docker wait llmlab-lab-drive >/dev/null
  for c in llmlab-nearRT-RIC llmlab-oai-cu llmlab-oai-du-pci0 llmlab-oai-du-pci1 llmlab-oai-nr-ue llmlab-llm-bridge llmlab-ran-xapp llmlab-lab-drive; do
    docker logs "$c" > "out/${c#llmlab-}.log" 2>&1 || true
  done
  echo ci fetch_du_by_ue_id | nc -w 2 192.168.71.150 9090 > out/cu_fetch_du_by_ue_id.txt 2>&1 || true

  log "checks (logs in $(pwd)/out)"
  FAILS=0
  check "E2 Setup: CU connected to FlexRIC"                   grep -q "E2 SETUP-REQUEST rx" out/nearRT-RIC.log
  check "llm_bridge found the CU (RC Handover Control)"       grep -q "using E2 node" out/llm-bridge.log
  check "llm_bridge: UE context over E2 (Style 5)"            grep -q "attached" out/llm-bridge.log
  check "lab drive: synthetic reports sent to ran-xapp"       python3 -c "import json,sys; s=json.load(open('out/lab_drive_summary.json')); sys.exit(s['reports_synthetic'] < 100)"
  check "ran-xapp: model issued handover commands"            python3 -c "import json,sys; s=json.load(open('out/lab_drive_summary.json')); sys.exit(s['ho_commands'] < 1)"
  check "llm_bridge: sent RC Handover Control"                grep -q "Handover Control" out/llm-bridge.log
  check "RIC: CONTROL acknowledged by the CU"                 grep -q "CONTROL ACKNOWLEDGE rx" out/nearRT-RIC.log
  check "CU: E2 Handover Control -> F1 handover triggered"    grep -q "RC Control: F1 Handover Control" out/oai-cu.log
  check "CU: F1 handover completed (source released)"         grep -q "Handover: trigger release on cell PCI" out/oai-cu.log
  check "E2: UE seen on the new cell (serving changed)"       python3 -c "import json,sys; s=json.load(open('out/lab_drive_summary.json')); sys.exit(s['handovers_executed'] < 1)"
  check "UE still attached at the end"                        docker exec llmlab-oai-nr-ue ip -4 addr show oaitun_ue1

  log "summary"
  python3 - <<'PY'
import json
s = json.load(open("out/lab_drive_summary.json"))
print(f"  drive {s['elapsed_s']} s | reports {s['reports_synthetic']} synthetic, {s['reports_real']} real | "
      f"decisions {s['decisions']} | ho_commands {s['ho_commands']} | serving changes over E2 {s['handovers_executed']} | "
      f"outcomes {s['ho_outcomes']}")
for e in s["events"]:
    if e["event"] in ("ho_command", "serving_changed", "ho_outcome"):
        print("   ", e)
PY
  grep -h "F1 Handover Control\|Handover triggered\|trigger release on cell" out/oai-cu.log | tail -8 | sed 's/^/    CU: /'
  if ((FAILS)); then log "$FAILS check(s) failed"; exit 1; else log "all checks passed"; fi
}

case "${1:-all}" in
  up) up ;;
  test) test_run ;;
  all) up; test_run ;;
  logs) docker logs -f --tail 20 llmlab-llm-bridge & docker logs -f --tail 20 llmlab-lab-drive & docker logs -f --tail 5 llmlab-oai-cu 2>&1 | grep --line-buffered -E "E2 AGENT|Handover|HO " ;;
  down) $DC down -v --remove-orphans ;;
  *) echo "usage: $0 up|test|all|logs|down"; exit 2 ;;
esac
