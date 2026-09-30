#!/usr/bin/env bash
# Download the OAI configuration files this testbed reuses from OAI's F1 RF-simulator CI
# (ci-scripts/yaml_files/5g_f1_rfsimulator), at a pinned OAI commit, into ./oai-conf/.
# The CU config gets an e2_agent section pointing at the nearRT-RIC (192.168.71.160).
set -euo pipefail
OAI_COMMIT="${OAI_COMMIT:-29d5fa7d38bb1ee8935ef7ae30a396d53772ea4f}"
BASE="https://raw.githubusercontent.com/OPENAIRINTERFACE/openairinterface5g/${OAI_COMMIT}"
cd "$(dirname "$0")"
mkdir -p oai-conf
for f in ci-scripts/conf_files/gnb-cu.sa.band78.106prb.conf \
         ci-scripts/conf_files/gnb-du.sa.band78.106prb.rfsim.conf \
         ci-scripts/conf_files/nrue.uicc.conf \
         ci-scripts/conf_files/neighbour-config.conf \
         ci-scripts/yaml_files/5g_rfsimulator/oai_db.sql \
         ci-scripts/yaml_files/5g_rfsimulator/mysql-healthcheck.sh \
         ci-scripts/yaml_files/5g_rfsimulator_e1/mini_nonrf_config_3slices.yaml; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "oai-conf/$(basename "$f")"
done
chmod +x oai-conf/mysql-healthcheck.sh
if ! grep -q "^e2_agent" oai-conf/gnb-cu.sa.band78.106prb.conf; then
  cat >> oai-conf/gnb-cu.sa.band78.106prb.conf <<'CONF'

e2_agent = {
  near_ric_ip_addr = "192.168.71.160";
  sm_dir = "/usr/local/lib/flexric/";
};
CONF
fi
echo "OAI config files (OAI ${OAI_COMMIT}) are in $(pwd)/oai-conf"
