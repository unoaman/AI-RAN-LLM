/*
 * llm_bridge: FlexRIC xApp between OAI's E2 agent and the HandoverLLM xApp (ran-xapp).
 *
 * docs/RAN_INTEGRATION.md §9. Builds inside a FlexRIC source tree as
 * examples/xApp/c/llm_bridge (see CMakeLists.txt and ../testbed/Dockerfile.llm_bridge).
 *
 *   E2 node (OAI CU / gNB)                      llm_bridge                        ran-xapp / lab drive
 *   RC REPORT Style 5 (on demand, polled) ----> ue_context {ue_id, serving NR-CGI} ---->
 *   RC REPORT Style 1 (message copy:      ----> meas_report (TS 38.133 -> dBm)      ---->
 *     UL-DCCH MeasurementReport)
 *   RC CONTROL Style 3 / Action 1         <---- ho_command {ue_id, target plmn+nci} <----
 *     (Handover Control, target NR-CGI)          ho_outcome on refusal              ---->
 *
 * The link to ran-xapp (or to `ran-lab-drive`) is the ran-bridge protocol: NDJSON over TCP.
 * Environment: LLM_BRIDGE_XAPP=host:port (default 127.0.0.1:7001), LLM_BRIDGE_POLL_MS (500).
 * FlexRIC's own options (-c flexric.conf, ...) are parsed by init_fr_args as usual.
 *
 * OAI specifics this relies on (openair2/E2AP/RAN_FUNCTION/O-RAN/ran_func_rc.c):
 *  - Style 5 answers each subscription once, immediately, with every UE and its PCell NR-CGI;
 *    UE ID = GNB_UE_ID_E2SM with ran_ue_id = RRC UE id. The bridge polls by re-subscribing.
 *  - Handover Control needs that GNB_UE_ID_E2SM with ran_ue_id; the CU runs an F1 handover to
 *    one of its own cells, or N2 to a configured neighbour.
 *  - Message-copy indications carry only the RRC message, no UE ID, so a copied
 *    MeasurementReport is attributed only when exactly one UE is on that E2 node.
 *
 * Status: compiled against FlexRIC d7a71285 (the commit OAI pins); not yet run against a live
 * nearRT-RIC in this repository.
 */

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/sm/rc_sm/ie/rc_data_ie.h"
#include "../../../../src/sm/rc_sm/rc_sm_id.h"
#include "../../../../src/sm/rc_sm/ie/ir/ran_param_struct.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/conversions.h"

#include "NR_UL-DCCH-Message.h"
#include "NR_MeasurementReport.h"
#include "NR_MeasResults.h"
#include "NR_MeasResultListNR.h"

#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define MAX_NODES 8
#define MAX_UES 64

static const uint32_t RC_CTRL_STYLE_CONN_MODE_MOBILITY = 3;   // E2SM-RC 7.6.4
static const uint32_t RC_REPORT_STYLE_MESSAGE_COPY = 1;       // E2SM-RC 7.4.2
static const uint32_t RC_REPORT_STYLE_ON_DEMAND = 5;          // E2SM-RC 7.4.6
static const uint16_t EV_COND_MEAS_REPORT = 2;
static const uint32_t RRC_MSG_ID_MEASUREMENT_REPORT = 1;      // UL-DCCH c1 choice index

/* ------------------------------------------------------------------ state */

typedef struct {
  bool used;
  bool seen;                 // present in the latest Style 5 poll
  int node;                  // index into nodes[]
  uint64_t rrc_ue_id;
  ue_id_e2sm_t ue_id;        // echoed back in Handover Control
  nr_cgi_t serving;
  char key[64];              // ran-bridge ue_id
} ue_t;

typedef struct {
  global_e2_node_id_t id;
  char name[48];
  bool msg_copy;
  sem_t poll_sem;
} node_t;

static node_t nodes[MAX_NODES];
static int n_nodes;
static ue_t ues[MAX_UES];
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t stop_flag;

static int sock_fd = -1;
static pthread_mutex_t sock_mtx = PTHREAD_MUTEX_INITIALIZER;
static char xapp_host[128] = "127.0.0.1";
static int xapp_port = 7001;

static double now_s(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------ ran-bridge link */

static int connect_xapp(void)
{
  char port[16];
  snprintf(port, sizeof(port), "%d", xapp_port);
  struct addrinfo hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM}, *res = NULL;
  for (;;) {
    if (stop_flag)
      return -1;
    if (getaddrinfo(xapp_host, port, &hints, &res) == 0) {
      int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
        freeaddrinfo(res);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        printf("[llm_bridge] connected to %s:%d\n", xapp_host, xapp_port);
        return fd;
      }
      if (fd >= 0)
        close(fd);
      freeaddrinfo(res);
    }
    fprintf(stderr, "[llm_bridge] waiting for %s:%d ...\n", xapp_host, xapp_port);
    sleep(1);
  }
}

static void send_line(const char* line)
{
  pthread_mutex_lock(&sock_mtx);
  if (sock_fd >= 0) {
    size_t len = strlen(line), off = 0;
    while (off < len) {
      ssize_t n = send(sock_fd, line + off, len - off, MSG_NOSIGNAL);
      if (n <= 0) {
        fprintf(stderr, "[llm_bridge] send failed: %s\n", strerror(errno));
        break;
      }
      off += (size_t)n;
    }
  }
  pthread_mutex_unlock(&sock_mtx);
}

/* Minimal JSON field extraction for the fixed-shape messages ran-xapp sends. */
static const char* json_find(const char* s, const char* key)
{
  char pat[64];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(s, pat);
  if (p == NULL)
    return NULL;
  p += strlen(pat);
  while (*p == ' ' || *p == '\t')
    p++;
  if (*p != ':')
    return NULL;
  p++;
  while (*p == ' ' || *p == '\t')
    p++;
  return p;
}

static bool json_str(const char* s, const char* key, char* out, size_t sz)
{
  const char* p = json_find(s, key);
  if (p == NULL || *p != '"')
    return false;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i + 1 < sz)
    out[i++] = *p++;
  out[i] = '\0';
  return *p == '"';
}

static bool json_u64(const char* s, const char* key, uint64_t* out)
{
  const char* p = json_find(s, key);
  if (p == NULL)
    return false;
  char* end = NULL;
  *out = strtoull(p, &end, 0);
  return end != p;
}

/* ------------------------------------------------------------- identities */

static void node_name(const global_e2_node_id_t* id, char* out, size_t sz)
{
  snprintf(out, sz, "%03d-%02d/%" PRIu32, id->plmn.mcc, id->plmn.mnc, id->nb_id.nb_id);
  if (id->cu_du_id != NULL) {
    size_t n = strlen(out);
    snprintf(out + n, sz - n, "/%" PRIu64, *id->cu_du_id);
  }
}

static byte_array_t encode_nr_cgi(const nr_cgi_t* nr_cgi)
{
  byte_array_t dst = {.len = 8};
  dst.buf = calloc(dst.len, 1);
  assert(dst.buf != NULL && "Memory exhausted");
  const int mcc = nr_cgi->plmn_id.mcc, mnc = nr_cgi->plmn_id.mnc, mnc_digit_len = nr_cgi->plmn_id.mnc_digit_len;
  dst.buf[0] = (MCC_MNC_DECIMAL(mcc) << 4) | MCC_HUNDREDS(mcc);
  dst.buf[1] = (MNC_HUNDREDS(mnc, mnc_digit_len) << 4) | MCC_MNC_DIGIT(mcc);
  dst.buf[2] = (MCC_MNC_DIGIT(mnc) << 4) | MCC_MNC_DECIMAL(mnc);
  const uint64_t cell_id = nr_cgi->nr_cell_id;
  dst.buf[3] = (cell_id >> 28) & 0xFF;
  dst.buf[4] = (cell_id >> 20) & 0xFF;
  dst.buf[5] = (cell_id >> 12) & 0xFF;
  dst.buf[6] = (cell_id >> 4) & 0xFF;
  dst.buf[7] = (cell_id & 0xF) << 4;
  return dst;
}

/* "00101" -> MCC 001, MNC 01 (2 digits); "001001" -> MNC 001 (3 digits). */
static bool parse_plmn(const char* s, e2sm_plmn_t* out)
{
  size_t n = strlen(s);
  if (n != 5 && n != 6)
    return false;
  char mcc[4] = {0}, mnc[4] = {0};
  memcpy(mcc, s, 3);
  memcpy(mnc, s + 3, n - 3);
  out->mcc = (uint16_t)atoi(mcc);
  out->mnc = (uint16_t)atoi(mnc);
  out->mnc_digit_len = (uint8_t)(n - 3);
  return true;
}

/* -------------------------------------------------------- E2 RC plumbing */

static ran_param_val_type_t wrap_in_struct(seq_ran_param_t inner)
{
  ran_param_val_type_t dst = {.type = STRUCTURE_RAN_PARAMETER_VAL_TYPE};
  dst.strct = calloc(1, sizeof(ran_param_struct_t));
  assert(dst.strct != NULL && "Memory exhausted");
  dst.strct->sz_ran_param_struct = 1;
  dst.strct->ran_param_struct = calloc(1, sizeof(seq_ran_param_t));
  assert(dst.strct->ran_param_struct != NULL && "Memory exhausted");
  dst.strct->ran_param_struct[0] = inner;
  return dst;
}

/* Target Primary Cell ID > CHOICE Target Cell > NR Cell ID > NR CGI (E2SM-RC 8.4.4.1). */
static rc_ctrl_req_data_t gen_handover_ctrl(const ue_id_e2sm_t* ue_id, byte_array_t nr_cgi)
{
  seq_ran_param_t nr_cgi_p = {.ran_param_id = NR_CGI_8_4_4_1};
  nr_cgi_p.ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
  nr_cgi_p.ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
  assert(nr_cgi_p.ran_param_val.flag_false != NULL && "Memory exhausted");
  nr_cgi_p.ran_param_val.flag_false->type = OCTET_STRING_RAN_PARAMETER_VALUE;
  nr_cgi_p.ran_param_val.flag_false->octet_str_ran = nr_cgi;
  seq_ran_param_t nr_cell = {.ran_param_id = NR_CELL_8_4_4_1, .ran_param_val = wrap_in_struct(nr_cgi_p)};
  seq_ran_param_t choice = {.ran_param_id = CHOICE_TARGET_CELL_8_4_4_1, .ran_param_val = wrap_in_struct(nr_cell)};
  seq_ran_param_t target = {.ran_param_id = TARGET_PRIMARY_CELL_ID_8_4_4_1, .ran_param_val = wrap_in_struct(choice)};

  rc_ctrl_req_data_t dst = {0};
  dst.hdr.format = FORMAT_1_E2SM_RC_CTRL_HDR;
  dst.hdr.frmt_1.ric_style_type = RC_CTRL_STYLE_CONN_MODE_MOBILITY;
  dst.hdr.frmt_1.ctrl_act_id = HANDOVER_CONTROL_7_6_4_1;
  dst.hdr.frmt_1.ue_id = cp_ue_id_e2sm(ue_id);
  dst.msg.format = FORMAT_1_E2SM_RC_CTRL_MSG;
  dst.msg.frmt_1.sz_ran_param = 1;
  dst.msg.frmt_1.ran_param = calloc(1, sizeof(seq_ran_param_t));
  assert(dst.msg.frmt_1.ran_param != NULL && "Memory exhausted");
  dst.msg.frmt_1.ran_param[0] = target;
  return dst;
}

static param_report_def_t param_report(uint32_t id)
{
  param_report_def_t p = {0};
  p.ran_param_id = id;
  return p;
}

static rc_sub_data_t gen_sub_on_demand(void)
{
  rc_sub_data_t s = {0};
  s.et.format = FORMAT_5_E2SM_RC_EV_TRIGGER_FORMAT;
  s.et.frmt_5.on_demand = TRUE_ON_DEMAND_FRMT_5;
  s.sz_ad = 1;
  s.ad = calloc(1, sizeof(e2sm_rc_action_def_t));
  assert(s.ad != NULL && "Memory exhausted");
  s.ad[0].ric_style_type = RC_REPORT_STYLE_ON_DEMAND;
  s.ad[0].format = FORMAT_1_E2SM_RC_ACT_DEF;
  s.ad[0].frmt_1.sz_param_report_def = 1;
  s.ad[0].frmt_1.param_report_def = calloc(1, sizeof(param_report_def_t));
  assert(s.ad[0].frmt_1.param_report_def != NULL && "Memory exhausted");
  s.ad[0].frmt_1.param_report_def[0] = param_report(E2SM_RC_RS5_UE_CONTEXT_INFORMATION);
  return s;
}

static rc_sub_data_t gen_sub_meas_report_copy(void)
{
  rc_sub_data_t s = {0};
  s.et.format = FORMAT_1_E2SM_RC_EV_TRIGGER_FORMAT;
  s.et.frmt_1.sz_msg_ev_trg = 1;
  s.et.frmt_1.msg_ev_trg = calloc(1, sizeof(msg_ev_trg_t));
  assert(s.et.frmt_1.msg_ev_trg != NULL && "Memory exhausted");
  msg_ev_trg_t* t = &s.et.frmt_1.msg_ev_trg[0];
  t->ev_trigger_cond_id = EV_COND_MEAS_REPORT;
  t->msg_type = RRC_MSG_MSG_TYPE_EV_TRG;
  t->rrc_msg.type = NR_RRC_MESSAGE_ID;
  t->rrc_msg.nr = UL_DCCH_NR_RRC_CLASS;
  t->rrc_msg.rrc_msg_id = RRC_MSG_ID_MEASUREMENT_REPORT;
  s.sz_ad = 1;
  s.ad = calloc(1, sizeof(e2sm_rc_action_def_t));
  assert(s.ad != NULL && "Memory exhausted");
  s.ad[0].ric_style_type = RC_REPORT_STYLE_MESSAGE_COPY;
  s.ad[0].format = FORMAT_1_E2SM_RC_ACT_DEF;
  s.ad[0].frmt_1.sz_param_report_def = 1;
  s.ad[0].frmt_1.param_report_def = calloc(1, sizeof(param_report_def_t));
  assert(s.ad[0].frmt_1.param_report_def != NULL && "Memory exhausted");
  s.ad[0].frmt_1.param_report_def[0] = param_report(E2SM_RC_RS1_RRC_MESSAGE);
  return s;
}

static bool supports(const sm_ran_function_t* rf, size_t len, bool* ho, bool* on_demand, bool* msg_copy)
{
  *ho = *on_demand = *msg_copy = false;
  for (size_t i = 0; i < len; i++) {
    if (rf[i].id != SM_RC_ID)
      continue;
    const ran_func_def_ctrl_t* c = rf[i].defn.rc.ctrl;
    for (size_t k = 0; c != NULL && k < c->sz_seq_ctrl_style; k++)
      if (c->seq_ctrl_style[k].style_type == RC_CTRL_STYLE_CONN_MODE_MOBILITY)
        for (size_t j = 0; j < c->seq_ctrl_style[k].sz_seq_ctrl_act; j++)
          if (c->seq_ctrl_style[k].seq_ctrl_act[j].id == HANDOVER_CONTROL_7_6_4_1)
            *ho = true;
    const ran_func_def_report_t* r = rf[i].defn.rc.report;
    for (size_t k = 0; r != NULL && k < r->sz_seq_report_sty; k++) {
      if (r->seq_report_sty[k].report_type == RC_REPORT_STYLE_ON_DEMAND)
        *on_demand = true;
      if (r->seq_report_sty[k].report_type == RC_REPORT_STYLE_MESSAGE_COPY)
        *msg_copy = true;
    }
    return true;
  }
  return false;
}

/* ---------------------------------------------------------- UE context (Style 5) */

static ue_t* find_ue_key(const char* key)
{
  for (int i = 0; i < MAX_UES; i++)
    if (ues[i].used && strcmp(ues[i].key, key) == 0)
      return &ues[i];
  return NULL;
}

static void on_ue_context(int node, const e2sm_rc_ind_msg_frmt_4_t* msg)
{
  char line[768];
  pthread_mutex_lock(&mtx);
  for (int i = 0; i < MAX_UES; i++)
    if (ues[i].used && ues[i].node == node)
      ues[i].seen = false;
  for (size_t k = 0; k < msg->sz_seq_ue_info; k++) {
    const seq_ue_info_t* info = &msg->seq_ue_info[k];
    if (info->ue_id.type != GNB_UE_ID_E2SM || info->ue_id.gnb.ran_ue_id == NULL)
      continue;                                   // Handover Control needs GNB_UE_ID with ran_ue_id
    const uint64_t rrc_ue_id = *info->ue_id.gnb.ran_ue_id;
    char key[64];
    snprintf(key, sizeof(key), "rrc_ue_id=%" PRIu64 "@%s", rrc_ue_id, nodes[node].name);
    ue_t* ue = find_ue_key(key);
    if (ue == NULL) {
      for (int i = 0; i < MAX_UES && ue == NULL; i++)
        if (!ues[i].used)
          ue = &ues[i];
      if (ue == NULL)
        continue;
      memset(ue, 0, sizeof(*ue));
      ue->used = true;
      ue->node = node;
      ue->rrc_ue_id = rrc_ue_id;
      snprintf(ue->key, sizeof(ue->key), "%s", key);
      printf("[llm_bridge] UE %s attached\n", key);
    } else {
      free_ue_id_e2sm(&ue->ue_id);
    }
    ue->seen = true;
    ue->ue_id = cp_ue_id_e2sm(&info->ue_id);
    ue->serving = info->cell_global_id.nr_cgi;
    snprintf(line, sizeof(line),
             "{\"type\":\"ue_context\",\"ue_id\":\"%s\",\"timestamp_s\":%.3f,"
             "\"ue_ids\":{\"ran_ue_id\":%" PRIu64 ",\"amf_ue_ngap_id\":%" PRIu64 ",\"e2_node\":\"%s\"},"
             "\"serving\":{\"nci\":%" PRIu64 ",\"plmn\":\"%03d%0*d\"}}\n",
             ue->key, now_s(), rrc_ue_id, info->ue_id.gnb.amf_ue_ngap_id, nodes[node].name,
             (uint64_t)ue->serving.nr_cell_id, ue->serving.plmn_id.mcc, ue->serving.plmn_id.mnc_digit_len,
             ue->serving.plmn_id.mnc);
    send_line(line);
  }
  for (int i = 0; i < MAX_UES; i++)
    if (ues[i].used && ues[i].node == node && !ues[i].seen) {
      snprintf(line, sizeof(line), "{\"type\":\"ue_release\",\"ue_id\":\"%s\"}\n", ues[i].key);
      send_line(line);
      printf("[llm_bridge] UE %s released\n", ues[i].key);
      free_ue_id_e2sm(&ues[i].ue_id);
      ues[i].used = false;
    }
  pthread_mutex_unlock(&mtx);
}

/* ------------------------------------------- message copy: MeasurementReport */

static int append_cell(char* out, size_t sz, const NR_MeasResultNR_t* r, bool serving, uint64_t nci)
{
  const NR_MeasQuantityResults_t* q = r->measResult.cellResults.resultsSSB_Cell;
  if (q == NULL || q->rsrp == NULL)
    return 0;
  int n = snprintf(out, sz, "{");
  if (r->physCellId != NULL)
    n += snprintf(out + n, sz - n, "\"pci\":%ld,", *r->physCellId);
  if (serving && nci != 0)
    n += snprintf(out + n, sz - n, "\"nci\":%" PRIu64 ",", nci);
  n += snprintf(out + n, sz - n, "\"rsrp_dbm\":%ld", *q->rsrp - 157);          // TS 38.133 10.1.6
  if (q->rsrq != NULL)
    n += snprintf(out + n, sz - n, ",\"rsrq_db\":%.1f", (*q->rsrq - 87) / 2.0);  // TS 38.133 10.1.11
  if (q->sinr != NULL)
    n += snprintf(out + n, sz - n, ",\"sinr_db\":%.1f", (*q->sinr - 47) / 2.0);  // TS 38.133 10.1.16
  n += snprintf(out + n, sz - n, "}");
  return n;
}

static void on_meas_report_copy(int node, const byte_array_t* rrc)
{
  NR_UL_DCCH_Message_t* msg = NULL;
  asn_dec_rval_t rv = uper_decode(NULL, &asn_DEF_NR_UL_DCCH_Message, (void**)&msg, rrc->buf, rrc->len, 0, 0);
  defer({ ASN_STRUCT_FREE(asn_DEF_NR_UL_DCCH_Message, msg); });
  if (rv.code != RC_OK || msg->message.present != NR_UL_DCCH_MessageType_PR_c1
      || msg->message.choice.c1->present != NR_UL_DCCH_MessageType__c1_PR_measurementReport)
    return;
  const NR_MeasurementReport_t* mr = msg->message.choice.c1->choice.measurementReport;
  if (mr->criticalExtensions.present != NR_MeasurementReport__criticalExtensions_PR_measurementReport)
    return;
  const NR_MeasResults_t* res = &mr->criticalExtensions.choice.measurementReport->measResults;
  if (res->measResultServingMOList.list.count < 1)
    return;

  pthread_mutex_lock(&mtx);
  ue_t* ue = NULL;
  int n_on_node = 0;
  for (int i = 0; i < MAX_UES; i++)
    if (ues[i].used && ues[i].node == node) {
      n_on_node++;
      ue = &ues[i];
    }
  if (n_on_node != 1) {                           // OAI's copy has no UE ID: attribute only if unambiguous
    pthread_mutex_unlock(&mtx);
    static unsigned dropped;
    if (dropped++ % 50 == 0)
      fprintf(stderr, "[llm_bridge] MeasurementReport not attributable (%d UEs on node %s), dropped\n",
              n_on_node, nodes[node].name);
    return;
  }
  char line[4096];
  int n = snprintf(line, sizeof(line), "{\"type\":\"meas_report\",\"ue_id\":\"%s\",\"timestamp_s\":%.3f,"
                   "\"ue_ids\":{\"ran_ue_id\":%" PRIu64 ",\"e2_node\":\"%s\"},\"source\":\"e2-msg-copy\",\"serving\":",
                   ue->key, now_s(), ue->rrc_ue_id, nodes[node].name);
  const uint64_t serving_nci = ue->serving.nr_cell_id;
  pthread_mutex_unlock(&mtx);

  int c = append_cell(line + n, sizeof(line) - n, &res->measResultServingMOList.list.array[0]->measResultServingCell,
                      true, serving_nci);
  if (c == 0)
    return;
  n += c;
  n += snprintf(line + n, sizeof(line) - n, ",\"neighbours\":[");
  bool first = true;
  if (res->measResultNeighCells != NULL
      && res->measResultNeighCells->present == NR_MeasResults__measResultNeighCells_PR_measResultListNR) {
    const NR_MeasResultListNR_t* l = res->measResultNeighCells->choice.measResultListNR;
    for (int i = 0; i < l->list.count && n < (int)sizeof(line) - 128; i++) {
      char cell[160];
      if (append_cell(cell, sizeof(cell), l->list.array[i], false, 0) == 0 || l->list.array[i]->physCellId == NULL)
        continue;
      n += snprintf(line + n, sizeof(line) - n, "%s%s", first ? "" : ",", cell);
      first = false;
    }
  }
  snprintf(line + n, sizeof(line) - n, "]}\n");
  send_line(line);
}

/* ------------------------------------------------------ indication callbacks */

static void handle_indication(int node, sm_ag_if_rd_t const* rd)
{
  if (rd->type != INDICATION_MSG_AGENT_IF_ANS_V0 || rd->ind.type != RAN_CTRL_STATS_V1_03)
    return;
  const rc_ind_data_t* ind = &rd->ind.rc.ind;
  if (ind->msg.format == FORMAT_4_E2SM_RC_IND_MSG) {
    on_ue_context(node, &ind->msg.frmt_4);
    sem_post(&nodes[node].poll_sem);
  } else if (ind->msg.format == FORMAT_1_E2SM_RC_IND_MSG) {
    for (size_t i = 0; i < ind->msg.frmt_1.sz_seq_ran_param; i++) {
      const seq_ran_param_t* p = &ind->msg.frmt_1.seq_ran_param[i];
      if (p->ran_param_id == E2SM_RC_RS1_RRC_MESSAGE && p->ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE
          && p->ran_param_val.flag_false->type == OCTET_STRING_RAN_PARAMETER_VALUE)
        on_meas_report_copy(node, &p->ran_param_val.flag_false->octet_str_ran);
    }
  }
}

/* sm_cb carries no user data: one trampoline per node slot. */
#define CB(i) static void cb_##i(sm_ag_if_rd_t const* rd) { handle_indication(i, rd); }
CB(0) CB(1) CB(2) CB(3) CB(4) CB(5) CB(6) CB(7)
static const sm_cb node_cb[MAX_NODES] = {cb_0, cb_1, cb_2, cb_3, cb_4, cb_5, cb_6, cb_7};

/* --------------------------------------------------------- ho_command -> E2 */

static void send_outcome(const char* cmd_id, const char* ue_id, const char* status, const char* detail)
{
  char line[512];
  snprintf(line, sizeof(line), "{\"type\":\"ho_outcome\",\"command_id\":\"%s\",\"ue_id\":\"%s\",\"status\":\"%s\",\"detail\":\"%s\"}\n",
           cmd_id, ue_id, status, detail);
  send_line(line);
}

static void on_ho_command(const char* line)
{
  char cmd_id[64] = "", ue_key[64] = "", plmn_s[8] = "";
  uint64_t nci = 0;
  const char* tgt = strstr(line, "\"target_cell\"");
  if (!json_str(line, "command_id", cmd_id, sizeof(cmd_id)) || !json_str(line, "ue_id", ue_key, sizeof(ue_key))
      || tgt == NULL || !json_u64(tgt, "nci", &nci) || !json_str(tgt, "plmn", plmn_s, sizeof(plmn_s))) {
    fprintf(stderr, "[llm_bridge] ho_command without command_id / ue_id / target nci / plmn: %s", line);
    if (cmd_id[0] && ue_key[0])
      send_outcome(cmd_id, ue_key, "rejected", "target cell needs nci and plmn in the cell map");
    return;
  }
  nr_cgi_t target = {.nr_cell_id = nci};
  if (!parse_plmn(plmn_s, &target.plmn_id)) {
    send_outcome(cmd_id, ue_key, "rejected", "bad plmn");
    return;
  }

  pthread_mutex_lock(&mtx);
  ue_t* ue = find_ue_key(ue_key);
  if (ue == NULL) {
    pthread_mutex_unlock(&mtx);
    send_outcome(cmd_id, ue_key, "rejected", "unknown UE");
    return;
  }
  const int node = ue->node;
  rc_ctrl_req_data_t ctrl = gen_handover_ctrl(&ue->ue_id, encode_nr_cgi(&target));
  pthread_mutex_unlock(&mtx);
  defer({ free_rc_ctrl_req_data(&ctrl); });

  printf("[llm_bridge] Handover Control %s: %s -> NR cell %" PRIu64 " (PLMN %s) on %s\n",
         cmd_id, ue_key, nci, plmn_s, nodes[node].name);
  const sm_ans_xapp_t ans = control_sm_xapp_api(&nodes[node].id, SM_RC_ID, &ctrl);
  if (!ans.success)
    send_outcome(cmd_id, ue_key, "failure", "E2 node refused the RC CONTROL");
  /* Success is not reported here: OAI answers the CONTROL before the handover runs.
   * ran-xapp infers it from the next ue_context / meas_report with the new serving cell. */
}

static void* rx_thread(void* arg)
{
  (void)arg;
  char buf[8192];
  size_t used = 0;
  while (!stop_flag) {
    ssize_t n = recv(sock_fd, buf + used, sizeof(buf) - 1 - used, 0);
    if (n <= 0) {
      fprintf(stderr, "[llm_bridge] connection to ran-xapp lost, reconnecting\n");
      pthread_mutex_lock(&sock_mtx);
      close(sock_fd);
      sock_fd = -1;
      pthread_mutex_unlock(&sock_mtx);
      int fd = connect_xapp();
      pthread_mutex_lock(&sock_mtx);
      sock_fd = fd;
      pthread_mutex_unlock(&sock_mtx);
      send_line("{\"type\":\"hello\",\"node\":\"llm_bridge\",\"protocol\":\"ai-ran-llm/ran-bridge/1\"}\n");
      used = 0;
      continue;
    }
    used += (size_t)n;
    buf[used] = '\0';
    char* start = buf;
    char* nl;
    while ((nl = strchr(start, '\n')) != NULL) {
      *nl = '\0';
      char type[32] = "";
      if (json_str(start, "type", type, sizeof(type)) && strcmp(type, "ho_command") == 0) {
        strcat(start, "\n");
        on_ho_command(start);
      }
      start = nl + 1;
    }
    used = strlen(start);
    memmove(buf, start, used);
    if (used == sizeof(buf) - 1)
      used = 0;                                   // oversized line: drop
  }
  return NULL;
}

/* ------------------------------------------------------------------- main */

static void on_signal(int sig)
{
  (void)sig;
  stop_flag = 1;
}

int main(int argc, char* argv[])
{
  const char* addr = getenv("LLM_BRIDGE_XAPP");
  if (addr != NULL) {
    const char* colon = strrchr(addr, ':');
    if (colon != NULL) {
      snprintf(xapp_host, sizeof(xapp_host), "%.*s", (int)(colon - addr), addr);
      xapp_port = atoi(colon + 1);
    }
  }
  const char* poll = getenv("LLM_BRIDGE_POLL_MS");
  const int poll_ms = poll != NULL ? atoi(poll) : 500;
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);

  fr_args_t args = init_fr_args(argc, argv);
  init_xapp_api(&args);
  sleep(1);

  e2_node_arr_xapp_t arr = e2_nodes_xapp_api();
  defer({ free_e2_node_arr_xapp(&arr); });
  for (int i = 0; i < arr.len && n_nodes < MAX_NODES; i++) {
    const e2_node_connected_xapp_t* n = &arr.n[i];
    bool ho, on_demand, msg_copy;
    if (n->id.type != ngran_gNB && n->id.type != ngran_gNB_CU && n->id.type != ngran_gNB_CUCP)
      continue;                                   // the CU(-CP) owns UE contexts and handovers
    if (!supports(n->rf, n->len_rf, &ho, &on_demand, &msg_copy) || !ho || !on_demand) {
      printf("[llm_bridge] E2 node %d skipped: no RC Handover Control / On Demand report\n", n->id.nb_id.nb_id);
      continue;
    }
    node_t* nd = &nodes[n_nodes];
    nd->id = cp_global_e2_node_id(&n->id);
    node_name(&nd->id, nd->name, sizeof(nd->name));
    nd->msg_copy = msg_copy;
    sem_init(&nd->poll_sem, 0, 0);
    printf("[llm_bridge] using E2 node %s (message copy: %s)\n", nd->name, msg_copy ? "yes" : "no");
    n_nodes++;
  }
  if (n_nodes == 0) {
    fprintf(stderr, "[llm_bridge] no E2 node with RC Handover Control; is the gNB/CU connected?\n");
    return EXIT_FAILURE;
  }

  sock_fd = connect_xapp();
  if (sock_fd < 0)
    return EXIT_FAILURE;
  send_line("{\"type\":\"hello\",\"node\":\"llm_bridge\",\"protocol\":\"ai-ran-llm/ran-bridge/1\"}\n");
  pthread_t rx;
  pthread_create(&rx, NULL, rx_thread, NULL);

  int copy_handle[MAX_NODES];
  for (int i = 0; i < n_nodes; i++) {
    copy_handle[i] = -1;
    if (!nodes[i].msg_copy)
      continue;
    rc_sub_data_t s = gen_sub_meas_report_copy();
    const sm_ans_xapp_t a = report_sm_xapp_api(&nodes[i].id, SM_RC_ID, &s, node_cb[i]);
    free_rc_sub_data(&s);
    if (a.success)
      copy_handle[i] = a.u.handle;
    printf("[llm_bridge] MeasurementReport copy on %s: %s\n", nodes[i].name, a.success ? "subscribed" : "refused");
  }

  while (!stop_flag) {
    for (int i = 0; i < n_nodes; i++) {
      rc_sub_data_t s = gen_sub_on_demand();
      const sm_ans_xapp_t a = report_sm_xapp_api(&nodes[i].id, SM_RC_ID, &s, node_cb[i]);
      free_rc_sub_data(&s);
      if (!a.success)
        continue;
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 2;
      sem_timedwait(&nodes[i].poll_sem, &ts);
      rm_report_sm_xapp_api(a.u.handle);
    }
    usleep((useconds_t)poll_ms * 1000);
  }

  for (int i = 0; i < n_nodes; i++)
    if (copy_handle[i] >= 0)
      rm_report_sm_xapp_api(copy_handle[i]);
  while (try_stop_xapp_api() == false)
    usleep(1000);
  return EXIT_SUCCESS;
}
