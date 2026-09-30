/*
 * Offline tests for llm_bridge.c (no RIC, no E2 node): the pieces between E2 and the ran-bridge.
 * Build: target llm_bridge_test (CMakeLists.txt). Prints the JSON lines it produced, one per line,
 * so a script can check them with the Python protocol classes (see testbed/check_bridge_test.py).
 */

#define main llm_bridge_main
#include "llm_bridge.c"
#undef main

#include "NR_MeasurementReport-IEs.h"
#include "NR_MeasResultServMO.h"
#include "NR_MeasQuantityResults.h"

static int fails;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static long* lv(long v) { long* p = calloc(1, sizeof(long)); *p = v; return p; }

static NR_MeasResultNR_t* cell(long pci, long rsrp, long rsrq, long sinr)
{
  NR_MeasResultNR_t* r = calloc(1, sizeof(*r));
  r->physCellId = lv(pci);
  r->measResult.cellResults.resultsSSB_Cell = calloc(1, sizeof(NR_MeasQuantityResults_t));
  r->measResult.cellResults.resultsSSB_Cell->rsrp = lv(rsrp);
  r->measResult.cellResults.resultsSSB_Cell->rsrq = lv(rsrq);
  r->measResult.cellResults.resultsSSB_Cell->sinr = lv(sinr);
  return r;
}

static byte_array_t encode_meas_report(void)
{
  NR_UL_DCCH_Message_t msg = {0};
  msg.message.present = NR_UL_DCCH_MessageType_PR_c1;
  msg.message.choice.c1 = calloc(1, sizeof(*msg.message.choice.c1));
  msg.message.choice.c1->present = NR_UL_DCCH_MessageType__c1_PR_measurementReport;
  NR_MeasurementReport_t* mr = calloc(1, sizeof(*mr));
  msg.message.choice.c1->choice.measurementReport = mr;
  mr->criticalExtensions.present = NR_MeasurementReport__criticalExtensions_PR_measurementReport;
  NR_MeasurementReport_IEs_t* ies = calloc(1, sizeof(*ies));
  mr->criticalExtensions.choice.measurementReport = ies;
  ies->measResults.measId = 1;
  NR_MeasResultServMO_t* serv = calloc(1, sizeof(*serv));
  serv->servCellId = 0;
  NR_MeasResultNR_t* s = cell(0, 60, 70, 40);                 // -97 dBm, -8.5 dB, -3.5 dB
  serv->measResultServingCell = *s;
  ASN_SEQUENCE_ADD(&ies->measResults.measResultServingMOList.list, serv);
  ies->measResults.measResultNeighCells = calloc(1, sizeof(*ies->measResults.measResultNeighCells));
  ies->measResults.measResultNeighCells->present = NR_MeasResults__measResultNeighCells_PR_measResultListNR;
  NR_MeasResultListNR_t* l = calloc(1, sizeof(*l));
  ies->measResults.measResultNeighCells->choice.measResultListNR = l;
  ASN_SEQUENCE_ADD(&l->list, cell(1, 66, 75, 50));             // -91 dBm
  ASN_SEQUENCE_ADD(&l->list, cell(2, 40, 60, 30));             // -117 dBm

  uint8_t buf[1024];
  asn_enc_rval_t er = uper_encode_to_buffer(&asn_DEF_NR_UL_DCCH_Message, NULL, &msg, buf, sizeof(buf));
  CHECK(er.encoded > 0);
  byte_array_t ba = {.len = (size_t)((er.encoded + 7) / 8)};
  ba.buf = malloc(ba.len);
  memcpy(ba.buf, buf, ba.len);
  return ba;
}

static char out[1 << 16];

static void drain(int fd)
{
  usleep(10000);
  ssize_t n = recv(fd, out, sizeof(out) - 1, MSG_DONTWAIT);
  out[n > 0 ? n : 0] = '\0';
  fputs(out, stdout);
}

int main(void)
{
  int sv[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  sock_fd = sv[0];
  n_nodes = 1;
  nodes[0].id.plmn.mcc = 1;
  nodes[0].id.plmn.mnc = 1;
  nodes[0].id.plmn.mnc_digit_len = 2;
  nodes[0].id.nb_id.nb_id = 3584;
  node_name(&nodes[0].id, nodes[0].name, sizeof(nodes[0].name));
  CHECK(strcmp(nodes[0].name, "001-01/3584") == 0);

  /* Style 5 UE context: one UE (RRC UE id 1) on NR cell 0x12345678. */
  e2sm_rc_ind_msg_frmt_4_t f4 = {0};
  f4.sz_seq_ue_info = 1;
  f4.seq_ue_info = calloc(1, sizeof(seq_ue_info_t));
  f4.seq_ue_info[0].ue_id.type = GNB_UE_ID_E2SM;
  f4.seq_ue_info[0].ue_id.gnb.amf_ue_ngap_id = 7;
  f4.seq_ue_info[0].ue_id.gnb.ran_ue_id = calloc(1, sizeof(uint64_t));
  *f4.seq_ue_info[0].ue_id.gnb.ran_ue_id = 1;
  f4.seq_ue_info[0].cell_global_id.type = NR_CGI_RAT_TYPE;
  f4.seq_ue_info[0].cell_global_id.nr_cgi.plmn_id = (e2sm_plmn_t){.mcc = 1, .mnc = 1, .mnc_digit_len = 2};
  f4.seq_ue_info[0].cell_global_id.nr_cgi.nr_cell_id = 0x12345678;
  on_ue_context(0, &f4);
  drain(sv[1]);
  CHECK(strstr(out, "\"type\":\"ue_context\"") && strstr(out, "\"ue_id\":\"rrc_ue_id=1@001-01/3584\""));
  CHECK(strstr(out, "\"nci\":305419896") && strstr(out, "\"plmn\":\"00101\"") && strstr(out, "\"amf_ue_ngap_id\":7"));

  /* Message copy: MeasurementReport -> meas_report with TS 38.133 conversions. */
  byte_array_t ba = encode_meas_report();
  on_meas_report_copy(0, &ba);
  drain(sv[1]);
  CHECK(strstr(out, "\"type\":\"meas_report\"") && strstr(out, "\"source\":\"e2-msg-copy\""));
  CHECK(strstr(out, "\"serving\":{\"pci\":0,\"nci\":305419896,\"rsrp_dbm\":-97,\"rsrq_db\":-8.5,\"sinr_db\":-3.5}"));
  CHECK(strstr(out, "{\"pci\":1,\"rsrp_dbm\":-91,\"rsrq_db\":-6.0,\"sinr_db\":1.5}"));
  CHECK(strstr(out, "{\"pci\":2,\"rsrp_dbm\":-117"));

  /* ho_command from ran-xapp (exact bytes of ai_ran_llm.ran.messages.encode) -> E2 target NR-CGI. */
  const char* cmd = "{\"type\":\"ho_command\",\"command_id\":\"9e64bd6a3d0f\",\"ue_id\":\"rrc_ue_id=1@001-01/3584\","
                    "\"ue_ids\":{\"ran_ue_id\":1,\"e2_node\":\"001-01/3584\"},\"source_cell\":{\"index\":0,\"pci\":0,"
                    "\"nci\":305419896,\"plmn\":\"00101\"},\"target_cell\":{\"index\":1,\"pci\":1,\"nci\":286331153,"
                    "\"plmn\":\"00101\",\"gnb\":\"oai-cu\"},\"confidence\":0.55,\"rationale\":\"x\",\"decided_by\":\"llm\","
                    "\"issued_at\":1.4,\"dry_run\":false}";
  char id[64], key[64], plmn_s[8];
  uint64_t nci = 0;
  const char* tgt = strstr(cmd, "\"target_cell\"");
  CHECK(json_str(cmd, "command_id", id, sizeof(id)) && strcmp(id, "9e64bd6a3d0f") == 0);
  CHECK(json_str(cmd, "ue_id", key, sizeof(key)) && find_ue_key(key) != NULL);
  CHECK(json_u64(tgt, "nci", &nci) && nci == 286331153);
  CHECK(json_str(tgt, "plmn", plmn_s, sizeof(plmn_s)) && strcmp(plmn_s, "00101") == 0);
  nr_cgi_t target = {.nr_cell_id = nci};
  CHECK(parse_plmn(plmn_s, &target.plmn_id) && target.plmn_id.mcc == 1 && target.plmn_id.mnc == 1);
  byte_array_t cgi = encode_nr_cgi(&target);
  /* 3GPP PLMN BCD 001/01 = 00 F1 10; NR cell identity 0x011111111 (36 bits) left-aligned in 5 octets. */
  const uint8_t want[8] = {0x00, 0xF1, 0x10, 0x01, 0x11, 0x11, 0x11, 0x10};
  CHECK(cgi.len == 8 && memcmp(cgi.buf, want, 8) == 0);
  /* Round trip through the decoder OAI's E2 agent uses (ran_func_rc.c nr_cgi_cell_id). */
  const uint8_t* b = cgi.buf;
  const uint64_t back = ((uint64_t)b[3] << 28) | ((uint64_t)b[4] << 20) | ((uint64_t)b[5] << 12) | ((uint64_t)b[6] << 4) | ((uint64_t)b[7] >> 4);
  CHECK(back == 286331153);
  rc_ctrl_req_data_t ctrl = gen_handover_ctrl(&find_ue_key(key)->ue_id, cgi);
  CHECK(ctrl.hdr.frmt_1.ric_style_type == 3 && ctrl.hdr.frmt_1.ctrl_act_id == HANDOVER_CONTROL_7_6_4_1);
  CHECK(ctrl.hdr.frmt_1.ue_id.type == GNB_UE_ID_E2SM && *ctrl.hdr.frmt_1.ue_id.gnb.ran_ue_id == 1);
  CHECK(ctrl.msg.frmt_1.ran_param[0].ran_param_id == TARGET_PRIMARY_CELL_ID_8_4_4_1);
  free_rc_ctrl_req_data(&ctrl);

  /* A second UE makes copied reports ambiguous (no UE ID in OAI's copy): dropped. */
  f4.seq_ue_info = realloc(f4.seq_ue_info, 2 * sizeof(seq_ue_info_t));
  f4.sz_seq_ue_info = 2;
  f4.seq_ue_info[1] = f4.seq_ue_info[0];
  f4.seq_ue_info[1].ue_id.gnb.ran_ue_id = calloc(1, sizeof(uint64_t));
  *f4.seq_ue_info[1].ue_id.gnb.ran_ue_id = 2;
  on_ue_context(0, &f4);
  drain(sv[1]);
  on_meas_report_copy(0, &ba);
  drain(sv[1]);
  CHECK(out[0] == '\0');

  /* UE 1 gone from the next poll -> ue_release. */
  f4.seq_ue_info[0] = f4.seq_ue_info[1];
  f4.sz_seq_ue_info = 1;
  on_ue_context(0, &f4);
  drain(sv[1]);
  CHECK(strstr(out, "{\"type\":\"ue_release\",\"ue_id\":\"rrc_ue_id=1@001-01/3584\"}"));

  fprintf(stderr, fails ? "llm_bridge_test: %d FAILED\n" : "llm_bridge_test: all checks passed\n", fails);
  return fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
