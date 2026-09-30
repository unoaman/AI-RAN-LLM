# llm_bridge: HandoverLLM <-> FlexRIC bridge xApp (docs/RAN_INTEGRATION.md §9).
# Copy this directory to <flexric>/examples/xApp/c/llm_bridge and add
#   add_subdirectory(llm_bridge)
# to <flexric>/examples/xApp/c/CMakeLists.txt (testbed/Dockerfile.llm_bridge does both).
# Needs the NR RRC ASN.1 library built by examples/xApp/c/monitor (target asn1_nr_rrc).

add_executable(llm_bridge
  llm_bridge.c
  ../../../../src/util/alg_ds/alg/defer.c
  )

target_link_libraries(llm_bridge
                      PUBLIC
                      asn1_nr_rrc
                      e42_xapp
                      -pthread
                      -lsctp
                      -ldl
                      )

# Offline tests (no RIC needed): ninja llm_bridge_test && ./llm_bridge_test
add_executable(llm_bridge_test
  test_llm_bridge.c
  ../../../../src/util/alg_ds/alg/defer.c
  )
target_link_libraries(llm_bridge_test PUBLIC asn1_nr_rrc e42_xapp -pthread -lsctp -ldl)
