"""Command line: python -m ai_ran_llm <command> ..."""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(prog="ai_ran_llm", description="HandoverLLM for AI-RAN mobility management")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="simulate drives and build a tokenised corpus")
    _city_args(g)
    g.add_argument("--episodes", type=int, default=60)
    g.add_argument("--ues", type=int, default=32)
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/handover_corpus.npz")
    g.add_argument("--from-drives", nargs="+", metavar="NPZ",
                   help="build the corpus from these drive files (real traces or export-raw output) "
                        "instead of simulating; see README 'Using real network data'")
    g.add_argument("--serving", choices=["logged", "replay"], default="logged",
                   help="with --from-drives: use the logged serving cells, or replay simulated policies")

    t = sub.add_parser("train", help="train HandoverGPT")
    t.add_argument("--data", default="data/handover_corpus.npz")
    t.add_argument("--out", default="checkpoints/handover_llm.pt")
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--embd", type=int, default=128)

    e = sub.add_parser("evaluate", help="closed-loop benchmark vs A3")
    _city_args(e)
    e.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    e.add_argument("--episodes", type=int, default=5)
    e.add_argument("--ues", type=int, default=64)
    e.add_argument("--seed", type=int, default=10_000)
    e.add_argument("--ho-threshold", type=float, default=0.35,
                   help="hand over when P(handover to best neighbour) >= this")
    e.add_argument("--json", help="also write results to this file")

    i = sub.add_parser("infer", help="decide for one JSON measurement report")
    i.add_argument("report", help="path to a JSON report, or '-' for the built-in example")
    i.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    i.add_argument("--ho-threshold", type=float, default=0.35)

    s = sub.add_parser("serve", help="HTTP endpoint for a near-RT RIC xApp")
    s.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--ho-threshold", type=float, default=0.35,
                   help="hand over when P(handover to best neighbour) >= this")
    s.add_argument("--min-confidence", type=float, default=0.3,
                   help="below this, the A3 fallback decides instead of the model")

    r = sub.add_parser("export-raw", help="export the raw drives and readable measurement reports")
    _city_args(r)
    r.add_argument("--episodes", type=int, default=1, help="first N drives of the corpus")
    r.add_argument("--ues", type=int, default=32)
    r.add_argument("--steps", type=int, default=600)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="data/raw")

    rx = sub.add_parser("ran-xapp", help="run the handover xApp against a real RAN (OCUDU/srsRAN, OAI, ...)")
    rx.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    rx.add_argument("--cells", required=True, help="cell map JSON (see integrations/), or 'sim' for fake-gnb")
    rx.add_argument("--auto-add-cells", action="store_true", help="index unknown PCIs on the fly (lab only)")
    rx.add_argument("--bridge", default="127.0.0.1:7000", help="ran-bridge listen HOST:PORT ('off' to disable)")
    rx.add_argument("--rrc-log", help="tail this gNB/CU-CP log for RRC MeasurementReports (JSON or XER)")
    rx.add_argument("--rrc-log-from-start", action="store_true", help="also process what is already in the log")
    rx.add_argument("--id-regex", action="append", default=[], metavar="KEY=REGEX",
                    help="extract a UE id from the log header line, e.g. amf_ue_ngap_id='amf_ue_id=(\\d+)'")
    rx.add_argument("--pci-regex", default=r"\bpci[= ]?(\d+)", help="serving PCI in the log header line")
    rx.add_argument("--serv-cell-pci", action="append", default=[], metavar="SERVCELLID=PCI",
                    help="PCI of a servCellId when reports omit it")
    rx.add_argument("--rrc-header-filter", help="only use blocks whose header line matches this regex")
    rx.add_argument("--learn-id", action="append", default=[], metavar="REGEX",
                    help="learn UE ids from any log line: named groups, one named 'ue', e.g. "
                         "'ue=(?P<ue>\\d+).*?amf_ue_id=(?P<amf_ue_ngap_id>\\d+)'")
    rx.add_argument("--actuator", default="log", choices=["log", "bridge", "oai-telnet", "console", "command"])
    rx.add_argument("--oai-telnet", default="127.0.0.1:9090", help="OAI telnet server HOST:PORT")
    rx.add_argument("--oai-mode", default="auto", choices=["auto", "f1", "n2"])
    rx.add_argument("--console-fifo", help="FIFO feeding the srsRAN/OCUDU gnb console")
    rx.add_argument("--console-template", default="ho {serving_pci} {rnti_hex} {target_pci}")
    rx.add_argument("--command", help="command template per handover (no shell), e.g. a FlexRIC xApp")
    rx.add_argument("--live", action="store_true", help="actually send handovers (default: shadow mode)")
    rx.add_argument("--ho-threshold", type=float, default=0.35)
    rx.add_argument("--min-confidence", type=float, default=0.3)
    rx.add_argument("--confirm", type=int, default=1, help="consecutive identical recommendations required")
    rx.add_argument("--hold-off-s", type=float, default=0.0, help="minimum time between handovers of a UE")
    rx.add_argument("--a3-override-db", type=float, default=6.0,
                    help="hand over despite the model when a neighbour leads by this many dB in the last 3 samples "
                         "(safety net for out-of-distribution input; negative value disables)")
    rx.add_argument("--max-commands-per-s", type=float, default=50.0)
    rx.add_argument("--no-neighbour-check", action="store_true")
    rx.add_argument("--explain", action="store_true", help="generate model rationales for every report (slower)")
    rx.add_argument("--audit", default="ran_audit.jsonl", help="JSONL decision log ('' to disable)")
    rx.add_argument("--duration", type=float, help="stop after this many seconds")
    rx.add_argument("-v", "--verbose", action="store_true")

    fg = sub.add_parser("fake-gnb", help="simulated gNB that talks to a running ran-xapp over the bridge")
    _city_args(fg)
    fg.add_argument("--xapp", default="127.0.0.1:7000", help="ran-xapp bridge HOST:PORT")
    fg.add_argument("--ues", type=int, default=16)
    fg.add_argument("--steps", type=int, default=600)
    fg.add_argument("--seed", type=int, default=10_000)
    fg.add_argument("--realistic", action="store_true",
                    help="200 ms reports, 8 neighbours, RRC-quantised values (instead of every cell every 100 ms)")

    rp = sub.add_parser("ran-parse", help="parse a gNB log offline and print the measurement reports found")
    rp.add_argument("log")
    rp.add_argument("--id-regex", action="append", default=[], metavar="KEY=REGEX")
    rp.add_argument("--pci-regex", default=r"\bpci[= ]?(\d+)")
    rp.add_argument("--serv-cell-pci", action="append", default=[], metavar="SERVCELLID=PCI")
    rp.add_argument("--rrc-header-filter")
    rp.add_argument("--learn-id", action="append", default=[], metavar="REGEX")
    rp.add_argument("--cells", help="also resolve cells against this cell map")

    x = sub.add_parser("export-jsonl", help="export chat-format data to fine-tune a general LLM")
    x.add_argument("--data", default="data/handover_corpus.npz")
    x.add_argument("--out", default="data/handover_sft.jsonl")
    x.add_argument("--limit", type=int, default=None)

    a = p.parse_args(argv)
    import os

    if a.cmd == "gen-data":
        from .dataset import generate_dataset
        d = generate_dataset(a.episodes, a.ues, a.steps, a.seed, sim=_sim(a),
                             drives=sorted(a.from_drives) if a.from_drives else None, serving=a.serving)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, **d)
        print(f"wrote {len(d['tokens'])} samples to {a.out}")

    elif a.cmd == "train":
        from .config import ModelConfig
        from .train import train
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        train(a.data, a.out, a.epochs, a.batch_size, a.lr,
              model_cfg=ModelConfig(n_layer=a.layers, n_head=a.heads, n_embd=a.embd))

    elif a.cmd == "evaluate":
        from .evaluate import benchmark, default_policies, format_table
        from .inference import HandoverLLM
        llm = HandoverLLM.load(a.ckpt)
        res = benchmark(default_policies(llm, llm.tok.obs_cfg, a.ho_threshold), a.episodes, a.ues, seed=a.seed,
                        sim=_sim(a))
        print(format_table(res))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=2)

    elif a.cmd == "infer":
        from .inference import HandoverLLM
        report = EXAMPLE_REPORT if a.report == "-" else json.load(open(a.report))
        print(json.dumps(HandoverLLM.load(a.ckpt).handle_report(report, a.ho_threshold), indent=2))

    elif a.cmd == "serve":
        from .serve import serve
        serve(a.ckpt, a.host, a.port, a.ho_threshold, a.min_confidence)

    elif a.cmd == "export-raw":
        from .dataset import export_raw
        n = export_raw(a.out, a.episodes, a.ues, a.steps, a.seed, sim=_sim(a))
        print(f"wrote {n['reports']} reports ({n['in_corpus']} in the corpus) and {a.episodes} drive(s) to {a.out}/")

    elif a.cmd == "ran-xapp":
        _ran_xapp(a)

    elif a.cmd == "fake-gnb":
        from .ran.fake_gnb import FakeGnb
        from .simulator import generate_episode
        from .evaluate import COLUMNS
        host, _, port = a.xapp.rpartition(":")
        ep = generate_episode(a.ues, a.steps, np.random.default_rng(a.seed), _sim(a))
        kw = dict(report_every=2, report_all_cells=False, max_neighbours=8, quantize_rrc=True) if a.realistic else {}
        gnb = FakeGnb(ep, host or "127.0.0.1", int(port), **kw)
        m, _ = gnb.run()
        print(f"fake gNB: {a.ues} UEs x {a.steps * 0.1:.0f} s, {gnb.decisions} decisions, {gnb.commands} commands")
        r = m.summary()
        print("  ".join(f"{h} {r[k]:.3f}" for k, h in COLUMNS))

    elif a.cmd == "ran-parse":
        from .ran.sources import DEFAULT_ID_PATTERNS, RrcLogSource
        from .ran.messages import encode
        src = RrcLogSource(a.log, follow=False, id_patterns=_kv(a.id_regex, dict(DEFAULT_ID_PATTERNS)),
                           pci_pattern=a.pci_regex, serv_cell_pci={int(k): int(v) for k, v in _kv(a.serv_cell_pci).items()},
                           header_filter=a.rrc_header_filter, clock=lambda: 0.0, learn_patterns=a.learn_id)
        cells = None
        if a.cells:
            from .ran.app import load_cells
            cells = load_cells(a.cells)
        n = 0
        for rep in src.reports():
            n += 1
            line = encode(rep).decode().rstrip()
            if cells is not None:
                res = [cells.resolve(c) for c in [rep.serving, *rep.neighbours]]
                line += "  # model cells: " + ",".join("?" if r is None else str(r.index) for r in res)
            print(line)
        print(f"# {n} measurement reports, {src.skipped} blocks skipped (no UE id or serving PCI)")

    elif a.cmd == "export-jsonl":
        from .dataset import export_jsonl
        d = np.load(a.data)
        n = export_jsonl(d["tokens"], int(d["prompt_len"]), a.out, a.limit)
        print(f"wrote {n} examples to {a.out}")


def _city_args(p):
    c = p.add_argument_group("city model (docs/LOCATION_AWARE_HANDOVER.md; defaults = original simulator)")
    c.add_argument("--mobility", default="random", choices=["random", "roads"],
                   help="random: Gauss-Markov heading; roads: popular routes on a road network")
    c.add_argument("--shadowing", default="per_ue", choices=["per_ue", "spatial"],
                   help="per_ue: random along each path; spatial: fixed field per cell, tied to places")
    c.add_argument("--map-seed", type=int, default=1, help="which city (roads, routes, shadowing field)")


def _sim(a):
    from .config import SimConfig
    return SimConfig(mobility=a.mobility, shadowing=a.shadowing, map_seed=a.map_seed)


def _kv(items, base=None):
    out = dict(base or {})
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def _ran_xapp(a):
    import logging
    from .ran.app import XAppOptions, build_xapp
    from .ran.controller import ControllerConfig
    from .ran.sources import DEFAULT_ID_PATTERNS
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    opt = XAppOptions(
        ckpt=a.ckpt, cells=a.cells, auto_add_cells=a.auto_add_cells,
        bridge=None if a.bridge == "off" else a.bridge, rrc_log=a.rrc_log, rrc_log_from_start=a.rrc_log_from_start,
        id_patterns=_kv(a.id_regex, dict(DEFAULT_ID_PATTERNS)), pci_pattern=a.pci_regex,
        serv_cell_pci={int(k): int(v) for k, v in _kv(a.serv_cell_pci).items()},
        rrc_header_filter=a.rrc_header_filter, learn_patterns=a.learn_id, actuator=a.actuator, oai_telnet=a.oai_telnet, oai_mode=a.oai_mode,
        console_fifo=a.console_fifo, console_template=a.console_template, command=a.command, audit=a.audit or None,
        controller=ControllerConfig(ho_threshold=a.ho_threshold, min_confidence=a.min_confidence,
                                    confirm_count=a.confirm, hold_off_s=a.hold_off_s,
                                    a3_override_db=a.a3_override_db if a.a3_override_db >= 0 else None,
                                    max_commands_per_s=a.max_commands_per_s,
                                    require_neighbour=not a.no_neighbour_check, dry_run=not a.live,
                                    explain=a.explain))
    runtime, sources = build_xapp(opt)
    try:
        runtime.run(a.duration)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
        for s in sources:
            s.close()
        st = runtime.controller.stats
        print(f"reports {st.reports}, evaluated {st.evaluated}, recommended {st.recommended}, "
              f"commands {st.commands}, blocked {st.blocked}, outcomes {st.outcomes}")


EXAMPLE_REPORT = {
    "ue_id": "ue-42",
    "serving_cell": 9,
    "speed_kmh": 90,
    "sinr_db": -1.0,
    "serving_rsrp": [-88, -90, -92, -95, -97],
    "neighbors": [
        {"cell_id": 4, "rsrp": [-99, -96, -94, -92, -89]},
        {"cell_id": 10, "rsrp": [-97, -97, -98, -98, -99]},
        {"cell_id": 3, "rsrp": [-104, -103, -104, -105, -104]},
        {"cell_id": 8, "rsrp": [-108, -109, -107, -108, -110]},
    ],
}
