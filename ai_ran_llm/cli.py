"""Command line: python -m ai_ran_llm <command> ..."""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(prog="ai_ran_llm", description="HandoverLLM for AI-RAN mobility management")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="simulate drives and build a tokenised corpus")
    g.add_argument("--episodes", type=int, default=60)
    g.add_argument("--ues", type=int, default=32)
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/handover_corpus.npz")

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
    e.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    e.add_argument("--episodes", type=int, default=5)
    e.add_argument("--ues", type=int, default=64)
    e.add_argument("--seed", type=int, default=10_000)
    e.add_argument("--min-confidence", type=float, default=0.5)
    e.add_argument("--json", help="also write results to this file")

    i = sub.add_parser("infer", help="decide for one JSON measurement report")
    i.add_argument("report", help="path to a JSON report, or '-' for the built-in example")
    i.add_argument("--ckpt", default="checkpoints/handover_llm.pt")

    s = sub.add_parser("serve", help="HTTP endpoint for a near-RT RIC xApp")
    s.add_argument("--ckpt", default="checkpoints/handover_llm.pt")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--min-confidence", type=float, default=0.5)

    x = sub.add_parser("export-jsonl", help="export chat-format data to fine-tune a general LLM")
    x.add_argument("--data", default="data/handover_corpus.npz")
    x.add_argument("--out", default="data/handover_sft.jsonl")
    x.add_argument("--limit", type=int, default=None)

    a = p.parse_args(argv)
    import os

    if a.cmd == "gen-data":
        from .dataset import generate_dataset
        d = generate_dataset(a.episodes, a.ues, a.steps, a.seed)
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
        res = benchmark(default_policies(llm, llm.tok.obs_cfg, a.min_confidence), a.episodes, a.ues, seed=a.seed)
        print(format_table(res))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=2)

    elif a.cmd == "infer":
        from .inference import HandoverLLM
        report = EXAMPLE_REPORT if a.report == "-" else json.load(open(a.report))
        print(json.dumps(HandoverLLM.load(a.ckpt).handle_report(report), indent=2))

    elif a.cmd == "serve":
        from .serve import serve
        serve(a.ckpt, a.host, a.port, a.min_confidence)

    elif a.cmd == "export-jsonl":
        from .dataset import export_jsonl
        d = np.load(a.data)
        n = export_jsonl(d["tokens"], int(d["prompt_len"]), a.out, a.limit)
        print(f"wrote {n} examples to {a.out}")


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
