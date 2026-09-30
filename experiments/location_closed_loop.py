"""Closed-loop test of location context in the simulated city (docs/LOCATION_AWARE_HANDOVER.md §12).

    PYTHONPATH=. python experiments/location_closed_loop.py [--episodes 40] [--epochs 1]
    # longer training, one process per variant, then the benchmark:
    PYTHONPATH=. python experiments/location_closed_loop.py --epochs 3 --tag _e3 --only all --threads 1
    PYTHONPATH=. python experiments/location_closed_loop.py --epochs 3 --tag _e3 --bench-only

Steps (each skipped if its output exists, so the script can be resumed):

1. Location service from 20 *history* drives (seed 777) under classical A3: radio map from
   the reports at estimated positions (10 m error) and handover-sequence statistics.
2. Five city corpora with identical drives and labels (seed 0) that differ only in context:
   base, +position, +radio map, +trajectory prior, all.
3. One model per corpus, identical architecture and training budget.
4. Closed-loop benchmark on unseen city drives (seed 10000, 5 x 64 UEs x 60 s) against A3,
   the shipped model (trained on the original simulator) and the non-causal teacher.

Outputs go to data/city/ and checkpoints/city_*.pt (git-ignored); results to
data/city/closed_loop.json. ``--tag`` suffixes checkpoints and results (e.g. ``_e3``) so runs with
different training budgets sit side by side.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ai_ran_llm.config import ObsConfig, SimConfig
from ai_ran_llm.dataset import generate_dataset
from ai_ran_llm.evaluate import benchmark, default_policies, format_table
from ai_ran_llm.inference import HandoverLLM, LLMPolicy
from ai_ran_llm.location import LocationAwarePolicy, LocationService
from ai_ran_llm.train import train

CITY = SimConfig(mobility="roads", shadowing="spatial")
VARIANTS = {
    "base": ObsConfig(),
    "position": ObsConfig(use_position=True),
    "radio_map": ObsConfig(use_radio_map=True),
    "trajectory": ObsConfig(use_trajectory=True),
    "all": ObsConfig(use_position=True, use_radio_map=True, use_trajectory=True),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--out", default="data/city")
    ap.add_argument("--tag", default="", help="suffix for checkpoints and results, e.g. _e3")
    ap.add_argument("--only", choices=list(VARIANTS), help="build/train this variant only, no benchmark")
    ap.add_argument("--bench-only", action="store_true", help="skip corpus/training, benchmark only")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    svc_dir = os.path.join(a.out, "location_service")
    if not os.path.exists(os.path.join(svc_dir, "service.json")):
        log("building location service from 20 history drives")
        LocationService.build(CITY, n_drives=20, n_ue=32, n_steps=600, seed=777).save(svc_dir)
    svc = LocationService.load(svc_dir)
    log(f"location service: radio-map coverage {svc.radio_map.coverage():.0%}")

    for name, obs_cfg in VARIANTS.items():
        if a.bench_only or (a.only and name != a.only):
            continue
        corpus = os.path.join(a.out, f"corpus_{name}.npz")
        if not os.path.exists(corpus):
            log(f"corpus {name}")
            d = generate_dataset(a.episodes, 32, 600, seed=0, sim=CITY, obs_cfg=obs_cfg,
                                 location=svc if obs_cfg.uses_context else None, verbose=False)
            np.savez_compressed(corpus, **d)
            log(f"  {len(d['tokens'])} samples, prompt {int(d['prompt_len'])} tokens")
        ckpt = f"checkpoints/city_{name}{a.tag}.pt"
        if not os.path.exists(ckpt):
            log(f"training {name}")
            train(corpus, ckpt, epochs=a.epochs, log_every=400)

    if a.only:
        return
    log("closed-loop benchmark (city)")
    shipped = HandoverLLM.load("checkpoints/handover_llm.pt")
    pols = default_policies(None, ObsConfig())
    oracle = pols.pop("Oracle (non-causal)")
    pols["Shipped model (original sim)"] = lambda: LLMPolicy(shipped, 0.35)
    for name in VARIANTS:
        llm = HandoverLLM.load(f"checkpoints/city_{name}{a.tag}.pt")
        if llm.tok.obs_cfg.uses_context:
            pols[f"City model: {name}"] = (lambda llm=llm: LocationAwarePolicy(llm, svc, a.threshold))
        else:
            pols[f"City model: {name}"] = (lambda llm=llm: LLMPolicy(llm, a.threshold))
    pols["Oracle (non-causal)"] = oracle
    res = benchmark(pols, a.eval_episodes, 64, seed=10_000, sim=CITY)
    print(format_table(res))
    thr = "" if a.threshold == 0.35 else f"_t{a.threshold:g}"
    with open(os.path.join(a.out, f"closed_loop{a.tag}{thr}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
