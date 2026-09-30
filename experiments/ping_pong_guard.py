"""Model-side hysteresis against ping-pong: ReturnGuard sweep (docs/DESIGN.md §21.17).

    PYTHONPATH=. python experiments/ping_pong_guard.py [--part city|original|both] [--threads 4]

No retraining: the guard sits in the decision layer (ai_ran_llm.inference.ReturnGuard).
* city: the 3-epoch city models (checkpoints/city_*_e3.pt, threshold 0.5) and the shipped model
  (threshold 0.35) on the same unseen city drives as docs/LOCATION_AWARE_HANDOVER.md §13.1.
* original: the shipped model on the original simulator (the README benchmark drives).
Results: experiments/results/ping_pong_guard_<part>.json; "return5s %" guards against a policy
that only postpones returns past the 1 s ping-pong window.
"""

import argparse
import json
import os
import time

import torch

from ai_ran_llm.config import ObsConfig, SimConfig
from ai_ran_llm.evaluate import benchmark, format_table
from ai_ran_llm.inference import HandoverLLM, LLMPolicy, ReturnGuard
from ai_ran_llm.location import LocationAwarePolicy, LocationService
from ai_ran_llm.policies import A3Policy

GUARDS = {
    "": None,
    " +guard(2s,0.8,3dB)": dict(window_s=2.0, threshold=0.8, margin_db=3.0),
    " +guard(2s,0.9,5dB)": dict(window_s=2.0, threshold=0.9, margin_db=5.0),
    " +guard(5s,0.8,3dB)": dict(window_s=5.0, threshold=0.8, margin_db=3.0),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def guarded(make_policy):
    """name suffix -> factory, one per guard setting."""
    return {suffix: (lambda g=g: make_policy(ReturnGuard(**g) if g else None)) for suffix, g in GUARDS.items()}


def run(part, pols, sim):
    log(f"benchmark: {part} ({len(pols)} policies)")
    res = benchmark(pols, 5, 64, seed=10_000, sim=sim)
    print(format_table(res), flush=True)
    os.makedirs("experiments/results", exist_ok=True)
    with open(f"experiments/results/ping_pong_guard_{part}.json", "w") as f:
        json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["city", "original", "both"], default="both")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    shipped = HandoverLLM.load("checkpoints/handover_llm.pt")
    a3 = {"A3 (1dB, 200ms)": lambda: A3Policy(1.0, 2), "A3 (2dB, 300ms)": lambda: A3Policy(2.0, 3)}

    if a.part in ("original", "both"):
        pols = dict(a3)
        for suffix, make in guarded(lambda g: LLMPolicy(shipped, 0.35, g)).items():
            pols["Shipped @0.35" + suffix] = make
        run("original", pols, SimConfig())

    if a.part in ("city", "both"):
        svc = LocationService.load("data/city/location_service")
        pols = dict(a3)
        for suffix, make in guarded(lambda g: LLMPolicy(shipped, 0.35, g)).items():
            pols["Shipped @0.35" + suffix] = make
        for name in ("base", "radio_map", "all"):
            llm = HandoverLLM.load(f"checkpoints/city_{name}_e3.pt")
            if llm.tok.obs_cfg.uses_context:
                factory = lambda g, llm=llm: LocationAwarePolicy(llm, svc, 0.5, guard=g)
            else:
                factory = lambda g, llm=llm: LLMPolicy(llm, 0.5, g)
            for suffix, make in guarded(factory).items():
                pols[f"City {name} @0.5" + suffix] = make
        run("city", pols, SimConfig(mobility="roads", shadowing="spatial"))
    log("done")


if __name__ == "__main__":
    main()
