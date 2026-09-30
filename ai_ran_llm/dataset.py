"""Build supervised handover corpora from simulated drives."""

import gzip
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .config import ObsConfig, SimConfig
from .policies import A3Policy, OraclePolicy, label_decision
from .simulator import (Episode, Observation, build_observation, generate_episode, load_episode,
                        run_policy, save_episode)
from .tokenizer import HandoverTokenizer


def _behaviour_policy(rng: np.random.Generator, obs_cfg: ObsConfig):
    """Mix of policies driving the rollout, so the corpus covers good *and*
    off-optimal serving-cell states (late, early and missed handovers)."""
    r = rng.uniform()
    if r < 0.35:
        return OraclePolicy(obs_cfg)
    if r < 0.6:
        return OraclePolicy(obs_cfg, exec_prob=rng.uniform(0.1, 0.5), seed=int(rng.integers(1 << 31)))
    return A3Policy(hyst_db=float(rng.uniform(0.0, 5.0)), ttt_steps=int(rng.integers(1, 8)))


def describe_policy(policy) -> str:
    if policy is None:
        return "logged serving cells"
    if isinstance(policy, A3Policy):
        return f"A3(hyst={policy.hyst_db:.2f} dB, ttt={policy.ttt_steps} steps)"
    if policy.exec_prob < 1.0:
        return f"delayed teacher (executes each handover with p={policy.exec_prob:.2f} per step)"
    return "teacher"


@dataclass
class LabelledStep:
    """All UEs of one drive at one step: the report, the teacher label and whether
    each UE's sample was kept in the corpus."""

    t: int
    obs: Observation
    target: np.ndarray     # (U,) teacher target cell, -1 = stay
    gain: np.ndarray       # (U,) teacher's predicted RSRP gain of the best neighbour (dB)
    keep: np.ndarray       # (U,) bool, sample included in the corpus


def _label_steps(ep: Episode, seen, rng: np.random.Generator, obs_cfg: ObsConfig,
                 easy_keep_prob: float) -> list[LabelledStep]:
    lookahead = max(obs_cfg.oracle_horizon + obs_cfg.label_window, obs_cfg.label_confirm_horizon)
    steps = []
    for t, obs in seen:
        if t < obs_cfg.hist_stride * (obs_cfg.hist_len - 1) or t + lookahead >= ep.n_steps:
            continue
        target, gain = label_decision(ep, t, obs, obs_cfg)
        margin = obs.nbr_hist[:, 0, -1] - obs.serving_hist[:, -1]
        keep = (target >= 0) | (margin > -6.0) | (rng.uniform(size=len(target)) < easy_keep_prob)
        steps.append(LabelledStep(t, obs, target, gain, keep))
    return steps


def _logged_reports(ep: Episode, obs_cfg: ObsConfig):
    """Reports along the serving cells recorded in a trace (no policy in the loop)."""
    seen = []
    for t in range(ep.n_steps):
        obs = build_observation(ep, t, ep.logged_serving[:, t], obs_cfg)
        if ep.logged_sinr_db is not None:
            obs.sinr_db = ep.logged_sinr_db[:, t].copy()
        seen.append((t, obs))
    return seen


def iter_labelled_drives(n_episodes: int, n_ue: int, n_steps: int, rng: np.random.Generator,
                         sim: SimConfig, obs_cfg: ObsConfig, easy_keep_prob: float = 0.1,
                         drives: list[str] | None = None, serving: str = "logged"):
    """Label every report of a set of drives. Yields (episode index, Episode,
    behaviour policy or None, serving-cell trajectory (U, T), [LabelledStep, ...]).

    Without `drives`, it simulates `n_episodes` new drives. Both `generate_dataset`
    and `export_raw` consume it with the same random stream, so a raw export with
    the same seed contains exactly the drives behind the corpus.

    With `drives` (paths to .npz files, see `simulator.load_episode`), it labels
    those instead, e.g. real traces:

    * ``serving="logged"`` (default): use the serving cells recorded in each file
      (``serving`` array), i.e. the states the real network visited.
    * ``serving="replay"``: replay the simulated behaviour-policy mix on the file's
      RSRP, as for simulated drives.

    Samples where every neighbour is >6 dB below the serving cell are trivially
    STAY; only `easy_keep_prob` of them are kept.
    """
    if serving not in ("logged", "replay"):
        raise ValueError("serving must be 'logged' or 'replay'")
    sources = drives if drives is not None else [None] * n_episodes
    for e, path in enumerate(sources):
        if path is None:
            ep = generate_episode(n_ue, n_steps, rng, sim)
        else:
            ep = load_episode(path, sim)
        if path is not None and serving == "logged":
            if ep.logged_serving is None:
                raise ValueError(f"{path} has no 'serving' array; use serving='replay'")
            policy, seen = None, _logged_reports(ep, obs_cfg)
            traj = ep.logged_serving
        else:
            policy = _behaviour_policy(rng, obs_cfg)
            _, traj, seen = run_policy(ep, policy, obs_cfg, record=True)
        yield e, ep, policy, traj, _label_steps(ep, seen, rng, obs_cfg, easy_keep_prob)


def generate_dataset(n_episodes: int, n_ue: int = 32, n_steps: int = 600, seed: int = 0,
                     sim: SimConfig | None = None, obs_cfg: ObsConfig | None = None,
                     block_size: int = 64, easy_keep_prob: float = 0.1,
                     verbose: bool = True, drives: list[str] | None = None, serving: str = "logged",
                     location=None) -> dict:
    """Returns {"tokens": (N, block_size) int64 padded with <pad>, "prompt_len": int}
    (plus "obs_cfg" as JSON when location context is enabled).

    Labels come from the look-ahead teacher (`label_decision`); see
    `iter_labelled_drives` for how reports are produced (simulated, or from the
    `drives` files) and subsampled.

    With ``obs_cfg.uses_context``, every report also gets location context
    (``ai_ran_llm.location.compute_context``) from `location`, a LocationService
    built from separate history drives; position-only context works without it.
    A share of samples (``LocationConfig.context_dropout``) has its context hidden
    so the model also learns to decide without it. The context uses its own
    random generators, so the drives and labels are the same as without context.
    """
    sim = sim or SimConfig()
    obs_cfg = obs_cfg or ObsConfig()
    tok = HandoverTokenizer(obs_cfg)
    block_size = max(block_size, tok.prompt_len + 16)       # room for the longest answer
    rng = np.random.default_rng(seed)
    rows, n_ho = [], 0
    if obs_cfg.uses_context:
        from .location import LocationConfig, compute_context, drop_context, position_track, previous_cells
        loc_cfg = location.cfg if location is not None else LocationConfig()
        if (obs_cfg.use_radio_map or obs_cfg.use_trajectory) and location is None:
            raise ValueError("radio-map / trajectory context needs a LocationService (build-location-service)")

    n_total = len(drives) if drives is not None else n_episodes
    for e, ep, _, traj, steps in iter_labelled_drives(n_episodes, n_ue, n_steps, rng, sim, obs_cfg,
                                                      easy_keep_prob, drives, serving):
        if obs_cfg.uses_context:
            est = position_track(ep, np.random.default_rng([seed, e, 7]), loc_cfg)
            prev = previous_cells(traj)
            drop_rng = np.random.default_rng([seed, e, 8])
        for st in steps:
            obs = st.obs
            if obs_cfg.uses_context:
                ctx = compute_context(ep, st.t, obs, est, prev[:, st.t], location, loc_cfg)
                obs.context = drop_context(ctx, drop_rng.uniform(size=ep.n_ue) < loc_cfg.context_dropout)
            prompts = tok.encode_prompts(obs)
            for u in np.nonzero(st.keep)[0]:
                ans = tok.encode_answer(int(st.target[u]), float(st.gain[u]), obs.serving_hist[u],
                                        obs.nbr_ids[u], obs.nbr_hist[u], float(obs.sinr_db[u]))
                row = np.full(block_size, tok.PAD, dtype=np.int64)
                seq = np.concatenate([prompts[u], ans])
                row[: len(seq)] = seq
                rows.append(row)
                n_ho += int(st.target[u] >= 0)
        if verbose:
            print(f"episode {e + 1}/{n_total}: {len(rows)} samples ({n_ho} handovers)", flush=True)

    if not rows:
        raise ValueError("no samples: drives are shorter than the report history + teacher look-ahead")
    tokens = np.stack(rows)
    rng.shuffle(tokens)
    out = {"tokens": tokens, "prompt_len": np.int64(tok.prompt_len)}
    if obs_cfg.uses_context:                 # the default corpus format stays unchanged
        out["obs_cfg"] = np.array(json.dumps(asdict(obs_cfg)))
    return out


def export_raw(out_dir: str, n_episodes: int = 1, n_ue: int = 32, n_steps: int = 600, seed: int = 0,
               sim: SimConfig | None = None, obs_cfg: ObsConfig | None = None,
               easy_keep_prob: float = 0.1, verbose: bool = True) -> dict:
    """Write the raw simulation behind the corpus in readable form.

    With the same seed / UE count / steps as `gen-data`, episode k here is
    exactly episode k of the corpus. Writes:

    * ``cells.csv``: cell id and site coordinates (m).
    * ``drives.json``: seed, configs and the behaviour policy of each drive.
    * ``drive_XXX.npz``: per-step channel of the drive (float32). ``pos`` (U,T,2) m,
      ``speed_kmh`` (U,), ``rsrp_true`` / ``rsrp_inst`` / ``rsrp_meas`` (U,T,C) dBm
      (large-scale, with fast fading, L3-filtered measurement), ``serving`` (U,T)
      and ``sinr_db`` (U,T): serving cell and SINR at each report under the
      behaviour policy. Load it with :func:`ai_ran_llm.simulator.load_episode`;
      it is also the template for real traces (``gen-data --from-drives``).
    * ``reports.jsonl.gz``: one measurement report per UE per labelled step, in
      the xApp request format (``serving_rsrp``, ``neighbors[].rsrp``, ...) with
      unrounded values, plus the teacher ``label`` and ``in_corpus``.
    """
    sim = sim or SimConfig()
    obs_cfg = obs_cfg or ObsConfig()
    tok = HandoverTokenizer(obs_cfg)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"seed": seed, "n_ue": n_ue, "n_steps": n_steps, "dt_s": sim.dt_s,
                "sim_config": asdict(sim), "obs_config": asdict(obs_cfg), "drives": []}
    n_reports = n_kept = 0
    r1 = lambda x: round(float(x), 1)

    with gzip.open(os.path.join(out_dir, "reports.jsonl.gz"), "wt") as f:
        for e, ep, policy, traj, steps in iter_labelled_drives(n_episodes, n_ue, n_steps, rng, sim,
                                                               obs_cfg, easy_keep_prob):
            if e == 0:
                with open(os.path.join(out_dir, "cells.csv"), "w") as c:
                    c.write("cell_id,x_m,y_m\n")
                    for i, (x, y) in enumerate(ep.sites):
                        c.write(f"{i},{x:.1f},{y:.1f}\n")
            sinr = np.stack([ep.sinr_db(t, traj[:, t]) for t in range(ep.n_steps)], axis=1)   # as in the reports
            save_episode(os.path.join(out_dir, f"drive_{e:03d}.npz"), ep, serving=traj, sinr_db=sinr)
            manifest["drives"].append({"episode": e, "file": f"drive_{e:03d}.npz",
                                       "behaviour_policy": describe_policy(policy)})
            for st in steps:
                obs = st.obs
                for u in range(ep.n_ue):
                    tgt = int(st.target[u])
                    ans = tok.encode_answer(tgt, float(st.gain[u]), obs.serving_hist[u], obs.nbr_ids[u],
                                            obs.nbr_hist[u], float(obs.sinr_db[u]))
                    rec = {
                        "episode": e, "ue_id": f"ue-{u}", "t": st.t, "time_s": round(st.t * sim.dt_s, 1),
                        "serving_cell": int(obs.serving[u]), "speed_kmh": r1(obs.speed_kmh[u]),
                        "sinr_db": r1(obs.sinr_db[u]),
                        "serving_rsrp": [r1(v) for v in obs.serving_hist[u]],
                        "neighbors": [{"cell_id": int(c), "rsrp": [r1(v) for v in h]}
                                      for c, h in zip(obs.nbr_ids[u], obs.nbr_hist[u])],
                        "label": {"action": "HANDOVER" if tgt >= 0 else "STAY",
                                  "target_cell": tgt if tgt >= 0 else None,
                                  "gain_db": r1(st.gain[u]),
                                  "rationale": tok.explain(ans)["rationale"]},
                        "in_corpus": bool(st.keep[u]),
                    }
                    f.write(json.dumps(rec) + "\n")
                    n_reports += 1
                    n_kept += int(st.keep[u])
            if verbose:
                print(f"drive {e + 1}/{n_episodes}: {n_reports} reports ({n_kept} in corpus)", flush=True)

    with open(os.path.join(out_dir, "drives.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return {"reports": n_reports, "in_corpus": n_kept}


_CTX_TEXT = {
    "L": lambda v: f"log-distance ratio serving/neighbour {int(v) / 10:+.1f}",
    "S": lambda v: f"radial speed {int(v) * 4:+d} m/s",
    "M": lambda v: f"radio-map gain {int(v):+d} dB",
    "P": lambda v: f"next-cell probability {int(v) * 10}%",
    "N": lambda v: f"route-history confidence {int(v)}/6",
}


def _ctx_text(tokens) -> str:
    parts = ["unknown" if x == "<unk>" else _CTX_TEXT[x[0]](x[1:]) for x in tokens]
    return f" [{'; '.join(parts)}]" if parts else ""


def prompt_to_text(tok: HandoverTokenizer, prompt_ids) -> str:
    """Render a tokenised report as plain English (for general-purpose LLM fine-tuning)."""
    t = [tok.itos[int(i)] for i in prompt_ids]
    h = tok.obs_cfg.hist_len
    speed_bin = int(t[2][1:])
    i = 7 + h
    j = i + tok.serving_extra
    lines = [f"UE speed {speed_bin * 10}-{speed_bin * 10 + 9} km/h, serving SINR {t[4][1:]} dB.",
             f"Serving cell {t[6][1:]}, L3-filtered RSRP history (dBm, oldest first): "
             + ", ".join(x[1:] for x in t[7:7 + h]) + "." + _ctx_text(t[i:j])]
    i = j
    while i < len(t) and t[i] == "<nbr>":
        j = i + 2 + h + tok.neighbour_extra
        lines.append(f"Neighbour cell {t[i + 1][1:]}, RSRP relative to serving (dB): "
                     + ", ".join(x[1:] for x in t[i + 2:i + 2 + h]) + "." + _ctx_text(t[i + 2 + h:j]))
        i = j
    return "\n".join(lines)


def export_jsonl(tokens: np.ndarray, prompt_len: int, path: str, limit: int | None = None,
                 obs_cfg: ObsConfig | None = None) -> int:
    """Write chat-style instruction data usable to fine-tune any open LLM."""
    tok = HandoverTokenizer(obs_cfg)
    system = ("You are a near-RT RIC mobility xApp. Given a UE measurement report, decide whether to "
              "hand over and to which reported neighbour cell, and explain why.")
    n = 0
    with open(path, "w") as f:
        for row in tokens[:limit]:
            ans = [i for i in row[prompt_len:] if i != tok.PAD]
            rec = {"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_to_text(tok, row[:prompt_len])},
                {"role": "assistant", "content": tok.explain(ans)["rationale"]},
            ]}
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
