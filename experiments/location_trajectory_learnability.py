"""Location, radio-map and trajectory-mining learnability: original simulator vs city.

Reproduces the results in docs/LOCATION_AWARE_HANDOVER.md §9:

    PYTHONPATH=. python experiments/location_trajectory_learnability.py 5   # also run with 6 and 7

For each world (original simulator, and the city model with road mobility + spatial shadowing)
it simulates 16 drives x 32 UEs x 60 s with the corpus's behaviour-policy mix, labels every
report with the look-ahead teacher, and trains a small MLP to predict, for each of the 4
reported neighbours, "the teacher hands over to this cell now". Feature sets:

* report only: today's model inputs
* + position & heading: distance ratios and radial speeds from positions with 10 m
  (temporally correlated) error
* + radio-map forecast: RSRP map (20 m grid) learned from the 12 training drives; for each
  neighbour, map(neighbour) - map(serving) at the current and the predicted (+1 s) position
* + trajectory prior: P(next serving cell | previous, current) mined from the handover
  sequences of the training drives; needs no location
* all combined

Maps and transition statistics use the 12 training drives only; scores come from the 4 held-out
drives. Learnability only: no closed-loop KPIs.
"""
import sys, json, time
import numpy as np, torch
from collections import defaultdict
from ai_ran_llm.config import ObsConfig, SimConfig
from ai_ran_llm.simulator import generate_episode, run_policy
from ai_ran_llm.dataset import _behaviour_policy
from ai_ran_llm.policies import label_decision

N_EP, N_TRAIN, U, T, SIGMA, G_RES = 16, 12, 32, 600, 10.0, 20.0
G_N = int(2560 / G_RES)
o = ObsConfig()

def pos_error(rng, s):
    z = np.zeros((U, T, 2)); x = rng.normal(0, s, (U, 2))
    for t in range(T):
        z[:, t] = x; x = 0.9 * x + np.sqrt(1 - 0.81) * rng.normal(0, s, (U, 2))
    return z

def grid(p):
    g = np.clip(np.floor(p / G_RES + G_N / 2).astype(int), 0, G_N - 1)
    return g[..., 0], g[..., 1]

def run_world(name, sim, seed):
    t0 = time.time(); rng = np.random.default_rng(seed)
    samples = []                    # per episode: dict of arrays
    map_sum = map_cnt = None; trans3 = defaultdict(lambda: defaultdict(int)); trans2 = defaultdict(lambda: defaultdict(int))
    for e in range(N_EP):
        ep = generate_episode(U, T, rng, sim)
        if map_sum is None:
            map_sum = np.zeros((ep.n_cells, G_N, G_N)); map_cnt = np.zeros_like(map_sum)
        est = ep.pos + pos_error(rng, SIGMA)                          # what a Location xApp would see
        _, traj, seen = run_policy(ep, _behaviour_policy(rng, o), o, record=True)
        # previous distinct serving cell at every step, per UE
        prev = np.full((U, T), -1); stint_prev = np.full(U, -1)
        for t in range(1, T):
            ch = traj[:, t] != traj[:, t - 1]
            stint_prev = np.where(ch, traj[:, t - 1], stint_prev); prev[:, t] = stint_prev
        if e < N_TRAIN:                                               # history for maps / mining
            gx, gy = grid(est)
            top = np.argsort(-ep.rsrp_meas, axis=2)[:, :, :9]         # a UE reports ~serving + 8 strongest
            for k in range(9):
                c = top[:, :, k]
                np.add.at(map_sum, (c, gx, gy), np.take_along_axis(ep.rsrp_meas, c[..., None], 2)[..., 0])
                np.add.at(map_cnt, (c, gx, gy), 1)
            for u in range(U):
                seq = [traj[u, 0]] + [traj[u, t] for t in range(1, T) if traj[u, t] != traj[u, t - 1]]
                for i in range(1, len(seq)):
                    trans2[seq[i - 1]][seq[i]] += 1
                    if i >= 2:
                        trans3[(seq[i - 2], seq[i - 1])][seq[i]] += 1
        rec = defaultdict(list)
        for t, obs in seen:
            if t < 8 or t + 10 >= T: continue
            d = obs.nbr_hist - obs.serving_hist[:, None, :]
            rec['base'].append(np.concatenate([d[:, :, -1], d[:, :, -1] - d[:, :, 0],
                (obs.serving_hist[:, -1] - obs.serving_hist[:, 0])[:, None], obs.serving_hist[:, -1:] / 10 + 9,
                obs.sinr_db[:, None] / 10, obs.speed_kmh[:, None] / 100], 1))
            cells = np.concatenate([obs.serving[:, None], obs.nbr_ids], 1)
            site = ep.sites[cells]
            dn = np.linalg.norm(est[:, t][:, None] - site, axis=-1) + 1
            do = np.linalg.norm(est[:, t - 8][:, None] - site, axis=-1) + 1
            rec['geo'].append(np.concatenate([np.log10(dn[:, :1] / dn[:, 1:]), (dn - do) / 0.8 / 30], 1))
            rec['pos_now'].append(est[:, t]); rec['vel'].append((est[:, t] - est[:, t - 8]) / 0.8)
            rec['cells'].append(cells); rec['prev'].append(prev[:, t])
            tg, _ = label_decision(ep, t, obs, o)
            rec['y'].append((obs.nbr_ids == tg[:, None]).astype(np.float32))
        samples.append({k: np.concatenate(v) for k, v in rec.items()})
        print(f'  [{name}] episode {e + 1}/{N_EP} ({time.time() - t0:.0f}s)', flush=True)

    rmap = np.where(map_cnt > 0, map_sum / np.maximum(map_cnt, 1), np.nan)
    def radio_feats(s):
        out = []
        for lead in (0.0, 1.0):
            gx, gy = grid(s['pos_now'] + lead * s['vel'])
            v = rmap[s['cells'], gx[:, None], gy[:, None]]            # (N, 5)
            ok = np.isfinite(v)
            diff = np.where(ok[:, 1:] & ok[:, :1], v[:, 1:] - v[:, :1], 0.0) / 10
            out += [diff, (ok[:, 1:] & ok[:, :1]).astype(float)]
        return np.concatenate(out, 1)
    def traj_feats(s):
        n = len(s['y']); f = np.zeros((n, 8))
        for i in range(n):
            cur, pv = s['cells'][i, 0], s['prev'][i]
            dist = trans3.get((pv, cur)) if pv >= 0 else None
            if not dist or sum(dist.values()) < 5:
                dist = trans2.get(cur, {})
            tot = sum(dist.values())
            for k in range(4):
                c = s['cells'][i, k + 1]
                f[i, k] = dist.get(c, 0) / tot if tot else 0.0
            f[i, 4:] = np.log1p(tot) / 5
        return f
    for s in samples:
        s['radio'] = radio_feats(s); s['traj'] = traj_feats(s)
    sets = {'report only (today)': ['base'], '+ position & heading (σ=10 m)': ['base', 'geo'],
            '+ radio-map forecast (σ=10 m)': ['base', 'radio'], '+ trajectory prior (no location)': ['base', 'traj'],
            'all combined': ['base', 'geo', 'radio', 'traj']}
    tr, te = samples[:N_TRAIN], samples[N_TRAIN:]
    Ytr = torch.tensor(np.concatenate([s['y'] for s in tr])); Yte = np.concatenate([s['y'] for s in te]).astype(bool)
    res = {}
    for label, keys in sets.items():
        torch.manual_seed(0)
        X = torch.tensor(np.concatenate([np.concatenate([s[k] for k in keys], 1) for s in tr]), dtype=torch.float32)
        Xte = torch.tensor(np.concatenate([np.concatenate([s[k] for k in keys], 1) for s in te]), dtype=torch.float32)
        m = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 64),
                                torch.nn.ReLU(), torch.nn.Linear(64, 4))
        opt = torch.optim.Adam(m.parameters(), 3e-3)
        for i in range(800):
            idx = torch.randint(0, len(X), (8192,))
            l = torch.nn.functional.binary_cross_entropy_with_logits(m(X[idx]), Ytr[idx]); opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad(): P = torch.sigmoid(m(Xte)).numpy()
        def auc(s_, y):
            o_ = np.argsort(s_); r = np.empty(len(s_)); r[o_] = np.arange(len(s_)); p = y.sum()
            return (r[y].sum() - p * (p - 1) / 2) / (p * (len(y) - p))
        def rec50(s_, y):
            o_ = np.argsort(-s_); tp = np.cumsum(y[o_]); prec = tp / np.arange(1, len(s_) + 1)
            return (tp / y.sum())[prec >= 0.5].max() if (prec >= 0.5).any() else 0.0
        pair_s, pair_y = P.ravel(), Yte.ravel()
        any_s, any_y = P.max(1), Yte.any(1)
        hit = (P.argmax(1) == Yte.argmax(1))[any_y].mean()
        res[label] = dict(pair_auc=auc(pair_s, pair_y), pair_r50=rec50(pair_s, pair_y), any_auc=auc(any_s, any_y),
                          any_r50=rec50(any_s, any_y), target_acc=hit)
        print(f'  [{name}] {label:34s} ' + '  '.join(f'{k} {v:.3f}' for k, v in res[label].items()), flush=True)
    return res

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
worlds = [('original sim', SimConfig(), SEED), ('city: roads + spatial shadowing', SimConfig(mobility='roads', shadowing='spatial'), SEED)]
out = {n: run_world(n, c, s) for n, c, s in worlds}
json.dump(out, open('location_trajectory_%d.json' % SEED, 'w'), indent=1)
print('done')
