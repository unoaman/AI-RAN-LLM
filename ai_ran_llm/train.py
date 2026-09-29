"""Train HandoverGPT on a tokenised handover corpus."""

import math
import time
from dataclasses import asdict

import numpy as np
import torch

from .config import ModelConfig, ObsConfig, SimConfig
from .model import HandoverGPT
from .tokenizer import MAX_CELLS, HandoverTokenizer


def make_batch(tokens: torch.Tensor, prompt_len: int, pad_id: int, prompt_weight: float):
    """Next-token targets; answer tokens get weight 1, report tokens `prompt_weight`
    (a light telemetry-modelling objective: predicting how RSRP evolves)."""
    x, y = tokens[:, :-1], tokens[:, 1:].clone()
    y[y == pad_id] = -100
    w = torch.full(y.shape, prompt_weight)
    w[:, prompt_len - 1:] = 1.0
    w[y == -100] = 0.0
    return x, y, w


@torch.no_grad()
def evaluate_split(model, tokens, prompt_len, tok: HandoverTokenizer, batch_size=1024):
    """Teacher-forced decision accuracy (STAY/HO + target cell) and answer loss."""
    model.eval()
    correct = total = ho_correct = ho_total = 0
    loss_sum = 0.0
    for i in range(0, len(tokens), batch_size):
        b = tokens[i:i + batch_size]
        x, y, w = make_batch(b, prompt_len, tok.PAD, 0.0)
        logits, loss = model(x, y, w.to(b.device))
        loss_sum += loss.item() * len(b)
        act_true = b[:, prompt_len]
        act_pred = logits[:, prompt_len - 1, [tok.STAY, tok.HO]].argmax(-1)
        act_pred = torch.where(act_pred == 1, tok.HO, tok.STAY)
        ok = act_pred == act_true
        is_ho = act_true == tok.HO
        cell_pred = logits[:, prompt_len, tok.cell_base:tok.cell_base + MAX_CELLS].argmax(-1) + tok.cell_base
        ok = ok & (~is_ho | (cell_pred == b[:, prompt_len + 1]))
        correct += ok.sum().item()
        total += len(b)
        ho_correct += (ok & is_ho).sum().item()
        ho_total += is_ho.sum().item()
    model.train()
    return {"answer_loss": loss_sum / total, "decision_acc": correct / total,
            "ho_recall": ho_correct / max(ho_total, 1)}


def train(data_path: str, out_path: str, epochs: int = 3, batch_size: int = 256, lr: float = 1e-3,
          prompt_weight: float = 0.1, val_frac: float = 0.05, seed: int = 0,
          model_cfg: ModelConfig | None = None, device: str | None = None, log_every: int = 200):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(data_path)
    tokens = torch.from_numpy(data["tokens"])
    prompt_len = int(data["prompt_len"])
    tok = HandoverTokenizer()
    assert tok.prompt_len == prompt_len, "dataset built with a different ObsConfig"

    n_val = max(1, int(len(tokens) * val_frac))
    val, trn = tokens[:n_val], tokens[n_val:]
    cfg = model_cfg or ModelConfig()
    cfg.vocab_size = tok.vocab_size
    cfg.block_size = tokens.shape[1] - 1
    model = HandoverGPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M params | train {len(trn)} | val {len(val)} | device {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(trn) / batch_size)
    total, warmup = epochs * steps_per_epoch, min(500, epochs * steps_per_epoch // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))) * 0.95 + 0.05)

    step, t0 = 0, time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(trn))
        for i in range(0, len(trn), batch_size):
            x, y, w = make_batch(trn[perm[i:i + batch_size]], prompt_len, tok.PAD, prompt_weight)
            _, loss = model(x.to(device), y.to(device), w.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0:
                print(f"epoch {ep + 1} step {step}/{total} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} ({time.time() - t0:.0f}s)", flush=True)
        stats = evaluate_split(model, val.to(device), prompt_len, tok)
        print(f"epoch {ep + 1} val: " + ", ".join(f"{k} {v:.4f}" for k, v in stats.items()), flush=True)

    torch.save({"model_cfg": cfg.to_dict(), "obs_cfg": asdict(ObsConfig()), "sim_cfg": asdict(SimConfig()),
                "state_dict": model.state_dict(), "val": stats}, out_path)
    print(f"saved {out_path}")
    return model, stats
