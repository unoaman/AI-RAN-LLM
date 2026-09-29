"""Shared configuration for the radio simulator, tokenizer and model."""

from dataclasses import dataclass, asdict


@dataclass
class SimConfig:
    """Radio / mobility parameters (3GPP-style macro deployment)."""

    rings: int = 2                 # hex rings around the centre site -> 19 cells
    isd_m: float = 500.0           # inter-site distance
    dt_s: float = 0.1              # simulation step (100 ms)
    tx_power_dbm: float = 15.0     # per-RE reference signal power
    noise_dbm: float = -125.0      # per-RE noise incl. noise figure
    load: float = 0.7              # neighbour-cell load factor for interference
    shadow_sigma_db: float = 6.0
    shadow_decorr_m: float = 50.0
    fading_sigma_db: float = 2.0   # instantaneous fast-fading spread (dB domain)
    meas_sigma_db: float = 1.5     # UE measurement error
    l3_alpha: float = 0.5          # L3 filter coefficient (filterCoefficient k=4)
    min_speed_kmh: float = 3.0
    max_speed_kmh: float = 120.0
    # radio link monitoring / handover outcome modelling
    q_out_db: float = -8.0         # out-of-sync threshold
    t310_steps: int = 5            # consecutive out-of-sync steps before RLF
    hof_sinr_db: float = -10.0     # serving SINR below this at HO command -> HO failure
    ping_pong_steps: int = 10      # return to previous cell within 1 s = ping-pong
    ho_interruption_s: float = 0.05


@dataclass
class ObsConfig:
    """What a measurement report exposes to the policy."""

    n_neighbors: int = 4           # neighbours reported (strongest first)
    hist_len: int = 5              # RSRP snapshots per cell in the report
    hist_stride: int = 2           # steps between snapshots (200 ms)
    oracle_horizon: int = 10       # teacher looks 1 s into the future
    oracle_margin_db: float = 2.0
    # training-label smoothing (see policies.label_decision)
    label_window: int = 0           # HO label if the teacher would hand over within this many steps
    label_confirm_horizon: int = 20 # ...and the target still beats serving on average over 2 s


@dataclass
class ModelConfig:
    vocab_size: int = 0            # filled in from the tokenizer
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)
