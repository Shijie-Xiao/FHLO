"""Train per-step Markov conditional Gaussians for each case dir.

Verbatim method from Reproduce/track_model/train_markov.py (FHLO paper sec.3a):
at EACH lead time fit a k=1 Gaussian (single mixture component) to that
step's P(u_{t-1}, v_{t-1}, u_t, v_t) rows. Time-pooled stationary fits are
NOT used -- they bias the synthetic mean on recurving storms.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PROCESSED_TRACKS_DIR


def _fit_gaussian(velocity_pairs: np.ndarray):
    """Fit Gaussian to joint (u_{t-1}, v_{t-1}, u_t, v_t) and compute
    conditional parameters."""
    mu_joint = np.mean(velocity_pairs, axis=0)
    Sigma_joint = np.cov(velocity_pairs, rowvar=False)
    mu_old, mu_new = mu_joint[:2], mu_joint[2:]
    Sigma_oo, Sigma_on = Sigma_joint[:2, :2], Sigma_joint[:2, 2:]
    Sigma_no, Sigma_nn = Sigma_joint[2:, :2], Sigma_joint[2:, 2:]

    try:
        L_oo = np.linalg.cholesky(Sigma_oo + np.eye(2) * 1e-8)
        Sigma_oo_inv = np.linalg.solve(L_oo.T, np.linalg.solve(L_oo, np.eye(2)))
    except np.linalg.LinAlgError:
        Sigma_oo_inv = np.linalg.pinv(Sigma_oo)

    A = Sigma_no @ Sigma_oo_inv
    Sigma_cond = Sigma_nn - A @ Sigma_on
    Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2.0
    eig = np.linalg.eigvals(Sigma_cond)
    if np.any(eig <= 0):
        Sigma_cond += np.eye(2) * max(1e-6, float(-np.min(eig) + 1e-6))

    return {
        "mu_old": mu_old,
        "mu_new": mu_new,
        "Sigma_oo": Sigma_oo,
        "A": A,
        "Sigma_cond": Sigma_cond,
    }


def _fit_per_step(step_indices: np.ndarray, velocity_pairs: np.ndarray,
                  min_samples: int = 5):
    """Fit per-step Gaussian for each unique step index."""
    step_params = {}
    for s in np.unique(step_indices):
        pairs_s = velocity_pairs[step_indices == s]
        if len(pairs_s) >= min_samples:
            step_params[int(s)] = _fit_gaussian(pairs_s)
    return step_params


def train_case(case_dir: Path):
    """pairs_6h.pkl -> markov_params_6h.pkl for one case dir."""
    pairs_file = case_dir / "pairs_6h.pkl"
    if not pairs_file.exists():
        return None
    with open(pairs_file, "rb") as f:
        payload = pickle.load(f)
    velocity_pairs = payload["velocity_pairs"]
    step_indices = payload.get("step_indices")
    if step_indices is None or len(step_indices) == 0:
        return None

    step_params = _fit_per_step(step_indices, velocity_pairs)
    if not step_params:
        return None

    first_step = min(step_params.keys())
    params = {
        "mu_old": step_params[first_step]["mu_old"],
        "Sigma_oo": step_params[first_step]["Sigma_oo"],
        "dt_hours": payload.get("dt_hours", 6.0),
        "reference_time": payload.get("forced_init_time"),
        "step_params": step_params,
        "max_reliable_step": int(step_indices.max()),
    }
    with open(case_dir / "markov_params_6h.pkl", "wb") as f:
        pickle.dump({
            "storm_config": payload.get("storm_config", {}),
            "markov_params": params,
            "velocity_pairs": velocity_pairs,
            "step_indices": step_indices,
            "dt_hours": payload.get("dt_hours", 6.0),
            "reference_time": payload.get("forced_init_time"),
        }, f)
    return len(step_params)


def run_train_markov(storm=None):
    root = PROCESSED_TRACKS_DIR / storm.lower() if storm else PROCESSED_TRACKS_DIR
    n = 0
    for case in sorted(root.glob("*/") if storm else root.glob("*/*/")):
        if not case.is_dir():
            continue
        r = train_case(case)
        if r:
            n += 1
            print(f"  {case.parent.name}/{case.name}: {r} steps fitted")
    print(f"train_markov: {n} case(s)")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--storm", default="")
    args = ap.parse_args()
    run_train_markov(args.storm or None)
