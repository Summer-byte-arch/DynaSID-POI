"""排序、可达性和用户级不确定性评估指标。"""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd


def ranking_metrics(
    ranked_ids: Iterable,
    target_id,
    distances_km: Iterable[float],
    *,
    top_k: int = 10,
    infeasible_threshold_km: float = 10.0,
) -> dict[str, float]:
    ranked = list(ranked_ids)[:top_k]
    distances = np.asarray(list(distances_km), dtype=float)[:top_k]
    if len(ranked) != len(distances):
        raise ValueError("ranked_ids 与 distances_km 的长度必须一致")
    rank = ranked.index(target_id) + 1 if target_id in ranked else np.inf
    return {
        "Acc@1": float(rank == 1),
        f"Recall@{top_k}": float(rank <= top_k),
        f"NDCG@{top_k}": float(1.0 / np.log2(rank + 1)) if rank <= top_k else 0.0,
        f"Infeasible@{top_k}": float(np.mean(distances > infeasible_threshold_km)) if len(distances) else 0.0,
        f"MeanDist@{top_k}": float(np.mean(distances)) if len(distances) else 0.0,
    }


def user_paired_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    *,
    user_col: str = "user",
    new_col: str = "new",
    base_col: str = "base",
    samples: int = 5000,
    seed: int = 20260809,
) -> dict[str, float | int]:
    required = {user_col, new_col, base_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"缺少必要字段：{sorted(missing)}")
    per_user = frame.assign(delta=frame[new_col] - frame[base_col]).groupby(user_col).delta.mean().to_numpy()
    if len(per_user) == 0:
        raise ValueError("Bootstrap 至少需要一名用户")
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for i in range(samples):
        boot[i] = rng.choice(per_user, len(per_user), replace=True).mean()
    return {
        "metric": metric,
        "mean_delta": float(per_user.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_users": int(len(per_user)),
        "bootstrap_samples": int(samples),
    }
