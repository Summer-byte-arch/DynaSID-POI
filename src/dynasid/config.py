"""带类型约束和合法性校验的 JSON 实验配置。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    city_prefix: str
    seed: int
    codebook_sizes: tuple[int, ...]
    embedding_dim: int
    epochs: int
    batch_size: int
    patience: int
    top_k: int
    infeasible_threshold_km: float
    bootstrap_samples: int
    spatial_radii_km: tuple[float, ...]
    frozen_weights: tuple[float, ...]
    confirmation_protocol: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sid = raw["sid"]
        evaluation = raw["evaluation"]
        config = cls(
            city_prefix=str(raw["city_prefix"]),
            seed=int(raw["seed"]),
            codebook_sizes=tuple(int(x) for x in sid["codebook_sizes"]),
            embedding_dim=int(sid["embedding_dim"]),
            epochs=int(sid["epochs"]),
            batch_size=int(sid["batch_size"]),
            patience=int(sid["patience"]),
            top_k=int(evaluation["top_k"]),
            infeasible_threshold_km=float(evaluation["infeasible_threshold_km"]),
            bootstrap_samples=int(evaluation["bootstrap_samples"]),
            spatial_radii_km=tuple(float(x) for x in raw["spatial_radii_km"]),
            frozen_weights=tuple(float(x) for x in raw["frozen_weights"]),
            confirmation_protocol=raw.get("confirmation_protocol"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.city_prefix or not self.city_prefix.isalnum():
            raise ValueError("city_prefix 必须是非空的字母数字标识")
        if len(self.codebook_sizes) < 2 or any(x <= 1 for x in self.codebook_sizes):
            raise ValueError("至少需要两个大小大于 1 的码本")
        if self.top_k <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("top_k 和 bootstrap_samples 必须为正数")
        if self.infeasible_threshold_km <= 0:
            raise ValueError("infeasible_threshold_km 必须为正数")
        if tuple(sorted(self.spatial_radii_km)) != self.spatial_radii_km:
            raise ValueError("spatial_radii_km 必须按升序排列")
        if len(self.frozen_weights) != 17:
            raise ValueError("当前发布的特征评分器需要 17 个权重")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return ExperimentConfig.from_mapping(json.load(stream))
