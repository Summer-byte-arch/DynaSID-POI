"""数据格式、时间划分与数据泄漏检查。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

REQUIRED_COLUMNS = {
    "user_id", "POI_id", "POI_catid_code", "latitude", "longitude",
    "trajectory_id", "local_time",
}


@dataclass(frozen=True)
class SplitAudit:
    split: str
    rows: int
    users: int
    pois: int
    trajectories: int
    start_time: str
    end_time: str
    missing_required_values: int


def read_split(path: Path, split: str) -> tuple[pd.DataFrame, SplitAudit]:
    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} 缺少字段：{sorted(missing_columns)}")
    missing_values = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())
    frame["local_time"] = pd.to_datetime(frame["local_time"], utc=True, errors="coerce")
    if frame.local_time.isna().any():
        raise ValueError(f"{path.name} 包含无效的 local_time 值")
    if not frame.latitude.between(-90, 90).all() or not frame.longitude.between(-180, 180).all():
        raise ValueError(f"{path.name} 包含无效经纬度")
    audit = SplitAudit(
        split=split, rows=len(frame), users=frame.user_id.nunique(), pois=frame.POI_id.nunique(),
        trajectories=frame.trajectory_id.nunique(), start_time=str(frame.local_time.min()),
        end_time=str(frame.local_time.max()), missing_required_values=missing_values,
    )
    return frame, audit


def audit_city(data_dir: Path, prefix: str) -> dict:
    frames, audits = {}, []
    for split in ("train", "val", "test"):
        frames[split], audit = read_split(data_dir / f"{prefix}_{split}.csv", split)
        audits.append(audit)
    # Official temporal splits may divide one long trajectory across boundaries.
    # Report that overlap, but fail only on duplicated observed events.
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_keys = set(zip(frames[left].user_id, frames[left].trajectory_id))
        right_keys = set(zip(frames[right].user_id, frames[right].trajectory_id))
        overlaps[f"{left}_{right}_trajectory_overlap"] = len(left_keys & right_keys)
        event_columns = ["user_id", "POI_id", "local_time"]
        left_events = set(map(tuple, frames[left][event_columns].itertuples(index=False, name=None)))
        right_events = set(map(tuple, frames[right][event_columns].itertuples(index=False, name=None)))
        duplicate_events = left_events & right_events
        overlaps[f"{left}_{right}_duplicate_events"] = len(duplicate_events)
        if duplicate_events:
            raise ValueError(f"{left} 与 {right} 之间存在重复事件：{len(duplicate_events)}")
    overlaps["chronological_boundaries"] = {
        "train_end_le_val_start": bool(frames["train"].local_time.max() <= frames["val"].local_time.min()),
        "val_end_le_test_start": bool(frames["val"].local_time.max() <= frames["test"].local_time.min()),
    }
    return {"city": prefix, "splits": [asdict(x) for x in audits], **overlaps}


def write_audit(audit: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
