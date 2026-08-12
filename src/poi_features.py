"""Leakage-safe POI feature construction for the GNPR-SID stage."""
from __future__ import annotations

import numpy as np


def build_poi_features(train, split_name="train"):
    poi = train.groupby("POI_id", sort=False).agg(
        latitude=("latitude", "first"), longitude=("longitude", "first"),
        category=("POI_catid_code", "first"), visits=("POI_id", "size")
    ).reset_index()
    pidx = {p: i for i, p in enumerate(poi.POI_id)}
    cats = sorted(train.POI_catid_code.unique()); cidx = {c: i for i, c in enumerate(cats)}
    users = sorted(train.user_id.astype(str).unique()); uidx = {u: i for i, u in enumerate(users)}
    lat_edges = np.unique(np.quantile(poi.latitude, np.linspace(0, 1, 11)))
    lon_edges = np.unique(np.quantile(poi.longitude, np.linspace(0, 1, 11)))
    lat_bin = np.clip(np.searchsorted(lat_edges[1:-1], poi.latitude), 0, 9)
    lon_bin = np.clip(np.searchsorted(lon_edges[1:-1], poi.longitude), 0, 9)
    region_raw = lat_bin * 10 + lon_bin; regions = sorted(np.unique(region_raw)); ridx = {r: i for i, r in enumerate(regions)}
    category = np.zeros((len(poi), len(cats)), np.float32); region = np.zeros((len(poi), len(regions)), np.float32)
    hour = np.zeros((len(poi), 24), np.float32); user = np.zeros((len(poi), len(users)), np.float32)
    for i, (c, r) in enumerate(zip(poi.category, region_raw)):
        category[i, cidx[c]] = 1; region[i, ridx[r]] = 1
    for row in train.itertuples():
        i = pidx[row.POI_id]; hour[i, int(row.local_time.hour)] += 1; user[i, uidx[str(row.user_id)]] += 1
    user = (user > 0).astype(np.float32); hour = (hour > 0).astype(np.float32)
    x = np.concatenate([category, region, hour, user], axis=1)
    meta = {"n_pois": len(poi), "n_categories": len(cats), "n_regions": len(regions), "n_hours": 24,
            "n_users": len(users), "input_dim": x.shape[1], "block_order": ["category", "region", "hour", "user_multi_hot"],
            "fitted_on": f"{split_name} only"}
    return poi, x, meta
