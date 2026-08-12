"""明确使用距离单位的向量化地理计算工具。"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """计算标量或数组输入的球面距离，单位为千米。"""
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(
        np.asarray(lat1, dtype=float), np.asarray(lon1, dtype=float),
        np.asarray(lat2, dtype=float), np.asarray(lon2, dtype=float),
    )
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    result = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return float(result) if result.ndim == 0 else result


def pairwise_haversine_km(latitudes, longitudes):
    """返回 N×N 对称距离矩阵。"""
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) != len(lon):
        raise ValueError("纬度和经度必须是长度相同的一维数组")
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
