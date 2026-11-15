"""Mock backend for interface development.

Returns responses shaped exactly like the proposed FastAPI contract, built from
a small synthetic dataset. Every response carries a MOCK warning, and nothing
here is a research result. Acquisition dates are kept null, matching the real
dataset's limitation.
"""
import io
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

MOCK_WARNING = "MOCK DATA: illustrative values for interface development, not research results."
DEFAULT_UNCERTAINTY_KM = 0.25
EARTH_RADIUS_KM = 6371.0088


def _envelope(data, warnings=None):
    return {
        "status": "success",
        "data": data,
        "metadata": {"analysis_id": "MOCK_RUN_001", "generated_at": None, "source": "MOCK"},
        "warnings": [MOCK_WARNING] + list(warnings or []),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@lru_cache(maxsize=1)
def _detections():
    rng = np.random.default_rng(7)
    hotspots = [(10.12, 20.80), (10.45, 21.20), (9.85, 20.40), (10.70, 20.55), (10.05, 21.45)]
    rows = []
    for sensor in ("MODIS", "VIIRS"):
        for i in range(100):
            if rng.random() < 0.85:
                c = hotspots[rng.integers(len(hotspots))]
                lat, lon = c[0] + rng.normal(0, 0.02), c[1] + rng.normal(0, 0.02)
            else:
                lat, lon = rng.uniform(9.5, 11.0), rng.uniform(20.0, 21.8)
            if sensor == "MODIS":
                sat = str(rng.choice(["Terra", "Aqua"]))
                conf = int(rng.integers(30, 100))
                frp = round(float(rng.gamma(1.5, 20)), 1)
                scan, track = round(float(rng.uniform(1.0, 2.5)), 2), round(float(rng.uniform(1.0, 1.8)), 2)
            else:
                sat = "N20"
                conf = str(rng.choice(["l", "n", "h"], p=[0.15, 0.6, 0.25]))
                frp = round(float(rng.gamma(1.2, 3)), 2)
                scan, track = round(float(rng.uniform(0.32, 0.8)), 2), round(float(rng.uniform(0.36, 0.75)), 2)
            rows.append({
                "detection_id": f"{sensor}_{i + 1:03d}", "sensor": sensor, "satellite": sat,
                "latitude": round(float(lat), 4), "longitude": round(float(lon), 4),
                "frp": frp, "confidence": conf, "daynight": str(rng.choice(["D", "N"], p=[0.7, 0.3])),
                "scan_km": scan, "track_km": track, "footprint_area_km2": round(scan * track, 3),
                "acq_date": None,
                "acq_time_utc": f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}",
                "timestamp_source": "UNAVAILABLE",
            })
    return rows


def _half_diag_km(d):
    return 0.5 * math.hypot(d["scan_km"], d["track_km"])


@lru_cache(maxsize=16)
def _associations(uncertainty_km=DEFAULT_UNCERTAINTY_KM):
    dets = _detections()
    modis = [d for d in dets if d["sensor"] == "MODIS"]
    viirs = [d for d in dets if d["sensor"] == "VIIRS"]
    pairs = []
    for m in modis:
        best = min(viirs, key=lambda v: haversine_km(m["latitude"], m["longitude"], v["latitude"], v["longitude"]))
        dist = haversine_km(m["latitude"], m["longitude"], best["latitude"], best["longitude"])
        thr = _half_diag_km(m) + _half_diag_km(best) + uncertainty_km
        if dist <= 2 * thr:
            pairs.append({"m": m, "v": best, "dist": dist, "thr": thr})
    viirs_use = pd.Series([p["v"]["detection_id"] for p in pairs if p["dist"] <= p["thr"]]).value_counts()
    out = []
    for i, p in enumerate(sorted(pairs, key=lambda x: x["dist"])):
        within = p["dist"] <= p["thr"]
        if not within:
            status = "UNRESOLVED"
        elif viirs_use.get(p["v"]["detection_id"], 0) > 1:
            status = "AMBIGUOUS"
        else:
            status = "SPATIAL_CANDIDATE"
        out.append({
            "association_id": f"ASSOC_{i + 1:03d}",
            "modis_id": p["m"]["detection_id"], "viirs_id": p["v"]["detection_id"],
            "distance_m": round(p["dist"] * 1000, 1), "spatial_threshold_m": round(p["thr"] * 1000, 1),
            "spatial_candidate": bool(within), "temporal_status": "UNAVAILABLE",
            "association_status": status,
            "similarity_score": round(max(0.0, 1 - p["dist"] / p["thr"]), 3) if within else None,
        })
    return out


@lru_cache(maxsize=1)
def _clusters(eps_km=4.0, min_samples=4):
    dets = _detections()
    n = len(dets)
    coords = [(d["latitude"], d["longitude"]) for d in dets]
    neigh = [[j for j in range(n) if haversine_km(*coords[i], *coords[j]) <= eps_km] for i in range(n)]
    labels = [None] * n
    cid = 0
    for i in range(n):
        if labels[i] is not None:
            continue
        if len(neigh[i]) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cid
        queue = list(neigh[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cid
            if labels[j] is not None:
                continue
            labels[j] = cid
            if len(neigh[j]) >= min_samples:
                queue.extend(neigh[j])
        cid += 1
    df = pd.DataFrame(dets)
    df["label"] = labels
    clusters = []
    for k, g in df[df["label"] >= 0].groupby("label"):
        clusters.append({
            "cluster_id": f"CLUSTER_{k + 1:03d}",
            "latitude": round(g["latitude"].mean(), 4), "longitude": round(g["longitude"].mean(), 4),
            "detection_count": int(len(g)), "modis_count": int((g["sensor"] == "MODIS").sum()),
            "viirs_count": int((g["sensor"] == "VIIRS").sum()), "cluster_type": "SPATIAL_DETECTION_CLUSTER",
        })
    return clusters, int((df["label"] == -1).sum())


def _paginate(rows, params):
    page = int(params.get("page", 1))
    size = int(params.get("page_size", 50))
    return {"data": rows[(page - 1) * size: page * size],
            "pagination": {"page": page, "page_size": size, "total_records": len(rows)}}


def _filter_detections(params):
    rows = _detections()
    if params.get("sensor"):
        rows = [r for r in rows if r["sensor"] == params["sensor"]]
    if params.get("satellite"):
        rows = [r for r in rows if r["satellite"].upper() == str(params["satellite"]).upper()]
    if params.get("daynight"):
        rows = [r for r in rows if r["daynight"] == params["daynight"]]
    if params.get("min_frp") is not None:
        rows = [r for r in rows if r["frp"] >= float(params["min_frp"])]
    if params.get("max_frp") is not None:
        rows = [r for r in rows if r["frp"] <= float(params["max_frp"])]
    return rows


def handle_get(path, params):
    """Route a GET request to the matching mock response."""
    if path == "/health":
        return _envelope({"service": "mock", "healthy": True})
    if path == "/dashboard/summary":
        assoc = _associations()
        clusters, _ = _clusters()
        dets = _detections()
        return _envelope({
            "total_detections": len(dets),
            "modis_detections": sum(d["sensor"] == "MODIS" for d in dets),
            "viirs_detections": sum(d["sensor"] == "VIIRS" for d in dets),
            "spatial_candidate_pairs": sum(a["association_status"] == "SPATIAL_CANDIDATE" for a in assoc),
            "spatiotemporal_candidate_pairs": None,
            "spatial_clusters": len(clusters),
            "temporal_analysis_status": "UNAVAILABLE",
            "dataset_status": "PROCESSED",
        })
    if path == "/detections":
        return _envelope(_paginate(_filter_detections(params), params))
    if path.startswith("/detections/"):
        det_id = path.split("/")[-1]
        match = [d for d in _detections() if d["detection_id"] == det_id]
        if not match:
            return {"status": "error", "error": {"code": "NOT_FOUND", "message": f"{det_id} not found"}}
        return _envelope(match[0])
    if path == "/associations":
        rows = _associations()
        if params.get("status"):
            rows = [r for r in rows if r["association_status"] == params["status"]]
        if params.get("min_distance_m") is not None:
            rows = [r for r in rows if r["distance_m"] >= float(params["min_distance_m"])]
        if params.get("max_distance_m") is not None:
            rows = [r for r in rows if r["distance_m"] <= float(params["max_distance_m"])]
        return _envelope(_paginate(rows, params))
    if path == "/clusters":
        clusters, noise = _clusters()
        return _envelope({"data": clusters, "noise_count": noise, "algorithm": "DBSCAN",
                          "parameters": {"eps_km": 4.0, "min_samples": 4}})
    if path == "/analysis/frp":
        df = pd.DataFrame(_detections())
        out = {s: {"mean": round(g["frp"].mean(), 2), "median": round(g["frp"].median(), 2),
                   "maximum": round(g["frp"].max(), 2)} for s, g in df.groupby("sensor")}
        out["unit"] = "MW"
        return _envelope(out)
    if path == "/analysis/sensitivity":
        dets = _detections()
        modis = [d for d in dets if d["sensor"] == "MODIS"]
        viirs = [d for d in dets if d["sensor"] == "VIIRS"]
        nearest = [min(haversine_km(m["latitude"], m["longitude"], v["latitude"], v["longitude"]) for v in viirs)
                   for m in modis]
        data = [{"threshold_m": t, "candidate_pairs": int(sum(d * 1000 <= t for d in nearest))}
                for t in (250, 500, 1000, 1500, 2000, 3000, 5000)]
        return _envelope({"data": data})
    if path == "/analysis/temporal":
        return _envelope({"status": "UNAVAILABLE", "reason": "ACQUISITION_DATES_MISSING",
                          "available_records": 0,
                          "message": "Temporal analysis requires valid acquisition dates."})
    if path == "/data-quality":
        dets = _detections()
        return _envelope({
            "modis_records": sum(d["sensor"] == "MODIS" for d in dets),
            "viirs_records": sum(d["sensor"] == "VIIRS" for d in dets),
            "missing_dates": sum(d["acq_date"] is None for d in dets),
            "invalid_coordinates": 0, "invalid_frp_values": 0, "invalid_footprints": 0,
            "unknown_confidence": 0, "duplicate_records": 0,
            "synthetic_dates_used": False, "temporal_analysis": "BLOCKED",
            "warnings": ["Acquisition dates are unavailable.", "Temporal matching cannot be validated."],
        })
    if path == "/analysis/runs":
        return _envelope({"data": [{
            "analysis_id": "MOCK_RUN_001", "dataset_versions": ["modis_mock_v1", "viirs_mock_v1"],
            "processed_at": None, "algorithm": "Footprint-based spatial association + DBSCAN",
            "parameters": {"uncertainty_km": DEFAULT_UNCERTAINTY_KM, "eps_km": 4.0, "min_samples": 4},
            "status": "COMPLETED", "records_processed": len(_detections()),
            "warnings": ["Acquisition dates are missing."]}]})
    return {"status": "error", "error": {"code": "NOT_FOUND", "message": f"Mock has no endpoint {path}"}}


def handle_post(path, body):
    """Route a POST request to the matching mock response."""
    if path == "/analysis/associations/run":
        unc = float(body.get("uncertainty_km") or DEFAULT_UNCERTAINTY_KM)
        assoc = _associations(round(unc, 3))
        return _envelope({
            "analysis_id": "MOCK_RUN_002", "status": "COMPLETED",
            "applied_parameters": {"uncertainty_km": unc,
                                   "distance_threshold_mode": body.get("distance_threshold_mode", "FOOTPRINT_BASED"),
                                   "time_window_minutes": None},
            "spatial_candidate_pairs": sum(a["association_status"] == "SPATIAL_CANDIDATE" for a in assoc),
            "temporal_analysis_status": "UNAVAILABLE",
        }, warnings=["Acquisition dates are missing."])
    return {"status": "error", "error": {"code": "NOT_FOUND", "message": f"Mock has no endpoint {path}"}}


def handle_export(fmt):
    """Mock export: processed detections as CSV or JSON bytes."""
    rows = _detections()
    if fmt == "json":
        return json.dumps({"source": "MOCK", "warning": MOCK_WARNING, "data": rows}, indent=2).encode()
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()
