"""Compatibility routes for the Streamlit frontend (app/services/api.py).

These reuse the existing pipeline logic and reshape results into what the
frontend pages read. Calls without country/year default to DEFAULT_COUNTRY /
DEFAULT_YEAR so pages that send no filters still get data.
"""
import io
import json
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response
from sklearn.neighbors import BallTree

from app.services import pipeline

DEFAULT_COUNTRY = "Chad"
DEFAULT_YEAR = 2018
SENSITIVITY_THRESHOLDS_M = (250, 500, 1000, 1500, 2000, 3000, 5000)

router = APIRouter(prefix="/api/v1")


def _response(data, warnings=None):
    return {
        "status": "success",
        "data": data,
        "metadata": {"generated_at": datetime.now(timezone.utc).isoformat()},
        "warnings": [w for w in (warnings or []) if w],
    }


def _unavailable(data, message="Not available yet"):
    return _response(data, [message])


def _defaults(country, year):
    country = (country or DEFAULT_COUNTRY).title()
    year = int(year or DEFAULT_YEAR)
    if country not in ("Bolivia", "Chad"):
        raise HTTPException(status_code=400, detail="Country must be Bolivia or Chad.")
    return country, year


def _paginate(rows, page, page_size):
    start = (page - 1) * page_size
    return {
        "data": rows[start:start + page_size],
        "pagination": {"page": page, "page_size": page_size, "total_records": len(rows)},
    }


@lru_cache(maxsize=16)
def _associations(country, year, max_time_minutes, uncertainty_km):
    """Run pipeline.associate and reshape candidates for the frontend."""
    result = pipeline.associate(country, year, max_time_minutes, uncertainty_km)
    cands = result.get("candidates", [])
    viirs_use = pd.Series(
        [f"{c['viirs_file']}#{c['viirs_row']}" for c in cands], dtype=object
    ).value_counts()
    rows = []
    for i, c in enumerate(sorted(cands, key=lambda c: c["distance_km"])):
        viirs_id = f"{c['viirs_file']}#{c['viirs_row']}"
        status = "AMBIGUOUS" if viirs_use.get(viirs_id, 0) > 1 else "SPATIOTEMPORAL_CANDIDATE"
        rows.append({
            "association_id": f"ASSOC_{i + 1:04d}",
            "modis_id": f"{c['modis_file']}#{c['modis_row']}",
            "viirs_id": viirs_id,
            "distance_m": round(c["distance_km"] * 1000, 1),
            "spatial_threshold_m": round(c["footprint_threshold_km"] * 1000, 1),
            "time_difference_minutes": round(c["time_difference_minutes"], 1),
            "spatial_candidate": True,
            "temporal_status": "WITHIN_WINDOW",
            "association_status": status,
            "similarity_score": round(c["spatial_similarity"], 3),
        })
    return rows, result.get("candidate_count", 0), result.get("parameters"), result.get("warning")


@lru_cache(maxsize=4)
def associated_detections(country=DEFAULT_COUNTRY, year=DEFAULT_YEAR):
    """Detection rows referenced by the default /associations pairs, with
    detection_id, so the frontend can draw relationship lines."""
    rows, _, _, _ = _associations(country, year, 60, 1.0)
    wanted = {r["modis_id"] for r in rows} | {r["viirs_id"] for r in rows}
    out = []
    for sensor in ("MODIS", "VIIRS"):
        for d in pipeline.get_detections(country, year, sensor, 3000):
            det_id = f"{d['source_file']}#{d['source_row']}"
            if det_id in wanted:
                out.append({**d, "detection_id": det_id})
    return out


@router.get("/associations")
def associations(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    max_time_minutes: int = Query(default=60, ge=0, le=1440),
    uncertainty_km: float = Query(default=1.0, ge=0, le=20),
    status: str | None = None,
    min_distance_m: float | None = None,
    max_distance_m: float | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=500, ge=1, le=1000),
):
    """MODIS-VIIRS candidate pairs (reuses pipeline.associate)."""
    country, year = _defaults(country, year)
    try:
        rows, total, params, warning = _associations(
            country, year, max_time_minutes, float(uncertainty_km))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if status:
        rows = [r for r in rows if r["association_status"] == status]
    if min_distance_m is not None:
        rows = [r for r in rows if r["distance_m"] >= min_distance_m]
    if max_distance_m is not None:
        rows = [r for r in rows if r["distance_m"] <= max_distance_m]
    out = _paginate(rows, page, page_size)
    out.update({"candidate_count": total, "parameters": params})
    warnings = [warning]
    if total > len(rows):
        warnings.append(f"Showing {len(rows)} of {total} candidates (sampled detections).")
    return _response(out, warnings)


@lru_cache(maxsize=16)
def _clusters(country, year, sensor, eps_km, min_samples):
    return pipeline.spatial_clusters(country, year, sensor, eps_km, min_samples)


@router.get("/clusters")
def clusters(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str = Query(default="VIIRS"),
    eps_km: float = Query(default=2.0, gt=0, le=50),
    min_samples: int = Query(default=5, ge=2, le=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=500, ge=1, le=1000),
):
    """Per-day DBSCAN clusters (reuses pipeline.spatial_clusters)."""
    country, year = _defaults(country, year)
    sensor = sensor.upper()
    if sensor not in ("MODIS", "VIIRS"):
        raise HTTPException(status_code=400, detail="Sensor must be MODIS or VIIRS.")
    try:
        result = _clusters(country, year, sensor, float(eps_km), int(min_samples))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    rows = [{
        "cluster_id": f"CLUSTER_{c['cluster_id'] + 1:04d}",
        "acquisition_date": c["acquisition_date"],
        "latitude": round(c["centroid_latitude"], 4),
        "longitude": round(c["centroid_longitude"], 4),
        "detection_count": c["observation_count"],
        "modis_count": c["observation_count"] if sensor == "MODIS" else 0,
        "viirs_count": c["observation_count"] if sensor == "VIIRS" else 0,
        "mean_frp": c["mean_frp"],
        "cluster_type": "SPATIAL_DETECTION_CLUSTER",
    } for c in result.get("clusters", [])]
    out = _paginate(rows, page, page_size)
    out.update({
        "noise_count": result.get("noise_count"),
        "algorithm": "DBSCAN",
        "parameters": {"country": country, "year": year, "sensor": sensor,
                       "eps_km": eps_km, "min_samples": min_samples},
    })
    return _response(out, [result.get("warning")])


@lru_cache(maxsize=16)
def _nearest_distances_m(country, year):
    """Nearest VIIRS distance (m) for each sampled MODIS detection."""
    modis = pd.DataFrame(pipeline.get_detections(country, year, "MODIS", 3000))
    viirs = pd.DataFrame(pipeline.get_detections(country, year, "VIIRS", 3000))
    if modis.empty or viirs.empty:
        return None
    tree = BallTree(np.radians(viirs[["latitude", "longitude"]].to_numpy(float)),
                    metric="haversine")
    dist, _ = tree.query(np.radians(modis[["latitude", "longitude"]].to_numpy(float)), k=1)
    return dist[:, 0] * pipeline.EARTH_RADIUS_KM * 1000


@router.get("/analysis/sensitivity")
def sensitivity(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
    min_samples: int = Query(default=5, ge=2, le=100),
):
    """Candidate pairs vs distance threshold. With country+year+sensor given,
    the original clustering sensitivity result is included too."""
    explicit = country is not None and year is not None and sensor is not None
    country, year = _defaults(country, year)
    try:
        dists = _nearest_distances_m(country, year)
        extra = {}
        if explicit:
            extra = pipeline.clustering_sensitivity(country, year, sensor.upper(), min_samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if dists is None:
        return _unavailable({"data": [], **extra},
                            "Both sensors need detections for this country/year.")
    rows = [{"threshold_m": t, "candidate_pairs": int((dists <= t).sum())}
            for t in SENSITIVITY_THRESHOLDS_M]
    return _response(
        {**extra, "data": rows, "country": country, "year": year},
        ["Spatial-only nearest-neighbour counts on sampled detections (no time window).",
         extra.get("warning")])


@lru_cache(maxsize=16)
def _temporal(country, year):
    """Daily valid-detection counts per sensor."""
    counts, total = [], 0
    for chunk in pipeline.iter_observations(country, year):
        valid = chunk[chunk["timestamp_valid"] & chunk["coordinate_valid"]]
        total += len(valid)
        if valid.empty:
            continue
        day = valid["acq_datetime"].dt.strftime("%Y-%m-%d")
        counts.append(valid.assign(date=day).groupby(["date", "sensor"]).size())
    if not counts:
        return [], 0
    series = pd.concat(counts).groupby(level=[0, 1]).sum().rename("count").reset_index()
    return series.to_dict(orient="records"), total


@router.get("/analysis/temporal")
def temporal(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
):
    """Daily detection counts per sensor."""
    country, year = _defaults(country, year)
    try:
        rows, total = _temporal(country, year)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not rows:
        return _unavailable({"status": "UNAVAILABLE", "reason": "NO_VALID_DATES",
                             "available_records": 0, "data": [],
                             "message": "No records with valid acquisition dates."})
    return _response({"status": "AVAILABLE", "reason": None, "available_records": total,
                      "country": country, "year": year, "data": rows})


@router.post("/analysis/associations/run")
def run_associations(body: dict = Body(default={})):
    """Recompute associations with the requested parameters."""
    country, year = _defaults(body.get("country"), body.get("year"))
    unc = body.get("uncertainty_km")
    unc = float(unc) if unc is not None else 1.0
    window = body.get("time_window_minutes")
    window = int(window) if window else 60
    try:
        rows, total, params, warning = _associations(country, year, window, unc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return _response({
        "analysis_id": f"RUN_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
        "status": "COMPLETED",
        "applied_parameters": {**(params or {}),
                               "distance_threshold_mode":
                                   body.get("distance_threshold_mode", "FOOTPRINT_BASED")},
        "spatial_candidate_pairs": total,
        "temporal_analysis_status": "AVAILABLE",
    }, [warning])


@router.get("/export")
def export(
    format: str = Query(default="csv"),
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
):
    """Download sampled processed detections as CSV or JSON."""
    country, year = _defaults(country, year)
    try:
        rows = pipeline.get_detections(country, year, sensor, 1000)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if format == "json":
        payload = json.dumps({"country": country, "year": year, "data": rows}, default=str)
        return Response(payload, media_type="application/json")
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return Response(buf.getvalue(), media_type="text/csv")
