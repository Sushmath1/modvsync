from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timezone

from app.services import pipeline
from app.routes.compat import router as compat_router

app = FastAPI(
    title="ModVSync API",
    description=(
        "MODIS–VIIRS active-fire data inspection, "
        "spatiotemporal candidate association and analysis."
    ),
    version="1.0.0",
)

# Registered first so its /analysis/sensitivity (optional params, superset of
# the original response) takes precedence; other compat routes are new paths.
app.include_router(compat_router)


def response(data, warnings=None):
    return {
        "status": "success",
        "data": data,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "warnings": warnings or [],
    }


@app.get("/api/v1/health")
def health():
    return response({"message": "ModVSync backend is running"})


@app.get("/api/v1/datasets")
def datasets():
    try:
        items = pipeline.inventory()
        return response({
            "dataset_count": len(items),
            "datasets": items,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/data-quality")
def quality(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
):
    try:
        return response(_cached_quality(country, year, sensor))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@lru_cache(maxsize=32)
def _cached_quality(country, year, sensor):
    """Existing counts plus the per-sensor fields the frontend page reads."""
    per_sensor = {}
    for s in ("MODIS", "VIIRS"):
        if sensor and sensor.upper() != s:
            per_sensor[s] = {"rows": 0, "invalid_coordinates": 0,
                             "invalid_timestamps": 0, "missing_frp": 0}
        else:
            per_sensor[s] = pipeline.data_quality(country, year, s)
    q = {k: per_sensor["MODIS"][k] + per_sensor["VIIRS"][k] for k in per_sensor["MODIS"]}
    warnings = []
    if q["invalid_timestamps"]:
        warnings.append(f"{q['invalid_timestamps']} records have invalid acquisition timestamps.")
    if q["invalid_coordinates"]:
        warnings.append(f"{q['invalid_coordinates']} records have invalid coordinates.")
    if q["missing_frp"]:
        warnings.append(f"{q['missing_frp']} records have missing FRP.")
    warnings.append("Footprint, confidence and duplicate checks are not computed yet.")
    q.update({
        "modis_records": per_sensor["MODIS"]["rows"],
        "viirs_records": per_sensor["VIIRS"]["rows"],
        "missing_dates": q["invalid_timestamps"],
        "invalid_frp_values": q["missing_frp"],
        "invalid_footprints": None,
        "unknown_confidence": None,
        "duplicate_records": None,
        "synthetic_dates_used": False,
        "temporal_analysis": "AVAILABLE" if q["rows"] > q["invalid_timestamps"] else "BLOCKED",
        "warnings": warnings,
    })
    return q


@lru_cache(maxsize=32)
def _cached_detection_sample(country, year, sensor):
    rows = pipeline.get_detections(country, year, sensor, 1000)
    for r in rows:
        # Same id format as /associations modis_id / viirs_id.
        r["detection_id"] = f"{r['source_file']}#{r['source_row']}"
    if country is None and year is None and sensor is None:
        # Include the detections that the default /associations pairs reference.
        from app.routes.compat import associated_detections
        seen = {r["detection_id"] for r in rows}
        rows = rows + [d for d in associated_detections() if d["detection_id"] not in seen]
    return rows

@app.get("/api/v1/detections")
def detections(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=500, ge=1, le=1000),
):
    try:
        sample_limit = 1000
        rows = _cached_detection_sample(country, year, sensor)

        start = (page - 1) * page_size
        page_rows = rows[start:start + page_size]

        return response({
            "data": page_rows,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_records": len(rows),
            },
            "sampled": True,
            "sample_size": len(rows),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@app.get("/api/v1/analysis/frp")
def frp(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
):
    try:
        return response(*_cached_frp(country, year, sensor))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@lru_cache(maxsize=32)
def _cached_frp(country, year, sensor):
    """Per-sensor summary for the frontend; original list kept under 'groups'."""
    groups = pipeline.frp_statistics(country, year, sensor)
    out = {"unit": "MW", "groups": groups}
    for s in ("MODIS", "VIIRS"):
        rows = [g for g in groups if g["sensor"] == s]
        if not rows:
            continue
        total = sum(g["count"] for g in rows)
        # Count-weighted median of group medians (approximate overall median).
        ordered = sorted(rows, key=lambda g: g["median"])
        acc, median = 0, ordered[-1]["median"]
        for g in ordered:
            acc += g["count"]
            if acc >= total / 2:
                median = g["median"]
                break
        out[s] = {
            "count": total,
            "mean": round(sum(g["mean"] * g["count"] for g in rows) / total, 2),
            "median": round(median, 2),
            "maximum": round(max(g["max"] for g in rows), 2),
        }
    warnings = ["Per-sensor median is a count-weighted median of per-country/year/day-night "
                "group medians (approximate)."]
    return out, warnings


@app.get("/api/v1/analysis/associations")
def associations(
    country: str = Query(..., description="Bolivia or Chad"),
    year: int = Query(..., ge=2018, le=2024),
    max_time_minutes: int = Query(default=60, ge=0, le=1440),
    uncertainty_km: float = Query(default=1.0, ge=0, le=20),
):
    if country.title() not in ("Bolivia", "Chad"):
        raise HTTPException(status_code=400, detail="Country must be Bolivia or Chad.")
    try:
        result = pipeline.associate(
            country.title(), year, max_time_minutes, uncertainty_km
        )
        return response(result, [result["warning"]])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/analysis/clusters")
def clusters(
    country: str = Query(...),
    year: int = Query(..., ge=2018, le=2024),
    sensor: str = Query(default="VIIRS"),
    eps_km: float = Query(default=2.0, gt=0, le=50),
    min_samples: int = Query(default=5, ge=2, le=100),
):
    if country.title() not in ("Bolivia", "Chad"):
        raise HTTPException(status_code=400, detail="Country must be Bolivia or Chad.")
    if sensor.upper() not in ("MODIS", "VIIRS"):
        raise HTTPException(status_code=400, detail="Sensor must be MODIS or VIIRS.")
    try:
        result = pipeline.spatial_clusters(
            country.title(), year, sensor.upper(), eps_km, min_samples
        )
        return response(result, [result["warning"]])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def root():
    return {
        "name": "ModVSync API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

@app.get("/api/v1/analysis/sensitivity")
def sensitivity(
    country: str = Query(...),
    year: int = Query(..., ge=2018, le=2024),
    sensor: str = Query(default="VIIRS"),
    min_samples: int = Query(default=5, ge=2, le=100),
):
    if country.title() not in ("Bolivia", "Chad"):
        raise HTTPException(
            status_code=400,
            detail="Country must be Bolivia or Chad."
        )

    if sensor.upper() not in ("MODIS", "VIIRS"):
        raise HTTPException(
            status_code=400,
            detail="Sensor must be MODIS or VIIRS."
        )

    try:
        result = pipeline.clustering_sensitivity(
            country.title(),
            year,
            sensor.upper(),
            min_samples,
        )
        return response(result, [result["warning"]])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/dashboard/summary")
@lru_cache(maxsize=1)
def dashboard_summary():
    try:
        datasets = pipeline.inventory()
        quality = pipeline.data_quality()

        total = sum(item.get("rows", 0) for item in datasets)
        modis = sum(
            item.get("rows", 0)
            for item in datasets
            if item.get("sensor") == "MODIS"
        )
        viirs = sum(
            item.get("rows", 0)
            for item in datasets
            if item.get("sensor") == "VIIRS"
        )

        valid_timestamps = (
            quality["rows"] - quality["invalid_timestamps"]
        )

        return response({
            "total_detections": total,
            "modis_detections": modis,
            "viirs_detections": viirs,
            "spatial_clusters": None,
            "spatial_candidate_pairs": None,
            "spatiotemporal_candidate_pairs": None,
            "dataset_status": "AVAILABLE" if datasets else "UNAVAILABLE",
            "temporal_analysis_status": (
                "AVAILABLE" if valid_timestamps > 0 else "UNAVAILABLE"
            ),
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@app.get("/api/v1/analysis/runs")
def analysis_runs():
    return response({
        "data": [],
        "message": "No analysis runs have been recorded yet."
    })
