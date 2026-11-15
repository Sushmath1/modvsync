from functools import lru_cache
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timezone

from app.services import pipeline

app = FastAPI(
    title="ModVSync API",
    description=(
        "MODIS–VIIRS active-fire data inspection, "
        "spatiotemporal candidate association and analysis."
    ),
    version="1.0.0",
)


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
        return response(pipeline.data_quality(country, year, sensor))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@lru_cache(maxsize=32)
def _cached_detection_sample(country, year, sensor):
    return pipeline.get_detections(country, year, sensor, 1000)

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
        return response(pipeline.frp_statistics(country, year, sensor))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
