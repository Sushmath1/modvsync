
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


@app.get("/api/v1/detections")
def detections(
    country: str | None = None,
    year: int | None = Query(default=None, ge=2018, le=2024),
    sensor: str | None = None,
    limit: int = Query(default=100, ge=1, le=10000),
):
    try:
        rows = pipeline.get_detections(country, year, sensor, limit)
        return response({"count": len(rows), "detections": rows})
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
