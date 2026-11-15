
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree


BACKEND_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BACKEND_DIR / "data" / "raw" / "incoming"
EARTH_RADIUS_KM = 6371.0

REQUIRED_COLUMNS = {
    "latitude", "longitude", "acq_date", "acq_time",
    "satellite", "instrument", "frp", "daynight",
    "scan", "track",
}


def discover_files():
    """Find source CSVs and infer sensor, country and year from filenames."""
    datasets = []

    for path in sorted(RAW_DIR.rglob("*.csv")):
        name = path.stem.lower()
        sensor = "MODIS" if name.startswith("modis_") else (
            "VIIRS" if name.startswith("viirs") else None
        )
        year_match = re.search(r"(?:^|_)(20\d{2})(?:_|$)", name)
        country_match = re.search(r"_(Bolivia|Chad)$", path.stem, re.I)

        if not sensor or not year_match or not country_match:
            continue

        datasets.append({
            "dataset_id": path.stem,
            "sensor": sensor,
            "country": country_match.group(1).title(),
            "year": int(year_match.group(1)),
            "filename": path.name,
            "path": path,
            "size_bytes": path.stat().st_size,
        })

    return datasets


def inventory():
    """Inspect every CSV without loading all observations into memory."""
    results = []

    for item in discover_files():
        path = item["path"]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        missing_columns = sorted(REQUIRED_COLUMNS - set(header))
        row_count = 0

        for chunk in pd.read_csv(path, usecols=lambda c: c in header,
                                 chunksize=200_000):
            row_count += len(chunk)

        results.append({
            "dataset_id": item["dataset_id"],
            "filename": item["filename"],
            "sensor": item["sensor"],
            "country": item["country"],
            "year": item["year"],
            "rows": row_count,
            "size_mb": round(item["size_bytes"] / (1024 * 1024), 2),
            "columns": header,
            "missing_required_columns": missing_columns,
            "valid_schema": not missing_columns,
        })

    return results


def _normalise(chunk: pd.DataFrame, item: dict, row_offset: int):
    """Create common fields while retaining original sensor columns."""
    df = chunk.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in ("latitude", "longitude", "frp", "scan", "track"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # HHMM must be a whole number with valid minute and hour components.
    raw_time = pd.to_numeric(df["acq_time"], errors="coerce")
    valid_integer = raw_time.notna() & (raw_time % 1 == 0)
    hhmm = raw_time.fillna(-1).astype("int64")
    hours = hhmm // 100
    minutes = hhmm % 100
    valid_time = valid_integer & hours.between(0, 23) & minutes.between(0, 59)

    date = pd.to_datetime(df["acq_date"], errors="coerce")
    timestamp = (
        date + pd.to_timedelta(hours.where(valid_time, 0), unit="h")
        + pd.to_timedelta(minutes.where(valid_time, 0), unit="m")
    )
    timestamp = timestamp.dt.tz_localize("UTC")

    df["acq_datetime"] = timestamp.where(valid_time)
    df["sensor"] = item["sensor"]
    df["country"] = item["country"]
    df["year"] = item["year"]
    df["source_file"] = item["filename"]
    df["source_row"] = np.arange(row_offset, row_offset + len(df)) + 1

    valid_coordinates = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )
    df["coordinate_valid"] = valid_coordinates
    df["timestamp_valid"] = df["acq_datetime"].notna()
    return df


def iter_observations(country=None, year=None, sensor=None,
                      chunksize=100_000):
    """Yield validated, normalised chunks from the original CSV files."""
    for item in discover_files():
        if country and item["country"].lower() != country.lower():
            continue
        if year and item["year"] != int(year):
            continue
        if sensor and item["sensor"].lower() != sensor.lower():
            continue

        offset = 0
        for chunk in pd.read_csv(item["path"], chunksize=chunksize):
            normalised = _normalise(chunk, item, offset)
            offset += len(chunk)
            yield normalised


def get_detections(country=None, year=None, sensor=None, limit=1000):
    """Return a deterministic uniform sample across all valid observations."""
    limit = max(1, min(int(limit), 10_000))
    rng = np.random.default_rng(42)
    sample = pd.DataFrame()

    for chunk in iter_observations(country, year, sensor):
        valid = chunk[
            chunk["coordinate_valid"] & chunk["timestamp_valid"]
        ]

        if valid.empty:
            continue

        candidate = valid.copy()
        candidate["_sample_priority"] = rng.random(len(candidate))

        if not sample.empty:
            candidate = pd.concat(
                [sample, candidate],
                ignore_index=True
            )

        sample = candidate.nsmallest(
            limit, "_sample_priority"
        ).reset_index(drop=True)

    if sample.empty:
        return []

    return sample.drop(columns=["_sample_priority"]).replace(
        {np.nan: None}
    ).to_dict(orient="records")


def data_quality(country=None, year=None, sensor=None):
    counts = {
        "rows": 0, "invalid_coordinates": 0,
        "invalid_timestamps": 0, "missing_frp": 0,
    }

    for chunk in iter_observations(country, year, sensor):
        counts["rows"] += len(chunk)
        counts["invalid_coordinates"] += int((~chunk["coordinate_valid"]).sum())
        counts["invalid_timestamps"] += int((~chunk["timestamp_valid"]).sum())
        counts["missing_frp"] += int(chunk["frp"].isna().sum())

    return counts


def frp_statistics(country=None, year=None, sensor=None):
    """Calculate FRP statistics separately for each sensor."""
    groups = {}

    for chunk in iter_observations(country, year, sensor):
        valid = chunk.dropna(subset=["frp"])
        for key, group in valid.groupby(["sensor", "country", "year", "daynight"]):
            stats = groups.setdefault(key, {
                "count": 0, "sum": 0.0, "values": [],
                "minimum": float("inf"), "maximum": float("-inf"),
            })
            values = group["frp"].astype(float).to_numpy()
            stats["count"] += len(values)
            stats["sum"] += float(values.sum())
            stats["minimum"] = min(stats["minimum"], float(values.min()))
            stats["maximum"] = max(stats["maximum"], float(values.max()))
            # Keep a bounded sample for median/percentiles on large files.
            # Retain all FRP values for exact median and percentile calculations.
            stats["values"].extend(values.tolist())

    output = []
    for (s, c, y, daynight), stat in groups.items():
        vals = np.asarray(stat["values"], dtype=float)
        output.append({
            "sensor": s, "country": c, "year": y, "daynight": daynight,
            "count": stat["count"],
            "mean": stat["sum"] / stat["count"],
            "median": float(np.median(vals)),
            "min": stat["minimum"], "max": stat["maximum"],
            "p90": float(np.percentile(vals, 90)),
            "percentile_observation_count": len(vals),
        })
    return output


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two latitude/longitude points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def associate(country, year, max_time_minutes=60, uncertainty_km=1.0,
              max_records_per_sensor=3000):
    """
    Exploratory cross-sensor candidate links.
    Uses approximate footprint radii, not exact pixel polygon overlap.
    """
    frames = []
    for sensor in ("MODIS", "VIIRS"):
        rows = get_detections(country, year, sensor, max_records_per_sensor)
        frames.append(pd.DataFrame(rows))

    if any(frame.empty for frame in frames):
        return {"candidate_count": 0, "candidates": [],
                "warning": "Both sensors need valid observations for this country/year."}

    modis, viirs = frames
    modis = modis.dropna(subset=["latitude", "longitude", "acq_datetime"])
    viirs = viirs.dropna(subset=["latitude", "longitude", "acq_datetime"])

    if modis.empty or viirs.empty:
        return {"candidate_count": 0, "candidates": []}

    # BallTree narrows possible neighbours; exact pair-specific thresholds follow.
    mcoords = np.radians(modis[["latitude", "longitude"]].to_numpy())
    vcoords = np.radians(viirs[["latitude", "longitude"]].to_numpy())
    m_radius = 0.5 * np.sqrt(modis["scan"].fillna(0)**2 +
                             modis["track"].fillna(0)**2)
    v_radius = 0.5 * np.sqrt(viirs["scan"].fillna(0)**2 +
                             viirs["track"].fillna(0)**2)
    max_radius_km = float(m_radius.max() + v_radius.max() + uncertainty_km)
    tree = BallTree(vcoords, metric="haversine")
    neighbours = tree.query_radius(mcoords, r=max_radius_km / EARTH_RADIUS_KM)

    mtime = pd.to_datetime(modis["acq_datetime"], utc=True).reset_index(drop=True)
    vtime = pd.to_datetime(viirs["acq_datetime"], utc=True).reset_index(drop=True)
    candidates = []

    for mi, indices in enumerate(neighbours):
        for vi in indices:
            distance = float(haversine_km(
                modis.iloc[mi]["latitude"], modis.iloc[mi]["longitude"],
                viirs.iloc[vi]["latitude"], viirs.iloc[vi]["longitude"]
            ))
            threshold = float(m_radius.iloc[mi] + v_radius.iloc[vi] + uncertainty_km)
            delta_minutes = abs((mtime.iloc[mi] - vtime.iloc[vi]).total_seconds()) / 60

            if distance <= threshold and delta_minutes <= max_time_minutes:
                similarity = math.exp(-(distance ** 2) / (2 * threshold ** 2))
                candidates.append({
                    "modis_file": modis.iloc[mi]["source_file"],
                    "modis_row": int(modis.iloc[mi]["source_row"]),
                    "viirs_file": viirs.iloc[vi]["source_file"],
                    "viirs_row": int(viirs.iloc[vi]["source_row"]),
                    "distance_km": distance,
                    "footprint_threshold_km": threshold,
                    "time_difference_minutes": delta_minutes,
                    "spatial_similarity": similarity,
                })

    return {
        "candidate_count": len(candidates),
        "candidates": candidates[:500],
        "parameters": {
            "country": country, "year": year,
            "max_time_minutes": max_time_minutes,
            "uncertainty_km": uncertainty_km,
            "max_records_per_sensor": max_records_per_sensor,
            "sampling_method": "deterministic_uniform",
            "sampled_records_per_sensor": {
                "MODIS": len(modis),
                "VIIRS": len(viirs),
            },
            "results_are_sampled": True,
        },
        "warning": "Exploratory candidates only; proximity is not proof of same-fire identity.",
    }



def spatial_clusters(country, year, sensor="VIIRS", eps_km=2.0,
                     min_samples=5, max_records=10_000):
    """Cluster detections spatially, separately for each acquisition date."""
    rows = get_detections(country, year, sensor, max_records)
    df = pd.DataFrame(rows)

    if df.empty:
        return {
            "observation_count": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "clusters": [],
        }

    df["acquisition_date"] = (
        pd.to_datetime(df["acq_datetime"], utc=True).dt.strftime("%Y-%m-%d")
    )

    clusters = []
    total_noise = 0
    next_cluster_id = 0

    for date, daily in df.groupby("acquisition_date"):
        if len(daily) < min_samples:
            total_noise += len(daily)
            continue

        coords = np.radians(
            daily[["latitude", "longitude"]].to_numpy()
        )

        labels = DBSCAN(
            eps=eps_km / EARTH_RADIUS_KM,
            min_samples=min_samples,
            metric="haversine",
            algorithm="ball_tree",
        ).fit_predict(coords)

        daily = daily.copy()
        daily["cluster_label"] = labels

        total_noise += int((labels == -1).sum())

        for label, group in daily[daily["cluster_label"] >= 0].groupby(
            "cluster_label"
        ):
            frp_values = group["frp"].dropna()

            clusters.append({
                "cluster_id": next_cluster_id,
                "acquisition_date": date,
                "observation_count": len(group),
                "centroid_latitude": float(group["latitude"].mean()),
                "centroid_longitude": float(group["longitude"].mean()),
                "mean_frp": (
                    float(frp_values.mean())
                    if not frp_values.empty else None
                ),
            })
            next_cluster_id += 1

    return {
        "observation_count": len(df),
        "cluster_count": len(clusters),
        "noise_count": total_noise,
        "clusters": clusters,
        "parameters": {
            "eps_km": eps_km,
            "min_samples": min_samples,
            "sampling_method": "deterministic_uniform",
            "sampled_observations": len(df),
            "results_are_sampled": True,
            "temporal_grouping": "acquisition_date",
        },
        "warning": (
            "Clusters are spatial detection groups calculated separately "
            "for each acquisition date. They are not confirmed fire events."
        ),
    }


def clustering_sensitivity(
    country, year, sensor="VIIRS", min_samples=5
):
    """Compare DBSCAN results across multiple spatial distance thresholds."""
    thresholds = [0.5, 1.0, 2.0, 5.0]
    results = []

    for eps in thresholds:
        analysis = spatial_clusters(
            country=country,
            year=year,
            sensor=sensor,
            eps_km=eps,
            min_samples=min_samples,
            max_records=10_000,
        )

        results.append({
            "eps_km": eps,
            "observation_count": analysis["observation_count"],
            "cluster_count": analysis["cluster_count"],
            "noise_count": analysis["noise_count"],
        })

    return {
        "country": country,
        "year": year,
        "sensor": sensor,
        "min_samples": min_samples,
        "sampling_method": "deterministic_uniform",
        "results_are_sampled": True,
        "comparisons": results,
        "warning": (
            "Exploratory sensitivity analysis. These thresholds are "
            "not validated universal fire-event parameters."
        ),
    }
