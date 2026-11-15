<div align="center">

# MOD-V SYNC

# Two satellites. One consistent fire record.

Harmonization of NASA MODIS and VIIRS Hotspots.

[![NASA Space Apps 2026](https://img.shields.io/badge/NASA%20Space%20Apps-2026-0B3D91?style=flat-square)](https://www.spaceappschallenge.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Data: NASA FIRMS](https://img.shields.io/badge/Data-NASA%20FIRMS-E4572E?style=flat-square)](https://firms.modaps.eosdis.nasa.gov/)

**Challenge:** Harmonization of MODIS and VIIRS Hot Spots  
**Team:** TEAM LILO, Chennai Institute of Technology

</div>

## The Problem

NASA monitors active fires from space with two instruments that often tell different stories about the same fires.

In our 2018–2024 data for Chad and Bolivia, VIIRS recorded about 3.9 times more fire detections than MODIS over the same areas and years. Most of the difference comes from small fires that only the sharper VIIRS sensor can see, and from the satellites passing overhead at different times of day.

Combining the two records naively double-counts fires. Choosing just one loses fires the other sensor caught. Either way, long-term fire analysis becomes unreliable.

## Our Solution

MOD-V SYNC treats the disagreement between the sensors as information instead of noise. It:

- **Cleans and aligns** both records while keeping each sensor's original values.
- **Matches detections across sensors**, pairing each MODIS detection with its nearest same-day VIIRS detection within a distance based on both sensors' pixel footprints.
- **Labels every pair honestly.** A pair is never called a "confirmed same fire."
- **Finds hotspot areas** with DBSCAN clustering.
- **Explains everything** in an interactive dashboard with maps, statistics and data-quality reporting.

### Association labels

| Label | Meaning |
|---|---|
| `SPATIOTEMPORAL_CANDIDATE` | Within the distance threshold and close in time |
| `SPATIAL_CANDIDATE` | Within the distance threshold, but not close in time |
| `AMBIGUOUS` | One VIIRS detection is the nearest match for several MODIS detections |
| `UNRESOLVED` | Nearest detection lies just outside the threshold |

## How It Works

1. **Collect.** NASA FIRMS active-fire archives for MODIS and VIIRS.
2. **Clean.** Validate coordinates, remove duplicates, and preserve sensor-specific confidence and fire radiative power (FRP).
3. **Match.** Pair each MODIS detection with its nearest same-day VIIRS detection using Haversine distance and a threshold based on both pixel footprints.
4. **Cluster.** DBSCAN groups detections into hotspot areas.
5. **Analyse.** FRP statistics per sensor, daily activity over time, and a sensitivity analysis of the matching threshold.
6. **Serve and display.** The FastAPI backend computes the results, and the Streamlit dashboard displays them.

## Dashboard

**Overview** - Detections per sensor, candidate pairs, hotspots, data-quality status 
**Detection Explorer** - Detections on a map with filters and full details 
**Sensor Harmonization** - MODIS–VIIRS pairs on the map, distances, labels, side-by-side comparison
**Statistical Analysis** - Sensor-specific FRP statistics and threshold sensitivity 
**Cluster Analysis** - Hotspot clusters and density heatmap
**Temporal Explorer** - Daily detection activity per sensor 
**Data Quality** - Validation results, limitations and data export 
**Methodology** - Equations and assumptions behind each method 

## Scientific Integrity

**Candidates, not conclusions.** A close pair is a possible match, never proof of the same fire.
**No double counting.** FRP values from a matched pair are never added together.
**Sensor-specific values.** MODIS and VIIRS confidence values are kept in their original form.
**Tested thresholds.** Sensitivity analysis shows how results depend on the chosen distance.
**No unsupported accuracy claims.** No accuracy metrics are reported without independent reference data.

## Data
Source - [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) 
Regions - Chad (Sahel savanna), Bolivia (Amazon edge) 
Years-2018–2024 
Cleaned detections - ~4.65 million
Raw data files are not stored in this repository because of their size.

## Tech Stack
Data processing - Python 3.12, pandas, NumPy, PyArrow 
Analysis - scikit-learn (BallTree, DBSCAN) 
Backend - FastAPI, Uvicorn
Dashboard - Streamlit, Plotly 
Workflow - Git, GitHub 

## Future Work
Machine-learning harmonization of weekly counts, satellite overpass and cloud data, and more regions.

## Team
**TEAM LILO**, Chennai Institute of Technology: Sushmathi V, Padmaja Vellayappan, R. Harshini, Suryaselvan K
