# ModVSync

Harmonizing MODIS and VIIRS active fire (hot spot) detections into one consistent dataset — NASA Space Apps 2026.

## Setup (Windows)

Use **Python 3.12** (on Windows, Smart App Control blocks pandas/pyarrow on newer versions). If `python` isn't 3.12, use `py -3.12 -m venv venv` for the first step.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download data from [DRIVE LINK] into `data/raw/modis`, `data/raw/viirs` or `data/processed/`.

## Run

Build the processed dataset (`data/processed/detections.parquet`):

```
python scripts/build_processed.py
```

Launch the app:

```
streamlit run app/streamlit_app.py
```
