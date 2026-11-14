# Mod-V Sync: Team Build Plan

This file describes the project, the existing code, the work split, the data contract between the two workstreams, and the git rules. Read it fully before writing any code.

## 1. Project summary

**Mod-V Sync** is a NASA Space Apps Challenge 2026 project for the challenge **"Harmonization of MODIS and VIIRS Hot Spots"**.

Goal: merge MODIS and VIIRS active-fire detections (from NASA FIRMS) into one consistent daily fire calendar per grid cell, and label every cell-day as one of:

| state | meaning |
|---|---|
| `fire` | at least one sensor detected a fire in this cell on this day |
| `clear` | at least one satellite passed over this cell on this day and no fire was detected |
| `not_observed` | no satellite passed over this cell on this day |

The final product is a Streamlit app with a map (where) linked to a calendar (when), plus an explanation of why the two sensors disagree.

Pilot regions: **Chad** (Sahel savanna) and **Bolivia** (Amazon edge). Data year in the current files: **2018**.

## 2. Environment and conventions

- **Python 3.12 only.** On Windows use `py -3.12 -m venv venv`. `pyarrow` is pinned to `21.0.0`. Do not change pinned versions.
- Always run commands from the **repo root**.
- Scripts in `scripts/` add the repo root to `sys.path` and import like `from src import config`. Follow the same pattern.
- All paths, region boxes and thresholds live in `src/config.py`. Add new constants there rather than hard-coding them.
- Keep the code simple and readable: small functions, a docstring on each, plain pandas. No classes unless clearly needed.
- **Never commit data.** `data/` (raw and processed), `.env` and `venv/` are git-ignored. Do not change that.
- All times are **UTC**. Dates are UTC calendar dates.

## 3. Existing code (already on `main`, do not rewrite)

```
src/config.py        paths, REGION_BBOX (Chad, Bolivia), PLATFORM_NAMES, GRID_SIZE_DEG = 0.1, confidence settings
src/ingest.py        loads FIRMS CSVs from data/raw/modis and data/raw/viirs, adds sensor/region/platform,
                     builds datetime_utc, unifies confidence into confidence_norm (low/nominal/high)
src/clean.py         drops invalid coords, duplicates, low-confidence and non-vegetation (type != 0) detections
src/grid.py          adds cell_id ("lat_lon" of the 0.1° cell's lower-left corner, e.g. "10.8_16.8") and date
scripts/build_processed.py   runs ingest -> clean -> grid and writes data/processed/detections.parquet
```

### `data/processed/detections.parquet` (exists now: one row per detection)

| column | type | example |
|---|---|---|
| `sensor` | str | `MODIS` / `VIIRS` |
| `platform` | str | `Terra`, `Aqua`, `Suomi-NPP`, `NOAA-20`, `NOAA-21` |
| `latitude`, `longitude` | float | `10.83`, `16.87` |
| `datetime_utc` | datetime64[ns, UTC] | `2018-01-14 12:35` |
| `date` | date | `2018-01-14` |
| `confidence_raw` | str | `87` or `n` |
| `confidence_norm` | category | `low` / `nominal` / `high` |
| `frp` | float | `12.4` |
| `daynight` | str | `D` / `N` |
| `region` | str | `Chad` / `Bolivia` |
| `cell_id` | str | `10.8_16.8` |

Raw CSV filenames follow `<sensor>_<year>_<region>.csv`, e.g. `modis_2018_Chad.csv`, `viirs-jpss1_2018_Chad.csv`.

## 4. Work split

Two people. Each person **owns specific files**. Do not edit the other person's files. If a change to a shared file (`src/config.py`, `requirements.txt`, `README.md`) is needed, tell the other person first and keep the change small.

### Person A: Sushmathi (data and science)

**Branch `feature/harmonize`.** File: `src/harmonize.py`
- Aggregate `detections.parquet` to one row per (`region`, `cell_id`, `date`).
- Columns: `modis_count`, `viirs_count`, `modis_platforms` (comma-joined), `viirs_platforms`, `max_frp`, `seen_by` (`both` / `modis` / `viirs`).
- Add cell-centre `lat`, `lon` (lower-left corner + `GRID_SIZE_DEG / 2`).
- Done when: function `harmonize(df) -> DataFrame` returns this table and a quick printout shows counts per `seen_by` per region.

**Branch `feature/coverage`.** Files: `src/coverage.py`, `scripts/build_calendar.py`, `data/tle/` (git-ignored if large)
- For every cell in each region's bounding box and every date in the data range, decide `modis_passed` and `viirs_passed`.
- Method: use Skyfield to propagate each satellite's orbit (Terra, Aqua for MODIS; Suomi NPP, NOAA-20 for VIIRS) across the day, and mark a cell as observed if the satellite's ground track comes within half the swath width of the cell centre (MODIS swath ~2330 km, VIIRS swath ~3060 km; store these in `config.py`).
- **Important:** the data is from 2018, so current TLEs from CelesTrak will not propagate accurately back to 2018. Use **historical TLEs** for the data dates (Space-Track.org general perturbations history, free account), and pick the TLE closest to each date. If historical TLEs are not available in time, fall back to a documented approximation and state the limitation clearly in the README.
- `build_calendar.py`: build the full cell × date grid, left-join the harmonized detections, add coverage, compute `state` and `reason`, write `data/processed/calendar.parquet`.
- Done when: `calendar.parquet` matches the contract in section 5 exactly.

### Person B: teammate (app and experience)

**Branch `feature/app`.** Files: `app/streamlit_app.py`, `app/components/*.py`, `scripts/make_fake_calendar.py`
- First write `scripts/make_fake_calendar.py`, which generates a fake `data/processed/calendar.parquet` that follows the contract in section 5 exactly (random but realistic values, both regions, one year). Build the whole app against this file so you never wait on Person A.
- App layout:
  - Sidebar: region selector, date range selector, sensor view toggle (`Harmonized` / `MODIS only` / `VIIRS only`).
  - Map (Plotly): cells for the selected day, coloured by `state` (fire = red/orange, clear = light, not_observed = grey).
  - Calendar heatmap (Plotly): days on the x-axis (or a GitHub-style day grid), coloured by the number of fire cells per day for the region.
  - Clicking or selecting a day on the calendar updates the map. Selecting a cell shows that cell's history.
- Load data with `@st.cache_data`. The app must also run on `detections.parquet` alone, showing a raw-detections map, if `calendar.parquet` is missing.
- Done when: `streamlit run app/streamlit_app.py` works end to end on the fake calendar.

**Branch `feature/explain`.** File: `app/components/explain.py`
- For a selected cell-day, show a plain-English explanation based on `seen_by`, `modis_passed`, `viirs_passed` and `reason`. Example: "Detected by VIIRS only. Likely a small fire below MODIS's 1 km resolution."
- Region-level stats: fire days by `seen_by`, and the share of days that were `not_observed`.

**Branch `feature/replay` (stretch, only if time allows).** File: `app/components/replay.py`
- Animate satellite ground tracks over the region for one day, with detections appearing as each satellite passes.

## 5. Data contract: `data/processed/calendar.parquet`

One row per (`region`, `cell_id`, `date`) for **every** cell in the region's bounding box and **every** date in the range, including days with no fire.

| column | type | allowed values / example |
|---|---|---|
| `region` | str | `Chad`, `Bolivia` |
| `cell_id` | str | `10.8_16.8` (same format as `src/grid.py`) |
| `lat` | float | cell centre latitude, e.g. `10.85` |
| `lon` | float | cell centre longitude, e.g. `16.85` |
| `date` | date | `2018-01-14` (UTC) |
| `modis_count` | int | `0` or more |
| `viirs_count` | int | `0` or more |
| `seen_by` | str | `both`, `modis`, `viirs`, `none` |
| `max_frp` | float | NaN if no detection |
| `modis_passed` | bool | |
| `viirs_passed` | bool | |
| `state` | str | `fire`, `clear`, `not_observed` |
| `reason` | str | short plain-English note, may be empty |

Rules:
- `state = "fire"` if `modis_count + viirs_count > 0`.
- else `state = "clear"` if `modis_passed or viirs_passed`.
- else `state = "not_observed"`.
- Column names and allowed values must match exactly. Do not rename anything without both people agreeing.

## 6. Git workflow

- `main` is always runnable. **Never commit directly to `main`.**
- Create a branch from an up-to-date `main` for each task:
  ```bash
  git checkout main
  git pull
  git checkout -b feature/<name>
  ```
- Commit small and often, with clear messages (e.g. `Add per-cell MODIS/VIIRS matching`). Commit at least every hour or two.
- Push the branch: `git push -u origin feature/<name>`
- When the task is done, open a **Pull Request into `main`** on GitHub. The other person reviews and merges.
- After any merge, both people run `git checkout main && git pull` before starting the next branch.
- To bring the latest `main` into your branch: `git merge main`.
- Merge to `main` at least twice a day (Saturday night, Sunday afternoon) so integration problems show up early.

## 7. Timeline

| When | Person A | Person B |
|---|---|---|
| Sat afternoon | `feature/harmonize` | `make_fake_calendar.py` + `feature/app` skeleton |
| Sat night | Merge harmonize, start `feature/coverage` | Merge app skeleton |
| Sun morning | Finish coverage, produce real `calendar.parquet`, merge | Switch app to real data, start `feature/explain` |
| Sun afternoon | Validate on one real fire event, update README results | Polish UI, `feature/replay` if time allows |
| Sun evening | Demo video, slides, submit (both) | |

## 8. Rules for AI coding assistants

- Read this file and the existing `src/` code before making changes.
- Only edit files owned by the person you are working for (section 4).
- Never invent results or numbers for the README or slides. Only report what the code actually produces.
- Do not add new dependencies without updating `requirements.txt` and telling the other person.
- Keep every change runnable from the repo root on Python 3.12 on Windows.
