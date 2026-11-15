"""Central API service layer.

Every page gets its data through this module. It talks to the FastAPI backend
(Live API mode) or to the built-in mock backend (Mock data mode), so pages never
need to know which one is in use. The frontend never calculates scientific
results itself: it only displays what the backend returns.
"""
import requests
import streamlit as st

from services import mock

DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
MODES = ["Live API", "Mock data"]


class ApiError(Exception):
    """Raised when the backend returns an error or cannot be reached."""


# --- Settings ------------------------------------------------------------

def _init_settings():
    st.session_state.setdefault("data_mode", "Live API")
    st.session_state.setdefault("api_base_url", DEFAULT_BASE_URL)


def is_mock():
    """True when pages should read from the mock backend."""
    _init_settings()
    return st.session_state["data_mode"] == "Mock data"


def base_url():
    _init_settings()
    return st.session_state["api_base_url"].rstrip("/")


def sidebar_settings():
    """Data-source controls shown at the bottom of every page's sidebar."""
    _init_settings()
    with st.sidebar:
        st.divider()
        st.caption("DATA SOURCE")
        st.session_state["data_mode"] = st.radio(
            "Data source", MODES, index=MODES.index(st.session_state["data_mode"]),
            label_visibility="collapsed")
        if not is_mock():
            st.session_state["api_base_url"] = st.text_input(
                "API base URL", st.session_state["api_base_url"])
            if st.button("Check connection", width="stretch"):
                try:
                    get("/health")
                    st.success("Backend reachable")
                except ApiError as exc:
                    st.error(str(exc))


# --- Response handling ---------------------------------------------------

def _unwrap(payload):
    """Accept both the common envelope ({status, data, warnings}) and raw JSON.

    Returns (data, warnings).
    """
    if isinstance(payload, dict) and payload.get("status") == "error":
        err = payload.get("error") or {}
        raise ApiError(f"{err.get('code', 'ERROR')}: {err.get('message', 'Unknown backend error')}")
    if isinstance(payload, dict) and payload.get("status") == "success" and "data" in payload:
        return payload["data"], list(payload.get("warnings") or [])
    warnings = list(payload.get("warnings") or []) if isinstance(payload, dict) else []
    return payload, warnings


def records(data):
    """Extract a list of records from a list response ({data: [...], pagination})."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    return []


@st.cache_data(ttl=60, show_spinner=False)
def _http_get(url, params):
    try:
        resp = requests.get(url, params=dict(params), timeout=30)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach backend at {url}: {exc}") from exc
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if resp.status_code >= 400:
        if isinstance(payload, dict) and payload.get("status") == "error":
            _unwrap(payload)
        detail = payload.get("detail") if isinstance(payload, dict) else resp.text[:200]
        raise ApiError(f"HTTP {resp.status_code} from {url}: {detail}")
    if payload is None:
        raise ApiError(f"Backend at {url} did not return JSON")
    return payload


def _clean(params):
    return {k: v for k, v in params.items() if v is not None and v != "" and v != "ALL"}


# --- Public helpers ------------------------------------------------------

def get(path, **params):
    """GET an endpoint (relative to /api/v1). Returns (data, warnings)."""
    params = _clean(params)
    if is_mock():
        return _unwrap(mock.handle_get(path, params))
    return _unwrap(_http_get(base_url() + path, tuple(sorted(params.items()))))


def get_all(path, page_size=500, max_pages=40, **params):
    """GET every page of a paginated list endpoint. Returns (records, warnings, extra).

    `extra` holds any non-list top-level fields (e.g. noise_count, algorithm).
    """
    all_rows, warnings, extra = [], [], {}
    page = 1
    while page <= max_pages:
        data, warns = get(path, page=page, page_size=page_size, **params)
        warnings.extend(w for w in warns if w not in warnings)
        rows = records(data)
        all_rows.extend(rows)
        if isinstance(data, dict):
            extra.update({k: v for k, v in data.items() if k not in ("data", "pagination")})
            pag = data.get("pagination")
            if not pag or page * pag.get("page_size", page_size) >= pag.get("total_records", 0):
                break
        else:
            break
        page += 1
    return all_rows, warnings, extra


def post(path, body):
    """POST JSON to an endpoint. Returns (data, warnings)."""
    if is_mock():
        return _unwrap(mock.handle_post(path, body))
    url = base_url() + path
    try:
        resp = requests.post(url, json=body, timeout=120)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach backend at {url}: {exc}") from exc
    try:
        payload = resp.json()
    except ValueError:
        raise ApiError(f"HTTP {resp.status_code} from {url}: response was not JSON")
    if resp.status_code >= 400 and not (isinstance(payload, dict) and payload.get("status") == "error"):
        raise ApiError(f"HTTP {resp.status_code} from {url}: {payload}")
    return _unwrap(payload)


def get_bytes(path, **params):
    """Download a file from an endpoint (used for exports). Returns bytes."""
    params = _clean(params)
    if is_mock():
        return mock.handle_export(params.get("format", "csv"))
    url = base_url() + path
    try:
        resp = requests.get(url, params=params, timeout=60)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach backend at {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ApiError(f"HTTP {resp.status_code} from {url}")
    return resp.content
