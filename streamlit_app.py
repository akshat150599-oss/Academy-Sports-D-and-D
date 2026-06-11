"""
Demurrage & Detention Analyzer — Academy Sports (Simplified)
============================================================
Standalone Streamlit app for ocean shipment demurrage and detention cost analysis,
simplified for a flat single-rate contract.

Contract model (Academy Sports):
  - ONE demurrage rate (USD/day). Used for BOTH POL demurrage and POD demurrage.
  - ONE detention rate (USD/day). Used for POD detention.
  - Optional demurrage free days and detention free days.
  - Optional combined free days for POD demurrage + POD detention.
  - No tiers. No POL/POD split inside the contract — the single demurrage rate covers both legs.

Calculation Logic:
  POL Demurrage = Container Loaded on Vessel - Container Gate In at POL  (CLL - CGI)  -> demurrage rate
  POD Demurrage = Gate Out Full from POD     - Discharge at POD          (CGO - CDD)  -> demurrage rate
  POD Detention = Container Empty Return     - Gate Out Full from POD    (CER - CGO)  -> detention rate

Free days:
  - POL demurrage and POD demurrage each deduct the demurrage free days.
  - POD detention deducts the detention free days.
  - If combined free days is set: POD demurrage consumes the pool first, POD detention
    receives the remainder. POL demurrage is never part of the combined pool.

Exclusion rule:
  CANCELLED -> shipment excluded from D&D entirely.

No CER handling:
  ACTIVE    -> detention accumulates to today's date (analysis run time)
  COMPLETED -> detention end = SHIPMENT_MODIFIED_DATE

Contract matching:
  Each contract row may name a carrierScac / ffwScac / POL / POD (any/all optional).
  A row matches a shipment if every identifier it specifies matches (blank = wildcard).
  The most specific matching row wins. A row with no identifiers is the GLOBAL default.

Run:
  streamlit run demurrage_detention_analyzer_academy.py
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from io import BytesIO

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="D&D Analyzer — Academy Sports",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# COLORS
# -----------------------------------------------------------------------------
POL_DEM_COLOR = "#00a6ff"
DEM_COLOR = "#f5a623"
DET_COLOR = "#7b61ff"
TOTAL_COLOR = "#00d4aa"
ALERT_COLOR = "#ff5c5c"

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1250px; }
    div[data-testid="stTabs"] { margin-top: 0.5rem !important; }
    div[data-testid="stTabs"] div[role="tablist"] {
        min-height: 64px !important; height: 64px !important;
        padding-top: 8px !important; padding-bottom: 14px !important;
        margin-bottom: 18px !important; border-bottom: 1px solid #2a2d3a !important;
        overflow: visible !important; gap: 18px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 48px !important; height: 48px !important;
        padding: 8px 8px 12px 8px !important; margin: 0 !important;
        overflow: visible !important; border-bottom: none !important; background: transparent !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: #cbd5e1 !important; font-size: 15px !important; font-weight: 800 !important;
        line-height: 22px !important; margin: 0 !important; padding: 0 !important;
        white-space: nowrap !important; overflow: visible !important; text-overflow: unset !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover p { color: #ffffff !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] p { color: #ff4b4b !important; font-weight: 900 !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom: 4px solid #ff4b4b !important; }
    div[data-testid="stTabs"] div { overflow: visible !important; }
    div[data-testid="stMetric"] {
        background: #111827; border: 1px solid #2a2d3a; border-radius: 10px; padding: 18px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #ffffff !important; font-size: 13px !important; text-transform: uppercase;
        letter-spacing: 0.8px; font-weight: 900 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 32px !important; font-weight: 900 !important;
    }
    div[data-testid="stMetricDelta"] { color: #22c55e !important; font-weight: 900 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# FLEXIBLE CONTRACT CSV PARSER (single flat rates)
# -----------------------------------------------------------------------------
# Canonical field -> list of accepted header spellings (matched case/punctuation-insensitive)
COLUMN_ALIASES = {
    "carrierScac": ["carrierscac", "carrier", "scac", "carriercode", "carriersymbol"],
    "ffwScac": ["ffwscac", "ffw", "forwarder", "forwarderscac", "freightforwarder",
                "freightforwarderscac", "freightforwardercode"],
    "pol": ["portofloadinglocode", "pol", "pollocode", "portofloading", "loadport"],
    "pod": ["portofdischargelocode", "pod", "podlocode", "portofdischarge", "dischargeport",
            "terminalidentifier", "terminal"],
    "demRate": ["demurragerate", "demrate", "demurragedailyrate", "demurrage",
                "demurrageratusdperday", "demurrageusdperday", "demurrageperday",
                "demurrageratusd", "demurrageamount"],
    "demFree": ["demurragefreedays", "freedemurragedays", "demfreedays", "demurragefree",
                "freedemurrage"],
    "detRate": ["detentionrate", "detrate", "detentiondailyrate", "detention",
                "detentionusdperday", "detentionperday", "detentionamount"],
    "detFree": ["detentionfreedays", "freedetentiondays", "detfreedays", "detentionfree",
                "freedetention"],
    "combinedFree": ["combinedfreedays", "combinedfree", "freedayscombined"],
    "currency": ["currency", "curr", "ccy"],
}


def _norm_key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower()) if text is not None else ""


def _to_num(val):
    if val is None:
        return None
    try:
        if isinstance(val, str) and val.strip() == "":
            return None
        num = float(val)
        if np.isnan(num):
            return None
        return num
    except (TypeError, ValueError):
        return None


def _clean_text(val):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def parse_contracts_csv(contract_file):
    """
    Read a simple flat-rate contract CSV.

    Accepts flexible header spellings. Recognized fields:
      carrierScac, ffwScac, pol, pod, demRate, demFree, detRate, detFree, combinedFree, currency

    Returns a list of normalized contract rows (dicts) and the raw dataframe.
    """
    cdf = pd.read_csv(contract_file)

    # Map each actual column to a canonical field name.
    resolved = {}
    for actual_col in cdf.columns:
        norm = _norm_key(actual_col)
        for canonical, spellings in COLUMN_ALIASES.items():
            if norm in spellings and canonical not in resolved:
                resolved[canonical] = actual_col
                break

    records = []
    for _, raw in cdf.iterrows():
        rec = {
            "carrierScac": _clean_text(raw[resolved["carrierScac"]]) if "carrierScac" in resolved else "",
            "ffwScac": _clean_text(raw[resolved["ffwScac"]]) if "ffwScac" in resolved else "",
            "pol": _clean_text(raw[resolved["pol"]]) if "pol" in resolved else "",
            "pod": _clean_text(raw[resolved["pod"]]) if "pod" in resolved else "",
            "demRate": _to_num(raw[resolved["demRate"]]) if "demRate" in resolved else None,
            "demFree": _to_num(raw[resolved["demFree"]]) if "demFree" in resolved else None,
            "detRate": _to_num(raw[resolved["detRate"]]) if "detRate" in resolved else None,
            "detFree": _to_num(raw[resolved["detFree"]]) if "detFree" in resolved else None,
            "combinedFree": _to_num(raw[resolved["combinedFree"]]) if "combinedFree" in resolved else None,
            "currency": _clean_text(raw[resolved["currency"]]) if "currency" in resolved else "",
        }
        records.append(rec)

    return records, cdf, resolved


def _to_profile(rec, is_estimate=False, label=None):
    """Convert a normalized contract row into a pricing profile."""
    identifier = label
    if identifier is None:
        bits = [b for b in [rec.get("carrierScac"), rec.get("ffwScac"),
                            rec.get("pol"), rec.get("pod")] if b]
        identifier = " | ".join(bits) if bits else "Global default"
    return {
        "dem_rate": rec.get("demRate"),
        "det_rate": rec.get("detRate"),
        "dem_free": rec.get("demFree") if rec.get("demFree") is not None else 0.0,
        "det_free": rec.get("detFree") if rec.get("detFree") is not None else 0.0,
        "combined_free": rec.get("combinedFree"),
        "currency": rec.get("currency") or "USD",
        "identifier": identifier,
        "is_estimate": is_estimate,
        "source": rec,
    }


def build_contract_candidates(contracts_list):
    """
    Build a scored list of contract candidates.
    Each candidate carries its identifiers and a specificity score.
    """
    candidates = []
    for rec in contracts_list or []:
        carrier = _clean_text(rec.get("carrierScac")).upper()
        ffw = _clean_text(rec.get("ffwScac")).upper()
        pol = _clean_text(rec.get("pol")).upper()
        pod = _clean_text(rec.get("pod")).upper()
        specificity = sum(1 for x in [carrier, ffw, pol, pod] if x)
        candidates.append({
            "carrier": carrier, "ffw": ffw, "pol": pol, "pod": pod,
            "specificity": specificity,
            "profile": _to_profile(rec),
        })
    # Most specific first so the best match wins.
    candidates.sort(key=lambda c: c["specificity"], reverse=True)
    return candidates


def match_contract(candidates, ship_carrier, ship_ffw, ship_pol, ship_pod):
    """
    Return (profile, matched_party_type) for the most specific matching contract row,
    or (None, "Missing") if nothing matches.
    """
    sc = _clean_text(ship_carrier).upper()
    sf = _clean_text(ship_ffw).upper()
    sp = _clean_text(ship_pol).upper()
    sd = _clean_text(ship_pod).upper()

    for cand in candidates:
        # Every identifier the row specifies must match the shipment (blank = wildcard).
        if cand["carrier"] and cand["carrier"] != sc:
            continue
        if cand["ffw"] and cand["ffw"] != sf:
            continue
        if cand["pol"] and cand["pol"] != sp:
            continue
        if cand["pod"] and cand["pod"] != sd:
            continue

        if cand["carrier"]:
            party = "Carrier"
        elif cand["ffw"]:
            party = "FFW"
        else:
            party = "Carrier" if sc else ("FFW" if sf else "Global")
        return cand["profile"], party

    return None, "Missing"


# -----------------------------------------------------------------------------
# CALCULATION HELPERS
# -----------------------------------------------------------------------------
def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        if np.isnan(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def _days_between(end_ts, start_ts):
    if pd.isna(end_ts) or pd.isna(start_ts):
        return None
    return max(0, (end_ts - start_ts).total_seconds() / 86400)


def _flat_cost(chargeable_days, rate):
    if chargeable_days is None or chargeable_days <= 0:
        return 0.0
    return round(chargeable_days * _safe(rate), 2)


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _shipment_match_identity(row):
    carrier = _clean_text(row.get("CARRIER_SCAC", ""))
    ffw = _clean_text(row.get("FFW_SCAC", ""))
    if carrier:
        return carrier, "Carrier"
    if ffw:
        return ffw, "FFW"
    return "", "Missing"


def normalize_required_columns(df):
    ffw_aliases = [
        "FFW_SCAC", "FFW", "FFW_SCAC_CODE", "FREIGHT_FORWARDER_SCAC",
        "FREIGHT_FORWARDER", "FORWARDER_SCAC", "FORWARDER", "FREIGHT_FORWARDER_CODE",
    ]
    ffw_col = _first_existing_column(df, ffw_aliases)
    if ffw_col is not None and ffw_col != "FFW_SCAC":
        df["FFW_SCAC"] = df[ffw_col]

    for col in [
        "SHIPMENT_ID", "CONTAINER_NUMBER", "CARRIER_SCAC", "CARRIER_NAME", "FFW_SCAC",
        "POL_LOCODE", "POL", "POD_LOCODE", "POD", "SUBSCRIPTION_STATUS",
        "LIFECYCLE_STATUS", "SHIPMENT_MODIFIED_DATE",
    ]:
        if col not in df.columns:
            df[col] = pd.NaT if col == "SHIPMENT_MODIFIED_DATE" else ""

    for col in ["CDD", "CGO", "CER", "VAD", "VDL", "CGI", "CEP", "CLL"]:
        if col not in df.columns:
            df[col] = pd.NaT

    return df


# -----------------------------------------------------------------------------
# D&D CALCULATION ENGINE (flat single rates)
# -----------------------------------------------------------------------------
def process_shipments(df, contracts_list=None, estimate_profile=None, use_estimate=False):
    df = normalize_required_columns(df.copy())

    event_cols = ["CDD", "CGO", "CER", "VAD", "VDL", "CGI", "CEP", "CLL"]
    for col in event_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if "REPORTING_DATE" in df.columns:
        df["REPORTING_DATE"] = pd.to_datetime(df["REPORTING_DATE"], errors="coerce", utc=True)
    df["SHIPMENT_MODIFIED_DATE"] = pd.to_datetime(df["SHIPMENT_MODIFIED_DATE"], errors="coerce", utc=True)

    analysis_run_date = pd.Timestamp.now(tz="UTC")
    candidates = build_contract_candidates(contracts_list or [])

    df["SUBSCRIPTION_STATUS"] = df["SUBSCRIPTION_STATUS"].fillna("").astype(str).str.upper()
    cancelled_count = (df["SUBSCRIPTION_STATUS"] == "CANCELLED").sum()
    original_count = len(df)
    df = df[df["SUBSCRIPTION_STATUS"] != "CANCELLED"].copy()

    ident = df.apply(_shipment_match_identity, axis=1, result_type="expand")
    df["CARRIER_FFW_SCAC"] = ident[0]
    df["MATCHED_PARTY_TYPE"] = ident[1]

    matched_results = []
    unmatched_results = []

    for _, row in df.iterrows():
        carrier_scac = _clean_text(row.get("CARRIER_SCAC", ""))
        ffw_scac = _clean_text(row.get("FFW_SCAC", ""))
        pol_locode = _clean_text(row.get("POL_LOCODE", ""))
        pod_locode = _clean_text(row.get("POD_LOCODE", ""))

        if use_estimate:
            profile = estimate_profile
            party_type = "Estimate"
        else:
            profile, party_type = match_contract(candidates, carrier_scac, ffw_scac, pol_locode, pod_locode)

        cgi, cll = row["CGI"], row["CLL"]
        cdd, cgo, cer = row["CDD"], row["CGO"], row["CER"]
        sub_status = row.get("SUBSCRIPTION_STATUS", "")

        pol_dem_total = _days_between(cll, cgi)
        pod_dem_total = _days_between(cgo, cdd)

        pod_det_total = None
        det_accumulating = False
        det_end_source = ""
        det_end_ts = pd.NaT
        if not pd.isna(cgo):
            if not pd.isna(cer):
                pod_det_total = _days_between(cer, cgo)
                det_end_source, det_end_ts = "CER", cer
            elif sub_status == "ACTIVE":
                pod_det_total = _days_between(analysis_run_date, cgo)
                det_accumulating = True
                det_end_source, det_end_ts = "TODAY", analysis_run_date
            elif sub_status == "COMPLETED":
                modified = row.get("SHIPMENT_MODIFIED_DATE", pd.NaT)
                if not pd.isna(modified):
                    pod_det_total = _days_between(modified, cgo)
                    det_end_ts = modified
                det_end_source = "MODIFIED_DATE"

        base = {
            "SHIPMENT_ID": row["SHIPMENT_ID"],
            "CONTAINER_NUMBER": row.get("CONTAINER_NUMBER", ""),
            "CARRIER_SCAC": carrier_scac,
            "CARRIER_NAME": row.get("CARRIER_NAME", ""),
            "FFW_SCAC": ffw_scac,
            "CARRIER_FFW_SCAC": row.get("CARRIER_FFW_SCAC", ""),
            "MATCHED_PARTY_TYPE": party_type,
            "POL_LOCODE": pol_locode,
            "POL": row.get("POL", ""),
            "POD_LOCODE": pod_locode,
            "POD": row.get("POD", ""),
            "SUBSCRIPTION_STATUS": sub_status,
            "LIFECYCLE_STATUS": row.get("LIFECYCLE_STATUS", ""),
            "CGI": cgi if not pd.isna(cgi) else pd.NaT,
            "CLL": cll if not pd.isna(cll) else pd.NaT,
            "CDD": cdd if not pd.isna(cdd) else pd.NaT,
            "CGO": cgo if not pd.isna(cgo) else pd.NaT,
            "CER": cer if not pd.isna(cer) else pd.NaT,
            "DET_END_TS": det_end_ts,
            "DD_ANCHOR_DATE": cdd if not pd.isna(cdd) else (
                cgo if not pd.isna(cgo) else (
                    cer if not pd.isna(cer) else (cll if not pd.isna(cll) else cgi))),
            "POL_DEM_TOTAL_DAYS": round(pol_dem_total, 2) if pol_dem_total is not None else None,
            "POD_DEM_TOTAL_DAYS": round(pod_dem_total, 2) if pod_dem_total is not None else None,
            "POD_DET_TOTAL_DAYS": round(pod_det_total, 2) if pod_det_total is not None else None,
            "DET_ACCUMULATING": det_accumulating,
            "DET_END_SOURCE": det_end_source,
            "LANE": f"{pol_locode} → {pod_locode}",
            "MATCH_KEY": f"{pod_locode}|{row.get('CARRIER_FFW_SCAC', '')}|{pol_locode}",
        }

        if profile is None:
            reason_parts = []
            if pd.isna(cgi) or pd.isna(cll):
                reason_parts.append("Cannot evaluate POL demurrage; missing CGI or CLL")
            if pd.isna(cdd) or pd.isna(cgo):
                reason_parts.append("Cannot evaluate POD demurrage; missing CDD or CGO")
            if pd.isna(cgo):
                reason_parts.append("Cannot evaluate POD detention; missing CGO")
            if pd.isna(cer) and sub_status not in ["ACTIVE", "COMPLETED"]:
                reason_parts.append("Cannot evaluate POD detention end; missing CER and status not ACTIVE/COMPLETED")
            urec = base.copy()
            urec.update({
                "MISSING_CONTRACT_REASON": "No matching contract row (carrier / FFW / POL / POD)",
                "DATA_LIMITATION": "; ".join(reason_parts) if reason_parts else
                                   "Dwell days available; fees cannot be calculated without a contract",
                "RISK_FLAG": False,
                "RISK_REASONS": "",
            })
            unmatched_results.append(urec)
            continue

        dem_rate = profile.get("dem_rate")
        det_rate = profile.get("det_rate")
        dem_free = _safe(profile.get("dem_free"), 0.0)
        det_free = _safe(profile.get("det_free"), 0.0)
        combined_free = profile.get("combined_free")
        has_combined = combined_free is not None

        # POL demurrage (always separate; uses the single demurrage rate)
        pol_dem_chargeable = 0.0
        if pol_dem_total is not None:
            pol_dem_chargeable = max(0.0, pol_dem_total - dem_free)
        pol_dem_cost = _flat_cost(pol_dem_chargeable, dem_rate)

        # POD demurrage + POD detention (combined pool optional)
        pod_dem_chargeable = 0.0
        pod_det_chargeable = 0.0
        remaining_free_for_det = 0.0

        if pod_dem_total is not None:
            if has_combined:
                pod_dem_chargeable = max(0.0, pod_dem_total - combined_free)
                remaining_free_for_det = max(0.0, combined_free - pod_dem_total)
            else:
                pod_dem_chargeable = max(0.0, pod_dem_total - dem_free)

        if pod_det_total is not None:
            if has_combined:
                pod_det_chargeable = max(0.0, pod_det_total - remaining_free_for_det)
            else:
                pod_det_chargeable = max(0.0, pod_det_total - det_free)

        pod_dem_cost = _flat_cost(pod_dem_chargeable, dem_rate)
        pod_det_cost = _flat_cost(pod_det_chargeable, det_rate)
        total_cost = round(pol_dem_cost + pod_dem_cost + pod_det_cost, 2)

        mrec = base.copy()
        mrec.update({
            "RATE_SOURCE": "Estimate" if use_estimate else "Contract",
            "CONTRACT_IDENTIFIER": profile.get("identifier", ""),
            "DEM_RATE": dem_rate,
            "DET_RATE": det_rate,
            "DEM_FREE_DAYS": dem_free,
            "DET_FREE_DAYS": det_free,
            "COMBINED_FREE_DAYS": combined_free,
            "CONTRACT_TYPE": "Estimate" if use_estimate else ("Combined" if has_combined else "Separate"),
            "POL_DEM_CHARGEABLE_DAYS": round(pol_dem_chargeable, 2),
            "POL_DEM_COST": pol_dem_cost,
            "POD_DEM_CHARGEABLE_DAYS": round(pod_dem_chargeable, 2),
            "POD_DEM_COST": pod_dem_cost,
            "POD_DET_CHARGEABLE_DAYS": round(pod_det_chargeable, 2),
            "POD_DET_COST": pod_det_cost,
            "DEM_COST": round(pol_dem_cost + pod_dem_cost, 2),
            "DET_COST": pod_det_cost,
            "TOTAL_DD_COST": total_cost,
        })
        matched_results.append(mrec)

    matched_df = pd.DataFrame(matched_results)
    unmatched_df = pd.DataFrame(unmatched_results)
    unmatched_df = enrich_unmatched_risk(unmatched_df, matched_df)
    return matched_df, unmatched_df, original_count, cancelled_count


def enrich_unmatched_risk(unmatched_df, matched_df):
    if unmatched_df.empty:
        return unmatched_df

    def positive_mean(df, col):
        if df.empty or col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[s > 0]
        return s.mean() if len(s) else np.nan

    avg_pol = positive_mean(matched_df, "POL_DEM_TOTAL_DAYS")
    avg_pod = positive_mean(matched_df, "POD_DEM_TOTAL_DAYS")
    avg_det = positive_mean(matched_df, "POD_DET_TOTAL_DAYS")
    if np.isnan(avg_pol):
        avg_pol = 3.0
    if np.isnan(avg_pod):
        avg_pod = 3.0
    if np.isnan(avg_det):
        avg_det = 5.0

    unmatched_df["AVG_POL_DEM_BENCHMARK"] = round(avg_pol, 2)
    unmatched_df["AVG_POD_DEM_BENCHMARK"] = round(avg_pod, 2)
    unmatched_df["AVG_POD_DET_BENCHMARK"] = round(avg_det, 2)

    flags, reasons, scores = [], [], []
    for _, row in unmatched_df.iterrows():
        rs, score = [], 0
        pol_d, pod_d, det_d = row.get("POL_DEM_TOTAL_DAYS"), row.get("POD_DEM_TOTAL_DAYS"), row.get("POD_DET_TOTAL_DAYS")
        if pd.notna(pol_d) and pol_d > avg_pol:
            rs.append(f"POL demurrage {pol_d:.1f}d vs avg {avg_pol:.1f}d"); score += 1
        if pd.notna(pod_d) and pod_d > avg_pod:
            rs.append(f"POD demurrage {pod_d:.1f}d vs avg {avg_pod:.1f}d"); score += 1
        if pd.notna(det_d) and det_d > avg_det:
            rs.append(f"POD detention {det_d:.1f}d vs avg {avg_det:.1f}d"); score += 1
        if bool(row.get("DET_ACCUMULATING")):
            rs.append("ACTIVE with no CER; POD detention still accumulating"); score += 1
        flags.append(score > 0)
        scores.append(score)
        reasons.append("; ".join(rs) if rs else "No above-average dwell risk detected")

    unmatched_df["RISK_FLAG"] = flags
    unmatched_df["RISK_SCORE"] = scores
    unmatched_df["RISK_REASONS"] = reasons
    return unmatched_df


# -----------------------------------------------------------------------------
# DOWNLOAD HELPERS
# -----------------------------------------------------------------------------
def format_datetime_cols(dl, cols):
    for col in cols:
        if col in dl.columns:
            dl[col] = pd.to_datetime(dl[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("")
    return dl


def build_download_df(data):
    dl = data.copy()
    dl = format_datetime_cols(dl, ["CGI", "CLL", "CDD", "CGO", "CER", "DET_END_TS"])
    rename_map = {
        "SHIPMENT_ID": "Shipment ID", "CONTAINER_NUMBER": "Container",
        "CARRIER_SCAC": "Carrier SCAC", "CARRIER_NAME": "Carrier Name",
        "FFW_SCAC": "Freight Forwarder SCAC", "CARRIER_FFW_SCAC": "Carrier / FFW SCAC",
        "MATCHED_PARTY_TYPE": "Matched Party Type", "POL_LOCODE": "Port of Loading",
        "POD_LOCODE": "Port of Discharge", "LANE": "Lane", "SUBSCRIPTION_STATUS": "Subscription Status",
        "CGI": "Gate In at POL (CGI)", "CLL": "Loaded on Vessel (CLL)",
        "CDD": "Discharge at POD (CDD)", "CGO": "Gate Out Full at POD (CGO)", "CER": "Empty Return (CER)",
        "DEM_RATE": "Demurrage Rate (USD/day)", "DET_RATE": "Detention Rate (USD/day)",
        "DEM_FREE_DAYS": "Free Demurrage Days", "DET_FREE_DAYS": "Free Detention Days",
        "COMBINED_FREE_DAYS": "Combined Free Days", "CONTRACT_TYPE": "Free Days Type",
        "CONTRACT_IDENTIFIER": "Matched Contract",
        "POL_DEM_TOTAL_DAYS": "POL Demurrage Total Days", "POL_DEM_CHARGEABLE_DAYS": "POL Demurrage Chargeable Days",
        "POL_DEM_COST": "POL Demurrage Cost (USD)",
        "POD_DEM_TOTAL_DAYS": "POD Demurrage Total Days", "POD_DEM_CHARGEABLE_DAYS": "POD Demurrage Chargeable Days",
        "POD_DEM_COST": "POD Demurrage Cost (USD)",
        "POD_DET_TOTAL_DAYS": "POD Detention Total Days", "POD_DET_CHARGEABLE_DAYS": "POD Detention Chargeable Days",
        "POD_DET_COST": "POD Detention Cost (USD)",
        "DEM_COST": "Total Demurrage Cost (USD)", "DET_COST": "Total Detention Cost (USD)",
        "TOTAL_DD_COST": "Total D&D Cost (USD)",
        "DET_ACCUMULATING": "Detention Still Accumulating", "DET_END_SOURCE": "Detention End Date Source",
        "MATCH_KEY": "Match Key",
    }
    dl = dl.rename(columns={k: v for k, v in rename_map.items() if k in dl.columns})
    dl = dl.drop(columns=[c for c in ["POL", "POD", "LIFECYCLE_STATUS"] if c in dl.columns], errors="ignore")

    desired_order = [
        "Shipment ID", "Container", "Carrier SCAC", "Carrier Name", "Freight Forwarder SCAC",
        "Carrier / FFW SCAC", "Matched Party Type", "Matched Contract", "Lane", "Port of Loading",
        "Port of Discharge", "Subscription Status", "Gate In at POL (CGI)", "Loaded on Vessel (CLL)",
        "Discharge at POD (CDD)", "Gate Out Full at POD (CGO)", "Empty Return (CER)",
        "Demurrage Rate (USD/day)", "Detention Rate (USD/day)", "Free Days Type",
        "Free Demurrage Days", "Free Detention Days", "Combined Free Days",
        "POL Demurrage Total Days", "POL Demurrage Chargeable Days", "POL Demurrage Cost (USD)",
        "POD Demurrage Total Days", "POD Demurrage Chargeable Days", "POD Demurrage Cost (USD)",
        "POD Detention Total Days", "POD Detention Chargeable Days", "POD Detention Cost (USD)",
        "Total Demurrage Cost (USD)", "Total Detention Cost (USD)", "Total D&D Cost (USD)",
        "Detention Still Accumulating", "Detention End Date Source", "Match Key",
    ]
    existing = [c for c in desired_order if c in dl.columns]
    remaining = [c for c in dl.columns if c not in existing]
    return dl[existing + remaining]


def build_unmatched_download_df(data):
    if data.empty:
        return data.copy()
    dl = data.copy()
    dl = format_datetime_cols(dl, ["CGI", "CLL", "CDD", "CGO", "CER", "DET_END_TS"])
    rename_map = {
        "SHIPMENT_ID": "Shipment ID", "CONTAINER_NUMBER": "Container",
        "CARRIER_SCAC": "Carrier SCAC", "CARRIER_NAME": "Carrier Name",
        "POL_LOCODE": "Port of Loading", "POD_LOCODE": "Port of Discharge", "LANE": "Lane",
        "SUBSCRIPTION_STATUS": "Subscription Status", "CGI": "Gate In at POL (CGI)",
        "CLL": "Loaded on Vessel (CLL)", "CDD": "Discharge at POD (CDD)",
        "CGO": "Gate Out Full at POD (CGO)", "CER": "Empty Return (CER)",
        "POL_DEM_TOTAL_DAYS": "POL Demurrage Days", "POD_DEM_TOTAL_DAYS": "POD Demurrage Days",
        "POD_DET_TOTAL_DAYS": "POD Detention Days", "MATCH_KEY": "Match Key",
        "RISK_FLAG": "Risk Flag", "RISK_SCORE": "Risk Score", "RISK_REASONS": "Risk Reasons",
        "DATA_LIMITATION": "Data Limitation",
    }
    dl = dl.rename(columns={k: v for k, v in rename_map.items() if k in dl.columns})
    dl = dl.drop(columns=[c for c in ["POL", "POD", "LIFECYCLE_STATUS"] if c in dl.columns], errors="ignore")
    return dl


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚢 D&D Analyzer — Academy Sports")
    st.caption("Single flat demurrage rate (POL + POD) · single flat detention rate")
    st.markdown("---")

    rate_source = st.radio(
        "Rate Source",
        ["Upload Contract CSV", "Estimate Rates"],
        help="Use uploaded contract terms, or estimate D&D exposure from manually entered flat rates.",
    )

    uploaded_contract_file = None
    estimate_profile = None

    if rate_source == "Upload Contract CSV":
        uploaded_contract_file = st.file_uploader(
            "Upload Contract CSV",
            type=["csv"],
            help=("Simple flat-rate contract. Recognized columns (flexible spelling): "
                  "demurrageRate, detentionRate, demurrageFreeDays, detentionFreeDays, "
                  "and optionally carrierScac / ffwScac / pol / pod / combinedFreeDays."),
            key="contract_uploader",
        )
    else:
        st.markdown("#### Estimate Rates")
        st.caption("Flat USD/day rates. The demurrage rate applies to both POL and POD demurrage.")
        est_dem_rate = st.number_input("Demurrage rate (USD/day) — POL & POD", min_value=0.0, value=0.0, step=25.0)
        est_det_rate = st.number_input("Detention rate (USD/day) — POD", min_value=0.0, value=0.0, step=25.0)
        use_combined = st.checkbox(
            "Use combined free days for POD demurrage + POD detention",
            value=False,
            help="POD demurrage consumes the pool first; POD detention gets the remainder. POL demurrage is separate.",
        )
        if use_combined:
            est_combined_free = st.number_input("Combined POD free days", min_value=0.0, value=0.0, step=1.0)
            est_dem_free = st.number_input("POL free demurrage days", min_value=0.0, value=0.0, step=1.0)
            est_det_free = 0.0
            estimate_profile = _to_profile({
                "demRate": est_dem_rate, "detRate": est_det_rate,
                "demFree": est_dem_free, "detFree": est_det_free,
                "combinedFree": est_combined_free,
            }, is_estimate=True, label="Estimate")
        else:
            est_dem_free = st.number_input("Free demurrage days (each leg)", min_value=0.0, value=0.0, step=1.0)
            est_det_free = st.number_input("Free detention days", min_value=0.0, value=0.0, step=1.0)
            estimate_profile = _to_profile({
                "demRate": est_dem_rate, "detRate": est_det_rate,
                "demFree": est_dem_free, "detFree": est_det_free,
                "combinedFree": None,
            }, is_estimate=True, label="Estimate")

    uploaded_file = st.file_uploader(
        "Upload Shipment CSV",
        type=["csv"],
        help="Upload the ocean shipment export CSV with milestone events.",
        key="shipment_uploader",
    )
    st.markdown("---")

contracts_list = None
contracts_df = None
contract_resolved = {}
if rate_source == "Upload Contract CSV" and uploaded_contract_file is not None:
    try:
        contracts_list, contracts_df, contract_resolved = parse_contracts_csv(uploaded_contract_file)
        with st.sidebar:
            st.success(f"✅ Loaded {len(contracts_list)} contract row(s)")
            mapped = ", ".join(f"{k}→{v}" for k, v in contract_resolved.items()) or "—"
            st.caption(f"Recognized columns: {mapped}")
            globals_rows = sum(1 for c in contracts_list
                               if not any([c.get("carrierScac"), c.get("ffwScac"), c.get("pol"), c.get("pod")]))
            dem_set = sum(1 for c in contracts_list if c.get("demRate") is not None)
            det_set = sum(1 for c in contracts_list if c.get("detRate") is not None)
            st.markdown(f"**Rows with demurrage rate:** {dem_set} | **with detention rate:** {det_set}")
            if globals_rows:
                st.markdown(f"**Global default rows (no identifiers):** {globals_rows}")
    except Exception as e:
        st.sidebar.error(f"❌ Error parsing contract CSV: {e}")
        contracts_list = None

# -----------------------------------------------------------------------------
# LANDING PAGE
# -----------------------------------------------------------------------------
missing_contract = rate_source == "Upload Contract CSV" and uploaded_contract_file is None
missing_shipments = uploaded_file is None
if missing_contract or missing_shipments:
    st.markdown("## 🚢 D&D Analyzer — Academy Sports")
    st.markdown("---")

    if missing_contract and missing_shipments:
        st.info("Upload both a **Contract CSV** and a **Shipment CSV** from the sidebar, or switch Rate Source to **Estimate Rates**.")
    elif missing_contract:
        st.info("Upload a **Contract CSV** from the sidebar, or switch Rate Source to **Estimate Rates**.")
    elif missing_shipments:
        st.info("Upload a **Shipment CSV** from the sidebar to continue.")

    st.markdown("**Simple flat-rate contract CSV** (one row is enough):")
    st.code(
        "demurrageRate,demurrageFreeDays,detentionRate,detentionFreeDays\n"
        "150,4,120,5\n\n"
        "# Optional identifier columns (any/all): carrierScac, ffwScac, pol, pod\n"
        "# Optional: combinedFreeDays (one pool for POD demurrage + POD detention)\n"
        "# A row with no identifiers = global default applied to every shipment.\n"
        "# Column spelling is flexible (case / underscores / spaces ignored).",
        language=None,
    )
    st.markdown("**Pricing logic:**")
    st.code(
        "POL Demurrage = CGI → CLL   priced at the demurrage rate\n"
        "POD Demurrage = CDD → CGO   priced at the demurrage rate (same rate)\n"
        "POD Detention = CGO → CER   priced at the detention rate",
        language=None,
    )
    st.stop()

if rate_source == "Upload Contract CSV" and contracts_list is None:
    st.error("Contract file could not be parsed. Check the format and re-upload, or switch to Estimate Rates.")
    st.stop()

# -----------------------------------------------------------------------------
# LOAD AND PROCESS
# -----------------------------------------------------------------------------
process_label = "estimate rates" if rate_source == "Estimate Rates" else "uploaded contract"
with st.spinner(f"Processing shipments against {process_label}..."):
    raw_df = pd.read_csv(uploaded_file)
    rdf, unmatched_df, total_shipments, cancelled_count = process_shipments(
        raw_df,
        contracts_list=contracts_list,
        estimate_profile=estimate_profile,
        use_estimate=(rate_source == "Estimate Rates"),
    )

if rdf.empty and unmatched_df.empty:
    st.error("No usable shipments found after excluding cancelled shipments.")
    st.stop()

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------
for _df in [rdf, unmatched_df]:
    if not _df.empty:
        if "DD_ANCHOR_DATE" not in _df.columns:
            _df["DD_ANCHOR_DATE"] = pd.NaT
        _df["DD_ANCHOR_DATE"] = pd.to_datetime(_df["DD_ANCHOR_DATE"], errors="coerce", utc=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### Filters")
    cols_needed = ["CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC", "POD_LOCODE", "POL_LOCODE", "DD_ANCHOR_DATE"]
    combined_for_filters = pd.concat(
        [
            rdf[cols_needed] if not rdf.empty else pd.DataFrame(),
            unmatched_df[cols_needed] if not unmatched_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    carriers = sorted(combined_for_filters.get("CARRIER_FFW_SCAC", pd.Series(dtype=str)).dropna().astype(str).unique())
    pods = sorted(combined_for_filters.get("POD_LOCODE", pd.Series(dtype=str)).dropna().astype(str).unique())
    pols = sorted(combined_for_filters.get("POL_LOCODE", pd.Series(dtype=str)).dropna().astype(str).unique())

    sel_carriers = st.multiselect("Carrier / FFW", carriers, default=carriers)
    sel_pods = st.multiselect("POD Terminal", pods, default=pods)
    sel_pols = st.multiselect("POL", pols, default=pols)
    show_zero = st.checkbox("Include $0 charge shipments", value=True)

    st.markdown("### Time Filters")
    trend_grain = st.radio("Trend View", ["Weekly", "Monthly"], horizontal=True)
    valid_dates = combined_for_filters["DD_ANCHOR_DATE"].dropna() if "DD_ANCHOR_DATE" in combined_for_filters.columns else pd.Series(dtype="datetime64[ns, UTC]")
    if not valid_dates.empty:
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
        date_range = st.date_input("D&D Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None
        st.caption("No valid D&D anchor dates found for date filtering.")


def apply_common_filters(data, require_cost_filter=False):
    if data.empty:
        return data.copy()
    out = data.copy()
    if sel_carriers and "CARRIER_FFW_SCAC" in out.columns:
        out = out[out["CARRIER_FFW_SCAC"].astype(str).isin(sel_carriers)]
    if sel_pods:
        out = out[out["POD_LOCODE"].astype(str).isin(sel_pods)]
    if sel_pols:
        out = out[out["POL_LOCODE"].astype(str).isin(sel_pols)]
    if date_range and len(date_range) == 2 and "DD_ANCHOR_DATE" in out.columns:
        start_date, end_date = date_range
        anchor = pd.to_datetime(out["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        out = out[(anchor.dt.date >= start_date) & (anchor.dt.date <= end_date)]
    if require_cost_filter and not show_zero and "TOTAL_DD_COST" in out.columns:
        out = out[out["TOTAL_DD_COST"] > 0]
    return out


def fill_grouping_blanks(data):
    if data.empty:
        return data
    out = data.copy()
    for col in ["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC",
                "CARRIER_NAME", "POD_LOCODE", "POL_LOCODE", "MATCH_KEY"]:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str)
    return out


fdf = apply_common_filters(rdf, require_cost_filter=True) if not rdf.empty else rdf.copy()
ufdf = apply_common_filters(unmatched_df, require_cost_filter=False) if not unmatched_df.empty else unmatched_df.copy()
fdf = fill_grouping_blanks(fdf)
ufdf = fill_grouping_blanks(ufdf)

# -----------------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------------
tab_overview, tab_trends, tab_carrier, tab_port, tab_ships, tab_gaps, tab_download = st.tabs(
    ["📊 Overview", "📈 Trends", "🚛 Carrier / FFW", "🏗️ Ports & Lanes",
     "📦 Shipments", "⚠️ Contract Gaps", "📥 Download"]
)

# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.markdown("### Executive Summary")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total D&D Cost", f"${fdf['TOTAL_DD_COST'].sum():,.0f}",
                  f"{len(fdf)} matched of {total_shipments:,}",
                  help="POL demurrage + POD demurrage + POD detention across matched shipments.")
        c2.metric("POL Demurrage", f"${fdf['POL_DEM_COST'].sum():,.0f}",
                  f"{(fdf['POL_DEM_COST'] > 0).sum()} shipments",
                  help="Container gate-in at POL to loaded-on-vessel, at the demurrage rate.")
        c3.metric("POD Demurrage", f"${fdf['POD_DEM_COST'].sum():,.0f}",
                  f"{(fdf['POD_DEM_COST'] > 0).sum()} shipments",
                  help="Discharge at POD to gate-out-full at POD, at the demurrage rate.")
        c4.metric("POD Detention", f"${fdf['POD_DET_COST'].sum():,.0f}",
                  f"{(fdf['POD_DET_COST'] > 0).sum()} shipments",
                  help="Gate-out-full at POD to empty return, at the detention rate.")

        c1, c2, c3, c4 = st.columns(4)
        avg_pol = fdf.loc[fdf["POL_DEM_COST"] > 0, "POL_DEM_CHARGEABLE_DAYS"].mean()
        avg_pod = fdf.loc[fdf["POD_DEM_COST"] > 0, "POD_DEM_CHARGEABLE_DAYS"].mean()
        avg_det = fdf.loc[fdf["POD_DET_COST"] > 0, "POD_DET_CHARGEABLE_DAYS"].mean()
        c1.metric("Avg POL Dem Days", f"{avg_pol:.1f}d" if not np.isnan(avg_pol) else "—")
        c2.metric("Avg POD Dem Days", f"{avg_pod:.1f}d" if not np.isnan(avg_pod) else "—")
        c3.metric("Avg POD Det Days", f"{avg_det:.1f}d" if not np.isnan(avg_det) else "—")
        c4.metric("⚠️ Accumulating", f"{fdf['DET_ACCUMULATING'].sum()}", "ACTIVE, no CER",
                  help="Active shipments with no empty return. Detention is calculated to today and keeps growing.")

        if cancelled_count > 0:
            st.caption(f"ℹ️ {cancelled_count} cancelled shipments excluded from analysis.")

        st.markdown("---")
        st.caption("💡 Cost split by carrier / FFW. Blue = POL demurrage, orange = POD demurrage, purple = POD detention.")
        carrier_agg = (
            fdf.groupby("CARRIER_FFW_SCAC")
            .agg(POL_Demurrage=("POL_DEM_COST", "sum"), POD_Demurrage=("POD_DEM_COST", "sum"),
                 POD_Detention=("POD_DET_COST", "sum"))
            .reset_index()
        )
        carrier_melt = carrier_agg.melt(id_vars="CARRIER_FFW_SCAC", var_name="Type", value_name="Cost")
        carrier_melt["Type"] = carrier_melt["Type"].replace(
            {"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage", "POD_Detention": "POD Detention"})
        chart_carrier = (
            alt.Chart(carrier_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                y=alt.Y("CARRIER_FFW_SCAC:N", sort="-x", title="Carrier / FFW"),
                x=alt.X("Cost:Q", title="Cost (USD)"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["POL Demurrage", "POD Demurrage", "POD Detention"],
                    range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                tooltip=["CARRIER_FFW_SCAC", "Type", alt.Tooltip("Cost:Q", format="$,.0f")],
            ).properties(title="D&D Cost by Carrier / FFW", height=280)
        )
        st.altair_chart(chart_carrier, use_container_width=True)

        st.caption("💡 Cost split by POD terminal. High POD demurrage = pickup/terminal delay. High detention = empty-return delay.")
        pod_agg = (
            fdf.groupby("POD_LOCODE")
            .agg(POL_Demurrage=("POL_DEM_COST", "sum"), POD_Demurrage=("POD_DEM_COST", "sum"),
                 POD_Detention=("POD_DET_COST", "sum"))
            .reset_index()
        )
        pod_melt = pod_agg.melt(id_vars="POD_LOCODE", var_name="Type", value_name="Cost")
        pod_melt["Type"] = pod_melt["Type"].replace(
            {"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage", "POD_Detention": "POD Detention"})
        chart_pod = (
            alt.Chart(pod_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                y=alt.Y("POD_LOCODE:N", sort="-x", title="POD Terminal"),
                x=alt.X("Cost:Q", title="Cost (USD)"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["POL Demurrage", "POD Demurrage", "POD Detention"],
                    range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                tooltip=["POD_LOCODE", "Type", alt.Tooltip("Cost:Q", format="$,.0f")],
            ).properties(title="D&D Cost by POD Terminal", height=250)
        )
        st.altair_chart(chart_pod, use_container_width=True)

# -----------------------------------------------------------------------------
# TRENDS
# -----------------------------------------------------------------------------
with tab_trends:
    st.markdown("### D&D Trends")
    st.caption("Trend date uses CDD when available, then CGO, then CER, then CLL/CGI as fallback.")
    if fdf.empty:
        st.warning("No matched/priced shipments available for the selected filters.")
    else:
        trend_df = fdf.copy()
        trend_df["DD_ANCHOR_DATE"] = pd.to_datetime(trend_df["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        trend_df = trend_df.dropna(subset=["DD_ANCHOR_DATE"])
        if trend_df.empty:
            st.warning("No valid D&D anchor dates found for the selected filters.")
        else:
            if trend_grain == "Weekly":
                trend_df["PERIOD"] = trend_df["DD_ANCHOR_DATE"].dt.to_period("W").apply(lambda r: r.start_time)
                period_title = "Week"
            else:
                trend_df["PERIOD"] = trend_df["DD_ANCHOR_DATE"].dt.to_period("M").apply(lambda r: r.start_time)
                period_title = "Month"
            trend_agg = (
                trend_df.groupby("PERIOD")
                .agg(Shipments=("SHIPMENT_ID", "count"), POL_Demurrage=("POL_DEM_COST", "sum"),
                     POD_Demurrage=("POD_DEM_COST", "sum"), Detention=("POD_DET_COST", "sum"),
                     Total=("TOTAL_DD_COST", "sum"), Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", "mean"),
                     Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", "mean"), Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", "mean"),
                     Accumulating=("DET_ACCUMULATING", "sum"))
                .reset_index().sort_values("PERIOD")
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Periods", f"{len(trend_agg):,}")
            c2.metric("Total Cost", f"${trend_agg['Total'].sum():,.0f}")
            c3.metric("Avg Cost / Shipment", f"${(trend_agg['Total'].sum() / max(trend_agg['Shipments'].sum(), 1)):,.0f}")
            c4.metric("Accumulating", f"{int(trend_agg['Accumulating'].sum()):,}", "ACTIVE, no CER")

            st.markdown("#### Cost Trend")
            cost_melt = trend_agg.melt(id_vars=["PERIOD"], value_vars=["POL_Demurrage", "POD_Demurrage", "Detention"],
                                       var_name="Charge Type", value_name="Cost")
            cost_melt["Charge Type"] = cost_melt["Charge Type"].replace(
                {"POL_Demurrage": "POL Demurrage", "POD_Demurrage": "POD Demurrage"})
            cost_chart = (
                alt.Chart(cost_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("PERIOD:T", title=period_title), y=alt.Y("Cost:Q", title="Cost"),
                    color=alt.Color("Charge Type:N", scale=alt.Scale(
                        domain=["POL Demurrage", "POD Demurrage", "Detention"],
                        range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                    tooltip=[alt.Tooltip("PERIOD:T", title=period_title), "Charge Type:N", alt.Tooltip("Cost:Q", format="$,.0f")],
                ).properties(height=350)
            )
            st.altair_chart(cost_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Shipment Volume")
                shipment_chart = (
                    alt.Chart(trend_agg).mark_line(point=True)
                    .encode(x=alt.X("PERIOD:T", title=period_title), y=alt.Y("Shipments:Q", title="Shipments"),
                            tooltip=[alt.Tooltip("PERIOD:T", title=period_title), "Shipments"])
                    .properties(height=280)
                )
                st.altair_chart(shipment_chart, use_container_width=True)
            with col2:
                st.markdown("#### Avg Chargeable Days")
                days_melt = trend_agg.melt(id_vars=["PERIOD"], value_vars=["Avg_POL_Dem_Days", "Avg_POD_Dem_Days", "Avg_Det_Days"],
                                           var_name="Metric", value_name="Days")
                days_melt["Metric"] = days_melt["Metric"].replace(
                    {"Avg_POL_Dem_Days": "POL Demurrage", "Avg_POD_Dem_Days": "POD Demurrage", "Avg_Det_Days": "Detention"})
                days_chart = (
                    alt.Chart(days_melt).mark_line(point=True)
                    .encode(x=alt.X("PERIOD:T", title=period_title), y=alt.Y("Days:Q", title="Avg Chargeable Days"),
                            color=alt.Color("Metric:N", scale=alt.Scale(
                                domain=["POL Demurrage", "POD Demurrage", "Detention"],
                                range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                            tooltip=[alt.Tooltip("PERIOD:T", title=period_title), "Metric:N", alt.Tooltip("Days:Q", format=".1f")])
                    .properties(height=280)
                )
                st.altair_chart(days_chart, use_container_width=True)

            st.markdown("#### Trend Summary")
            st.dataframe(
                trend_agg.style.format({
                    "POL_Demurrage": "${:,.0f}", "POD_Demurrage": "${:,.0f}", "Detention": "${:,.0f}",
                    "Total": "${:,.0f}", "Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_Det_Days": "{:.1f}"}),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")
    st.markdown("### Contract Gap Trend")
    if ufdf.empty:
        st.info("No unmatched/contract-gap shipments for the selected filters.")
    else:
        gap_trend = ufdf.copy()
        gap_trend["DD_ANCHOR_DATE"] = pd.to_datetime(gap_trend["DD_ANCHOR_DATE"], errors="coerce", utc=True)
        gap_trend = gap_trend.dropna(subset=["DD_ANCHOR_DATE"])
        if gap_trend.empty:
            st.info("Contract-gap shipments do not have valid anchor dates for trend analysis.")
        else:
            if trend_grain == "Weekly":
                gap_trend["PERIOD"] = gap_trend["DD_ANCHOR_DATE"].dt.to_period("W").apply(lambda r: r.start_time)
                period_title = "Week"
            else:
                gap_trend["PERIOD"] = gap_trend["DD_ANCHOR_DATE"].dt.to_period("M").apply(lambda r: r.start_time)
                period_title = "Month"
            gap_agg = (
                gap_trend.groupby("PERIOD")
                .agg(Unmatched_Shipments=("SHIPMENT_ID", "count"), Risk_Shipments=("RISK_FLAG", "sum"),
                     Missing_Contract_Keys=("MATCH_KEY", "nunique"), Avg_POL_Dem_Days=("POL_DEM_TOTAL_DAYS", "mean"),
                     Avg_POD_Dem_Days=("POD_DEM_TOTAL_DAYS", "mean"), Avg_POD_Det_Days=("POD_DET_TOTAL_DAYS", "mean"))
                .reset_index().sort_values("PERIOD")
            )
            gap_chart = (
                alt.Chart(gap_agg).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("PERIOD:T", title=period_title), y=alt.Y("Unmatched_Shipments:Q", title="Unmatched Shipments"),
                        tooltip=[alt.Tooltip("PERIOD:T", title=period_title), "Unmatched_Shipments:Q", "Risk_Shipments:Q", "Missing_Contract_Keys:Q"])
                .properties(height=260)
            )
            st.altair_chart(gap_chart, use_container_width=True)
            st.dataframe(
                gap_agg.style.format({"Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_POD_Det_Days": "{:.1f}"}),
                use_container_width=True, hide_index=True,
            )

# -----------------------------------------------------------------------------
# CARRIERS
# -----------------------------------------------------------------------------
with tab_carrier:
    st.markdown("### Carrier / FFW Summary")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        carrier_detail = (
            fdf.groupby(["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC", "CARRIER_NAME"], dropna=False)
            .agg(Ships=("SHIPMENT_ID", "count"),
                 POL_Dem_Ships=("POL_DEM_COST", lambda x: (x > 0).sum()),
                 POD_Dem_Ships=("POD_DEM_COST", lambda x: (x > 0).sum()),
                 Det_Ships=("POD_DET_COST", lambda x: (x > 0).sum()),
                 POL_Dem_Cost=("POL_DEM_COST", "sum"), POD_Dem_Cost=("POD_DEM_COST", "sum"), Det_Cost=("POD_DET_COST", "sum"),
                 Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
                 Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
                 Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", lambda x: x[x > 0].mean() if (x > 0).any() else 0))
            .reset_index()
        )
        carrier_detail["Total_Cost"] = carrier_detail["POL_Dem_Cost"] + carrier_detail["POD_Dem_Cost"] + carrier_detail["Det_Cost"]
        carrier_detail = carrier_detail.sort_values("Total_Cost", ascending=False)
        st.dataframe(
            carrier_detail.style.format({
                "POL_Dem_Cost": "${:,.0f}", "POD_Dem_Cost": "${:,.0f}", "Det_Cost": "${:,.0f}", "Total_Cost": "${:,.0f}",
                "Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}", "Avg_Det_Days": "{:.1f}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        st.markdown("#### Carrier / FFW × POD Breakdown")
        cp = (
            fdf.groupby(["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "POD_LOCODE"], dropna=False)
            .agg(Ships=("SHIPMENT_ID", "count"), POL_Dem=("POL_DEM_COST", "sum"),
                 POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum"))
            .reset_index()
        )
        cp["Total"] = cp["POL_Dem"] + cp["POD_Dem"] + cp["Det"]
        cp = cp[cp["Total"] > 0].sort_values("Total", ascending=False)
        if len(cp) > 0:
            heat = (
                alt.Chart(cp).mark_rect(cornerRadius=4)
                .encode(x=alt.X("POD_LOCODE:N", title="POD"), y=alt.Y("CARRIER_FFW_SCAC:N", title="Carrier / FFW"),
                        color=alt.Color("Total:Q", scale=alt.Scale(scheme="oranges"), title="Total D&D"),
                        tooltip=["CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "POD_LOCODE", "Ships",
                                 alt.Tooltip("POL_Dem:Q", format="$,.0f"), alt.Tooltip("POD_Dem:Q", format="$,.0f"),
                                 alt.Tooltip("Det:Q", format="$,.0f"), alt.Tooltip("Total:Q", format="$,.0f")])
                .properties(title="Cost Heatmap: Carrier / FFW × POD", height=280)
            )
            text = heat.mark_text(fontSize=11, fontWeight="bold").encode(
                text=alt.Text("Total:Q", format="$,.0f"),
                color=alt.condition(alt.datum.Total > cp["Total"].median(), alt.value("white"), alt.value("black")))
            st.altair_chart(heat + text, use_container_width=True)
        st.dataframe(
            cp.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

# -----------------------------------------------------------------------------
# PORTS & LANES
# -----------------------------------------------------------------------------
with tab_port:
    st.markdown("### Ports & Lanes")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### POD Terminal Summary")
            pod_sum = (
                fdf.groupby(["POD_LOCODE", "POD"])
                .agg(Ships=("SHIPMENT_ID", "count"), POL_Dem=("POL_DEM_COST", "sum"),
                     POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum"))
                .reset_index()
            )
            pod_sum["Total"] = pod_sum["POL_Dem"] + pod_sum["POD_Dem"] + pod_sum["Det"]
            pod_sum = pod_sum.sort_values("Total", ascending=False)
            st.dataframe(
                pod_sum.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
        with col2:
            st.markdown("#### Cost Split by POD")
            pod_melt2 = pod_sum.melt(id_vars="POD_LOCODE", value_vars=["POL_Dem", "POD_Dem", "Det"],
                                     var_name="Type", value_name="Cost")
            pod_melt2["Type"] = pod_melt2["Type"].replace(
                {"POL_Dem": "POL Demurrage", "POD_Dem": "POD Demurrage", "Det": "POD Detention"})
            ch = (
                alt.Chart(pod_melt2).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("POD_LOCODE:N", title="POD"), y=alt.Y("Cost:Q", title="Cost (USD)", stack=True),
                        color=alt.Color("Type:N", scale=alt.Scale(
                            domain=["POL Demurrage", "POD Demurrage", "POD Detention"],
                            range=[POL_DEM_COLOR, DEM_COLOR, DET_COLOR])),
                        tooltip=["POD_LOCODE", "Type", alt.Tooltip("Cost:Q", format="$,.0f")])
                .properties(height=320)
            )
            st.altair_chart(ch, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Top 20 Lanes by Total D&D Cost")
        lane_agg = (
            fdf.groupby("LANE")
            .agg(Ships=("SHIPMENT_ID", "count"),
                 Carrier_FFWs=("CARRIER_FFW_SCAC", lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))),
                 POL_Dem=("POL_DEM_COST", "sum"), POD_Dem=("POD_DEM_COST", "sum"), Det=("POD_DET_COST", "sum"),
                 Avg_POL_Dem_Days=("POL_DEM_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0),
                 Avg_POD_Dem_Days=("POD_DEM_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0),
                 Avg_Det_Days=("POD_DET_CHARGEABLE_DAYS", lambda x: round(x[x > 0].mean(), 1) if (x > 0).any() else 0))
            .reset_index()
        )
        lane_agg["Total"] = lane_agg["POL_Dem"] + lane_agg["POD_Dem"] + lane_agg["Det"]
        lane_agg = lane_agg.sort_values("Total", ascending=False).head(20)
        st.dataframe(
            lane_agg.style.format({"POL_Dem": "${:,.0f}", "POD_Dem": "${:,.0f}", "Det": "${:,.0f}", "Total": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

# -----------------------------------------------------------------------------
# SHIPMENT EXPLORER
# -----------------------------------------------------------------------------
with tab_ships:
    st.markdown("### Shipment-Level D&D Detail")
    if fdf.empty:
        st.warning("No matched shipments available for the selected filters.")
    else:
        st.caption(f"Showing {len(fdf)} matched shipments. Use sidebar filters to narrow.")
        sort_options = ["TOTAL_DD_COST", "POL_DEM_COST", "POD_DEM_COST", "POD_DET_COST",
                        "POL_DEM_CHARGEABLE_DAYS", "POD_DEM_CHARGEABLE_DAYS", "POD_DET_CHARGEABLE_DAYS"]
        sort_col = st.selectbox("Sort by", sort_options)
        top_n = st.slider("Show top N", 10, min(500, max(len(fdf), 10)), min(50, max(len(fdf), 10)))

        display_cols = [
            "CONTAINER_NUMBER", "SHIPMENT_ID", "CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC",
            "MATCHED_PARTY_TYPE", "LANE", "CGI", "CLL", "CDD", "CGO", "CER",
            "DEM_RATE", "DET_RATE",
            "POL_DEM_TOTAL_DAYS", "POL_DEM_CHARGEABLE_DAYS", "POL_DEM_COST",
            "POD_DEM_TOTAL_DAYS", "POD_DEM_CHARGEABLE_DAYS", "POD_DEM_COST",
            "POD_DET_TOTAL_DAYS", "POD_DET_CHARGEABLE_DAYS", "POD_DET_COST",
            "TOTAL_DD_COST", "CONTRACT_TYPE", "DET_ACCUMULATING", "DET_END_SOURCE",
        ]
        show_df = fdf[[c for c in display_cols if c in fdf.columns]].sort_values(sort_col, ascending=False).head(top_n).copy()
        for dc in ["CGI", "CLL", "CDD", "CGO", "CER"]:
            if dc in show_df.columns:
                show_df[dc] = pd.to_datetime(show_df[dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
        show_df["DET_STATUS"] = show_df.apply(
            lambda r: "⚠️ Active → Today" if r.get("DET_ACCUMULATING", False)
            else ("📅 Completed → Modified" if r.get("DET_END_SOURCE") == "MODIFIED_DATE" else "✓ CER"), axis=1)
        show_df = show_df.drop(columns=["DET_ACCUMULATING", "DET_END_SOURCE"], errors="ignore")
        st.dataframe(
            show_df.style.format({
                "DEM_RATE": "${:,.0f}", "DET_RATE": "${:,.0f}",
                "POL_DEM_COST": "${:,.2f}", "POD_DEM_COST": "${:,.2f}", "POD_DET_COST": "${:,.2f}", "TOTAL_DD_COST": "${:,.2f}"}),
            use_container_width=True, hide_index=True, height=600,
        )

        priced_download_df = fdf[fdf["TOTAL_DD_COST"] > 0].copy()
        if priced_download_df.empty:
            st.info("There's nothing to download because no priced shipments have D&D cost greater than $0.")
        else:
            download_df = priced_download_df[[c for c in display_cols if c in priced_download_df.columns]].copy()
            for dc in ["CGI", "CLL", "CDD", "CGO", "CER"]:
                if dc in download_df.columns:
                    download_df[dc] = pd.to_datetime(download_df[dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
            download_df["DET_STATUS"] = download_df.apply(
                lambda r: "⚠️ Active → Today" if r.get("DET_ACCUMULATING", False)
                else ("📅 Completed → Modified" if r.get("DET_END_SOURCE") == "MODIFIED_DATE" else "✓ CER"), axis=1)
            download_df = download_df.drop(columns=["DET_ACCUMULATING", "DET_END_SOURCE"], errors="ignore")
            st.download_button(
                label="📥 Download Full Priced Shipments CSV",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"DD_Academy_Priced_Shipments_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
            )

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            pol_data = fdf.loc[fdf["POL_DEM_COST"] > 0, ["POL_DEM_CHARGEABLE_DAYS"]].copy()
            if len(pol_data) > 0:
                st.altair_chart(
                    alt.Chart(pol_data).mark_bar(color=POL_DEM_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(x=alt.X("POL_DEM_CHARGEABLE_DAYS:Q", bin=alt.Bin(maxbins=20), title="Chargeable Days"),
                            y=alt.Y("count()", title="Shipments")).properties(title="POL Demurrage Days", height=230),
                    use_container_width=True)
        with col2:
            dem_data = fdf.loc[fdf["POD_DEM_COST"] > 0, ["POD_DEM_CHARGEABLE_DAYS"]].copy()
            if len(dem_data) > 0:
                st.altair_chart(
                    alt.Chart(dem_data).mark_bar(color=DEM_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(x=alt.X("POD_DEM_CHARGEABLE_DAYS:Q", bin=alt.Bin(maxbins=20), title="Chargeable Days"),
                            y=alt.Y("count()", title="Shipments")).properties(title="POD Demurrage Days", height=230),
                    use_container_width=True)
        with col3:
            det_data = fdf.loc[fdf["POD_DET_COST"] > 0, ["POD_DET_CHARGEABLE_DAYS"]].copy()
            if len(det_data) > 0:
                st.altair_chart(
                    alt.Chart(det_data).mark_bar(color=DET_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(x=alt.X("POD_DET_CHARGEABLE_DAYS:Q", bin=alt.Bin(maxbins=20), title="Chargeable Days"),
                            y=alt.Y("count()", title="Shipments")).properties(title="POD Detention Days", height=230),
                    use_container_width=True)

# -----------------------------------------------------------------------------
# CONTRACT GAPS
# -----------------------------------------------------------------------------
with tab_gaps:
    st.markdown("### ⚠️ Contract Gaps")
    st.caption("Shipments that did not match any contract row, so fees are not calculated. "
               "Containers with above-average dwell are flagged as potential risk.")
    if ufdf.empty:
        st.success("No unmatched shipments found for the selected filters. All visible non-cancelled shipments matched a contract row.")
    else:
        gap_source = fill_grouping_blanks(ufdf)
        risk_df = gap_source[gap_source["RISK_FLAG"] == True].copy()
        active_no_cer = gap_source[gap_source["DET_ACCUMULATING"] == True].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unmatched Shipments", f"{len(gap_source):,}")
        c2.metric("Risk Containers", f"{len(risk_df):,}", "above avg dwell")
        c3.metric("Missing Keys", f"{gap_source['MATCH_KEY'].nunique():,}")
        c4.metric("Active, No CER", f"{len(active_no_cer):,}", "detention may grow")

        st.markdown("---")
        st.markdown("#### Missing Contract Combinations")
        combo = (
            gap_source.groupby(["POD_LOCODE", "CARRIER_FFW_SCAC", "MATCHED_PARTY_TYPE", "CARRIER_SCAC", "FFW_SCAC", "POL_LOCODE", "MATCH_KEY"], dropna=False)
            .agg(Shipments=("SHIPMENT_ID", "count"), Containers=("CONTAINER_NUMBER", lambda x: x.nunique()),
                 Risk_Containers=("RISK_FLAG", lambda x: int(x.sum())),
                 Avg_POL_Dem_Days=("POL_DEM_TOTAL_DAYS", "mean"), Avg_POD_Dem_Days=("POD_DEM_TOTAL_DAYS", "mean"),
                 Avg_POD_Det_Days=("POD_DET_TOTAL_DAYS", "mean"), Max_POD_Det_Days=("POD_DET_TOTAL_DAYS", "max"),
                 Active_No_CER=("DET_ACCUMULATING", lambda x: int(x.sum())))
            .reset_index().sort_values(["Risk_Containers", "Shipments"], ascending=False)
        )
        st.dataframe(
            combo.style.format({"Avg_POL_Dem_Days": "{:.1f}", "Avg_POD_Dem_Days": "{:.1f}",
                                "Avg_POD_Det_Days": "{:.1f}", "Max_POD_Det_Days": "{:.1f}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("---")
        st.markdown("#### Container-Level Contract Gap Risk")
        gap_cols = ["CONTAINER_NUMBER", "SHIPMENT_ID", "CARRIER_SCAC", "FFW_SCAC", "CARRIER_FFW_SCAC",
                    "MATCHED_PARTY_TYPE", "LANE", "CGI", "CLL", "CDD", "CGO", "CER",
                    "POL_DEM_TOTAL_DAYS", "POD_DEM_TOTAL_DAYS", "POD_DET_TOTAL_DAYS", "DET_ACCUMULATING",
                    "RISK_SCORE", "RISK_REASONS", "MATCH_KEY", "DATA_LIMITATION"]
        gap_show = gap_source[[c for c in gap_cols if c in gap_source.columns]].sort_values(
            ["RISK_SCORE", "POD_DET_TOTAL_DAYS", "POD_DEM_TOTAL_DAYS", "POL_DEM_TOTAL_DAYS"], ascending=False).copy()
        for dc in ["CGI", "CLL", "CDD", "CGO", "CER"]:
            if dc in gap_show.columns:
                gap_show[dc] = pd.to_datetime(gap_show[dc], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")
        st.dataframe(gap_show, use_container_width=True, hide_index=True, height=520)

# -----------------------------------------------------------------------------
# DOWNLOAD
# -----------------------------------------------------------------------------
with tab_download:
    st.markdown("### 📥 Download Results")
    st.markdown("Download matched priced shipments, unmatched contract-gap shipments, and the contract used.")

    matched_dl = build_download_df(fdf) if not fdf.empty else pd.DataFrame()
    unmatched_dl = build_unmatched_download_df(unmatched_df) if not unmatched_df.empty else pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matched / Priced Shipments")
        st.markdown(f"**Rows:** {len(matched_dl)} | **Columns:** {len(matched_dl.columns)}")
        if not matched_dl.empty:
            st.dataframe(matched_dl.head(10), use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Matched Results CSV",
                data=matched_dl.to_csv(index=False).encode("utf-8"),
                file_name=f"DD_Academy_Matched_Results_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
            )
    with c2:
        st.markdown("#### Contract Gaps / Unmatched Shipments")
        st.markdown(f"**Rows:** {len(unmatched_dl)} | **Columns:** {len(unmatched_dl.columns)}")
        if not unmatched_dl.empty:
            st.dataframe(unmatched_dl.head(10), use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Contract Gaps CSV",
                data=unmatched_dl.to_csv(index=False).encode("utf-8"),
                file_name=f"DD_Academy_Contract_Gaps_{datetime.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown("#### Excel Workbook")
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            if not matched_dl.empty:
                matched_dl.to_excel(writer, sheet_name="Matched Results", index=False)
            if not unmatched_dl.empty:
                unmatched_dl.to_excel(writer, sheet_name="Contract Gaps", index=False)

            summary_data = {
                "Metric": ["Total Shipments in Upload", "Cancelled Excluded", "Matched / Priced Shipments",
                           "Unmatched Shipments", "Unmatched Risk Containers", "Total D&D Cost",
                           "POL Demurrage", "POD Demurrage", "POD Detention", "Detention Accumulating", "Analysis Date"],
                "Value": [
                    total_shipments, cancelled_count, len(rdf), len(unmatched_df),
                    int(unmatched_df["RISK_FLAG"].sum()) if not unmatched_df.empty else 0,
                    f"${rdf['TOTAL_DD_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POL_DEM_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POD_DEM_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    f"${rdf['POD_DET_COST'].sum():,.2f}" if not rdf.empty else "$0.00",
                    int(rdf["DET_ACCUMULATING"].sum()) if not rdf.empty else 0,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                ],
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            if contracts_df is not None:
                contracts_df.to_excel(writer, sheet_name="Contract", index=False)
            elif estimate_profile is not None:
                pd.DataFrame([{
                    "Demurrage Rate (USD/day, POL & POD)": estimate_profile.get("dem_rate"),
                    "Detention Rate (USD/day)": estimate_profile.get("det_rate"),
                    "Free Demurrage Days": estimate_profile.get("dem_free"),
                    "Free Detention Days": estimate_profile.get("det_free"),
                    "Combined Free Days": estimate_profile.get("combined_free"),
                }]).to_excel(writer, sheet_name="Estimate Rates", index=False)

        st.download_button(
            label="📥 Download Excel Workbook",
            data=buffer.getvalue(),
            file_name=f"DD_Academy_Analyzer_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.info("Excel download requires openpyxl. Add openpyxl to requirements.txt, or use the CSV downloads above.")
