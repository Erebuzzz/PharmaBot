"""
OpenFDA API client with in-memory TTL caching.
Provides FDA-approved drug labeling data — the highest-confidence source.
"""

import time
import logging
import requests
from typing import Optional
from urllib.parse import quote_plus
from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, dict]] = {}


def _cache_get(key: str) -> Optional[dict]:
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < settings.OPENFDA_CACHE_TTL:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val: dict):
    _cache[key] = (time.time(), val)


# ---------------------------------------------------------------------------
# FDA label fields we care about
# ---------------------------------------------------------------------------
_LABEL_FIELDS = [
    "indications_and_usage",
    "dosage_and_administration",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "contraindications",
    "mechanism_of_action",
    "description",
    "clinical_pharmacology",
    "use_in_specific_populations",
    "overdosage",
    "how_supplied",
]


def _first(lst: list | None) -> str:
    """Return first element of a list-field or empty string."""
    if lst and isinstance(lst, list):
        return lst[0]
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_drug(name: str) -> Optional[dict]:
    """
    Search OpenFDA drug/label endpoint by brand OR generic name.
    Returns a structured dict with label sections, or None if not found.
    """
    name = name.strip()
    if not name:
        return None

    cache_key = f"drug:{name.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Search both brand_name and generic_name with OR
    safe_name = quote_plus(name)
    url = (
        f"{settings.OPENFDA_BASE_URL}/drug/label.json"
        f"?search=openfda.brand_name:\"{safe_name}\""
        f"+openfda.generic_name:\"{safe_name}\""
        f"&limit=3"
    )

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            # Not found — cache the miss too (shorter TTL)
            _cache[cache_key] = (time.time(), None)  # type: ignore
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("OpenFDA request failed for '%s': %s", name, e)
        return None

    results = data.get("results", [])
    if not results:
        _cache[cache_key] = (time.time(), None)  # type: ignore
        return None

    # Pick the result with the most populated fields
    best = max(results, key=lambda r: sum(1 for f in _LABEL_FIELDS if f in r))

    info = _extract_label(best)
    _cache_set(cache_key, info)
    return info


def _extract_label(result: dict) -> dict:
    """Pull structured label sections from a raw FDA result."""
    openfda = result.get("openfda", {})

    info: dict = {
        "source": "fda",
        "brand_name": ", ".join(openfda.get("brand_name", [])),
        "generic_name": ", ".join(openfda.get("generic_name", [])),
        "manufacturer": ", ".join(openfda.get("manufacturer_name", [])),
        "route": ", ".join(openfda.get("route", [])),
        "substance_name": ", ".join(openfda.get("substance_name", [])),
        "product_type": ", ".join(openfda.get("product_type", [])),
        "pharm_class": openfda.get("pharm_class_epc", []),
    }

    for field in _LABEL_FIELDS:
        info[field] = _first(result.get(field))

    return info


def format_fda_context(info: dict) -> str:
    """Format FDA label info into a text block usable as RAG context."""
    if not info:
        return ""

    lines = ["[SOURCE: FDA-Approved Drug Label]"]
    if info.get("brand_name"):
        lines.append(f"Brand Name: {info['brand_name']}")
    if info.get("generic_name"):
        lines.append(f"Generic Name: {info['generic_name']}")
    if info.get("manufacturer"):
        lines.append(f"Manufacturer: {info['manufacturer']}")
    if info.get("route"):
        lines.append(f"Route: {info['route']}")
    if info.get("substance_name"):
        lines.append(f"Active Substance: {info['substance_name']}")
    if info.get("pharm_class"):
        lines.append(f"Pharmacological Class: {', '.join(info['pharm_class'])}")

    for field in _LABEL_FIELDS:
        val = info.get(field, "")
        if val:
            heading = field.replace("_", " ").title()
            # Truncate very long sections to keep context focused
            if len(val) > 2000:
                val = val[:2000] + " …[truncated]"
            lines.append(f"\n### {heading}\n{val}")

    return "\n".join(lines)


def search_drug_events(name: str, limit: int = 5) -> list[dict]:
    """
    Query the FDA adverse-event endpoint for a drug.
    Returns a list of event summaries.
    """
    name = name.strip()
    if not name:
        return []

    cache_key = f"events:{name.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached  # type: ignore

    safe_name = quote_plus(name)
    url = (
        f"{settings.OPENFDA_BASE_URL}/drug/event.json"
        f"?search=patient.drug.openfda.brand_name:\"{safe_name}\""
        f"+patient.drug.openfda.generic_name:\"{safe_name}\""
        f"&limit={limit}"
    )

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("OpenFDA events request failed for '%s': %s", name, e)
        return []

    events = []
    for r in data.get("results", []):
        reactions = [
            rx.get("reactionmeddrapt", "")
            for rx in r.get("patient", {}).get("reaction", [])
        ]
        events.append({
            "serious": r.get("serious", ""),
            "reactions": reactions,
            "outcome": r.get("patient", {}).get("patientonsetage", ""),
        })

    _cache_set(cache_key, events)  # type: ignore
    return events
