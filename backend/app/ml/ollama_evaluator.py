"""Offline LLM evaluation of live hazard telemetry via a local Ollama server.

The physics/ML engine remains the source of truth for the alert level; the LLM
only provides an independent second opinion and a plain-English explanation.
Config via env: OLLAMA_URL (default http://127.0.0.1:11434), OLLAMA_MODEL.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict

logger = logging.getLogger("nepal_mhews.ollama")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def ollama_status() -> Dict[str, Any]:
    """Return reachability and installed models."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return {"reachable": True, "url": OLLAMA_URL, "models": models, "selected": OLLAMA_MODEL}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"reachable": False, "url": OLLAMA_URL, "error": str(exc)}


def _compact(obj: Any, max_list: int = 6) -> Any:
    """Shrink long numeric lists (e.g. 72 hourly values) to keep the prompt short and fast."""
    if isinstance(obj, dict):
        return {k: _compact(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_list and all(isinstance(x, (int, float)) for x in obj):
            return {"n": len(obj), "min": min(obj), "max": max(obj), "sum": round(sum(obj), 2)}
        return [_compact(x, max_list) for x in obj]
    if isinstance(obj, float):
        return round(obj, 3)
    return obj


def _key_facts(t: Dict[str, Any]) -> str:
    """Distil the full alert into a few short lines (CPU inference is slow on long prompts)."""
    pd = t.get("precursor_details", {})
    hy = pd.get("hydromet_saturation", {})
    sar = pd.get("sar_creep", {})
    ro = t.get("runout_summary") or {}
    return (
        f"alert={t.get('alert_level')} dynFS={t.get('dynamic_fs')} staticFS={t.get('static_fs')} "
        f"ml_risk={round(float(t.get('ml_susceptibility_score', 0)), 2)}; "
        f"triggers={','.join(t.get('active_triggers', [])) or 'none'}; "
        f"rain72h={hy.get('cumulative_rain_72h_mm')}mm snowmelt={hy.get('snowmelt_mm_we')}mm "
        f"creep={sar.get('latest_velocity_mm_month')}mm/mo lake_growth={pd.get('glacial_lake_expansion_pct')}% "
        f"quake_M={pd.get('seismic', {}).get('max_magnitude')}; "
        f"runout={ro.get('runout_length_m', 'n/a')}m"
    )


def evaluate_with_ollama(telemetry: Dict[str, Any], timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT_S", "240"))) -> Dict[str, Any]:
    """Ask the local model to review a computed hazard assessment (JSON output)."""
    facts = _key_facts(telemetry)
    logger.info("Ollama evaluate starting for model '%s'. Facts payload: %s", OLLAMA_MODEL, facts)
    prompt = (
        "Himalayan hazard analyst. FS<1 failure, 1-1.25 marginal, >=1.25 stable. "
        "Reply ONLY JSON: {\"agrees_with_alert\":bool,\"suggested_level\":\"GREEN|YELLOW|ORANGE|RED\","
        "\"explanation\":\"max 2 short sentences\",\"recommended_actions\":[\"max 3 short items\"]}\n"
        f"DATA: {facts}"
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.1, "num_predict": 170},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = json.loads(r.read()).get("response", "")
        logger.info("Ollama raw generation completed (length: %d chars). Parsing response...", len(raw))
        review = _parse_review(raw)
        logger.info("Ollama review parsed successfully: level=%s, agrees=%s", review.get("suggested_level"), review.get("agrees_with_alert"))
        return review
    except urllib.error.URLError as exc:
        logger.error("Ollama network error connecting to %s: %s", OLLAMA_URL, exc)
        raise
    except Exception as exc:
        logger.error("Ollama evaluation unexpected failure: %s", exc, exc_info=True)
        raise


def explain_live_weather_and_factors(
    station_name: str,
    live_hydro: Dict[str, Any],
    slope_deg: float,
    elevation_m: float,
    timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT_S", "180")),
) -> Dict[str, Any]:
    """Ask local Ollama to provide a geotechnical interpretation of live rain and thermal factors."""
    rain_72h = live_hydro.get("cumulative_rain_72h_mm", 0.0)
    temp = live_hydro.get("mean_temp_c", 0.0)
    pdd = live_hydro.get("pdd_c_days", 0.0)
    
    prompt = (
        f"Himalayan geotechnical analyst. Explain current weather risk in 2 short sentences.\n"
        f"Station: {station_name} (Elev: {elevation_m}m, Slope: {slope_deg}°).\n"
        f"Live observations: 72h Rain: {rain_72h}mm, Mean Temp: {temp}°C, Melt PDD: {pdd}C-days.\n"
        f"Thresholds: Rain>120mm high landslide trigger, Temp>0C above 5000m causes permafrost thaw/snowmelt.\n"
        f"Reply ONLY JSON: {{\"risk_severity\":\"LOW|MODERATE|HIGH\",\"summary\":\"2 sentences max\",\"operational_advice\":\"1 short sentence\"}}"
    )
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.1, "num_predict": 120},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        raw = json.loads(r.read()).get("response", "")
    try:
        return json.loads(raw)
    except Exception:
        import re
        out = {"risk_severity": "MODERATE", "summary": raw.strip()}
        m = re.search(r'"risk_severity"\s*:\s*"(LOW|MODERATE|HIGH)"', raw)
        if m:
            out["risk_severity"] = m.group(1)
        m_sum = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        if m_sum:
            out["summary"] = m_sum.group(1)
        m_adv = re.search(r'"operational_advice"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
        if m_adv:
            out["operational_advice"] = m_adv.group(1)
        return out


def _parse_review(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON; salvage fields from truncated output (token cap)."""
    import re

    try:
        return json.loads(raw)
    except ValueError:
        pass
    out: Dict[str, Any] = {"agrees_with_alert": None, "truncated": True}
    m = re.search(r'"agrees_with_alert"\s*:\s*(true|false)', raw)
    if m:
        out["agrees_with_alert"] = m.group(1) == "true"
    m = re.search(r'"suggested_level"\s*:\s*"(GREEN|YELLOW|ORANGE|RED)"', raw)
    if m:
        out["suggested_level"] = m.group(1)
    m = re.search(r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)', raw)
    if m:
        out["explanation"] = m.group(1)
    acts = re.findall(r'"((?:[^"\\]|\\.)+)"\s*[,\]]', raw.split('"recommended_actions"')[-1]) if '"recommended_actions"' in raw else []
    out["recommended_actions"] = acts
    if "explanation" not in out:
        out["explanation"] = raw.strip()
    return out

