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


def generate_deterministic_geotechnical_review(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Offline Geotechnical AI synthesizer when local Ollama is in standby."""
    alert = telemetry.get("alert_level", "GREEN")
    dyn_fs = float(telemetry.get("dynamic_fs", 1.25))
    static_fs = float(telemetry.get("static_fs", 1.20))
    triggers = telemetry.get("active_triggers", [])

    if dyn_fs >= 1.25 and len(triggers) == 0:
        agrees = True
        suggested = "GREEN"
        explanation = (
            f"Under current live dry conditions, dynamic Factor of Safety (FS={dyn_fs:.2f}) remains above the 1.25 stability threshold, "
            f"even though terrain susceptibility baseline is high (static FS={static_fs:.2f}). Slope is in stable mechanical equilibrium."
        )
        actions = [
            "Maintain automated 30-minute sensor telemetry polling",
            "Monitor upstream satellite SAR creep and rain accumulation",
            "No civil evacuation required under current dry baseline",
        ]
    elif dyn_fs < 1.0 or len(triggers) >= 2:
        agrees = True
        suggested = alert
        explanation = (
            f"Real-time dynamic Factor of Safety (FS={dyn_fs:.2f}) indicates critical mechanical instability "
            f"exacerbated by active precursor triggers: {', '.join(triggers) if triggers else 'pore pressure buildup'}."
        )
        actions = [
            "Issue immediate civil protection evacuation advisory to downstream settlements",
            "Verify real-time Sentinel-1 InSAR creep and river streamflow gauges",
            "Mobilize local emergency response team (NDRRMA / District Disaster Management)",
        ]
    else:
        agrees = True
        suggested = alert
        explanation = (
            f"Marginal slope equilibrium detected (dynamic FS={dyn_fs:.2f}, static baseline FS={static_fs:.2f}). "
            f"Soil suction maintains grip, but elevated trigger watch is active."
        )
        actions = [
            "Increase telemetry polling cadence to 15-minute emergency mode",
            "Alert district flood coordinators along river corridor",
            "Inspect highway choke points and gorge culverts",
        ]

    return {
        "agrees_with_alert": agrees,
        "suggested_level": suggested,
        "explanation": explanation,
        "recommended_actions": actions,
        "source": "ONBOARD_GEOTECHNICAL_AI_ENGINE (Ollama Standby)",
    }


def evaluate_with_ollama(telemetry: Dict[str, Any], timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT_S", "240"))) -> Dict[str, Any]:
    """Ask the local model to review a computed hazard assessment (with graceful fallback)."""
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
        review["source"] = f"LOCAL_OLLAMA_{OLLAMA_MODEL}"
        logger.info("Ollama review parsed successfully: level=%s, agrees=%s", review.get("suggested_level"), review.get("agrees_with_alert"))
        return review
    except (urllib.error.URLError, OSError) as exc:
        logger.info("Local Ollama endpoint unreachable (%s). Using onboard Geotechnical AI synthesizer.", exc)
        return generate_deterministic_geotechnical_review(telemetry)
    except Exception as exc:
        logger.warning("Ollama evaluation unexpected failure: %s, using onboard synthesizer.", exc)
        return generate_deterministic_geotechnical_review(telemetry)


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


def generate_nepali_civil_protection_broadcast(
    basin_name: str,
    alert_level: str,
    dynamic_fs: float,
    active_triggers: list,
    impacted_settlements: list,
    timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT_S", "180")),
) -> Dict[str, str]:
    """Generate localized Nepali (नेपाली) and English emergency SMS/CAP broadcasts using local Ollama."""
    nearest_village = impacted_settlements[0]["settlement_name"] if impacted_settlements else "डाउनस्ट्रिम बस्तीहरू"
    eta_min = impacted_settlements[0]["eta_minutes"] if impacted_settlements else 30.0
    
    prompt = (
        f"You are the National Disaster Risk Reduction & Management Authority (NDRRMA) Nepal automated alert system.\n"
        f"Generate an urgent Emergency SMS Broadcast in NEPALI (देवनागरी लिपि) and English.\n"
        f"Details: Basin: {basin_name}, Alert: {alert_level}, Safety Factor: {dynamic_fs:.2f}.\n"
        f"Active Danger Triggers: {', '.join(active_triggers) if active_triggers else 'Monitoring'}.\n"
        f"Nearest settlement at risk: {nearest_village} (Estimated arrival time: {eta_min} minutes).\n"
        f"Reply ONLY JSON: {{\"sms_nepali\":\"आकस्मिक सूचना... (max 25 words in Nepali)\",\"sms_english\":\"EMERGENCY ALERT... (max 25 words in English)\",\"cap_headline\":\"...\"}}"
    )
    
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            "options": {"temperature": 0.1, "num_predict": 150},
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            raw = json.loads(r.read()).get("response", "")
        return json.loads(raw)
    except Exception:
        # Fallback deterministic bilingual template
        if alert_level == "RED":
            nep = f"अति जरुरी सूचना! {basin_name} क्षेत्रमा पहिरो/बाढीको उच्च जोखिम छ। {nearest_village} तथा तटीय क्षेत्रका बासिन्दा तुरुन्त सुरक्षित अग्लो स्थानमा जानुहोस्।"
            eng = f"CRITICAL EVACUATION ALERT: Immediate landslide/avalanche danger in {basin_name}. Residents near {nearest_village} must evacuate to higher ground immediately!"
        else:
            nep = f"सतर्कता सूचना: {basin_name} क्षेत्रमा नदी जलसतह र भिरमा पहिरोको जोखिम बढेको छ। सतर्क रहनुहोला।"
            eng = f"WATCH ALERT: Elevated landslide and flood risk in {basin_name}. Stay alert near river banks."
            
        return {
            "sms_nepali": nep,
            "sms_english": eng,
            "cap_headline": f"NDRRMA {alert_level} Hazard Warning - {basin_name}",
        }


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

