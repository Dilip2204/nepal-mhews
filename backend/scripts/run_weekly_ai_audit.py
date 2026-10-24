#!/usr/bin/env python3
"""Weekly AI Geohazard Audit & Continuous Calibration Script.

Runs once a week (cron, Sunday 02:00 UTC) so the 30-minute perpetual engine stays at zero LLM
cost during the week. Four pillars:

  1. NLP extraction of ground-truth landslide/GLOF events from staged disaster bulletins.
  2. Multimodal before/after screening of staged satellite thumbnail pairs.
  3. Residual-driven 7-factor equation refinement, gated by the Rule 8.4 validation gate.
  4. Executive markdown report + persisted audit run record.

Engines: ``deterministic`` (offline, zero API cost), ``ollama`` (local, zero API cost),
``gemini``, ``claude``, ``dual`` (Gemini **and** Claude, independent then cross-validated),
and ``auto`` which prefers ``dual`` when both keys are present.

DUAL CONSENSUS is the admission gate for the training catalog. Each model sees the same item
with no knowledge of the other's answer. Only when the two independently agree is a row written
to ``live_ground_truth_feedback.csv``; disagreements are quarantined as NEEDS_HUMAN_REVIEW.
This exists because unvalidated rows previously contaminated the ML training catalog
(HANDOFF.md Issue #4) and the system is about to run unattended for six months.

The physics engine remains the sole authority on alert levels. Models here only extract,
classify, review and propose.
"""
import os

# Enforce thread caps on shared 256-core server before importing numpy/torch
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
import argparse
import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

from app.core.config import settings
from app.core.database import get_time_series_store, BaseTimeSeriesStore
from app.ingestion.dem_processor import fetch_and_process_copernicus_dem_patch
from app.ingestion.ground_truth_feedback import (
    append_ground_truth_event,
    load_live_ground_truth_events,
)
from app.ingestion.unstructured_collector import (
    stage_unprocessed_bulletins,
    stage_satellite_visual_candidates,
    NEPAL_DISTRICTS,
    THUMBNAILS_DIR,
)
from app.ml.continuous_equation_optimizer import (
    ContinuousEquationOptimizer,
    ACTIVE_PARAMS_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nepal_mhews.weekly_ai_audit")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUDIT_REPORTS_DIR = DATA_DIR / "audit_reports"
AUDIT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Engine identifiers -----------------------------------------------------------------
ENGINE_DUAL = "DUAL_CONSENSUS"
ENGINE_GEMINI = "GEMINI_FLASH"
ENGINE_CLAUDE = "CLAUDE"
ENGINE_OLLAMA = "LOCAL_OLLAMA"
ENGINE_DETERMINISTIC = "DETERMINISTIC_GEOTECHNICAL_AI"

_GEMINI_ALIASES = {"GEMINI", "GEMINI_FLASH"}
_CLAUDE_ALIASES = {"CLAUDE", "ANTHROPIC"}
_DUAL_ALIASES = {"DUAL", "DUAL_CONSENSUS"}

# --- Model API constants ----------------------------------------------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Claude Opus 5 list price, USD per million tokens (input / output).
CLAUDE_PRICE_IN, CLAUDE_PRICE_OUT = 5.0, 25.0
# Gemini Flash rate-card ESTIMATE. Re-check against Google's published pricing before treating
# the reported cost as authoritative; the report labels it "estimated".
GEMINI_PRICE_IN, GEMINI_PRICE_OUT = 0.15, 0.60
CLAUDE_MAX_TOKENS = 16000
GEMINI_MAX_OUTPUT_TOKENS = 4096

# --- Consensus tuning -------------------------------------------------------------------
# Two independent extractions describe "the same event" when the hazard type matches and the
# coordinates fall within this distance (~5.5 km at Nepal's latitude).
BULLETIN_COORD_TOLERANCE_DEG = 0.05
# A candidate equation must beat the champion by more than this to be promoted. Equal fitness
# previously logged "Promoted! (0.9311 > 0.9311)" and overwrote the champion with a no-op.
PROMOTION_FITNESS_EPSILON = 1e-4

# --- Deterministic vision thresholds ----------------------------------------------------
# Calibrated against the measured output of
# unstructured_collector.generate_synthetic_thumbnail_pair() over the full 512x512 frame:
#   true-scar pair -> changed 14.5%, white-cloud 0.0%, bare-earth 14.4%, masked dR-dG = 72.5
#   cloud pair     -> changed 22.7%, white-cloud 11.6%, bare-earth 0.0%, darkened 8.3%
# The original thresholds (unmasked dR-dG > 12.0 against an actual 10.5) sat just above every
# real value, so every candidate -- including true scars -- fell through to SNOWMELT_RETREAT.
VISION_CHANGE_FRAC_MIN = 0.005
VISION_WHITE_CLOUD_FRAC_MIN = 0.03
VISION_BARE_EARTH_FRAC_MIN = 0.02
VISION_BARE_EARTH_INDEX_MIN = 25.0
VISION_BRIGHT_FRAC_MIN = 0.02
VISION_DARKENED_FRAC_MIN = 0.02
VISION_PIXEL_CHANGE_SUM = 20.0
VISION_DARKEN_DELTA = 12.0
VISION_MAX_CONFIDENCE = 0.90

VISION_LABELS = (
    "TRUE_LANDSLIDE_SCAR",
    "CLOUD_SHADOW_ARTIFACT",
    "SNOWMELT_RETREAT",
    "SENSOR_NOISE",
)

# Approximate geographical centroids for Nepal's high-risk mountain districts
DISTRICT_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    # District: (lat, lon, approx_elev_m)
    "Rasuwa": (28.152, 85.321, 2150.0),
    "Dhading": (27.850, 85.000, 1420.0),
    "Sindhupalchok": (27.945, 85.850, 1850.0),
    "Kavrepalanchok": (27.540, 85.650, 1320.0),
    "Gorkha": (28.320, 84.750, 2450.0),
    "Solukhumbu": (27.780, 86.720, 3600.0),
    "Kaski": (28.300, 83.980, 1750.0),
    "Myagdi": (28.450, 83.550, 1920.0),
    "Dolakha": (27.780, 86.150, 2250.0),
    "Manang": (28.660, 84.020, 3540.0),
    "Mustang": (28.820, 83.750, 3720.0),
    "Lamjung": (28.250, 84.400, 1680.0),
    "Sankhuwasabha": (27.600, 87.250, 2300.0),
    "Taplejung": (27.500, 87.750, 2400.0),
    "Chitwan": (27.600, 84.450, 350.0),
    "Makwanpur": (27.450, 85.100, 1100.0),
}

# --- JSON schemas shared by both providers ----------------------------------------------
BULLETIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_geohazard_event": {"type": "boolean"},
        "event_name": {"type": "string"},
        "hazard_type": {
            "type": "string",
            "enum": [
                "monsoon_landslide",
                "monsoon_cloudburst_landslide",
                "rockfall",
                "debris_flow",
                "glof",
                "rock_ice_avalanche",
                "not_a_geohazard",
            ],
        },
        "district": {"type": "string"},
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
        "elevation_m": {"type": "number"},
        "slope_deg": {"type": "number"},
        "rainfall_72h_mm": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": [
        "is_geohazard_event",
        "event_name",
        "hazard_type",
        "district",
        "latitude",
        "longitude",
        "elevation_m",
        "slope_deg",
        "rainfall_72h_mm",
        "confidence",
    ],
    "additionalProperties": False,
}

VISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_true_scar": {"type": "boolean"},
        "classification": {"type": "string", "enum": list(VISION_LABELS)},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
    },
    "required": ["is_true_scar", "classification", "confidence", "evidence"],
    "additionalProperties": False,
}

EQUATION_PARAM_KEYS = (
    "c_res_base",
    "w_cryo_plateau",
    "w_lowland_flat",
    "w_sti",
    "w_ommi",
    "w_gii",
    "w_s_perp",
    "w_tcwi",
    "w_fault_dmg",
    "prob_sigmoid_steepness",
)

EQUATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in EQUATION_PARAM_KEYS},
    "required": list(EQUATION_PARAM_KEYS),
    "additionalProperties": False,
}


@dataclass
class ModelVote:
    """One model's independent opinion on one item.

    ``status`` is ``OK`` when the model returned a usable payload, ``ABSTAINED`` when it
    errored, refused, was unavailable or returned unparseable output. An abstention is never
    fatal -- the dual design simply proceeds with fewer votes.
    """

    model: str
    status: str
    payload: Optional[Dict[str, Any]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "payload": self.payload,
            "reason": self.reason,
        }


@dataclass
class ConsensusDecision:
    """Outcome of cross-validating independent model votes."""

    status: str  # AGREED | NEEDS_HUMAN_REVIEW | SINGLE_VOTE | NO_VOTES | SINGLE_ENGINE
    confidence_tier: str  # HIGH | LOW | NONE
    admit: bool  # may this become a training label?
    payload: Optional[Dict[str, Any]]
    votes: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_status": self.status,
            "consensus_confidence": self.confidence_tier,
            "admitted_to_training_catalog": self.admit,
            "votes": self.votes,
            "note": self.note,
        }


@dataclass
class TokenAccountant:
    """Tracks real API token consumption and cost, per engine.

    Only genuine remote API calls go through :meth:`add`. Deterministic and local Ollama
    inference go through :meth:`record_local`, which records zero API tokens and zero cost --
    the deterministic engine previously reported ~2,965 fictitious tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cost_estimate_usd: float = 0.0
    per_engine: Dict[str, Dict[str, float]] = field(default_factory=dict)
    local_calls: Dict[str, int] = field(default_factory=dict)

    def add(
        self,
        inp: int,
        out: int,
        cost_per_m_in: float = 0.15,
        cost_per_m_out: float = 0.60,
        engine: str = "unspecified",
    ) -> None:
        """Record one billable API call using the token counts the provider reported."""
        inp = max(0, int(inp or 0))
        out = max(0, int(out or 0))
        cost = (inp / 1_000_000.0) * cost_per_m_in + (out / 1_000_000.0) * cost_per_m_out
        self.input_tokens += inp
        self.output_tokens += out
        self.api_calls += 1
        self.cost_estimate_usd += cost

        bucket = self.per_engine.setdefault(
            engine, {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cost_usd": 0.0}
        )
        bucket["input_tokens"] += inp
        bucket["output_tokens"] += out
        bucket["api_calls"] += 1
        bucket["cost_usd"] += cost

    def record_local(self, engine: str) -> None:
        """Record one local/offline inference: zero API tokens, zero cost."""
        self.local_calls[engine] = self.local_calls.get(engine, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "api_calls_count": self.api_calls,
            "estimated_cost_usd": round(self.cost_estimate_usd, 5),
            "by_engine": {
                k: {
                    "input_tokens": int(v["input_tokens"]),
                    "output_tokens": int(v["output_tokens"]),
                    "api_calls": int(v["api_calls"]),
                    "cost_usd": round(v["cost_usd"], 5),
                }
                for k, v in sorted(self.per_engine.items())
            },
            "local_inference_calls": dict(sorted(self.local_calls.items())),
        }


# =============================================================================
# PURE HELPERS (unit-testable without a pipeline instance)
# =============================================================================
def _normalize_hazard(value: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().upper()


def bulletin_votes_agree(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """Two bulletin extractions agree on hazard type AND coordinates within tolerance."""
    a_event = bool(a.get("is_geohazard_event"))
    b_event = bool(b.get("is_geohazard_event"))
    if a_event != b_event:
        return False, f"is_geohazard_event mismatch ({a_event} vs {b_event})"
    if not a_event:
        return True, "both models independently judged this bulletin to be a non-event"

    ha, hb = _normalize_hazard(a.get("hazard_type")), _normalize_hazard(b.get("hazard_type"))
    if ha != hb:
        return False, f"hazard type mismatch ('{ha}' vs '{hb}')"

    try:
        d_lat = abs(float(a["latitude"]) - float(b["latitude"]))
        d_lon = abs(float(a["longitude"]) - float(b["longitude"]))
    except (KeyError, TypeError, ValueError):
        return False, "at least one model returned unusable coordinates"

    if d_lat > BULLETIN_COORD_TOLERANCE_DEG or d_lon > BULLETIN_COORD_TOLERANCE_DEG:
        return (
            False,
            f"coordinates differ by {d_lat:.3f} deg lat / {d_lon:.3f} deg lon "
            f"(tolerance {BULLETIN_COORD_TOLERANCE_DEG} deg)",
        )
    return (
        True,
        f"hazard type '{ha}' matches and coordinates agree within {max(d_lat, d_lon):.3f} deg",
    )


def merge_bulletin_votes(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two agreeing extractions: mean coordinates, the lower of the two confidences."""
    merged = dict(a)
    if bool(a.get("is_geohazard_event")):
        try:
            merged["latitude"] = round((float(a["latitude"]) + float(b["latitude"])) / 2.0, 5)
            merged["longitude"] = round((float(a["longitude"]) + float(b["longitude"])) / 2.0, 5)
        except (KeyError, TypeError, ValueError):
            pass
    try:
        merged["confidence"] = min(
            float(a.get("confidence", 0.8)), float(b.get("confidence", 0.8))
        )
    except (TypeError, ValueError):
        merged["confidence"] = 0.8
    return merged


def vision_votes_agree(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """Two vision screenings agree only on an identical classification label."""
    la, lb = _normalize_label(a.get("classification")), _normalize_label(b.get("classification"))
    if not la or not lb:
        return False, "at least one model returned no classification label"
    if la != lb:
        return False, f"classification mismatch ('{la}' vs '{lb}')"
    return True, f"both models independently classified the pair as {la}"


def merge_vision_votes(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two agreeing screenings; is_true_scar is derived from the agreed label."""
    label = _normalize_label(a.get("classification"))
    merged = dict(a)
    merged["classification"] = label
    merged["is_true_scar"] = label == "TRUE_LANDSLIDE_SCAR"
    try:
        merged["confidence"] = min(
            float(a.get("confidence", 0.5)), float(b.get("confidence", 0.5))
        )
    except (TypeError, ValueError):
        merged["confidence"] = 0.5
    merged["evidence"] = (
        f"[gemini] {str(a.get('evidence', '')).strip()} "
        f"| [claude] {str(b.get('evidence', '')).strip()}"
    )
    return merged


def cross_validate_votes(
    votes: List[ModelVote],
    agree_fn: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, str]],
    merge_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    require_dual: bool,
) -> ConsensusDecision:
    """Cross-validate independent model votes into an admission decision.

    Agreement -> HIGH confidence, admitted. Disagreement -> quarantined for a human, never
    admitted. A lone surviving vote -> LOW confidence, not admitted while ``require_dual``.
    """
    recorded = [v.to_dict() for v in votes]
    usable = [v for v in votes if v.status == "OK" and isinstance(v.payload, dict)]

    if len(usable) >= 2:
        agreed, why = agree_fn(usable[0].payload or {}, usable[1].payload or {})
        if agreed:
            return ConsensusDecision(
                status="AGREED",
                confidence_tier="HIGH",
                admit=True,
                payload=merge_fn(usable[0].payload or {}, usable[1].payload or {}),
                votes=recorded,
                note=why,
            )
        return ConsensusDecision(
            status="NEEDS_HUMAN_REVIEW",
            confidence_tier="NONE",
            admit=False,
            payload=None,
            votes=recorded,
            note=f"Models disagreed: {why}. Quarantined for human adjudication.",
        )

    if len(usable) == 1:
        abstained = [v.model for v in votes if v.status != "OK"]
        return ConsensusDecision(
            status="SINGLE_VOTE",
            confidence_tier="LOW",
            admit=not require_dual,
            payload=usable[0].payload,
            votes=recorded,
            note=(
                f"Only {usable[0].model} returned a usable answer "
                f"(abstained: {', '.join(abstained) or 'none'}). "
                + (
                    "Dual consensus required, so not admitted."
                    if require_dual
                    else "Admitted under relaxed policy."
                )
            ),
        )

    return ConsensusDecision(
        status="NO_VOTES",
        confidence_tier="NONE",
        admit=False,
        payload=None,
        votes=recorded,
        note="No model returned a usable answer.",
    )


def compute_vision_features(before_path: Path, after_path: Path) -> Dict[str, float]:
    """Measure the before/after pixel statistics the deterministic classifier reasons over."""
    from PIL import Image
    import numpy as np

    before = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.float32)
    after = np.asarray(Image.open(after_path).convert("RGB"), dtype=np.float32)
    if before.shape != after.shape:
        raise ValueError(
            f"Thumbnail pair shape mismatch: {before.shape} vs {after.shape}; cannot difference."
        )

    diff = after - before
    changed = np.abs(diff).sum(axis=-1) > VISION_PIXEL_CHANGE_SUM

    # Chromaticity is measured over the changed footprint only: averaging an unchanged
    # background into the delta is what pushed the old bare-earth index below its threshold.
    if bool(changed.any()):
        delta_r, delta_g, delta_b = (float(diff[changed, i].mean()) for i in range(3))
    else:
        delta_r, delta_g, delta_b = (float(diff[:, :, i].mean()) for i in range(3))

    def _bright_achromatic(arr: "np.ndarray") -> "np.ndarray":
        lo = arr.min(axis=2)
        hi = arr.max(axis=2)
        return (lo > 200.0) & ((hi - lo) < 30.0)

    r, g, b = after[:, :, 0], after[:, :, 1], after[:, :, 2]
    # Exposed regolith: red-dominant, mid-tone, warm-desaturated and never near-white.
    bare_earth = (r > g + 22.0) & (r > b + 22.0) & (r > 60.0) & (r < 225.0)
    # Transient cloud top: saturated white across all three channels.
    white_cloud = (r > 235.0) & (g > 235.0) & (b > 235.0)
    darkened = (after.mean(axis=2) - before.mean(axis=2)) < -VISION_DARKEN_DELTA

    return {
        "delta_r": delta_r,
        "delta_g": delta_g,
        "delta_b": delta_b,
        "bare_earth_index": delta_r - delta_g,
        "brightness_shift": (delta_r + delta_g + delta_b) / 3.0,
        "changed_frac": float(changed.mean()),
        "bare_earth_frac": float(bare_earth.mean()),
        "white_cloud_frac": float(white_cloud.mean()),
        "bright_achromatic_frac_after": float(_bright_achromatic(after).mean()),
        "bright_achromatic_frac_before": float(_bright_achromatic(before).mean()),
        "darkened_frac": float(darkened.mean()),
    }


def classify_vision_features(f: Dict[str, float]) -> Dict[str, Any]:
    """Label a before/after pair from its measured pixel statistics.

    Ordered so the morphologically specific signatures (a saturated cloud top, a coherent
    bare-earth chute) are tested before the weaker residual cases. Confidence is capped at
    0.90: this is a pixel-statistics screen, not a geomorphologist.
    """
    bare = f.get("bare_earth_frac", 0.0)
    white = f.get("white_cloud_frac", 0.0)
    bright_after = f.get("bright_achromatic_frac_after", 0.0)
    bright_before = f.get("bright_achromatic_frac_before", 0.0)
    dark = f.get("darkened_frac", 0.0)
    changed = f.get("changed_frac", 0.0)
    bare_index = f.get("bare_earth_index", 0.0)

    def _result(label: str, scar: bool, conf: float, evidence: str) -> Dict[str, Any]:
        return {
            "is_true_scar": scar,
            "classification": label,
            "confidence": round(min(max(conf, 0.50), VISION_MAX_CONFIDENCE), 2),
            "evidence": evidence,
        }

    stats = (
        f"changed={changed * 100:.1f}%, bare_earth={bare * 100:.1f}%, "
        f"white_cloud={white * 100:.1f}%, darkened={dark * 100:.1f}%, "
        f"masked dR-dG={bare_index:.1f}"
    )

    if changed < VISION_CHANGE_FRAC_MIN:
        return _result(
            "SENSOR_NOISE", False, 0.55,
            f"No coherent change footprint between the two frames ({stats}).",
        )

    # A saturated cloud top dominating the scene; only outvoted by a larger bare-earth area.
    if white >= VISION_WHITE_CLOUD_FRAC_MIN and bare < max(white, VISION_BARE_EARTH_FRAC_MIN):
        return _result(
            "CLOUD_SHADOW_ARTIFACT", False, 0.70 + 1.0 * (white + dark),
            f"Transient saturated cloud top over {white * 100:.1f}% of the frame with no "
            f"competing bare-regolith footprint ({stats}).",
        )

    if bare >= VISION_BARE_EARTH_FRAC_MIN and bare_index > VISION_BARE_EARTH_INDEX_MIN:
        return _result(
            "TRUE_LANDSLIDE_SCAR", True, 0.70 + 1.2 * bare,
            f"Coherent red-dominant bare-regolith footprint covering {bare * 100:.1f}% of the "
            f"frame with a strong masked soil-reflectance shift ({stats}).",
        )

    if (bright_before - bright_after) >= VISION_BRIGHT_FRAC_MIN and bare < VISION_BARE_EARTH_FRAC_MIN:
        return _result(
            "SNOWMELT_RETREAT", False, 0.72,
            f"Bright achromatic snow cover shrank from {bright_before * 100:.1f}% to "
            f"{bright_after * 100:.1f}% with no debris accumulation ({stats}).",
        )

    if dark >= VISION_DARKENED_FRAC_MIN and abs(bare_index) < VISION_BARE_EARTH_INDEX_MIN:
        return _result(
            "CLOUD_SHADOW_ARTIFACT", False, 0.68,
            f"Achromatic terrain darkening over {dark * 100:.1f}% of the frame without a bare "
            f"scarp signature ({stats}).",
        )

    if bright_after >= VISION_BRIGHT_FRAC_MIN:
        return _result(
            "CLOUD_SHADOW_ARTIFACT", False, 0.62,
            f"Bright achromatic mass over {bright_after * 100:.1f}% of the frame with no "
            f"resolvable cast shadow ({stats}).",
        )

    return _result(
        "SENSOR_NOISE", False, 0.55,
        f"Change present but matching no bare-regolith, cloud or snow signature ({stats}).",
    )


def imagery_provenance(path: Optional[Path]) -> str:
    """Report where a thumbnail came from.

    Everything currently under the collector's thumbnails directory is drawn by Pillow, not
    downloaded from Sentinel-2. Labelling it honestly keeps synthetic fixtures out of the real
    training catalog.
    """
    if path is None:
        return "MISSING"
    try:
        if Path(path).resolve().parent == Path(THUMBNAILS_DIR).resolve():
            return "SYNTHETIC_FIXTURE"
    except OSError:
        pass
    return "UNVERIFIED_EXTERNAL"


def json_schema_to_gemini(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a JSON Schema into Gemini's OpenAPI-subset responseSchema dialect."""
    out: Dict[str, Any] = {}
    stype = schema.get("type")
    if stype:
        out["type"] = str(stype).upper()
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if "description" in schema:
        out["description"] = schema["description"]
    if stype == "object":
        props = schema.get("properties", {})
        out["properties"] = {k: json_schema_to_gemini(v) for k, v in props.items()}
        if schema.get("required"):
            out["required"] = list(schema["required"])
        out["propertyOrdering"] = list(props.keys())
    elif stype == "array" and "items" in schema:
        out["items"] = json_schema_to_gemini(schema["items"])
    return out


def parse_json_defensively(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse a model's text output into a dict, tolerating fences and surrounding prose."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


class WeeklyAIAuditPipeline:
    """Weekly data distiller and multi-model geohazard synthesis engine."""

    def __init__(
        self,
        engine: str = "auto",
        api_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        dry_run: bool = False,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        allow_synthetic_training_rows: bool = False,
    ):
        self.store: BaseTimeSeriesStore = get_time_series_store()
        self.optimizer = ContinuousEquationOptimizer()
        self.dry_run = dry_run
        self.allow_synthetic_training_rows = allow_synthetic_training_rows
        self.token_accountant = TokenAccountant()
        self._anthropic_client = None

        requested = (engine or "auto").strip().upper()
        self.gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        if api_key:
            # Legacy single-key flag: route it to whichever single engine was requested.
            if requested in _CLAUDE_ALIASES:
                self.anthropic_key = api_key
            else:
                self.gemini_key = api_key

        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        self.require_dual_consensus = bool(settings.audit_require_dual_consensus)

        if requested == "AUTO":
            if self.gemini_key and self.anthropic_key:
                self.engine = ENGINE_DUAL
            elif self.gemini_key:
                self.engine = ENGINE_GEMINI
            elif self.anthropic_key:
                self.engine = ENGINE_CLAUDE
            elif self._is_ollama_reachable():
                self.engine = ENGINE_OLLAMA
            else:
                self.engine = ENGINE_DETERMINISTIC
        elif requested in _DUAL_ALIASES:
            self.engine = ENGINE_DUAL
        else:
            self.engine = requested

        if self.is_dual and not (self.gemini_key and self.anthropic_key):
            logger.warning(
                "DUAL consensus requested but not both keys are present (gemini=%s, claude=%s). "
                "The missing model abstains; nothing will be admitted to the training catalog "
                "while AUDIT_REQUIRE_DUAL_CONSENSUS is set.",
                bool(self.gemini_key), bool(self.anthropic_key),
            )

        logger.info(
            "Weekly AI Audit Pipeline initialized with Engine: %s "
            "(Dry Run: %s, Dual consensus required: %s)",
            self.engine, self.dry_run, self.require_dual_consensus,
        )

    # ------------------------------------------------------------------
    # Engine predicates
    # ------------------------------------------------------------------
    @property
    def is_dual(self) -> bool:
        return self.engine in _DUAL_ALIASES

    @property
    def uses_gemini(self) -> bool:
        return (self.engine in _GEMINI_ALIASES or self.is_dual) and bool(self.gemini_key)

    @property
    def uses_claude(self) -> bool:
        return (self.engine in _CLAUDE_ALIASES or self.is_dual) and bool(self.anthropic_key)

    @property
    def uses_ollama(self) -> bool:
        return self.engine == ENGINE_OLLAMA and self._is_ollama_reachable()

    def _is_ollama_reachable(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    # =========================================================================
    # PILLAR 1: NLP Field Incident Extraction from Bulletins
    # =========================================================================
    def run_bulletin_nlp_audit(
        self, limit: int = 25
    ) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert unstructured bulletins to GPS ground truth behind the consensus gate."""
        logger.info("[Pillar 1/4] Scanning unprocessed disaster news & field bulletins...")
        pending = self.store.get_pending_bulletins(limit=limit)

        if not pending:
            logger.info("No pending bulletins in queue. Staging live bulletins from open feeds...")
            stage_unprocessed_bulletins(self.store)
            pending = self.store.get_pending_bulletins(limit=limit)

        logger.info("Found %d pending bulletin(s) ready for NLP parsing.", len(pending))
        extracted_events: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []

        for bul in pending:
            bid = bul["bulletin_id"]
            title = bul["title"]
            text = bul["raw_text"]
            district_hint = bul.get("district_hint") or ""

            logger.info("Parsing bulletin [%s]: '%s'...", bid, title[:60])
            decision = self._extract_disaster_consensus(title, text, district_hint)

            if decision.status == "NEEDS_HUMAN_REVIEW":
                logger.warning("--> [QUARANTINED] Bulletin %s: %s", bid, decision.note)
                quarantined.append({
                    "item_id": bid,
                    "pillar": "bulletin_nlp",
                    "title": title[:120],
                    **decision.to_dict(),
                })
                if not self.dry_run:
                    self.store.update_bulletin_status(
                        bid, "NEEDS_HUMAN_REVIEW", parsed_json=decision.to_dict()
                    )
                continue

            parsed = decision.payload
            has_coords = bool(
                parsed
                and parsed.get("is_geohazard_event")
                and parsed.get("latitude") is not None
                and parsed.get("longitude") is not None
            )

            if not has_coords:
                logger.info("Bulletin %s contains no geocoded failure. Marking as SKIPPED.", bid)
                if not self.dry_run:
                    self.store.update_bulletin_status(
                        bid, "SKIPPED_NO_GEOCODE", parsed_json=decision.to_dict()
                    )
                continue

            lat = float(parsed["latitude"])
            lon = float(parsed["longitude"])

            # Spatial bounding box validation for Nepal
            if not (26.2 <= lat <= 30.6 and 80.0 <= lon <= 88.3):
                logger.warning(
                    "Coordinates (%.3f, %.3f) out of Nepal bounds. Rejecting bulletin %s.",
                    lat, lon, bid,
                )
                if not self.dry_run:
                    self.store.update_bulletin_status(
                        bid, "REJECTED_OUT_OF_BOUNDS", parsed_json=decision.to_dict()
                    )
                continue

            elev, slope, curv, relief = self._enrich_with_dem(parsed, lat, lon, bid)
            event_name = parsed.get("event_name") or title
            hazard_type = parsed.get("hazard_type") or "monsoon_landslide"

            if not decision.admit:
                logger.warning(
                    "--> [HELD] Bulletin %s not admitted to the training catalog: %s",
                    bid, decision.note,
                )
                quarantined.append({
                    "item_id": bid,
                    "pillar": "bulletin_nlp",
                    "title": title[:120],
                    "proposed_event": {
                        "event_name": event_name, "latitude": lat, "longitude": lon
                    },
                    **decision.to_dict(),
                })
                if not self.dry_run:
                    self.store.update_bulletin_status(
                        bid, "NEEDS_HUMAN_REVIEW", parsed_json=decision.to_dict()
                    )
                continue

            if not self.dry_run:
                ev_record = append_ground_truth_event(
                    event_name=event_name,
                    latitude=lat,
                    longitude=lon,
                    elevation_m=float(elev),
                    slope_deg=float(slope),
                    label=1,
                    source=f"AI_AUDIT_{self.engine}",
                    hazard_type=hazard_type,
                    profile_curvature=float(curv) if curv else -0.48,
                    local_relief_m=float(relief) if relief else 250.0,
                    rainfall_72h_mm=parsed.get("rainfall_72h_mm", 160.0),
                    notes=(
                        f"NLP-extracted from bulletin '{title}' by {self.engine} audit. "
                        f"Consensus: {decision.status}/{decision.confidence_tier} "
                        f"({decision.note}). "
                        f"Model confidence: {float(parsed.get('confidence', 0.85)):.2f}."
                    ),
                )
                self.store.update_bulletin_status(
                    bulletin_id=bid,
                    status="PARSED_VERIFIED",
                    extracted_event_id=ev_record.get("event_id"),
                    parsed_json={**decision.to_dict(), "accepted_payload": parsed},
                )
                extracted_events.append({**ev_record, "consensus_status": decision.status})
            else:
                extracted_events.append({
                    "event_name": event_name,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation_m": elev,
                    "slope_deg": slope,
                    "hazard_type": hazard_type,
                    "label": 1,
                    "consensus_status": decision.status,
                })

            logger.info(
                "--> Extracted event '%s' at (%.3fN, %.3fE, elev=%.0fm) [consensus=%s/%s]",
                event_name, lat, lon, elev, decision.status, decision.confidence_tier,
            )

        return len(pending), extracted_events, quarantined

    def _enrich_with_dem(
        self, parsed: Dict[str, Any], lat: float, lon: float, bid: str
    ) -> Tuple[float, float, Optional[float], Optional[float]]:
        """Fill missing terrain derivatives from the Copernicus DEM, with offline fallbacks."""
        elev = parsed.get("elevation_m")
        slope = parsed.get("slope_deg")
        curv = parsed.get("profile_curvature")
        relief = parsed.get("local_relief_m")

        if elev is not None and slope is not None:
            return float(elev), float(slope), curv, relief

        try:
            dem_patch = fetch_and_process_copernicus_dem_patch(center_lat=lat, center_lon=lon)
            px = dem_patch.get("center_pixel", {})
            elev = elev or px.get("elevation_m", 1600.0)
            slope = slope or px.get("slope_deg", 36.0)
            curv = curv or px.get("profile_curvature", -0.45)
            relief = relief or dem_patch.get("grid_summary", {}).get(
                "max_elevation_m", elev + 200.0
            ) - elev
        except Exception as e:
            logger.debug("DEM patch query fallback (%s): %s", bid, e)
            elev = elev or 1600.0
            slope = slope or 36.0
            curv = curv or -0.45
            relief = relief or 280.0

        return float(elev), float(slope), curv, relief

    def _bulletin_prompt(self, title: str, text: str, district_hint: str) -> str:
        return (
            "You are an expert Nepal Geohazard Disaster Information Analyst. "
            "Extract disaster facts from this incident report in Nepal (Nepali or English). "
            "Reply strictly with JSON:\n"
            "{\n"
            '  "is_geohazard_event": true,\n'
            '  "event_name": "Short descriptive title with location",\n'
            '  "hazard_type": "monsoon_cloudburst_landslide|rockfall|debris_flow|glof|rock_ice_avalanche",\n'
            '  "district": "Nepal District name",\n'
            '  "latitude": 27.xxxx,\n'
            '  "longitude": 85.xxxx,\n'
            '  "elevation_m": 1200.0,\n'
            '  "slope_deg": 38.0,\n'
            '  "rainfall_72h_mm": 180.0,\n'
            '  "confidence": 0.90\n'
            "}\n"
            "If coordinates are not explicitly given, use the exact district or village centroid.\n"
            "If the report is not a geohazard event, set is_geohazard_event to false and "
            'hazard_type to "not_a_geohazard".\n'
            f"REPORT TITLE: {title}\n"
            f"DISTRICT HINT: {district_hint}\n"
            f"REPORT TEXT: {text[:1500]}"
        )

    def _extract_disaster_consensus(
        self, title: str, text: str, district_hint: str
    ) -> ConsensusDecision:
        """Gather independent extractions and cross-validate them."""
        prompt = self._bulletin_prompt(title, text, district_hint)

        if self.is_dual:
            # Each model sees only the bulletin -- never the other model's answer.
            votes = [
                self._gemini_vote(prompt, BULLETIN_SCHEMA),
                self._claude_vote(prompt, BULLETIN_SCHEMA),
            ]
            decision = cross_validate_votes(
                votes, bulletin_votes_agree, merge_bulletin_votes, self.require_dual_consensus
            )
            if decision.status == "NO_VOTES":
                decision.payload = self._extract_deterministic_nlp(title, text, district_hint)
                decision.status = "NO_VOTES_DETERMINISTIC_FALLBACK"
                decision.note += (
                    " Fell back to the offline deterministic parser for reporting only; "
                    "not admitted to the training catalog."
                )
            return decision

        vote = self._single_engine_vote(prompt, BULLETIN_SCHEMA)
        if vote.status != "OK" or not isinstance(vote.payload, dict):
            vote = ModelVote(
                model="deterministic",
                status="OK",
                payload=self._extract_deterministic_nlp(title, text, district_hint),
                reason=vote.reason or "deterministic offline parser",
            )
        return ConsensusDecision(
            status="SINGLE_ENGINE",
            confidence_tier="LOW",
            admit=True,
            payload=vote.payload,
            votes=[vote.to_dict()],
            note=f"Single-engine mode ({self.engine}); no cross-validation performed.",
        )

    def _extract_disaster_from_text(
        self, title: str, text: str, district_hint: str
    ) -> Optional[Dict[str, Any]]:
        """Back-compatible single-answer extraction (consensus metadata discarded)."""
        return self._extract_disaster_consensus(title, text, district_hint).payload

    def _extract_deterministic_nlp(
        self, title: str, text: str, district_hint: str
    ) -> Dict[str, Any]:
        """Regex and district-centroid deterministic NLP parser (zero token cost / offline)."""
        combined = title + " " + text
        self.token_accountant.record_local("deterministic")

        # 1. Regex search for explicit coordinates like 27.712N, 85.218E or 27.712, 85.218
        lat, lon = None, None
        coord_match = re.search(
            r"(\d{2}\.\d{2,4})\s*(?:N|°\s*N|उत्तर)?\s*[,/]\s*(\d{2}\.\d{2,4})\s*(?:E|°\s*E|पूर्व)?",
            combined,
            re.IGNORECASE,
        )
        if coord_match:
            try:
                lat = float(coord_match.group(1))
                lon = float(coord_match.group(2))
            except ValueError:
                pass

        # 2. If no explicit coordinates, look up district centroid
        dist = district_hint
        if not dist:
            for d in NEPAL_DISTRICTS:
                if re.search(r"\b" + re.escape(d) + r"\b", combined, re.IGNORECASE):
                    dist = d
                    break

        elev_m, slope_deg = 1500.0, 35.0
        if (lat is None or lon is None) and dist in DISTRICT_CENTROIDS:
            lat, lon, elev_m = DISTRICT_CENTROIDS[dist]

        # 3. Classify hazard type
        hazard = "monsoon_landslide"
        if any(w in combined.lower() for w in ["debris flow", "लेदो", "boulder"]):
            hazard = "debris_flow"
        elif any(w in combined.lower() for w in ["rockfall", "सुक्खा पहिरो", "ढुङ्गा"]):
            hazard = "rockfall"
        elif any(w in combined.lower() for w in ["glof", "moraine", "outburst", "ताल"]):
            hazard = "glof"
        elif any(w in combined.lower() for w in ["icefall", "avalanche", "हिमपहिरो"]):
            hazard = "rock_ice_avalanche"

        is_event = any(
            w in combined.lower()
            for w in ["landslide", "पहिरो", "block", "अवरुद्ध", "breach", "surge", "flood"]
        )

        return {
            "is_geohazard_event": is_event,
            "event_name": title[:80],
            "hazard_type": hazard,
            "district": dist or "Central Nepal",
            "latitude": lat or 27.75,
            "longitude": lon or 85.35,
            "elevation_m": elev_m,
            "slope_deg": slope_deg,
            "rainfall_72h_mm": 160.0 if "cloudburst" in combined.lower() else 95.0,
            "confidence": 0.88 if coord_match else 0.72,
        }

    # =========================================================================
    # PILLAR 2: Multimodal Visual Satellite Screening (Cloud vs Scar)
    # =========================================================================
    def run_satellite_visual_audit(
        self, limit: int = 10
    ) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Screen staged before/after thumbnail pairs for genuine mass-movement scars."""
        logger.info("[Pillar 2/4] Inspecting satellite visual candidate thumbnail pairs...")
        candidates = self.store.get_pending_visual_candidates(limit=limit)

        if not candidates:
            logger.info("No visual candidates pending. Staging candidate anomalies...")
            stage_satellite_visual_candidates(self.store)
            candidates = self.store.get_pending_visual_candidates(limit=limit)

        logger.info("Found %d visual candidate(s) for multimodal inspection.", len(candidates))
        inspected_results: List[Dict[str, Any]] = []
        quarantined: List[Dict[str, Any]] = []

        for cand in candidates:
            cid = cand["candidate_id"]
            site_id = cand.get("site_id") or cid
            lat = cand["latitude"]
            lon = cand["longitude"]
            before_p = cand.get("thumbnail_before_path")
            after_p = cand.get("thumbnail_after_path")

            logger.info(
                "Visual inspection for candidate [%s] at (%.3fN, %.3fE)...", cid, lat, lon
            )
            decision, provenance = self._classify_satellite_change_consensus(
                before_p, after_p, cand
            )

            if decision.status == "NEEDS_HUMAN_REVIEW":
                logger.warning("--> [QUARANTINED] Candidate %s: %s", cid, decision.note)
                quarantined.append({
                    "item_id": cid,
                    "pillar": "satellite_vision",
                    "imagery_provenance": provenance,
                    **decision.to_dict(),
                })
                if not self.dry_run:
                    self.store.update_visual_candidate_status(
                        candidate_id=cid,
                        status="NEEDS_HUMAN_REVIEW",
                        ai_confidence=0.0,
                        ai_notes=decision.note[:900],
                    )
                inspected_results.append({
                    "candidate_id": cid,
                    "classification": "NEEDS_HUMAN_REVIEW",
                    "is_true_scar": False,
                    "confidence": 0.0,
                    "evidence": decision.note,
                    "imagery_provenance": provenance,
                    "consensus_status": decision.status,
                    "confirmed": False,
                })
                continue

            vis = decision.payload or {
                "is_true_scar": False,
                "classification": "UNKNOWN",
                "confidence": 0.0,
                "evidence": decision.note,
            }
            is_scar = bool(vis.get("is_true_scar"))
            classification = _normalize_label(vis.get("classification")) or "UNKNOWN"
            try:
                confidence = float(vis.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.0
            notes = str(vis.get("evidence", ""))

            # Synthetic fixtures must never become real training rows (HANDOFF Issues #4/#5).
            synthetic_block = (
                provenance == "SYNTHETIC_FIXTURE" and not self.allow_synthetic_training_rows
            )
            admit = is_scar and confidence >= 0.75 and decision.admit and not synthetic_block

            result = {
                **vis,
                "classification": classification,
                "candidate_id": cid,
                "imagery_provenance": provenance,
                "consensus_status": decision.status,
                "consensus_confidence": decision.confidence_tier,
                "confirmed": is_scar if self.dry_run else admit,
            }

            if not self.dry_run:
                status = classification
                if admit:
                    status = "CONFIRMED_SCAR"
                elif is_scar:
                    status = "SCAR_HELD_FOR_REVIEW"
                self.store.update_visual_candidate_status(
                    candidate_id=cid,
                    status=status,
                    ai_confidence=confidence,
                    ai_notes=f"[{provenance}] {classification}: {notes}"[:900],
                )

                if admit:
                    ev = append_ground_truth_event(
                        event_name=f"Multimodal AI Verified Scar: {site_id}",
                        latitude=lat,
                        longitude=lon,
                        elevation_m=cand.get("elevation_m") or 2200.0,
                        slope_deg=cand.get("slope_deg") or 37.0,
                        label=1,
                        source=f"MULTIMODAL_VISION_{self.engine}",
                        hazard_type="satellite_detected_scar",
                        ndvi_drop=cand.get("delta_ndvi") or -0.25,
                        rainfall_72h_mm=140.0,
                        notes=(
                            f"Confirmed by multimodal vision {self.engine} audit "
                            f"(consensus={decision.status}/{decision.confidence_tier}, "
                            f"imagery={provenance}). Classification: {classification}. {notes}"
                        ),
                    )
                    result["event_id"] = ev.get("event_id")
                    logger.info("--> [CONFIRMED SCAR] Ingested into ground-truth feedback catalog!")
                elif is_scar:
                    reason = (
                        "synthetic fixture imagery is not evidence of a real landslide"
                        if synthetic_block
                        else f"consensus {decision.status}/{decision.confidence_tier}"
                    )
                    logger.warning(
                        "--> [HELD] Scar at %s not admitted to the training catalog (%s).",
                        cid, reason,
                    )
                    quarantined.append({
                        "item_id": cid,
                        "pillar": "satellite_vision",
                        "imagery_provenance": provenance,
                        "reason": reason,
                        **decision.to_dict(),
                    })
                else:
                    logger.info(
                        "--> [FILTERED OUT] Artifact identified: %s (confidence=%.2f).",
                        classification, confidence,
                    )

            inspected_results.append(result)

        return len(candidates), inspected_results, quarantined

    def _vision_prompt(self) -> str:
        return (
            "You are a Senior Satellite Geomorphologist analyzing high-mountain Himalayan hazards. "
            "Examine this before and after satellite image pair (512x512 thumbnail crops). "
            "Image 1 is PRE-EVENT; Image 2 is POST-EVENT. "
            "Determine if Image 2 shows an actual physical landslide scar (bare earth crown "
            "rupture with downslope debris trail) OR a false-positive artifact (cloud shadow, "
            "ephemeral snowfield retreat, sensor noise). "
            "Reply strictly with JSON:\n"
            "{\n"
            '  "is_true_scar": true|false,\n'
            '  "classification": "TRUE_LANDSLIDE_SCAR|CLOUD_SHADOW_ARTIFACT|SNOWMELT_RETREAT|SENSOR_NOISE",\n'
            '  "confidence": 0.95,\n'
            '  "evidence": "Brief description of morphological features observed"\n'
            "}"
        )

    def _classify_satellite_change_consensus(
        self,
        before_path_str: Optional[str],
        after_path_str: Optional[str],
        metadata: Dict[str, Any],
    ) -> Tuple[ConsensusDecision, str]:
        """Independently screen a thumbnail pair with each model, then cross-validate."""
        before_p = Path(before_path_str) if before_path_str else None
        after_p = Path(after_path_str) if after_path_str else None

        if not (before_p and before_p.exists() and after_p and after_p.exists()):
            missing = {
                "is_true_scar": False,
                "classification": "MISSING_THUMBNAIL_DATA",
                "confidence": 0.0,
                "evidence": "Thumbnail files not found on disk.",
            }
            return (
                ConsensusDecision(
                    status="SINGLE_ENGINE",
                    confidence_tier="NONE",
                    admit=False,
                    payload=missing,
                    votes=[],
                    note="Thumbnail pair unavailable.",
                ),
                "MISSING",
            )

        provenance = imagery_provenance(after_p)
        prompt = self._vision_prompt()
        if provenance == "SYNTHETIC_FIXTURE":
            prompt += (
                "\nNOTE: this pair is a SYNTHETIC rendering used for pipeline validation, not a "
                "real Sentinel-2 chip. Classify the rendered morphology as presented."
            )
        images = [before_p, after_p]

        if self.is_dual:
            votes = [
                self._gemini_vote(prompt, VISION_SCHEMA, images=images),
                self._claude_vote(prompt, VISION_SCHEMA, images=images),
            ]
            decision = cross_validate_votes(
                votes, vision_votes_agree, merge_vision_votes, self.require_dual_consensus
            )
            if decision.status == "NO_VOTES":
                decision.payload = self._classify_deterministic_vision(
                    before_p, after_p, metadata
                )
                decision.status = "NO_VOTES_DETERMINISTIC_FALLBACK"
                decision.note += " Fell back to the offline deterministic vision classifier."
            return decision, provenance

        vote = self._single_engine_vote(prompt, VISION_SCHEMA, images=images)
        if vote.status != "OK" or not isinstance(vote.payload, dict):
            vote = ModelVote(
                model="deterministic",
                status="OK",
                payload=self._classify_deterministic_vision(before_p, after_p, metadata),
                reason=vote.reason or "deterministic offline vision classifier",
            )
        return (
            ConsensusDecision(
                status="SINGLE_ENGINE",
                confidence_tier="LOW",
                admit=True,
                payload=vote.payload,
                votes=[vote.to_dict()],
                note=f"Single-engine mode ({self.engine}); no cross-validation performed.",
            ),
            provenance,
        )

    def _classify_satellite_change(
        self,
        before_path_str: Optional[str],
        after_path_str: Optional[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Back-compatible single-answer vision classification."""
        decision, _ = self._classify_satellite_change_consensus(
            before_path_str, after_path_str, metadata
        )
        return decision.payload or {
            "is_true_scar": False,
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "evidence": decision.note,
        }

    def _classify_deterministic_vision(
        self, before_path: Path, after_path: Path, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Offline chromaticity + morphology classifier (zero token cost).

        Thresholds are calibrated against the measured output of the synthetic thumbnail
        generator. This is a pixel-statistics screen over rendered fixtures, not real
        Sentinel-2 interpretation, and its evidence string says so.
        """
        self.token_accountant.record_local("deterministic")
        try:
            features = compute_vision_features(before_path, after_path)
        except (OSError, ValueError) as e:
            logger.warning("Deterministic vision could not read the thumbnail pair: %s", e)
            return {
                "is_true_scar": False,
                "classification": "SENSOR_NOISE",
                "confidence": 0.0,
                "evidence": f"Thumbnail pair unreadable: {e}",
            }

        result = classify_vision_features(features)
        provenance = imagery_provenance(after_path)
        result["features"] = {k: round(v, 4) for k, v in features.items()}
        result["imagery_provenance"] = provenance
        if provenance == "SYNTHETIC_FIXTURE":
            result["evidence"] += " [SYNTHETIC fixture imagery, not a real Sentinel-2 chip.]"
        return result

    # =========================================================================
    # PILLAR 3: Residual Extremes & AI Geomechanical Equation Discovery
    # =========================================================================
    def run_equation_discovery_audit(self) -> Dict[str, Any]:
        """Analyze residual outliers and evaluate proposed equation refinements."""
        logger.info("[Pillar 3/4] Running residual error analysis & symbolic equation evolution...")
        self.optimizer.reload_dataset()
        dataset = self.optimizer._dataset

        if dataset.empty:
            logger.warning("Benchmark dataset is empty. Skipping equation discovery.")
            return {"status": "SKIPPED_EMPTY_DATASET"}

        active_params = dict(self.optimizer.active_params)
        acc, rec, auc, f1, cryo, fb, avg_dev = self.optimizer.evaluate_parameter_set(active_params)
        baseline_fitness = self.optimizer._compute_fitness(auc, rec, avg_dev, cryo, fb)

        logger.info(
            "Current Production Active Baseline: Fitness=%.4f | Recall=%.2f%% | ROC-AUC=%.4f | "
            "Cryo=%.1f%% | LiveGT=%.1f%% | FS-Dev=%.2f%%",
            baseline_fitness, rec * 100.0, auc, cryo * 100.0, fb * 100.0, avg_dev,
        )

        false_negatives: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []

        for idx, row in dataset.iterrows():
            lbl = int(row["label"])
            fs, prob = self.optimizer.compute_candidate_fs(row, active_params)
            site_info = {
                "name": str(row.get("event_name", f"Site #{idx}")),
                "lat": round(float(row["latitude"]), 3),
                "lon": round(float(row["longitude"]), 3),
                "elev_m": round(float(row["elevation_m"]), 0),
                "slope_deg": round(float(row["slope_deg"]), 1),
                "curv": round(float(row["profile_curvature"]), 2),
                "relief_m": round(float(row.get("local_relief_m", 120.0)), 0),
                "static_fs": round(fs, 2),
                "failure_prob": round(prob, 2),
                "actual_label": lbl,
            }
            if lbl == 1 and fs >= 1.15:
                false_negatives.append(site_info)
            elif lbl == 0 and fs <= 0.85:
                false_positives.append(site_info)

        logger.info(
            "Identified %d False Negative(s) and %d False Positive(s) in the residual queue.",
            len(false_negatives), len(false_positives),
        )

        telemetry_extremes = self.store.get_telemetry_outliers_for_audit(window_hours=168)

        proposals = self._propose_equation_refinements(
            active_params=active_params,
            false_negatives=false_negatives[:6],
            false_positives=false_positives[:6],
            telemetry_extremes=telemetry_extremes,
        )

        # Rule 8.4 Validation Gate: Recall >= 91.36%, Cryo = 100%, FS Deviation <= 5.0%.
        # Promotion additionally requires a strictly meaningful fitness gain.
        evaluated: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        for prop in proposals:
            params = prop["parameters"]
            (
                c_acc, c_rec, c_auc, c_f1, c_cryo, c_fb, c_dev
            ) = self.optimizer.evaluate_parameter_set(params)
            fitness = self.optimizer._compute_fitness(c_auc, c_rec, c_dev, c_cryo, c_fb)
            passes_gate = c_rec >= 0.9136 and c_cryo >= 0.999 and c_dev <= 5.0
            beats_champion = fitness > baseline_fitness + PROMOTION_FITNESS_EPSILON

            record = {
                "source": prop["source"],
                "fitness": fitness,
                "metrics": {
                    "recall": c_rec, "roc_auc": c_auc, "cryo": c_cryo,
                    "live_gt": c_fb, "fs_dev": c_dev,
                },
                "passes_rule_8_4": passes_gate,
                "beats_champion": beats_champion,
                "parameters": params,
                "note": prop.get("note", ""),
            }
            evaluated.append(record)
            logger.info(
                "Proposal [%s]: Fitness=%.4f (baseline %.4f, delta %+.6f) | Rule 8.4=%s",
                prop["source"], fitness, baseline_fitness, fitness - baseline_fitness, passes_gate,
            )
            if passes_gate and beats_champion and (best is None or fitness > best["fitness"]):
                best = record

        promoted = False
        if best is not None:
            promoted = True
            reason = (
                f"PROMOTED! Proposal from '{best['source']}' surpassed the champion "
                f"(Fitness {best['fitness']:.4f} > {baseline_fitness:.4f} + "
                f"{PROMOTION_FITNESS_EPSILON:g}) with {best['metrics']['cryo']*100:.0f}% Cryo, "
                f"{best['metrics']['live_gt']*100:.1f}% Live-GT, "
                f"{best['metrics']['recall']*100:.2f}% Recall, "
                f"{best['metrics']['fs_dev']:.2f}% FS-Dev"
            )
            if not self.dry_run:
                self.optimizer._save_active_params(best["parameters"])
                logger.info("New champion parameters saved to %s!", ACTIVE_PARAMS_PATH)
        elif not evaluated:
            reason = "No candidate equations were proposed this week."
        elif any(e["passes_rule_8_4"] for e in evaluated):
            reason = (
                f"Passed Rule 8.4 Gate, but no proposal improved on the active champion fitness "
                f"({baseline_fitness:.4f}) by more than {PROMOTION_FITNESS_EPSILON:g}."
            )
        else:
            top = max(evaluated, key=lambda e: e["fitness"])
            reason = (
                f"Blocked by Rule 8.4 Gate (best proposal '{top['source']}': "
                f"Recall={top['metrics']['recall']*100:.1f}%, "
                f"Cryo={top['metrics']['cryo']*100:.0f}%, Dev={top['metrics']['fs_dev']:.2f}%)."
            )

        logger.info("Equation Discovery Result: Promoted=%s | Reason: %s", promoted, reason)

        headline = best or (max(evaluated, key=lambda e: e["fitness"]) if evaluated else None)
        baseline_metrics = {
            "recall": rec, "roc_auc": auc, "cryo": cryo, "live_gt": fb, "fs_dev": avg_dev
        }
        return {
            "baseline_fitness": baseline_fitness,
            "baseline_metrics": baseline_metrics,
            "candidate_fitness": headline["fitness"] if headline else baseline_fitness,
            "candidate_metrics": headline["metrics"] if headline else dict(baseline_metrics),
            "promoted": promoted,
            "reason": reason,
            "promoted_parameters": best["parameters"] if best else active_params,
            "residual_outliers_count": len(false_negatives) + len(false_positives),
            "proposals": [{k: v for k, v in e.items() if k != "parameters"} for e in evaluated],
            "promotion_epsilon": PROMOTION_FITNESS_EPSILON,
        }

    def _equation_prompt(
        self,
        active_params: Dict[str, float],
        false_negatives: List[Dict[str, Any]],
        false_positives: List[Dict[str, Any]],
    ) -> str:
        return (
            "You are a Principal Himalayan Geotechnical Engineer and Numerical Modeler. "
            "Our 7-Factor Himalayan Infinite Slope stability equation has the following "
            "active weights:\n"
            f"{json.dumps(active_params, indent=2)}\n\n"
            "Below are specific residual failure cases where the model struggled this week:\n"
            f"FALSE NEGATIVES (Actual failures, but FS was too stable):\n"
            f"{json.dumps(false_negatives, indent=2)}\n\n"
            f"FALSE POSITIVES (Actually stable, but FS was too low):\n"
            f"{json.dumps(false_positives, indent=2)}\n\n"
            "Propose a calibrated parameter dictionary. Adjust weights (within +/- 5% range) to "
            "increase driving stress on concave steep scarps while maintaining resistance on "
            "flat alluvial terraces. Reply strictly with JSON containing exactly these keys: "
            f"{', '.join(EQUATION_PARAM_KEYS)}."
        )

    def _propose_equation_refinements(
        self,
        active_params: Dict[str, float],
        false_negatives: List[Dict[str, Any]],
        false_positives: List[Dict[str, Any]],
        telemetry_extremes: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Collect candidate parameter sets. In dual mode both models propose independently."""
        prompt = self._equation_prompt(active_params, false_negatives, false_positives)
        proposals: List[Dict[str, Any]] = []

        def _coerce(payload: Optional[Dict[str, Any]], source: str) -> None:
            if not payload or "c_res_base" not in payload:
                return
            try:
                params = {
                    k: round(float(payload[k]), 6) for k in active_params.keys() if k in payload
                }
            except (TypeError, ValueError):
                logger.warning("Proposal from %s had non-numeric parameters; discarded.", source)
                return
            if not params:
                return
            merged = dict(active_params)
            merged.update(params)
            proposals.append({"source": source, "parameters": merged, "note": "model proposal"})

        votes: List[ModelVote] = []
        if self.is_dual:
            votes = [
                self._gemini_vote(prompt, EQUATION_SCHEMA),
                self._claude_vote(prompt, EQUATION_SCHEMA),
            ]
        elif self.uses_gemini or self.uses_claude or self.uses_ollama:
            votes = [self._single_engine_vote(prompt, EQUATION_SCHEMA)]

        for vote in votes:
            if vote.status == "OK":
                _coerce(vote.payload, vote.model)
            else:
                logger.info("Equation proposal: %s abstained (%s).", vote.model, vote.reason)

        proposals.append({
            "source": "deterministic_sensitivity",
            "parameters": self._deterministic_equation_nudge(
                active_params, false_negatives, false_positives
            ),
            "note": "Offline geomechanical sensitivity nudge (zero API tokens).",
        })
        return proposals

    def _deterministic_equation_nudge(
        self,
        active_params: Dict[str, float],
        false_negatives: List[Dict[str, Any]],
        false_positives: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Offline sensitivity nudge used as the always-available baseline proposal."""
        self.token_accountant.record_local("deterministic")
        nudged = dict(active_params)
        # False negatives dominating on steep slopes -> lift gorge incision + orographic moisture.
        if len(false_negatives) >= len(false_positives):
            nudged["w_ommi"] = round(nudged["w_ommi"] * 1.012, 6)
            nudged["w_gii"] = round(nudged["w_gii"] * 1.008, 6)
            nudged["w_tcwi"] = round(nudged["w_tcwi"] * 1.015, 6)
        else:
            nudged["c_res_base"] = round(nudged["c_res_base"] * 1.010, 6)
            nudged["w_lowland_flat"] = round(nudged["w_lowland_flat"] * 1.015, 6)
        return nudged

    def _propose_equation_refinement(
        self,
        active_params: Dict[str, float],
        false_negatives: List[Dict[str, Any]],
        false_positives: List[Dict[str, Any]],
        telemetry_extremes: Dict[str, Any],
    ) -> Dict[str, float]:
        """Back-compatible single-proposal entry point."""
        proposals = self._propose_equation_refinements(
            active_params, false_negatives, false_positives, telemetry_extremes
        )
        return proposals[0]["parameters"] if proposals else dict(active_params)

    # =========================================================================
    # PILLAR 4: Executive Disaster Resilience Report & Run Persistence
    # =========================================================================
    def generate_and_save_audit_report(
        self,
        b_scanned: int,
        events_extracted: List[Dict[str, Any]],
        v_inspected: int,
        v_results: List[Dict[str, Any]],
        eq_result: Dict[str, Any],
        quarantined: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Compile weekly findings, real token consumption and the executive report."""
        quarantined = quarantined or []
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        report_path = AUDIT_REPORTS_DIR / f"weekly_ai_audit_{timestamp_str}.md"
        tokens = self.token_accountant.to_dict()

        confirmed_visual_count = sum(1 for v in v_results if v.get("confirmed"))
        consensus_mode = "DUAL (Gemini + Claude, independent)" if self.is_dual else "SINGLE ENGINE"

        report_md = f"""# Weekly AI Geohazard Audit & Continuous Learning Report
**Execution Timestamp:** {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
**AI Engine:** `{self.engine}` | **Dry Run Mode:** `{self.dry_run}`
**Cross-validation:** {consensus_mode} | **Dual consensus required for admission:** `{self.require_dual_consensus}`
**Real API Token Consumption:** {tokens['total_tokens']:,} tokens (Input: {tokens['total_input_tokens']:,}, Output: {tokens['total_output_tokens']:,}) across {tokens['api_calls_count']} API call(s) | **Est. Cost:** \\${tokens['estimated_cost_usd']:.4f}

---

## 1. Executive Summary
- **Unstructured News Bulletins Scanned:** {b_scanned} items
- **Verified Ground-Truth Landslides Ingested:** {len(events_extracted)} events
- **Items Quarantined for Human Review:** {len(quarantined)}
- **Satellite Visual Anomaly Thumbnails Screened:** {v_inspected} candidate pairs
- **Confirmed True Landslide Scars:** {confirmed_visual_count} scars
- **False Alarms Filtered Out (Cloud Shadows & Snow Retreat):** {v_inspected - confirmed_visual_count} artifacts
- **Geomechanical Equation Evolution:** {'PROMOTED NEW FORMULA' if eq_result.get('promoted') else 'RETAINED ACTIVE CHAMPION'}

---

## 2. Unstructured Bulletin NLP Field Ingestion
{f"Ingested {len(events_extracted)} verified disaster events into the training feedback catalog (`live_ground_truth_feedback.csv`):" if events_extracted else "No new landslide events admitted in the current bulletin batch."}

| Event Name | District / Hazard | Lat, Lon | Elevation | Slope | Consensus |
|---|---|---|---|---|---|
"""
        for ev in events_extracted[:8]:
            report_md += (
                f"| {ev.get('event_name', 'Event')} | {ev.get('hazard_type', 'Landslide')} | "
                f"{ev.get('latitude', 0):.3f}°N, {ev.get('longitude', 0):.3f}°E | "
                f"{ev.get('elevation_m', 0):.0f}m | {ev.get('slope_deg', 0):.1f}° | "
                f"{ev.get('consensus_status', 'n/a')} |\n"
            )

        report_md += f"""
---

## 3. Multimodal Satellite Vision Verification (Cloud vs. Scar)
Screened {v_inspected} pre- and post-event thumbnail pairs ($512\\times512$) to distinguish physical mass-movements from ephemeral atmospheric artifacts:

| Candidate ID | Classification | Confidence | Imagery | Consensus | Morphological Evidence |
|---|---|---|---|---|---|
"""
        for v in v_results[:6]:
            report_md += (
                f"| `{v.get('candidate_id', 'CANDIDATE')}` | **{v.get('classification', 'UNKNOWN')}** | "
                f"{float(v.get('confidence', 0) or 0)*100:.1f}% | {v.get('imagery_provenance', 'n/a')} | "
                f"{v.get('consensus_status', 'n/a')} | {str(v.get('evidence', 'Processed'))[:220]} |\n"
            )

        report_md += """
---

## 3b. Quarantine — Items Needing Human Adjudication
Models disagreed, abstained, or the supporting imagery was synthetic. **None of these became training labels.**

| Item | Pillar | Consensus | Why |
|---|---|---|---|
"""
        if quarantined:
            for q in quarantined[:12]:
                report_md += (
                    f"| `{q.get('item_id', '?')}` | {q.get('pillar', '?')} | "
                    f"{q.get('consensus_status', '?')} | "
                    f"{str(q.get('note', q.get('reason', '')))[:220]} |\n"
                )
        else:
            report_md += "| _(none)_ | | | |\n"

        proposal_rows = ""
        for p in eq_result.get("proposals", []):
            m = p.get("metrics", {})
            proposal_rows += (
                f"| {p.get('source')} | {p.get('fitness', 0):.4f} | {m.get('recall', 0)*100:.2f}% | "
                f"{m.get('roc_auc', 0):.4f} | {m.get('cryo', 0)*100:.0f}% | "
                f"{m.get('fs_dev', 0):.2f}% | {'yes' if p.get('passes_rule_8_4') else 'no'} | "
                f"{'yes' if p.get('beats_champion') else 'no'} |\n"
            )
        if not proposal_rows:
            proposal_rows = "| _(no proposals)_ | | | | | | | |\n"

        report_md += f"""
---

## 4. 7-Factor Geomechanical Equation Optimization & Validation
- **Status:** `{eq_result.get('reason', 'Evaluation complete')}`
- **Active champion fitness:** {eq_result.get('baseline_fitness', 0):.4f}
- **Promotion epsilon (strict improvement required):** {eq_result.get('promotion_epsilon', PROMOTION_FITNESS_EPSILON):g}
- **Baseline Recall:** {eq_result.get('baseline_metrics', {}).get('recall', 0)*100:.2f}% -> **Best Candidate Recall:** {eq_result.get('candidate_metrics', {}).get('recall', 0)*100:.2f}%
- **Baseline ROC-AUC:** {eq_result.get('baseline_metrics', {}).get('roc_auc', 0):.4f} -> **Best Candidate ROC-AUC:** {eq_result.get('candidate_metrics', {}).get('roc_auc', 0):.4f}
- **High-Mountain Cryosphere Accuracy:** {eq_result.get('candidate_metrics', {}).get('cryo', 1.0)*100:.1f}% (Rule 8.4 Gate: 100%)
- **Live Ground-Truth Accuracy:** {eq_result.get('candidate_metrics', {}).get('live_gt', 1.0)*100:.1f}%
- **Physical Factor of Safety ($FS$) Deviation Envelope:** {eq_result.get('candidate_metrics', {}).get('fs_dev', 1.5):.2f}% (Rule 8.4 Gate: $\\le 5.0\\%$)

| Proposal source | Fitness | Recall | ROC-AUC | Cryo | FS-Dev | Rule 8.4 | Beats champion |
|---|---|---|---|---|---|---|---|
{proposal_rows}
---

## 5. Token Efficiency & API Resource Summary
Only genuine remote API calls are counted. Deterministic and local Ollama inference consume **zero** API tokens and appear under `local_inference_calls`.

| Engine | API calls | Input tokens | Output tokens | Est. cost (USD) |
|---|---|---|---|---|
"""
        by_engine = tokens.get("by_engine", {})
        if by_engine:
            for name, usage in by_engine.items():
                report_md += (
                    f"| {name} | {usage['api_calls']} | {usage['input_tokens']:,} | "
                    f"{usage['output_tokens']:,} | {usage['cost_usd']:.5f} |\n"
                )
        else:
            report_md += "| _(no remote API calls this run)_ | 0 | 0 | 0 | 0.00000 |\n"

        report_md += f"""
```json
{json.dumps(tokens, indent=2)}
```
*Cost basis: Claude `{settings.anthropic_model}` at \\${CLAUDE_PRICE_IN:.2f}/\\${CLAUDE_PRICE_OUT:.2f} per MTok; Gemini `{settings.gemini_model}` at an estimated \\${GEMINI_PRICE_IN:.2f}/\\${GEMINI_PRICE_OUT:.2f} per MTok (verify against the current Google rate card). Weekly batch processing keeps usage within the frugal quota target.*
"""
        report_path.write_text(report_md, encoding="utf-8")
        logger.info("Weekly AI Audit Report generated: %s", report_path)

        if not self.dry_run:
            run_record = {
                "run_id": f"audit_{timestamp_str}",
                "audit_timestamp_utc": time.time(),
                "engine_used": self.engine,
                "bulletins_scanned": b_scanned,
                "events_extracted": len(events_extracted),
                "visual_candidates_inspected": v_inspected,
                "visual_confirmed": confirmed_visual_count,
                "equations_evaluated": len(eq_result.get("proposals", [])) or 1,
                "promoted_to_production": bool(eq_result.get("promoted")),
                "promoted_parameters": eq_result.get("promoted_parameters"),
                "summary_report": report_md,
                "token_usage_json": tokens,
            }
            self.store.insert_ai_audit_run(run_record)

        return report_path

    # =========================================================================
    # MULTI-MODEL API ADAPTERS
    # =========================================================================
    def _single_engine_vote(
        self, prompt: str, schema: Dict[str, Any], images: Optional[List[Path]] = None
    ) -> ModelVote:
        """Dispatch one request to whichever single engine is configured."""
        if self.uses_gemini:
            return self._gemini_vote(prompt, schema, images=images)
        if self.uses_claude:
            return self._claude_vote(prompt, schema, images=images)
        if self.uses_ollama and not images:
            return self._ollama_vote(prompt)
        return ModelVote(
            model="none",
            status="ABSTAINED",
            reason=f"No remote engine configured for {self.engine}.",
        )

    @staticmethod
    def _encode_images(images: Optional[List[Path]]) -> List[str]:
        """Base64-encode PNG thumbnails with no embedded newlines."""
        return [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in images or []]

    # -- Gemini: raw HTTP against the generativelanguage REST API ------------------------
    def _gemini_vote(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        images: Optional[List[Path]] = None,
    ) -> ModelVote:
        """Independent Gemini opinion via models/{model}:generateContent."""
        if not self.gemini_key:
            return ModelVote("gemini", "ABSTAINED", reason="GEMINI_API_KEY not set")

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        try:
            for b64 in self._encode_images(images):
                parts.append({"inlineData": {"mimeType": "image/png", "data": b64}})
        except OSError as e:
            return ModelVote("gemini", "ABSTAINED", reason=f"could not read thumbnail: {e}")

        generation_config: Dict[str, Any] = {
            "temperature": 0.1,
            "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        }
        if schema:
            generation_config["responseSchema"] = json_schema_to_gemini(schema)

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        # Popular Gemini models return 503 UNAVAILABLE under load. Losing this vote
        # silently costs the audit its second independent opinion, so a transient
        # overload fails over to a less contended model rather than abstaining.
        model_candidates = [settings.gemini_model]
        model_candidates += [
            m for m in settings.gemini_fallback_models if m != settings.gemini_model
        ]

        data = None
        last_error = ""
        for attempt, model_name in enumerate(model_candidates):
            url = f"{GEMINI_API_BASE}/models/{model_name}:generateContent"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_key},
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if attempt:
                    logger.info("Gemini answered on fallback model %s", model_name)
                break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last_error = f"HTTP {e.code}: {detail}"
                logger.warning("Gemini (%s) HTTP %s: %s", model_name, e.code, detail)
                if e.code not in (429, 500, 502, 503, 504):
                    return ModelVote("gemini", "ABSTAINED", reason=last_error)
                continue
            except TimeoutError as e:
                # An overloaded model stalls as often as it returns 503, so a timeout
                # is treated as retryable and fails over to the next model.
                last_error = f"timeout: {e}"
                logger.warning("Gemini (%s) timed out; trying next model", model_name)
                continue
            except urllib.error.URLError as e:
                if isinstance(getattr(e, "reason", None), TimeoutError):
                    last_error = f"timeout: {e.reason}"
                    logger.warning("Gemini (%s) timed out; trying next model", model_name)
                    continue
                # A genuine connectivity failure hits every model equally - do not retry.
                logger.warning("Gemini (%s) unreachable: %s", model_name, e.reason)
                return ModelVote("gemini", "ABSTAINED", reason=f"unreachable: {e.reason}")
            except (json.JSONDecodeError, ValueError, OSError) as e:
                last_error = f"unusable response: {e}"
                logger.warning("Gemini (%s) response unusable: %s", model_name, e)
                continue

        if data is None:
            return ModelVote(
                "gemini",
                "ABSTAINED",
                reason=f"all models unavailable ({', '.join(model_candidates)}): {last_error}",
            )

        usage = data.get("usageMetadata", {}) or {}
        self.token_accountant.add(
            usage.get("promptTokenCount", 0),
            usage.get("candidatesTokenCount", 0),
            cost_per_m_in=GEMINI_PRICE_IN,
            cost_per_m_out=GEMINI_PRICE_OUT,
            engine="gemini",
        )

        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            return ModelVote(
                "gemini", "ABSTAINED", reason=f"no candidates returned (blockReason={block})"
            )

        finish = candidates[0].get("finishReason")
        if finish in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION"):
            return ModelVote(
                "gemini", "ABSTAINED", reason=f"refused/blocked (finishReason={finish})"
            )

        text = "".join(
            p.get("text", "")
            for p in (candidates[0].get("content", {}) or {}).get("parts", []) or []
        )
        parsed = parse_json_defensively(text)
        if parsed is None:
            return ModelVote(
                "gemini", "ABSTAINED", reason=f"unparseable JSON (finishReason={finish})"
            )
        return ModelVote("gemini", "OK", payload=parsed, reason=f"finishReason={finish}")

    def _call_gemini_json(self, prompt: str, max_tokens: int = 350) -> Optional[Dict[str, Any]]:
        """Back-compatible helper returning Gemini's payload or None."""
        return self._gemini_vote(prompt, None).payload

    def _call_gemini_multimodal_vision(
        self, prompt: str, before_p: Path, after_p: Path
    ) -> Dict[str, Any]:
        """Back-compatible helper; falls back to the deterministic classifier on abstention."""
        vote = self._gemini_vote(prompt, VISION_SCHEMA, images=[before_p, after_p])
        if vote.status == "OK" and isinstance(vote.payload, dict):
            return vote.payload
        return self._classify_deterministic_vision(before_p, after_p, {})

    # -- Claude: official anthropic SDK --------------------------------------------------
    def _get_anthropic_client(self):
        """Lazily construct the SDK client so the module imports without the SDK present."""
        if self._anthropic_client is not None:
            return self._anthropic_client
        import anthropic

        # An organization-scoped key must name a workspace or every request 400s with
        # "not scoped to a workspace". A workspace-scoped key needs no header.
        headers = {}
        workspace_id = (settings.anthropic_workspace_id or "").strip()
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id
        self._anthropic_client = anthropic.Anthropic(
            api_key=self.anthropic_key or None,
            default_headers=headers or None,
        )
        return self._anthropic_client

    def _claude_vote(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        images: Optional[List[Path]] = None,
    ) -> ModelVote:
        """Independent Claude opinion via the official SDK with adaptive thinking."""
        try:
            import anthropic
        except ImportError:
            return ModelVote("claude", "ABSTAINED", reason="anthropic SDK not installed")

        if not self.anthropic_key:
            return ModelVote("claude", "ABSTAINED", reason="ANTHROPIC_API_KEY not set")

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        try:
            for b64 in self._encode_images(images):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                })
        except OSError as e:
            return ModelVote("claude", "ABSTAINED", reason=f"could not read thumbnail: {e}")

        output_config: Dict[str, Any] = {"effort": "medium"}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        request: Dict[str, Any] = {
            "model": settings.anthropic_model,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
            "messages": [{"role": "user", "content": content}],
        }

        try:
            resp = self._get_anthropic_client().messages.create(**request)
        except anthropic.NotFoundError as e:
            logger.warning("Claude model '%s' not found: %s", settings.anthropic_model, e)
            return ModelVote("claude", "ABSTAINED", reason=f"model not found: {e}")
        except anthropic.RateLimitError as e:
            logger.warning("Claude rate limited after SDK retries: %s", e)
            return ModelVote("claude", "ABSTAINED", reason=f"rate limited: {e}")
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", None) == 400 and schema:
                # Retry once without structured output, then parse the text defensively.
                logger.warning("Claude rejected the structured-output request; retrying plain: %s", e)
                request["output_config"] = {"effort": "medium"}
                try:
                    resp = self._get_anthropic_client().messages.create(**request)
                except anthropic.APIStatusError as e2:
                    return ModelVote("claude", "ABSTAINED", reason=f"API error: {e2}")
                except anthropic.APIConnectionError as e2:
                    return ModelVote("claude", "ABSTAINED", reason=f"connection error: {e2}")
            else:
                logger.warning("Claude API error: %s", e)
                return ModelVote("claude", "ABSTAINED", reason=f"API error: {e}")
        except anthropic.APIConnectionError as e:
            logger.warning("Claude unreachable: %s", e)
            return ModelVote("claude", "ABSTAINED", reason=f"connection error: {e}")

        usage = getattr(resp, "usage", None)
        self.token_accountant.add(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            cost_per_m_in=CLAUDE_PRICE_IN,
            cost_per_m_out=CLAUDE_PRICE_OUT,
            engine="claude",
        )

        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "refusal":
            logger.info("Claude declined this item; treating it as an abstention.")
            return ModelVote("claude", "ABSTAINED", reason="refusal")

        # response.content is a list of blocks; thinking blocks carry no .text.
        text = next(
            (
                b.text
                for b in (getattr(resp, "content", None) or [])
                if getattr(b, "type", None) == "text"
            ),
            None,
        )
        parsed = parse_json_defensively(text)
        if parsed is None:
            return ModelVote(
                "claude", "ABSTAINED",
                reason=f"no parseable JSON text block (stop_reason={stop_reason})",
            )
        return ModelVote("claude", "OK", payload=parsed, reason=f"stop_reason={stop_reason}")

    def _call_claude_json(
        self, prompt: str, max_tokens: int = CLAUDE_MAX_TOKENS
    ) -> Optional[Dict[str, Any]]:
        """Back-compatible helper returning Claude's payload or None."""
        return self._claude_vote(prompt, None).payload

    def _call_claude_multimodal_vision(
        self, prompt: str, before_p: Path, after_p: Path
    ) -> Dict[str, Any]:
        """Back-compatible helper; falls back to the deterministic classifier on abstention."""
        vote = self._claude_vote(prompt, VISION_SCHEMA, images=[before_p, after_p])
        if vote.status == "OK" and isinstance(vote.payload, dict):
            return vote.payload
        return self._classify_deterministic_vision(before_p, after_p, {})

    # -- Ollama: local, zero API cost ----------------------------------------------------
    def _ollama_vote(self, prompt: str) -> ModelVote:
        """Local Ollama inference. Records zero API tokens because nothing is billed."""
        payload = {
            "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8")).get("response", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("Ollama call failed (%s); abstaining.", e)
            return ModelVote("ollama", "ABSTAINED", reason=str(e))

        self.token_accountant.record_local("ollama")
        parsed = parse_json_defensively(raw)
        if parsed is None:
            return ModelVote("ollama", "ABSTAINED", reason="unparseable JSON")
        return ModelVote("ollama", "OK", payload=parsed)

    def _call_ollama_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Back-compatible helper returning Ollama's payload or None."""
        return self._ollama_vote(prompt).payload

    # =========================================================================
    # MASTER AUDIT EXECUTION
    # =========================================================================
    def execute_complete_weekly_audit(
        self,
        limit_bulletins: int = 25,
        limit_visual: int = 10,
    ) -> Dict[str, Any]:
        """Execute the end-to-end 4-pillar weekly geohazard audit."""
        start_t = time.time()
        print("\n=======================================================", flush=True)
        print("🧠 HIMALAYAN-MHEWS WEEKLY AI AUDIT & LEARNING PIPELINE", flush=True)
        print(f"   Engine: {self.engine} | Dry Run: {self.dry_run}", flush=True)
        print(f"   Dual consensus required: {self.require_dual_consensus}", flush=True)
        print(f"   Database: {self.store.get_engine_name()}", flush=True)
        print("=======================================================", flush=True)

        b_scanned, events_extracted, b_quarantined = self.run_bulletin_nlp_audit(
            limit=limit_bulletins
        )
        v_inspected, v_results, v_quarantined = self.run_satellite_visual_audit(limit=limit_visual)
        eq_result = self.run_equation_discovery_audit()
        quarantined = b_quarantined + v_quarantined

        report_path = self.generate_and_save_audit_report(
            b_scanned=b_scanned,
            events_extracted=events_extracted,
            v_inspected=v_inspected,
            v_results=v_results,
            eq_result=eq_result,
            quarantined=quarantined,
        )

        duration = round(time.time() - start_t, 2)
        tokens = self.token_accountant.to_dict()
        confirmed = sum(1 for v in v_results if v.get("confirmed"))

        print("\n=======================================================", flush=True)
        print("✅ WEEKLY AI AUDIT SUCCESSFULLY COMPLETED", flush=True)
        print(f"   Duration: {duration}s", flush=True)
        print(f"   Disaster Bulletins Scanned: {b_scanned} -> Ingested: {len(events_extracted)} events", flush=True)
        print(f"   Satellite Anomalies Screened: {v_inspected} -> Confirmed: {confirmed}", flush=True)
        print(f"   Quarantined for human review: {len(quarantined)}", flush=True)
        print(f"   Equation Status: {eq_result.get('reason')}", flush=True)
        print(f"   Real API Token Usage: {tokens['total_tokens']:,} tokens (Cost: ${tokens['estimated_cost_usd']:.4f})", flush=True)
        print(f"   Executive Report: {report_path}", flush=True)
        print("=======================================================\n", flush=True)

        return {
            "status": "COMPLETED",
            "duration_sec": duration,
            "engine": self.engine,
            "consensus_required": self.require_dual_consensus,
            "bulletins_scanned": b_scanned,
            "events_extracted_count": len(events_extracted),
            "visual_screened_count": v_inspected,
            "visual_confirmed_count": confirmed,
            "quarantined_count": len(quarantined),
            "quarantined_items": quarantined,
            "equation_promoted": bool(eq_result.get("promoted")),
            "token_usage": tokens,
            "report_path": str(report_path),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Weekly AI Geohazard Audit & Continuous Calibration"
    )
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "dual", "gemini", "claude", "ollama", "deterministic"],
        help="AI engine. 'dual' runs Gemini and Claude independently and admits a training "
             "label only when they agree. 'auto' prefers 'dual' when both keys are present.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Legacy single key; applied to Claude when --engine claude, otherwise to Gemini.",
    )
    parser.add_argument("--gemini-api-key", default=None, help="Overrides GEMINI_API_KEY.")
    parser.add_argument("--anthropic-api-key", default=None, help="Overrides ANTHROPIC_API_KEY.")
    parser.add_argument("--ollama-url", default=None, help="Ollama server endpoint")
    parser.add_argument("--limit-bulletins", type=int, default=25, help="Max bulletins to parse")
    parser.add_argument("--limit-visual", type=int, default=10, help="Max visual pairs to inspect")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulate audit without mutating databases"
    )
    parser.add_argument(
        "--allow-synthetic-training-rows",
        action="store_true",
        help="Permit scars seen only in synthetic fixture thumbnails to enter the training "
             "catalog. Off by default: synthetic imagery is not evidence of a real landslide.",
    )
    args = parser.parse_args()

    pipeline = WeeklyAIAuditPipeline(
        engine=args.engine,
        api_key=args.api_key,
        gemini_api_key=args.gemini_api_key,
        anthropic_api_key=args.anthropic_api_key,
        ollama_url=args.ollama_url,
        dry_run=args.dry_run,
        allow_synthetic_training_rows=args.allow_synthetic_training_rows,
    )
    pipeline.execute_complete_weekly_audit(
        limit_bulletins=args.limit_bulletins,
        limit_visual=args.limit_visual,
    )


if __name__ == "__main__":
    main()
