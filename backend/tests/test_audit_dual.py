"""Tests for the dual-model (Gemini + Claude) consensus gate in the weekly AI audit.

Every test runs fully offline with no API keys: both model adapters are monkeypatched, so no
request ever leaves the host. The behaviour under test is the admission gate that keeps
unvalidated rows out of `live_ground_truth_feedback.csv` during the 6-month unattended run.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from app.ingestion.unstructured_collector import generate_synthetic_thumbnail_pair
from scripts.run_weekly_ai_audit import (
    ENGINE_DETERMINISTIC,
    ENGINE_DUAL,
    PROMOTION_FITNESS_EPSILON,
    ConsensusDecision,
    ModelVote,
    TokenAccountant,
    WeeklyAIAuditPipeline,
    bulletin_votes_agree,
    classify_vision_features,
    compute_vision_features,
    cross_validate_votes,
    imagery_provenance,
    json_schema_to_gemini,
    merge_bulletin_votes,
    parse_json_defensively,
    vision_votes_agree,
)
import scripts.run_weekly_ai_audit as audit_mod


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------
GEMINI_EVENT = {
    "is_geohazard_event": True,
    "event_name": "Jhyaple Khola debris flow",
    "hazard_type": "debris_flow",
    "district": "Dhading",
    "latitude": 27.712,
    "longitude": 85.218,
    "elevation_m": 1450.0,
    "slope_deg": 38.0,
    "rainfall_72h_mm": 180.0,
    "confidence": 0.91,
}

# Same hazard, 0.004 deg away -> inside the 0.05 deg agreement tolerance.
CLAUDE_EVENT_AGREE = {**GEMINI_EVENT, "latitude": 27.716, "longitude": 85.214, "confidence": 0.84}

# Same hazard but ~0.6 deg away (a different district entirely) -> disagreement.
CLAUDE_EVENT_DISAGREE = {**GEMINI_EVENT, "latitude": 28.320, "longitude": 84.750}


class FakeStore:
    """Minimal in-memory stand-in for the time-series store's audit queues."""

    def __init__(self, bulletins=None, candidates=None):
        self._bulletins = bulletins or []
        self._candidates = candidates or []
        self.bulletin_status_calls: List[Dict[str, Any]] = []
        self.candidate_status_calls: List[Dict[str, Any]] = []
        self.audit_runs: List[Dict[str, Any]] = []

    def get_engine_name(self) -> str:
        return "FAKE_IN_MEMORY"

    def get_pending_bulletins(self, limit: int = 50):
        return list(self._bulletins[:limit])

    def update_bulletin_status(self, bulletin_id, status, extracted_event_id=None, parsed_json=None):
        self.bulletin_status_calls.append({"bulletin_id": bulletin_id, "status": status})

    def get_pending_visual_candidates(self, limit: int = 20):
        return list(self._candidates[:limit])

    def update_visual_candidate_status(self, candidate_id, status, ai_confidence, ai_notes):
        self.candidate_status_calls.append({"candidate_id": candidate_id, "status": status})

    def insert_ai_audit_run(self, run):
        self.audit_runs.append(run)

    def get_telemetry_outliers_for_audit(self, window_hours: int = 168):
        return {}


class FakeOptimizer:
    """Optimizer stub with a scripted fitness for each candidate parameter set."""

    def __init__(self, baseline_fitness: float, candidate_delta: float):
        self.active_params = {
            "c_res_base": 1.0, "w_cryo_plateau": 1.0, "w_lowland_flat": 1.0,
            "w_sti": 1.0, "w_ommi": 1.0, "w_gii": 1.0, "w_s_perp": 1.0,
            "w_tcwi": 1.0, "w_fault_dmg": 1.0, "prob_sigmoid_steepness": 1.0,
        }
        self.baseline_fitness = baseline_fitness
        self.candidate_delta = candidate_delta
        self.saved: List[Dict[str, float]] = []
        self._dataset = pd.DataFrame([{
            "label": 1, "event_name": "Test site", "latitude": 28.0, "longitude": 85.0,
            "elevation_m": 2000.0, "slope_deg": 40.0, "profile_curvature": -0.5,
            "local_relief_m": 300.0,
        }])

    def reload_dataset(self) -> int:
        return len(self._dataset)

    def compute_candidate_fs(self, row, params):
        return 1.30, 0.20

    def evaluate_parameter_set(self, params):
        # Everything clears the Rule 8.4 gate so the test isolates the epsilon check.
        return (0.95, 0.95, 0.95, 0.95, 1.0, 1.0, 1.0)

    def _compute_fitness(self, auc, rec, dev, cryo, fb):
        # The pipeline evaluates the champion first, then each proposal.
        if not getattr(self, "_baseline_taken", False):
            self._baseline_taken = True
            return self.baseline_fitness
        return self.baseline_fitness + self.candidate_delta

    def _save_active_params(self, params):
        self.saved.append(dict(params))


@pytest.fixture(scope="module")
def pipeline() -> WeeklyAIAuditPipeline:
    """One deterministic pipeline reused across tests (construction is slow)."""
    return WeeklyAIAuditPipeline(engine="deterministic", dry_run=True)


@pytest.fixture
def dual_pipeline(pipeline: WeeklyAIAuditPipeline) -> WeeklyAIAuditPipeline:
    """The shared pipeline switched into dual mode with placeholder (unused) keys."""
    pipeline.engine = ENGINE_DUAL
    pipeline.gemini_key = "test-gemini-key"
    pipeline.anthropic_key = "test-anthropic-key"
    pipeline.require_dual_consensus = True
    pipeline.dry_run = True
    pipeline.token_accountant = TokenAccountant()
    yield pipeline
    pipeline.engine = ENGINE_DETERMINISTIC
    pipeline.gemini_key = ""
    pipeline.anthropic_key = ""
    pipeline.dry_run = True


def _stub_votes(monkeypatch, pipeline, gemini: ModelVote, claude: ModelVote) -> None:
    """Replace both model adapters so no network call is possible."""
    monkeypatch.setattr(
        pipeline, "_gemini_vote", lambda *a, **k: gemini, raising=True
    )
    monkeypatch.setattr(
        pipeline, "_claude_vote", lambda *a, **k: claude, raising=True
    )


# ---------------------------------------------------------------------------
# 1. Consensus agreement admits
# ---------------------------------------------------------------------------
def test_agreement_admits_with_high_confidence(monkeypatch, dual_pipeline):
    """Both models agree on hazard type and coordinates -> HIGH confidence, admitted."""
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "OK", payload=CLAUDE_EVENT_AGREE),
    )
    decision = dual_pipeline._extract_disaster_consensus("Jhyaple Khola", "Debris flow", "Dhading")

    assert decision.status == "AGREED"
    assert decision.confidence_tier == "HIGH"
    assert decision.admit is True
    assert len(decision.votes) == 2
    # Merged coordinates are the mean of the two independent answers.
    assert decision.payload["latitude"] == pytest.approx(27.714, abs=1e-3)
    assert decision.payload["longitude"] == pytest.approx(85.216, abs=1e-3)
    # Merged confidence is the more conservative of the two.
    assert decision.payload["confidence"] == pytest.approx(0.84)


def test_agreement_writes_the_training_row(monkeypatch, dual_pipeline):
    """An agreed bulletin reaches append_ground_truth_event exactly once."""
    appended: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        audit_mod, "append_ground_truth_event",
        lambda **kw: appended.append(kw) or {"event_id": "ev_test_1"},
    )
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "OK", payload=CLAUDE_EVENT_AGREE),
    )
    store = FakeStore(bulletins=[{
        "bulletin_id": "bul_agree",
        "title": "Debris flow blocks Prithvi Highway",
        "raw_text": "A debris flow blocked the highway in Dhading.",
        "district_hint": "Dhading",
    }])
    monkeypatch.setattr(dual_pipeline, "store", store)
    dual_pipeline.dry_run = False

    scanned, extracted, quarantined = dual_pipeline.run_bulletin_nlp_audit(limit=5)

    assert scanned == 1
    assert len(extracted) == 1
    assert quarantined == []
    assert len(appended) == 1
    assert appended[0]["label"] == 1
    assert store.bulletin_status_calls[0]["status"] == "PARSED_VERIFIED"


# ---------------------------------------------------------------------------
# 2. Disagreement quarantines and does NOT admit
# ---------------------------------------------------------------------------
def test_disagreement_quarantines_and_does_not_admit(monkeypatch, dual_pipeline):
    """Coordinates ~0.6 deg apart are two different events -> NEEDS_HUMAN_REVIEW."""
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "OK", payload=CLAUDE_EVENT_DISAGREE),
    )
    decision = dual_pipeline._extract_disaster_consensus("Landslide", "Somewhere", "")

    assert decision.status == "NEEDS_HUMAN_REVIEW"
    assert decision.admit is False
    # Nothing is handed downstream, but both answers are preserved for the human.
    assert decision.payload is None
    assert len(decision.votes) == 2
    assert {v["model"] for v in decision.votes} == {"gemini", "claude"}


def test_disagreement_never_reaches_the_training_catalog(monkeypatch, dual_pipeline):
    """A quarantined bulletin must not call append_ground_truth_event at all."""
    appended: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        audit_mod, "append_ground_truth_event",
        lambda **kw: appended.append(kw) or {"event_id": "ev_should_not_exist"},
    )
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "OK", payload=CLAUDE_EVENT_DISAGREE),
    )
    store = FakeStore(bulletins=[{
        "bulletin_id": "bul_conflict",
        "title": "Landslide reported",
        "raw_text": "Reports of a landslide.",
        "district_hint": "",
    }])
    monkeypatch.setattr(dual_pipeline, "store", store)
    dual_pipeline.dry_run = False

    scanned, extracted, quarantined = dual_pipeline.run_bulletin_nlp_audit(limit=5)

    assert scanned == 1
    assert extracted == []
    assert len(quarantined) == 1
    assert quarantined[0]["item_id"] == "bul_conflict"
    assert appended == []
    assert store.bulletin_status_calls[0]["status"] == "NEEDS_HUMAN_REVIEW"


def test_hazard_type_mismatch_is_a_disagreement():
    """Same coordinates but a different hazard type is still a disagreement."""
    agreed, why = bulletin_votes_agree(
        GEMINI_EVENT, {**GEMINI_EVENT, "hazard_type": "glof"}
    )
    assert agreed is False
    assert "hazard type mismatch" in why


def test_both_models_agreeing_on_a_non_event_is_agreement():
    """Agreeing that a bulletin is not a geohazard is consensus, not a conflict."""
    non_event = {"is_geohazard_event": False, "hazard_type": "not_a_geohazard"}
    agreed, _ = bulletin_votes_agree(non_event, dict(non_event))
    assert agreed is True
    merged = merge_bulletin_votes(non_event, dict(non_event))
    assert merged["is_geohazard_event"] is False


# ---------------------------------------------------------------------------
# 3. A single surviving vote does not auto-admit
# ---------------------------------------------------------------------------
def test_single_vote_is_low_confidence_and_not_admitted(monkeypatch, dual_pipeline):
    """Claude abstaining leaves one vote: LOW confidence, blocked by the consensus policy."""
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "ABSTAINED", reason="refusal"),
    )
    decision = dual_pipeline._extract_disaster_consensus("Landslide", "text", "Dhading")

    assert decision.status == "SINGLE_VOTE"
    assert decision.confidence_tier == "LOW"
    assert decision.admit is False
    assert decision.payload == GEMINI_EVENT


def test_single_vote_admits_only_when_consensus_is_not_required(monkeypatch, dual_pipeline):
    """With AUDIT_REQUIRE_DUAL_CONSENSUS off, a lone vote is admitted but stays LOW."""
    dual_pipeline.require_dual_consensus = False
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "ABSTAINED", reason="HTTP 503"),
        ModelVote("claude", "OK", payload=CLAUDE_EVENT_AGREE),
    )
    decision = dual_pipeline._extract_disaster_consensus("Landslide", "text", "Dhading")

    assert decision.status == "SINGLE_VOTE"
    assert decision.confidence_tier == "LOW"
    assert decision.admit is True


def test_both_models_abstaining_falls_back_without_admitting(monkeypatch, dual_pipeline):
    """No usable votes -> deterministic fallback for reporting only, never admitted."""
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "ABSTAINED", reason="GEMINI_API_KEY not set"),
        ModelVote("claude", "ABSTAINED", reason="anthropic SDK not installed"),
    )
    decision = dual_pipeline._extract_disaster_consensus(
        "Landslide blocks road in Rasuwa", "Debris blocked the highway.", "Rasuwa"
    )

    assert decision.status == "NO_VOTES_DETERMINISTIC_FALLBACK"
    assert decision.admit is False
    assert decision.payload is not None  # still reported, just not trusted


def test_claude_refusal_is_an_abstention_not_a_crash():
    """stop_reason 'refusal' must degrade to a vote of ABSTAINED."""
    votes = [
        ModelVote("gemini", "OK", payload=GEMINI_EVENT),
        ModelVote("claude", "ABSTAINED", reason="refusal"),
    ]
    decision = cross_validate_votes(
        votes, bulletin_votes_agree, merge_bulletin_votes, require_dual=True
    )
    assert decision.status == "SINGLE_VOTE"
    assert decision.admit is False
    assert any(v["reason"] == "refusal" for v in decision.votes)


# ---------------------------------------------------------------------------
# 4. Deterministic engine reports zero API tokens
# ---------------------------------------------------------------------------
def test_deterministic_paths_report_zero_api_tokens(pipeline, tmp_path):
    """The offline parser and vision classifier must bill nothing (HANDOFF Issue #7)."""
    pipeline.token_accountant = TokenAccountant()

    pipeline._extract_deterministic_nlp(
        "Landslide blocks highway in Dhading", "Debris flow after heavy rain.", "Dhading"
    )
    before, after = generate_synthetic_thumbnail_pair(
        site_id="test_audit_dual_tokens", lat=27.7, lon=85.2, is_true_scar=True
    )
    try:
        pipeline._classify_deterministic_vision(before, after, {})
    finally:
        before.unlink(missing_ok=True)
        after.unlink(missing_ok=True)

    usage = pipeline.token_accountant.to_dict()
    assert usage["total_tokens"] == 0
    assert usage["total_input_tokens"] == 0
    assert usage["total_output_tokens"] == 0
    assert usage["api_calls_count"] == 0
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["by_engine"] == {}
    # The work still happened; it is just recorded as local, non-billable inference.
    assert usage["local_inference_calls"]["deterministic"] == 2


def test_token_accountant_separates_engines():
    """Per-engine accounting uses each provider's own rate card."""
    acc = TokenAccountant()
    acc.add(1_000_000, 0, cost_per_m_in=5.0, cost_per_m_out=25.0, engine="claude")
    acc.add(1_000_000, 0, cost_per_m_in=0.15, cost_per_m_out=0.60, engine="gemini")
    acc.record_local("deterministic")

    usage = acc.to_dict()
    assert usage["api_calls_count"] == 2
    assert usage["by_engine"]["claude"]["cost_usd"] == pytest.approx(5.0)
    assert usage["by_engine"]["gemini"]["cost_usd"] == pytest.approx(0.15)
    assert usage["local_inference_calls"] == {"deterministic": 1}
    assert usage["estimated_cost_usd"] == pytest.approx(5.15)


# ---------------------------------------------------------------------------
# 5. Promotion requires a strictly meaningful improvement
# ---------------------------------------------------------------------------
def _run_equation_audit(monkeypatch, pipeline, baseline, delta):
    pipeline.optimizer = FakeOptimizer(baseline_fitness=baseline, candidate_delta=delta)
    monkeypatch.setattr(pipeline, "store", FakeStore())
    pipeline.dry_run = False
    try:
        return pipeline.run_equation_discovery_audit(), pipeline.optimizer
    finally:
        pipeline.dry_run = True


def test_equal_fitness_does_not_promote(monkeypatch, pipeline):
    """'Promoted! (0.9311 > 0.9311)' must not happen again (HANDOFF Issue #6)."""
    result, optimizer = _run_equation_audit(monkeypatch, pipeline, 0.9311, 0.0)

    assert result["promoted"] is False
    assert optimizer.saved == []
    assert "did not" in result["reason"] or "no proposal improved" in result["reason"]


def test_sub_epsilon_improvement_does_not_promote(monkeypatch, pipeline):
    """A gain smaller than the epsilon is numerical noise, not an improvement."""
    result, optimizer = _run_equation_audit(
        monkeypatch, pipeline, 0.9311, PROMOTION_FITNESS_EPSILON / 2.0
    )

    assert result["promoted"] is False
    assert optimizer.saved == []


def test_improvement_above_epsilon_promotes_and_saves(monkeypatch, pipeline):
    """A real gain clears the gate and the champion is rewritten exactly once."""
    result, optimizer = _run_equation_audit(monkeypatch, pipeline, 0.9311, 0.01)

    assert result["promoted"] is True
    assert len(optimizer.saved) == 1
    assert result["promotion_epsilon"] == PROMOTION_FITNESS_EPSILON
    assert result["proposals"], "the report needs the evaluated proposals"


# ---------------------------------------------------------------------------
# 6. The deterministic vision classifier labels the generated fixtures correctly
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def generated_pairs():
    """Render one scar pair and one cloud/shadow pair from the real generator."""
    scar = generate_synthetic_thumbnail_pair(
        site_id="test_audit_dual_scar", lat=27.968, lon=85.942, is_true_scar=True
    )
    cloud = generate_synthetic_thumbnail_pair(
        site_id="test_audit_dual_cloud", lat=28.215, lon=85.515, is_true_scar=False
    )
    yield {"scar": scar, "cloud": cloud}
    for pair in (scar, cloud):
        for p in pair:
            Path(p).unlink(missing_ok=True)


def test_generated_scar_is_classified_as_a_true_scar(generated_pairs):
    """A rendered bare-earth chute must come back TRUE_LANDSLIDE_SCAR, not SNOWMELT_RETREAT."""
    before, after = generated_pairs["scar"]
    features = compute_vision_features(before, after)
    result = classify_vision_features(features)

    assert result["classification"] == "TRUE_LANDSLIDE_SCAR"
    assert result["is_true_scar"] is True
    # Must clear the 0.75 bar the audit uses before considering admission.
    assert result["confidence"] >= 0.75
    assert features["bare_earth_frac"] > 0.05
    assert features["white_cloud_frac"] == pytest.approx(0.0, abs=1e-6)


def test_generated_cloud_shadow_is_classified_as_an_artifact(generated_pairs):
    """A rendered cloud top plus cast shadow must come back CLOUD_SHADOW_ARTIFACT."""
    before, after = generated_pairs["cloud"]
    features = compute_vision_features(before, after)
    result = classify_vision_features(features)

    assert result["classification"] == "CLOUD_SHADOW_ARTIFACT"
    assert result["is_true_scar"] is False
    assert features["white_cloud_frac"] > 0.03
    assert features["bare_earth_frac"] == pytest.approx(0.0, abs=1e-6)


def test_identical_frames_are_not_a_scar(generated_pairs):
    """No change at all must never be read as a landslide."""
    before, _ = generated_pairs["scar"]
    features = compute_vision_features(before, before)
    result = classify_vision_features(features)

    assert result["is_true_scar"] is False
    assert result["classification"] == "SENSOR_NOISE"


def test_vision_result_declares_synthetic_provenance(pipeline, generated_pairs):
    """Synthetic fixtures must be labelled as such, not passed off as Sentinel-2 chips."""
    before, after = generated_pairs["scar"]
    result = pipeline._classify_deterministic_vision(before, after, {})

    assert result["imagery_provenance"] == "SYNTHETIC_FIXTURE"
    assert "SYNTHETIC" in result["evidence"]
    assert imagery_provenance(after) == "SYNTHETIC_FIXTURE"


def test_synthetic_scar_is_not_admitted_to_the_training_catalog(monkeypatch, pipeline, generated_pairs):
    """A confirmed scar seen only in a rendered fixture is held, never ingested."""
    appended: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        audit_mod, "append_ground_truth_event",
        lambda **kw: appended.append(kw) or {"event_id": "ev_synthetic"},
    )
    before, after = generated_pairs["scar"]
    store = FakeStore(candidates=[{
        "candidate_id": "test_audit_dual_scar",
        "site_id": "test_audit_dual_scar",
        "latitude": 27.968,
        "longitude": 85.942,
        "thumbnail_before_path": str(before),
        "thumbnail_after_path": str(after),
    }])
    monkeypatch.setattr(pipeline, "store", store)
    pipeline.dry_run = False
    try:
        inspected, results, quarantined = pipeline.run_satellite_visual_audit(limit=5)
    finally:
        pipeline.dry_run = True

    assert inspected == 1
    assert results[0]["classification"] == "TRUE_LANDSLIDE_SCAR"
    assert appended == [], "synthetic imagery must not become a training label"
    assert len(quarantined) == 1
    assert store.candidate_status_calls[0]["status"] == "SCAR_HELD_FOR_REVIEW"


# ---------------------------------------------------------------------------
# Engine resolution, vision consensus and adapter plumbing
# ---------------------------------------------------------------------------
def test_explicit_dual_engine_resolves(monkeypatch):
    """--engine dual always selects the consensus engine."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = WeeklyAIAuditPipeline(engine="dual", dry_run=True)
    assert p.engine == ENGINE_DUAL
    assert p.is_dual is True
    # With no keys, both models abstain rather than erroring.
    assert p.uses_gemini is False and p.uses_claude is False


def test_auto_prefers_dual_when_both_keys_present(monkeypatch):
    """'auto' upgrades to dual consensus as soon as both providers are configured."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-test")
    p = WeeklyAIAuditPipeline(engine="auto", dry_run=True)
    assert p.engine == ENGINE_DUAL
    assert p.uses_gemini is True and p.uses_claude is True


def test_vision_consensus_requires_an_identical_label(monkeypatch, dual_pipeline, generated_pairs):
    """Different vision labels quarantine the candidate instead of confirming a scar."""
    before, after = generated_pairs["scar"]
    _stub_votes(
        monkeypatch, dual_pipeline,
        ModelVote("gemini", "OK", payload={
            "is_true_scar": True, "classification": "TRUE_LANDSLIDE_SCAR",
            "confidence": 0.9, "evidence": "bare chute",
        }),
        ModelVote("claude", "OK", payload={
            "is_true_scar": False, "classification": "CLOUD_SHADOW_ARTIFACT",
            "confidence": 0.8, "evidence": "cast shadow",
        }),
    )
    decision, provenance = dual_pipeline._classify_satellite_change_consensus(
        str(before), str(after), {}
    )

    assert decision.status == "NEEDS_HUMAN_REVIEW"
    assert decision.admit is False
    assert provenance == "SYNTHETIC_FIXTURE"


def test_vision_agreement_merges_to_the_shared_label():
    """Identical labels agree; is_true_scar is derived from the label, not from either vote."""
    a = {"is_true_scar": True, "classification": "TRUE_LANDSLIDE_SCAR", "confidence": 0.95, "evidence": "x"}
    b = {"is_true_scar": True, "classification": "true_landslide_scar", "confidence": 0.80, "evidence": "y"}
    agreed, _ = vision_votes_agree(a, b)
    assert agreed is True

    decision = cross_validate_votes(
        [ModelVote("gemini", "OK", payload=a), ModelVote("claude", "OK", payload=b)],
        vision_votes_agree, audit_mod.merge_vision_votes, require_dual=True,
    )
    assert decision.status == "AGREED"
    assert decision.payload["is_true_scar"] is True
    assert decision.payload["confidence"] == pytest.approx(0.80)


def test_claude_adapter_abstains_without_a_key(pipeline):
    """No ANTHROPIC_API_KEY must abstain cleanly rather than raise."""
    pipeline.anthropic_key = ""
    vote = pipeline._claude_vote("hello", None)
    assert vote.status == "ABSTAINED"
    assert vote.payload is None


def test_gemini_adapter_abstains_without_a_key(pipeline):
    """No GEMINI_API_KEY must abstain cleanly rather than raise."""
    pipeline.gemini_key = ""
    vote = pipeline._gemini_vote("hello", None)
    assert vote.status == "ABSTAINED"
    assert vote.payload is None


def test_gemini_schema_translation_uses_the_openapi_dialect():
    """Gemini's responseSchema needs uppercase types and no additionalProperties."""
    translated = json_schema_to_gemini(audit_mod.VISION_SCHEMA)
    assert translated["type"] == "OBJECT"
    assert translated["properties"]["is_true_scar"]["type"] == "BOOLEAN"
    assert translated["properties"]["confidence"]["type"] == "NUMBER"
    assert "TRUE_LANDSLIDE_SCAR" in translated["properties"]["classification"]["enum"]
    assert "additionalProperties" not in translated


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go: {"a": 1} hope that helps', {"a": 1}),
        ("not json at all", None),
        ("[1, 2, 3]", None),
        (None, None),
    ],
)
def test_defensive_json_parsing(raw, expected):
    """Model text is always parsed with json.loads, never by string matching."""
    assert parse_json_defensively(raw) == expected
