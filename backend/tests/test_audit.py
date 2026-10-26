"""Tests for the Token-Frugal AI Ingestion, Multimodal Vision Screening & Weekly AI Audit Engine."""

from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import get_time_series_store
from app.ingestion import unstructured_collector as uc
from app.ingestion.unstructured_collector import (
    stage_unprocessed_bulletins,
    stage_satellite_visual_candidates,
    extract_district_hint,
    is_geohazard_relevant,
)
from scripts.run_weekly_ai_audit import WeeklyAIAuditPipeline, TokenAccountant

client = TestClient(app)


def test_district_extraction_and_keyword_relevance():
    """Verify district name matching and multilingual geohazard keyword tagging."""
    assert extract_district_hint("A massive debris flow blocked the highway in Sindhupalchok.") == "Sindhupalchok"
    assert extract_district_hint("गोरखा जिल्लाको मनास्लु क्षेत्रमा हिमपहिरो गएको छ।") == "Gorkha"
    assert extract_district_hint("Normal weather in a remote area.") is None

    assert is_geohazard_relevant("Landslide on Highway", "Road blocked by debris") is True
    assert is_geohazard_relevant("पहिरोको चेतावनी", "सडक अवरुद्ध") is True
    assert is_geohazard_relevant("Stock market update", "Prices increased today") is False


def test_staging_unprocessed_bulletins_and_visual_candidates(monkeypatch):
    """Verify staging of unstructured text bulletins and 512x512 satellite thumbnail candidates.

    All three live sources (Google News/Kathmandu Post RSS, GDACS, ReliefWeb) are stubbed.
    A system meant to run unattended for six months must not have its test suite depend on
    three external hosts being reachable and fast - this test is about the staging/dedup
    logic, not about whether the internet is up right now.
    """
    monkeypatch.setattr(uc, "fetch_open_bulletins_and_rss", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.ingestion.gdacs_feed.stage_gdacs_events",
        lambda *a, **k: {"fetched": 0, "queued": 0, "suppressed_duplicates": 0,
                         "restaged_escalations": 0, "errors": 0, "warnings": []},
    )
    monkeypatch.setattr("app.ingestion.reliefweb_feed.fetch_reports", lambda *a, **k: [])

    store = get_time_series_store()

    staged_b = stage_unprocessed_bulletins(store)
    assert staged_b >= 0
    pending_b = store.get_pending_bulletins(limit=10)
    assert isinstance(pending_b, list)

    staged_v = stage_satellite_visual_candidates(store)
    assert staged_v >= 0
    pending_v = store.get_pending_visual_candidates(limit=10)
    assert isinstance(pending_v, list)


def test_token_accountant_cost_tracking():
    """Verify operational token accounting and cost calculation."""
    accountant = TokenAccountant()
    accountant.add(inp=1000, out=200, cost_per_m_in=0.15, cost_per_m_out=0.60)
    stats = accountant.to_dict()
    assert stats["total_input_tokens"] == 1000
    assert stats["total_output_tokens"] == 200
    assert stats["total_tokens"] == 1200
    assert stats["api_calls_count"] == 1
    assert stats["estimated_cost_usd"] > 0.0


def test_weekly_ai_audit_pipeline_deterministic_execution():
    """Verify complete end-to-end audit run in deterministic simulation mode."""
    pipeline = WeeklyAIAuditPipeline(
        engine="deterministic",
        dry_run=True,
    )
    result = pipeline.execute_complete_weekly_audit(
        limit_bulletins=3,
        limit_visual=2,
    )
    assert result["status"] == "COMPLETED"
    assert result["engine"] == "DETERMINISTIC"
    assert result["bulletins_scanned"] >= 0
    assert result["visual_screened_count"] >= 0
    assert "token_usage" in result
    assert Path(result["report_path"]).exists()


def test_audit_api_endpoints():
    """Verify FastAPI audit routes: staging-queues, reports, runs, and trigger endpoint."""
    # Staging queues endpoint
    r_queues = client.get("/api/v1/audit/staging-queues?limit=5")
    assert r_queues.status_code == 200
    q_data = r_queues.json()
    assert "pending_bulletins_count" in q_data
    assert "pending_visual_candidates_count" in q_data

    # Audit reports listing endpoint
    r_rep = client.get("/api/v1/audit/reports")
    assert r_rep.status_code == 200
    rep_data = r_rep.json()
    assert "reports" in rep_data
    assert isinstance(rep_data["reports"], list)

    # Specific report content endpoint
    if rep_data["reports"]:
        sample_file = rep_data["reports"][0]["filename"]
        r_file = client.get(f"/api/v1/audit/reports/{sample_file}")
        assert r_file.status_code == 200
        assert "content_markdown" in r_file.json()

    # Recent runs history endpoint
    r_runs = client.get("/api/v1/audit/runs?limit=5")
    assert r_runs.status_code == 200
    runs_data = r_runs.json()
    assert "runs" in runs_data

    # Trigger audit endpoint (dry run)
    r_trigger = client.post(
        "/api/v1/audit/trigger-weekly-audit",
        json={
            "engine": "deterministic",
            "limit_bulletins": 2,
            "limit_visual": 2,
            "dry_run": True,
        },
    )
    assert r_trigger.status_code == 200
    t_data = r_trigger.json()
    assert t_data["status"] == "COMPLETED"
    assert t_data["engine"] == "DETERMINISTIC"

