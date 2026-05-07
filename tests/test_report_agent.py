from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from agents.report_agent import report_agent
from core.state import ARIAState, PubMedReference
from models.diagnostic import DiagnosticLLM
from models.report import ARIAReport
from models.metrics import BiomechanicalMetrics

pytestmark = pytest.mark.asyncio


@pytest.fixture
def state_for_report() -> ARIAState:
    metrics = BiomechanicalMetrics(
        cadence=147.5,
        angle_attaque_pied=1.1,
        flexion_genou_impact=16.8,
        inclinaison_tronc=34.5,
        oscillation_verticale=2.1,
        ratio_contact_suspension=0.625,
        pelvic_drop=4.4,
        valgus_genou=7.7,
        asymetrie_charge=2.3,
        oscillation_laterale_hanche=1.7,
        pronation_pied=14.2,
        vue_posterieure_disponible=True,
    )
    diagnostic = DiagnosticLLM(
        pathologie="Lombalgie",
        confiance="élevée",
        justification="Inclinaison tronc massive et cadence basse",
    )
    rag_refs: list[PubMedReference] = [
        {
            "pmid": "12345678",
            "titre": "Trunk lean and lumbar load during running",
            "extrait": "Forward trunk lean above 15° significantly increases lumbar compressive forces.",
        },
        {
            "pmid": "87654321",
            "titre": "Step rate and running injury prevention",
            "extrait": "Increasing step rate by 10% reduces ground reaction forces and injury risk.",
        },
    ]
    return ARIAState(
        session_id="SES-test-001",
        patient_id="PAT-test-001",
        video_path="/tmp/test.mp4",
        video_path_posterior=None,
        pathologie_declaree=None,
        age=None,
        taille_cm=None,
        poids_kg=None,
        km_semaine=None,
        niveau_pratique=None,
        profil_chaussure=None,
        strava_charge=None,
        garmin_charge=None,
        key_frames=[],
        metrics=metrics,
        diagnostic=diagnostic,
        rag_refs=rag_refs,
        prompt=None,
        report=None,
        statut="rag",
        erreur=None,
    )


@patch("agents.report_agent.llm.generate_report", new_callable=AsyncMock)
async def test_report_agent_returns_valid_report(
    mock_generate: AsyncMock,
    state_for_report: ARIAState,
) -> None:
    mock_generate.return_value = ARIAReport(
        session_id="SES-test-001",
        patient_id="PAT-test-001",
        pathologie="Lombalgie",
        confiance="élevée",
        justification_diagnostic="Inclinaison tronc massive et cadence basse",
        metriques_anormales=["inclinaison_tronc = 34.5° (norme : 5–10°)"],
        recommandations=[
            "Renforcement des abducteurs de hanche (3×15 reps, 3×/semaine)"
        ],
        references_pubmed=["Trunk lean and lumbar load during running"],
        avertissement="Ce rapport est généré par un système d'IA.",
        date_generation=datetime.now(timezone.utc).isoformat(),
    ).model_dump_json()

    result = await report_agent(state_for_report)

    assert result["statut"] == "rapport"
    assert result["report"] is not None
    assert result["report"].pathologie == "Lombalgie"
    assert isinstance(result["report"].recommandations, list)
    assert len(result["report"].recommandations) > 0
    assert isinstance(result["report"].metriques_anormales, list)
    assert len(result["report"].metriques_anormales) > 0
