import pytest

from agents.diagnosis_agent import diagnosis_agent
from core.state import ARIAState
from models.metrics import BiomechanicalMetrics

pytestmark = pytest.mark.asyncio


@pytest.fixture
def state_with_metrics() -> ARIAState:
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
    return ARIAState(
        session_id="SES-test-001",
        patient_id="PAT-test-001",
        video_path="/tmp/test.mp4",
        video_path_posterior=None,
        pathologie_declaree=None,
        metrics=metrics,
        diagnostic=None,
        rag_refs=[],
        prompt=None,
        report=None,
        statut="video",
        erreur=None,
    )


async def test_diagnosis_agent_returns_diagnostic(
    state_with_metrics: ARIAState,
) -> None:
    result = await diagnosis_agent(state_with_metrics)

    assert result["statut"] == "diagnostic"
    assert result["diagnostic"] is not None
    assert isinstance(result["diagnostic"].pathologie, str)
    assert result["diagnostic"].pathologie != ""
    assert result["diagnostic"].confiance in ("élevée", "modérée", "faible")
