import pytest
from unittest.mock import AsyncMock, patch

from agents.rag_agent import rag_agent
from core.state import ARIAState, PubMedReference
from models.diagnostic import DiagnosticLLM

pytestmark = pytest.mark.asyncio

FAKE_REFS: list[PubMedReference] = [
    PubMedReference(
        pmid="12345678", titre="Running and low back pain", extrait="Abstract text."
    ),
    PubMedReference(
        pmid="87654321", titre="Gait biomechanics review", extrait="Another abstract."
    ),
]


def _state(diagnostic: DiagnosticLLM | None) -> ARIAState:
    return ARIAState(
        session_id="SES-test-rag",
        patient_id="PAT-test-rag",
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
        metrics=None,
        diagnostic=diagnostic,
        rag_refs=[],
        prompt=None,
        report=None,
        statut="diagnostic",
        erreur=None,
    )


@patch(
    "agents.rag_agent.ChromaDBService",
    return_value=AsyncMock(retrieve=AsyncMock(return_value=FAKE_REFS)),
)
async def test_rag_agent_returns_refs(_mock_service) -> None:
    diagnostic = DiagnosticLLM(
        pathologie="Lombalgie",
        confiance="élevée",
        justification="test",
    )
    result = await rag_agent(_state(diagnostic))

    assert result["statut"] == "rag"
    assert result["erreur"] is None
    assert isinstance(result["rag_refs"], list)

    for ref in result["rag_refs"]:
        assert "pmid" in ref
        assert "titre" in ref
        assert "extrait" in ref


async def test_rag_agent_missing_diagnostic() -> None:
    result = await rag_agent(_state(None))

    assert result["statut"] == "erreur"
    assert "diagnostic" in result["erreur"]
    assert result["rag_refs"] == []
