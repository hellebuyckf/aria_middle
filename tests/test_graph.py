import os

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import run_analysis, run_report
from core.state import ARIAState, PubMedReference
from models.metrics import BiomechanicalMetrics
from services.pose.mediapipe_service import Landmark, PoseLandmarks

pytestmark = pytest.mark.asyncio

_FAKE_LANDMARK = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99)
_FAKE_POSE = PoseLandmarks(frame_index=0, landmarks=[_FAKE_LANDMARK] * 33)
_FAKE_FRAMES = [np.zeros((480, 640, 3), dtype=np.uint8)] * 10
_FAKE_METRICS = BiomechanicalMetrics(
    cadence=172.0,
    angle_attaque_pied=8.5,
    flexion_genou_impact=18.0,
    inclinaison_tronc=7.0,
    oscillation_verticale=6.5,
    ratio_contact_suspension=0.60,
)
_FAKE_REFS: list[PubMedReference] = [
    PubMedReference(
        pmid="11111111",
        titre="Running cadence and injury risk",
        extrait="Higher cadence reduces impact loading.",
    ),
    PubMedReference(
        pmid="22222222",
        titre="Trunk lean biomechanics",
        extrait="Forward lean optimises energy transfer.",
    ),
]

_INITIAL_STATE = ARIAState(
    session_id="SES-graph-001",
    patient_id="PAT-042",
    video_path="data/sessions/sagittale_test.mp4",
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
    diagnostic=None,
    rag_refs=[],
    prompt=None,
    report=None,
    statut="idle",
    erreur=None,
)


@pytest.mark.skipif(
    not os.path.exists("data/sessions/sagittale_test.mp4"),
    reason="Vidéo de test absente — test ignoré",
)
@patch(
    "agents.rag_agent.ChromaDBService",
    return_value=AsyncMock(retrieve=AsyncMock(return_value=_FAKE_REFS)),
)
@patch("agents.video_agent.calculate_metrics", return_value=_FAKE_METRICS)
@patch(
    "agents.video_agent.detect_pose",
    return_value=[_FAKE_POSE] * 10,
)
@patch("agents.video_agent.extract_frames", return_value=_FAKE_FRAMES)
async def test_pipeline_analyse(
    _mock_extract: MagicMock,
    _mock_detect: MagicMock,
    _mock_calc: MagicMock,
    _mock_chroma: MagicMock,
) -> None:
    result = await run_analysis(_INITIAL_STATE)

    assert result["statut"] == "pret"
    assert result["metrics"] is not None
    assert result["diagnostic"] is not None
    assert isinstance(result["rag_refs"], list)
    assert len(result["rag_refs"]) > 0
    assert result["erreur"] is None


@pytest.mark.skipif(
    not os.path.exists("data/sessions/sagittale_test.mp4"),
    reason="Vidéo de test absente — test ignoré",
)
@patch(
    "agents.rag_agent.ChromaDBService",
    return_value=AsyncMock(retrieve=AsyncMock(return_value=_FAKE_REFS)),
)
@patch("agents.video_agent.calculate_metrics", return_value=_FAKE_METRICS)
@patch("agents.video_agent.detect_pose", return_value=[_FAKE_POSE] * 10)
@patch("agents.video_agent.extract_frames", return_value=_FAKE_FRAMES)
async def test_pipeline_complet(
    _mock_extract: MagicMock,
    _mock_detect: MagicMock,
    _mock_calc: MagicMock,
    _mock_chroma: MagicMock,
) -> None:
    analyse = await run_analysis(_INITIAL_STATE)
    assert analyse["statut"] == "pret"

    result = await run_report(analyse)
    assert result["statut"] == "rapport"
    assert result["report"] is not None
    assert result["erreur"] is None


async def test_pipeline_fichier_manquant() -> None:
    state = ARIAState(
        session_id="SES-graph-missing",
        patient_id="PAT-042",
        video_path="/inexistant/video.mp4",
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
        diagnostic=None,
        rag_refs=[],
        prompt=None,
        report=None,
        statut="idle",
        erreur=None,
    )
    result = await run_analysis(state)

    assert result["statut"] == "erreur"
    assert result["erreur"] is not None
    assert "introuvable" in result["erreur"]
