import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.video_agent import ARIAVideoError, video_agent
from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from services.pose.mediapipe_service import Landmark, PoseLandmarks

SAMPLE_VIDEO = "tests/fixtures/sample.mp4"

_BASE_STATE = ARIAState(
    session_id="SES-TEST",
    patient_id="PAT-001",
    video_path=SAMPLE_VIDEO,
    metrics=None,
    rag_refs=[],
    prompt=None,
    report=None,
    statut="idle",
    erreur=None,
)

_FAKE_LANDMARK = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99)
_FAKE_POSE = PoseLandmarks(frame_index=0, landmarks=[_FAKE_LANDMARK] * 33)
_FAKE_FRAMES = [np.zeros((480, 640, 3), dtype=np.uint8)] * 10


def _make_landmarks(nb_ok: int, nb_none: int) -> list[PoseLandmarks | None]:
    return [_FAKE_POSE] * nb_ok + [None] * nb_none


@patch("agents.video_agent.calculate_metrics")
@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_succes(
    mock_extract: MagicMock, mock_detect: MagicMock, mock_calc: MagicMock
) -> None:
    """Pipeline nominal : statut passe à 'rag' et metrics est renseigné."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(10, 0)
    mock_calc.return_value = BiomechanicalMetrics(
        cadence_spm=172.0,
        nb_frames_analysees=10,
        nb_frames_echec=0,
    )

    result = video_agent(_BASE_STATE)

    assert result["statut"] == "rag"
    assert result["metrics"] is not None
    assert result["metrics"].cadence_spm == 172.0
    assert result["erreur"] is None


@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_taux_echec_depasse_seuil(
    mock_extract: MagicMock, mock_detect: MagicMock
) -> None:
    """Plus de 20% de frames sans détection → ARIAVideoError levée."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(7, 3)  # 30% d'échec

    with pytest.raises(ARIAVideoError, match="Taux d'échec MediaPipe"):
        video_agent(_BASE_STATE)


@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_taux_echec_sous_seuil(
    mock_extract: MagicMock, mock_detect: MagicMock
) -> None:
    """Exactement 20% d'échec → ne lève pas ARIAVideoError (seuil strict >)."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(8, 2)  # exactement 20%

    with patch("agents.video_agent.calculate_metrics") as mock_calc:
        mock_calc.return_value = BiomechanicalMetrics(
            nb_frames_analysees=8, nb_frames_echec=2
        )
        result = video_agent(_BASE_STATE)

    assert result["statut"] == "rag"


@patch("agents.video_agent.extract_frames", side_effect=OSError("fichier introuvable"))
def test_video_agent_erreur_extraction(mock_extract: MagicMock) -> None:
    """Exception générique → statut 'erreur', pas de levée d'exception."""
    assert mock_extract.side_effect is not None
    result = video_agent(_BASE_STATE)

    assert result["statut"] == "erreur"
    assert result["erreur"] == "fichier introuvable"
    assert result["metrics"] is None


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_VIDEO), reason="sample.mp4 absent des fixtures"
)
def test_video_agent_integration() -> None:
    """Test d'intégration complet avec une vraie vidéo (optionnel)."""
    result = video_agent(_BASE_STATE)
    assert result["statut"] == "rag"
    assert result["metrics"] is not None


def test_import_propre() -> None:
    """Vérifie que tous les imports de l'agent sont résolus sans erreur."""
    assert callable(video_agent)
    assert issubclass(ARIAVideoError, Exception)
    assert isinstance(_BASE_STATE, dict)
