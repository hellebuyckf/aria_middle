import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.video_agent import ARIAVideoError, video_agent
from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from services.pose.mediapipe_service import Landmark, PoseLandmarks

_SAGITTALE_PATH = "data/video/lombalgie/sagittale_lomb.mp4"
_POSTERIEURE_PATH = "data/video/lombalgie/arriere_lomb.mp4"

_FAKE_LANDMARK = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.99)
_FAKE_POSE = PoseLandmarks(frame_index=0, landmarks=[_FAKE_LANDMARK] * 33)
_FAKE_FRAMES = [np.zeros((480, 640, 3), dtype=np.uint8)] * 10


def _make_landmarks(nb_ok: int, nb_none: int) -> list[PoseLandmarks | None]:
    return [_FAKE_POSE] * nb_ok + [None] * nb_none


# ---------------------------------------------------------------------------
# Fixture partagée
# ---------------------------------------------------------------------------


@pytest.fixture
def session_state() -> ARIAState:
    """ARIAState minimal pour les tests video_agent."""
    return ARIAState(
        session_id="SES-test-001",
        patient_id="PAT-042",
        video_path=_SAGITTALE_PATH,
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


# ---------------------------------------------------------------------------
# Tests unitaires (mocks)
# ---------------------------------------------------------------------------


@patch("agents.video_agent.os.path.exists", return_value=True)
@patch("agents.video_agent.calculate_metrics")
@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_succes(
    mock_extract: MagicMock,
    mock_detect: MagicMock,
    mock_calc: MagicMock,
    _mock_exists: MagicMock,
) -> None:
    """Pipeline nominal : statut passe à 'rag' et metrics est renseigné."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(10, 0)
    mock_calc.return_value = BiomechanicalMetrics(
        cadence=172.0,
        angle_attaque_pied=8.5,
        flexion_genou_impact=18.0,
        inclinaison_tronc=7.0,
        oscillation_verticale=6.5,
        ratio_contact_suspension=0.60,
    )

    state = ARIAState(
        session_id="SES-TEST",
        patient_id="PAT-001",
        video_path="tests/fixtures/sample.mp4",
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
    result = video_agent(state)

    assert result["statut"] == "rag"
    assert result["metrics"] is not None
    assert result["metrics"].cadence == 172.0
    assert result["erreur"] is None


@patch("agents.video_agent.os.path.exists", return_value=True)
@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_taux_echec_depasse_seuil(
    mock_extract: MagicMock, mock_detect: MagicMock, _mock_exists: MagicMock
) -> None:
    """Plus de 20% de frames sans détection → ARIAVideoError levée."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(7, 3)  # 30% d'échec

    state = ARIAState(
        session_id="SES-TEST",
        patient_id="PAT-001",
        video_path="tests/fixtures/sample.mp4",
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
    with pytest.raises(ARIAVideoError, match="Taux d'échec MediaPipe"):
        video_agent(state)


@patch("agents.video_agent.os.path.exists", return_value=True)
@patch("agents.video_agent.detect_pose")
@patch("agents.video_agent.extract_frames")
def test_video_agent_taux_echec_sous_seuil(
    mock_extract: MagicMock, mock_detect: MagicMock, _mock_exists: MagicMock
) -> None:
    """Exactement 20% d'échec → ne lève pas ARIAVideoError (seuil strict >)."""
    mock_extract.return_value = _FAKE_FRAMES
    mock_detect.return_value = _make_landmarks(8, 2)  # exactement 20%

    state = ARIAState(
        session_id="SES-TEST",
        patient_id="PAT-001",
        video_path="tests/fixtures/sample.mp4",
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
    with patch("agents.video_agent.calculate_metrics") as mock_calc:
        mock_calc.return_value = BiomechanicalMetrics(
            cadence=172.0,
            angle_attaque_pied=8.5,
            flexion_genou_impact=18.0,
            inclinaison_tronc=7.0,
            oscillation_verticale=6.5,
            ratio_contact_suspension=0.60,
        )
        result = video_agent(state)

    assert result["statut"] == "rag"


@patch("agents.video_agent.os.path.exists", return_value=True)
@patch("agents.video_agent.extract_frames", side_effect=OSError("fichier introuvable"))
def test_video_agent_erreur_extraction(
    mock_extract: MagicMock, _mock_exists: MagicMock
) -> None:
    """Exception générique → statut 'erreur', pas de levée d'exception."""
    assert mock_extract.side_effect is not None

    state = ARIAState(
        session_id="SES-TEST",
        patient_id="PAT-001",
        video_path="tests/fixtures/sample.mp4",
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
    result = video_agent(state)

    assert result["statut"] == "erreur"
    assert result["erreur"] == "fichier introuvable"
    assert result["metrics"] is None


def test_video_agent_fichier_introuvable() -> None:
    """video_path inexistant → statut 'erreur' immédiat, sans appeler extract_frames."""
    state = ARIAState(
        session_id="SES-TEST",
        patient_id="PAT-001",
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
    result = video_agent(state)

    assert result["statut"] == "erreur"
    assert result["erreur"] is not None and "introuvable" in result["erreur"]
    assert result["metrics"] is None


def test_import_propre() -> None:
    """Vérifie que tous les imports de l'agent sont résolus sans erreur."""
    assert callable(video_agent)
    assert issubclass(ARIAVideoError, Exception)


# ---------------------------------------------------------------------------
# Tests d'intégration (vraies vidéos)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(_SAGITTALE_PATH),
    reason=f"{_SAGITTALE_PATH} absent — test d'intégration ignoré",
)
def test_sagittal_seul(session_state: ARIAState) -> None:
    """Pipeline sagittal seul : métriques calculées, pas de vue postérieure."""
    result = video_agent(session_state)

    assert result["statut"] == "rag"
    assert result["metrics"] is not None
    assert result["metrics"].vue_posterieure_disponible is False
    assert result["erreur"] is None


@pytest.mark.skipif(
    not os.path.exists(_POSTERIEURE_PATH),
    reason=f"{_POSTERIEURE_PATH} absent — test d'intégration ignoré",
)
def test_bilateral(session_state: ARIAState) -> None:
    """Pipeline bilatéral : métriques postérieures fusionnées dans BiomechanicalMetrics."""
    state: ARIAState = {**session_state, "video_path_posterior": _POSTERIEURE_PATH}  # type: ignore[assignment]
    result = video_agent(state)

    assert result["statut"] == "rag"
    assert result["metrics"] is not None
    assert result["metrics"].vue_posterieure_disponible is True
    assert result["metrics"].pelvic_drop is not None
    assert result["metrics"].valgus_genou is not None
