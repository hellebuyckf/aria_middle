"""Tests pour services/pose/metrics_calculator.py.

Les landmarks synthétiques simulent une course sagittale à fréquence contrôlée.
Coordonnées normalisées MediaPipe : Y=0 haut, Y=1 bas.
"""

import math
from dataclasses import dataclass

import pytest

from services.pose.metrics_calculator import calculate_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class SyntheticLandmark:
    """Landmark minimal compatible avec l'interface de calculate_metrics."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.99


def _make_frame(
    hip_y: float,
    heel_y: float,
    shoulder_y: float = 0.2,
    knee_x: float = 0.55,
) -> list[SyntheticLandmark]:
    """Génère une frame avec les 33 landmarks, dont 6 sagittaux significatifs."""
    frame = [SyntheticLandmark(x=0.5, y=0.5) for _ in range(33)]

    frame[11] = SyntheticLandmark(x=0.5, y=shoulder_y)  # LEFT_SHOULDER
    frame[23] = SyntheticLandmark(x=0.5, y=hip_y)  # LEFT_HIP
    frame[25] = SyntheticLandmark(x=knee_x, y=0.62)  # LEFT_KNEE
    frame[27] = SyntheticLandmark(x=0.5, y=0.82)  # LEFT_ANKLE
    frame[29] = SyntheticLandmark(x=0.35, y=heel_y)  # LEFT_HEEL
    frame[31] = SyntheticLandmark(x=0.45, y=heel_y + 0.02)  # LEFT_FOOT_INDEX

    return frame


def _synthetic_sequence(
    n_seconds: float = 10.0,
    fps: float = 50.0,
    freq_hz: float = 3.0,
) -> list[list]:
    """Séquence synthétique à fréquence contrôlée.

    La hanche oscille à freq_hz → cadence attendue = freq_hz × 60 spm.
    Le talon oscille en opposition de phase pour fournir des IC détectables.
    """
    n_frames = int(n_seconds * fps)
    frames = []
    for i in range(n_frames):
        t = i / fps
        hip_y = 0.5 + 0.05 * math.sin(2 * math.pi * freq_hz * t)
        # Talon en opposition : minimum heel_y ≈ IC dans la convention ARIA
        heel_y = 0.75 - 0.08 * math.sin(2 * math.pi * freq_hz * t)
        frames.append(_make_frame(hip_y=hip_y, heel_y=heel_y))
    return frames


# ---------------------------------------------------------------------------
# Tests cadence
# ---------------------------------------------------------------------------


def test_cadence_180_spm() -> None:
    """Sinus à 3 Hz → cadence dans [175, 185] spm."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0, freq_hz=3.0)
    metrics = calculate_metrics(frames, fps=50.0)
    assert 175.0 <= metrics.cadence_spm <= 185.0, (
        f"Cadence hors plage : {metrics.cadence_spm}"
    )


def test_cadence_170_spm() -> None:
    """Sinus à 2.83 Hz → cadence dans [165, 175] spm."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0, freq_hz=170 / 60)
    metrics = calculate_metrics(frames, fps=50.0)
    assert 163.0 <= metrics.cadence_spm <= 177.0, (
        f"Cadence hors plage : {metrics.cadence_spm}"
    )


# ---------------------------------------------------------------------------
# Tests oscillation verticale
# ---------------------------------------------------------------------------


def test_oscillation_avec_taille() -> None:
    """Avec taille_patient, oscillation_verticale_cm est convertie et approximatif=False."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0, taille_patient_cm=175.0)

    assert metrics.oscillation_verticale_cm is not None
    assert metrics.approximatif is False
    assert metrics.oscillation_verticale_cm > 0


def test_oscillation_sans_taille() -> None:
    """Sans taille_patient, approximatif=True et valeur retournée est normalisée."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert metrics.oscillation_verticale_cm is not None
    assert metrics.approximatif is True
    # Valeur normalisée : doit être < 1.0 (coordonnées 0-1)
    assert 0.0 < metrics.oscillation_verticale_cm < 1.0


# ---------------------------------------------------------------------------
# Tests ratio contact/suspension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cadence, expected_min, expected_max",
    [
        (160.0, 0.598, 0.602),  # formule : 0.6 - (160-160)*0.002 = 0.6
        (180.0, 0.558, 0.562),  # 0.6 - (180-160)*0.002 = 0.56
        (100.0, 0.649, 0.651),  # clamp à 0.65
        (300.0, 0.349, 0.351),  # 0.6-(300-160)*0.002=0.32 → clamp à 0.35
    ],
)
def test_ratio_contact_approximation(
    cadence: float, expected_min: float, expected_max: float
) -> None:
    """Approximation Morin 2011 avec clamp [0.35, 0.65]."""
    from services.pose.metrics_calculator import _ratio_contact

    ratio = _ratio_contact(cadence)
    assert expected_min <= ratio <= expected_max, (
        f"ratio={ratio} hors [{expected_min}, {expected_max}]"
    )


# ---------------------------------------------------------------------------
# Tests cycles et IC
# ---------------------------------------------------------------------------


def test_nb_cycles_detectes() -> None:
    """10 secondes à 3 Hz → ~30 cycles détectés."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0, freq_hz=3.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert 25 <= metrics.nb_cycles_analyses <= 35, (
        f"Cycles : {metrics.nb_cycles_analyses}"
    )
    assert len(metrics.cycles) == metrics.nb_cycles_analyses


def test_cycles_contiennent_angles() -> None:
    """Chaque GaitCycle a angle_attaque et flexion_genou définis."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    for cycle in metrics.cycles:
        assert isinstance(cycle.angle_attaque, float)
        assert isinstance(cycle.flexion_genou, float)
        assert isinstance(cycle.frame_ic, int)


# ---------------------------------------------------------------------------
# Tests erreurs et cas limites
# ---------------------------------------------------------------------------


def test_erreur_sequence_vide() -> None:
    """Séquence vide → ValueError."""
    with pytest.raises(ValueError, match="aucune frame valide"):
        calculate_metrics([])


def test_erreur_trop_peu_ic() -> None:
    """Vidéo trop courte pour détecter 3 IC → ValueError."""
    # 0.3 seconde à 50fps = 15 frames → moins d'un cycle à 3 Hz → < 3 IC
    frames = _synthetic_sequence(n_seconds=0.3, fps=50.0, freq_hz=3.0)
    with pytest.raises(ValueError, match="Signal insuffisant"):
        calculate_metrics(frames, fps=50.0)


def test_to_dict() -> None:
    """to_dict() retourne un dict JSON-sérialisable."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)
    d = metrics.to_dict()

    assert isinstance(d, dict)
    assert "cadence_spm" in d
    assert "cycles" in d
    assert isinstance(d["cycles"], list)


def test_confiance_detection_vaut_1() -> None:
    """Toutes les frames valides → confiance_detection = 1.0."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)
    assert metrics.confiance_detection == 1.0
