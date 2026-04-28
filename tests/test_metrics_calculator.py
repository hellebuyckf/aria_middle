"""Tests pour services/pose/metrics_calculator.py.

Les landmarks synthétiques simulent une course sagittale à fréquence contrôlée.
Coordonnées normalisées MediaPipe : Y=0 haut, Y=1 bas.
"""

import math
from dataclasses import dataclass

import numpy as np
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
    frame[25] = SyntheticLandmark(x=knee_x, y=0.67)  # LEFT_KNEE
    frame[27] = SyntheticLandmark(x=0.5, y=0.85)  # LEFT_ANKLE
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
    assert metrics.cadence is not None
    assert 175.0 <= metrics.cadence <= 185.0, f"Cadence hors plage : {metrics.cadence}"


def test_cadence_170_spm() -> None:
    """Sinus à 2.83 Hz → cadence dans [163, 177] spm."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0, freq_hz=170 / 60)
    metrics = calculate_metrics(frames, fps=50.0)
    assert metrics.cadence is not None
    assert 163.0 <= metrics.cadence <= 177.0, f"Cadence hors plage : {metrics.cadence}"


# ---------------------------------------------------------------------------
# Tests oscillation verticale
# ---------------------------------------------------------------------------


def test_oscillation_verticale() -> None:
    """oscillation_verticale est calculée en cm depuis le midpoint 23/24."""
    frames = _synthetic_sequence(n_seconds=10.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert metrics.oscillation_verticale is not None
    assert 4.0 <= metrics.oscillation_verticale <= 10.0


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
# Tests erreurs et cas limites
# ---------------------------------------------------------------------------


def test_erreur_sequence_vide() -> None:
    """Séquence vide → ValueError."""
    with pytest.raises(ValueError, match="aucune frame valide"):
        calculate_metrics([])


def test_erreur_trop_peu_ic() -> None:
    """Vidéo trop courte pour détecter 3 IC → ValueError."""
    frames = _synthetic_sequence(n_seconds=0.3, fps=50.0, freq_hz=3.0)
    with pytest.raises(ValueError, match="Signal insuffisant"):
        calculate_metrics(frames, fps=50.0)


def test_to_dict() -> None:
    """to_dict() retourne un dict JSON-sérialisable avec les 6 clés."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)
    d = metrics.to_dict()

    assert isinstance(d, dict)
    assert "cadence" in d
    assert "angle_attaque_pied" in d
    assert "flexion_genou_impact" in d
    assert "inclinaison_tronc" in d
    assert "oscillation_verticale" in d
    assert "ratio_contact_suspension" in d


def test_tous_champs_renseignes() -> None:
    """Tous les champs sagittaux doivent être renseignés sur une séquence suffisante."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert metrics.cadence is not None
    assert metrics.angle_attaque_pied is not None
    assert metrics.flexion_genou_impact is not None
    assert metrics.inclinaison_tronc is not None
    assert metrics.oscillation_verticale is not None
    assert metrics.ratio_contact_suspension is not None


def test_angle_attaque_talon_positif() -> None:
    """Attaque talon : toe au-dessus du talon → angle positif (convention clinique)."""
    from services.pose.metrics_calculator import _angle_attaque_moyen

    # Coureur allant à droite : heel(0.35, Y-down=0.88), toe(0.47, Y-down=0.85)
    # En Y-up : heel_y=0.12, toe_y=0.15 → toe au-dessus → positif
    frame = [SyntheticLandmark(x=0.5, y=0.5) for _ in range(33)]
    frame[29] = SyntheticLandmark(x=0.35, y=0.88)  # LEFT_HEEL  (sol)
    frame[31] = SyntheticLandmark(x=0.47, y=0.85)  # LEFT_FOOT_INDEX (levé)
    angle = _angle_attaque_moyen([frame], np.array([0]))
    assert angle > 0, f"Attaque talon devrait être positif, obtenu {angle:.1f}°"
    assert -20.0 <= angle <= 20.0, f"Hors plage [-20°, +20°] : {angle:.1f}°"


def test_angle_attaque_avantpied_negatif() -> None:
    """Attaque avant-pied : toe en-dessous du talon → angle négatif."""
    from services.pose.metrics_calculator import _angle_attaque_moyen

    # heel(0.35, Y-down=0.85), toe(0.47, Y-down=0.88)
    # En Y-up : heel_y=0.15, toe_y=0.12 → toe en-dessous → négatif
    frame = [SyntheticLandmark(x=0.5, y=0.5) for _ in range(33)]
    frame[29] = SyntheticLandmark(x=0.35, y=0.85)  # LEFT_HEEL
    frame[31] = SyntheticLandmark(x=0.47, y=0.88)  # LEFT_FOOT_INDEX (plus bas)
    angle = _angle_attaque_moyen([frame], np.array([0]))
    assert angle < 0, f"Attaque avant-pied devrait être négatif, obtenu {angle:.1f}°"
    assert -20.0 <= angle <= 20.0, f"Hors plage [-20°, +20°] : {angle:.1f}°"


def test_angle_attaque_invariant_sens_deplacement() -> None:
    """Même angle absolu que le coureur aille à gauche ou à droite."""
    from services.pose.metrics_calculator import _angle_attaque_moyen

    def make_frame(heel_x: float, foot_x: float) -> list:
        f = [SyntheticLandmark(x=0.5, y=0.5) for _ in range(33)]
        f[29] = SyntheticLandmark(x=heel_x, y=0.88)
        f[31] = SyntheticLandmark(x=foot_x, y=0.85)
        return f

    ic = np.array([0])
    angle_droite = _angle_attaque_moyen([make_frame(0.35, 0.47)], ic)  # toe à droite
    angle_gauche = _angle_attaque_moyen([make_frame(0.65, 0.53)], ic)  # toe à gauche
    assert abs(angle_droite - angle_gauche) < 0.1, (
        f"Angles divergent selon le sens : {angle_droite:.2f}° vs {angle_gauche:.2f}°"
    )


def test_angle_attaque_plage_clinique() -> None:
    """angle_attaque_pied calculé aux IC (maxima heel_y) → plage [-15°, +25°]."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert metrics.angle_attaque_pied is not None
    assert -15.0 <= metrics.angle_attaque_pied <= 25.0, (
        f"Angle attaque hors plage [-15°, +25°] : {metrics.angle_attaque_pied}°"
    )


def test_flexion_genou_plage_clinique() -> None:
    """flexion_genou_impact = 180 - angle_géométrique → plage clinique [15°, 45°]."""
    frames = _synthetic_sequence(n_seconds=5.0, fps=50.0)
    metrics = calculate_metrics(frames, fps=50.0)

    assert metrics.flexion_genou_impact is not None
    assert 15.0 <= metrics.flexion_genou_impact <= 45.0, (
        f"Flexion genou hors plage clinique : {metrics.flexion_genou_impact}°"
    )
