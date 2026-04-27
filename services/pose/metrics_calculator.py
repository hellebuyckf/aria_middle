import math

import numpy as np
from loguru import logger
from scipy.signal import find_peaks, savgol_filter

from models.metrics import BiomechanicalMetrics, GaitCycle

# Indices BlazePose pour la vue sagittale gauche (CLAUDE.md §5)
_LEFT_SHOULDER = 11
_LEFT_HIP = 23
_LEFT_KNEE = 25
_LEFT_ANKLE = 27
_LEFT_HEEL = 29
_LEFT_FOOT_INDEX = 31

_MIN_IC = 3  # nombre minimum d'initial contacts pour une analyse valide


def _xy(frame: list, idx: int) -> tuple[float, float]:
    """Retourne (x, y) normalisés du landmark BlazePose d'index idx."""
    lm = frame[idx]
    return float(lm.x), float(lm.y)


def _smooth(signal: np.ndarray) -> np.ndarray:
    """Lisse un signal 1D avec Savitzky-Golay (window adaptatif, polyorder=3)."""
    n = len(signal)
    win = min(15, n)
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return signal
    return np.asarray(
        savgol_filter(signal, window_length=win, polyorder=min(3, win - 2))
    )


def _angle_horizontal(ax: float, ay: float, bx: float, by: float) -> float:
    """Angle en degrés entre le vecteur A→B et l'axe horizontal."""
    return math.degrees(math.atan2(by - ay, bx - ax))


def _angle_sommet(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    """Angle en degrés au sommet B formé par les vecteurs B→A et B→C."""
    v1 = np.array([ax - bx, ay - by], dtype=float)
    v2 = np.array([cx - bx, cy - by], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def _cadence(hip_y: np.ndarray, fps: float) -> float:
    """Cadence en spm depuis les minima de la série Y de la hanche.

    La hanche oscille à la fréquence des pas. Chaque minimum local = 1 pas.
    """
    hip_smooth = _smooth(hip_y)
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = find_peaks(-hip_smooth, distance=min_dist)
    duree_s = len(hip_y) / fps
    return (len(peaks) / duree_s) * 60.0 if duree_s > 0 and len(peaks) > 0 else 0.0


def _detecter_ic(heel_y: np.ndarray, fps: float) -> np.ndarray:
    """Détecte les frames d'initial contact via les minima locaux du talon.

    En coordonnées normalisées MediaPipe (Y=0 haut, Y=1 bas), un minimum
    de heel_y correspond au point de plus faible valeur Y, cohérent avec la
    détection du contact selon la convention de la spécification ARIA.
    """
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = find_peaks(-heel_y, distance=min_dist)
    return peaks


def _angle_attaque_moyen(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Angle moyen talon→pointe à l'impact (degrés)."""
    angles = [
        _angle_horizontal(
            *_xy(all_landmarks[f], _LEFT_HEEL), *_xy(all_landmarks[f], _LEFT_FOOT_INDEX)
        )
        for f in ic_frames
    ]
    return float(np.mean(angles)) if angles else 0.0


def _flexion_genou_moyenne(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Flexion genou moyenne hanche-genou-cheville à l'impact (degrés)."""
    angles = [
        _angle_sommet(
            *_xy(all_landmarks[f], _LEFT_HIP),
            *_xy(all_landmarks[f], _LEFT_KNEE),
            *_xy(all_landmarks[f], _LEFT_ANKLE),
        )
        for f in ic_frames
    ]
    return float(np.mean(angles)) if angles else 0.0


def _inclinaison_tronc(all_landmarks: list[list]) -> float:
    """Inclinaison tronc moyenne sur toutes les frames (degrés, positif = forward lean)."""
    angles = [
        math.degrees(
            math.atan2(
                _xy(frame, _LEFT_HIP)[0] - _xy(frame, _LEFT_SHOULDER)[0],
                _xy(frame, _LEFT_HIP)[1] - _xy(frame, _LEFT_SHOULDER)[1],
            )
        )
        for frame in all_landmarks
    ]
    return float(np.mean(angles)) if angles else 0.0


def _oscillation(
    hip_smooth: np.ndarray,
    ic_frames: np.ndarray,
    taille_patient_cm: float | None,
) -> tuple[float, bool]:
    """Oscillation verticale hanche sur un cycle. Retourne (valeur, approximatif)."""
    if len(ic_frames) >= 2:
        f0, f1 = int(ic_frames[0]), int(ic_frames[1])
        segment = hip_smooth[f0 : f1 + 1]
        osc_norm = float(np.max(segment) - np.min(segment)) if len(segment) > 1 else 0.0
    else:
        osc_norm = float(np.max(hip_smooth) - np.min(hip_smooth))

    if taille_patient_cm is not None:
        return osc_norm * taille_patient_cm * 0.52, False
    return osc_norm, True


def _ratio_contact(cadence_spm: float) -> float:
    """Approximation du ratio contact/suspension (Morin 2011)."""
    return float(np.clip(0.6 - (cadence_spm - 160.0) * 0.002, 0.35, 0.65))


def calculate_metrics(
    all_landmarks: list[list],
    fps: float = 50.0,
    taille_patient_cm: float | None = None,
) -> BiomechanicalMetrics:
    """Calcule les 6 métriques biomécaniques ARIA depuis les keypoints sagittaux.

    Utilise uniquement les landmarks gauches : LEFT_SHOULDER(11), LEFT_HIP(23),
    LEFT_KNEE(25), LEFT_ANKLE(27), LEFT_HEEL(29), LEFT_FOOT_INDEX(31).

    Args:
        all_landmarks: Séquence de frames valides. Chaque frame est une liste de
            33 landmarks avec attributs .x, .y, .z, .visibility (coords normalisées 0-1,
            Y=0 en haut, Y=1 en bas). Ne doit contenir que des frames sans échec.
        fps: Fréquence d'images en Hz (défaut 50fps = FZ200 sagittale ARIA).
        taille_patient_cm: Taille pour convertir l'oscillation en cm.
            Si None : valeur normalisée retournée avec approximatif=True.

    Returns:
        BiomechanicalMetrics avec les 6 métriques et les métadonnées de confiance.

    Raises:
        ValueError: Si les données sont insuffisantes pour l'analyse
            (vidéo trop courte, qualité médiocre, < 3 initial contacts détectés).
    """
    if not all_landmarks:
        raise ValueError(
            "Signal insuffisant pour l'analyse — aucune frame valide fournie."
        )

    hip_y = np.array([float(frame[_LEFT_HIP].y) for frame in all_landmarks])
    heel_y = np.array([float(frame[_LEFT_HEEL].y) for frame in all_landmarks])
    hip_smooth = _smooth(hip_y)

    cad = _cadence(hip_y, fps)
    logger.info(f"Cadence calculée : {cad:.1f} spm")

    ic_frames = _detecter_ic(heel_y, fps)
    if len(ic_frames) < _MIN_IC:
        raise ValueError(
            f"Signal insuffisant pour l'analyse — vidéo trop courte ou qualité médiocre "
            f"({len(ic_frames)} initial contact(s) détecté(s), minimum requis : {_MIN_IC})."
        )

    angle_att = _angle_attaque_moyen(all_landmarks, ic_frames)
    logger.info(f"Angle attaque pied moyen : {angle_att:.1f}°")

    flex_genou = _flexion_genou_moyenne(all_landmarks, ic_frames)
    logger.info(f"Flexion genou à l'impact : {flex_genou:.1f}°")

    inclin = _inclinaison_tronc(all_landmarks)
    logger.info(f"Inclinaison tronc moyenne : {inclin:.1f}°")

    osc, approximatif = _oscillation(hip_smooth, ic_frames, taille_patient_cm)
    logger.info(
        f"Oscillation verticale hanche : {osc:.4f} {'(normalisé)' if approximatif else 'cm'}"
    )

    ratio = _ratio_contact(cad)
    logger.info(f"Ratio contact/suspension : {ratio:.3f}")

    cycles = [
        GaitCycle(
            frame_ic=int(ic_frames[i]),
            angle_attaque=round(
                _angle_attaque_moyen(all_landmarks, ic_frames[i : i + 1]), 1
            ),
            flexion_genou=round(
                _flexion_genou_moyenne(all_landmarks, ic_frames[i : i + 1]), 1
            ),
        )
        for i in range(len(ic_frames))
    ]

    return BiomechanicalMetrics(
        cadence_spm=round(cad, 1),
        angle_attaque_pied_deg=round(angle_att, 1),
        flexion_genou_impact_deg=round(flex_genou, 1),
        inclinaison_tronc_deg=round(inclin, 1),
        oscillation_verticale_cm=round(osc, 4),
        ratio_contact_suspension=round(ratio, 3),
        nb_cycles_analyses=len(ic_frames),
        cycles=cycles,
        approximatif=approximatif,
        confiance_detection=1.0,
    )
