import math

import numpy as np
from loguru import logger
from scipy.signal import find_peaks, savgol_filter

from models.metrics import BiomechanicalMetrics

# Indices BlazePose pour la vue sagittale gauche (CLAUDE.md §5)
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HEEL = 29
_RIGHT_HEEL = 30
_LEFT_FOOT_INDEX = 31
_RIGHT_FOOT_INDEX = 32

_MIN_IC = 3  # nombre minimum d'initial contacts pour une analyse valide
_HIP_WIDTH_CM = 28.0  # largeur inter-hanches moyenne coureur (référence métrique)


def _xy(frame: list, idx: int) -> tuple[float, float]:
    """Retourne (x, y) normalisés du landmark BlazePose d'index idx."""
    lm = frame[idx]
    return float(lm.x), float(lm.y)


def _xy_up(frame: list, idx: int) -> tuple[float, float]:
    """Retourne (x, 1 - y) — Y orienté vers le haut (convention géométrique standard).

    MediaPipe/OpenCV ont Y=0 en haut, Y=1 en bas. Les calculs d'angle
    nécessitent le repère mathématique standard (Y vers le haut).
    """
    lm = frame[idx]
    return float(lm.x), 1.0 - float(lm.y)


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
    """Détecte les frames d'Initial Contact (IC) via les maxima locaux de heel_y.

    En coordonnées MediaPipe (Y=0 haut, Y=1 bas), le talon au sol correspond
    au maximum de heel_y (valeur Y la plus grande = position physique la plus basse).
    find_peaks(heel_y) détecte ces maxima = frames IC.
    """
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = find_peaks(heel_y, distance=min_dist)
    return peaks


def _angle_attaque_moyen(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Angle moyen talon→pointe à l'impact, en degrés, dans le sens du déplacement.

    Convention : positif = attaque talon (toe au-dessus du talon),
    négatif = attaque avant-pied (toe en-dessous du talon). Plage [-15°, +25°].

    abs(dx) rend le calcul invariant par rapport au sens de déplacement
    (coureur allant à gauche ou à droite dans le champ).
    """
    angles = []
    for f in ic_frames:
        hx, hy = _xy_up(all_landmarks[f], _LEFT_HEEL)
        fx, fy = _xy_up(all_landmarks[f], _LEFT_FOOT_INDEX)
        dx = abs(fx - hx)
        dy = fy - hy  # positif = toe au-dessus du talon (Y-up)
        angles.append(math.degrees(math.atan2(dy, dx)) if dx > 0 else 0.0)
    return float(np.mean(angles)) if angles else 0.0


def _flexion_genou_moyenne(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Flexion genou moyenne hanche-genou-cheville à l'impact (degrés)."""
    angles = [
        180.0
        - _angle_sommet(
            *_xy_up(all_landmarks[f], _LEFT_HIP),
            *_xy_up(all_landmarks[f], _LEFT_KNEE),
            *_xy_up(all_landmarks[f], _LEFT_ANKLE),
        )
        for f in ic_frames
    ]
    return float(np.mean(angles)) if angles else 0.0


def _inclinaison_tronc(all_landmarks: list[list]) -> float | None:
    """Inclinaison tronc moyenne sur toutes les frames, en degrés (0–45°).

    Détecte la direction du coureur via le déplacement X de LEFT_ANKLE(27) entre
    la première et la dernière frame. Si droite→gauche (dx < 0), signe de (hip_x -
    shoulder_x) inversé pour que le forward lean reste positif dans les deux sens.
    Retourne None si le résultat corrigé sort de [0°, 45°].
    """
    if not all_landmarks:
        return None

    x_start = float(all_landmarks[0][_LEFT_ANKLE].x)
    x_end = float(all_landmarks[-1][_LEFT_ANKLE].x)
    direction = 1.0 if x_end >= x_start else -1.0

    angles = [
        math.degrees(
            math.atan2(
                direction
                * (_xy_up(frame, _LEFT_SHOULDER)[0] - _xy_up(frame, _LEFT_HIP)[0]),
                _xy_up(frame, _LEFT_SHOULDER)[1] - _xy_up(frame, _LEFT_HIP)[1],
            )
        )
        for frame in all_landmarks
    ]
    result = float(np.mean(angles))
    return result if 0.0 <= result <= 45.0 else None


_FEMUR_CM = 45.0  # longueur moyenne fémur adulte (landmark 23→25)


def _oscillation(all_landmarks: list[list], fps: float) -> float | None:
    """Oscillation verticale pic-à-pic du centre du bassin sur un cycle, en cm.

    Midpoint Y entre LEFT_HIP(23) et RIGHT_HIP(24), converti en Y-up (bas = contact sol).
    Un cycle = deux minima locaux consécutifs de mid_y_up.
    Amplitude = médiane des (max - min) sur tous les cycles disponibles.

    Calibration : ratio_cm_par_pixel = 45 / dist_norm(LEFT_HIP, LEFT_KNEE).
    La distance hanche→genou (axe vertical dominant) est une référence anatomique
    stable indépendante de la largeur de cadre. Retourne None si moins de 2 cycles
    (3 minima) détectés ou si la référence fémur est nulle.
    """
    mid_y_up = _smooth(
        np.array(
            [
                1.0 - (float(f[_LEFT_HIP].y) + float(f[_RIGHT_HIP].y)) / 2.0
                for f in all_landmarks
            ]
        )
    )

    min_dist = max(1, int(fps * 0.25))
    troughs, _ = find_peaks(-mid_y_up, distance=min_dist)
    if len(troughs) < 3:
        return None

    cycle_amplitudes = [
        float(
            np.max(mid_y_up[troughs[i] : troughs[i + 1] + 1])
            - np.min(mid_y_up[troughs[i] : troughs[i + 1] + 1])
        )
        for i in range(len(troughs) - 1)
        if troughs[i + 1] > troughs[i]
    ]
    if not cycle_amplitudes:
        return None

    osc_norm = float(np.median(cycle_amplitudes))
    femur_dists = [
        math.sqrt(
            (float(f[_LEFT_HIP].x) - float(f[_LEFT_KNEE].x)) ** 2
            + (float(f[_LEFT_HIP].y) - float(f[_LEFT_KNEE].y)) ** 2
        )
        for f in all_landmarks
    ]
    avg_femur = float(np.mean(femur_dists)) if femur_dists else 0.0
    return (osc_norm / avg_femur) * _FEMUR_CM if avg_femur > 0 else None


def _ratio_contact(cadence_spm: float) -> float:
    """Approximation du ratio contact/suspension (Morin 2011)."""
    return float(np.clip(0.6 - (cadence_spm - 160.0) * 0.002, 0.35, 0.65))


def _pelvic_drop(all_landmarks: list[list]) -> float:
    """Chute du bassin côté oscillant, en degrés (95e percentile de l'angle de la ligne des hanches)."""
    tilts = []
    for frame in all_landmarks:
        lh_x, lh_y = _xy(frame, _LEFT_HIP)
        rh_x, rh_y = _xy(frame, _RIGHT_HIP)
        dx = abs(lh_x - rh_x)
        dy = abs(lh_y - rh_y)
        if dx > 0:
            tilts.append(math.degrees(math.atan2(dy, dx)))
    return float(np.percentile(tilts, 95)) if tilts else 0.0


def _valgus_genou_moyen(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Effondrement médial moyen du genou gauche à l'IC, en degrés (vue postérieure)."""
    angles = [
        180.0
        - _angle_sommet(
            *_xy_up(all_landmarks[f], _LEFT_HIP),
            *_xy_up(all_landmarks[f], _LEFT_KNEE),
            *_xy_up(all_landmarks[f], _LEFT_ANKLE),
        )
        for f in ic_frames
    ]
    return float(np.mean(angles)) if angles else 0.0


def _asymetrie_charge(all_landmarks: list[list], fps: float) -> float:
    """Différence D/G du nombre de contacts sol, en % (asymétrie de foulée)."""
    left_heel_y = np.array([float(frame[_LEFT_HEEL].y) for frame in all_landmarks])
    right_heel_y = np.array([float(frame[_RIGHT_HEEL].y) for frame in all_landmarks])
    left_ics = _detecter_ic(left_heel_y, fps)
    right_ics = _detecter_ic(right_heel_y, fps)
    n_left, n_right = len(left_ics), len(right_ics)
    mean_n = (n_left + n_right) / 2.0
    return float(abs(n_left - n_right) / mean_n * 100) if mean_n > 0 else 0.0


def _oscillation_laterale(all_landmarks: list[list], fps: float) -> float | None:
    """Amplitude latérale pic-à-pic du centre du bassin sur un cycle, en cm.

    Un cycle = deux impacts consécutifs du pied gauche (IC→IC suivant).
    L'amplitude est la médiane des amplitudes sur tous les cycles disponibles,
    ce qui évite la surestimation liée à la dérive globale du coureur dans le champ.

    Conversion : (osc_norm / avg_inter_hip_norm) × 28 cm.
    Retourne None si moins de 2 ICs détectés ou si la distance inter-hanches est nulle.
    """
    mid_x = _smooth(
        np.array(
            [
                (float(f[_LEFT_HIP].x) + float(f[_RIGHT_HIP].x)) / 2.0
                for f in all_landmarks
            ]
        )
    )

    heel_y_left = np.array([float(f[_LEFT_HEEL].y) for f in all_landmarks])
    ic_frames = _detecter_ic(heel_y_left, fps)
    if len(ic_frames) < 2:
        return None

    cycle_amplitudes = [
        float(
            np.max(mid_x[ic_frames[i] : ic_frames[i + 1] + 1])
            - np.min(mid_x[ic_frames[i] : ic_frames[i + 1] + 1])
        )
        for i in range(len(ic_frames) - 1)
        if ic_frames[i + 1] > ic_frames[i]
    ]
    if not cycle_amplitudes:
        return None

    osc_norm = float(np.median(cycle_amplitudes))
    hip_widths = [
        abs(float(f[_LEFT_HIP].x) - float(f[_RIGHT_HIP].x)) for f in all_landmarks
    ]
    avg_inter_hip = float(np.mean(hip_widths)) if hip_widths else 0.0
    return (osc_norm / avg_inter_hip) * _HIP_WIDTH_CM if avg_inter_hip > 0 else None


def _pronation_pied(all_landmarks: list[list], ic_frames: np.ndarray) -> float:
    """Angle de pronation du pied gauche à l'IC, en degrés (inclinaison talon/cheville, vue postérieure)."""
    angles = []
    for f in ic_frames:
        heel_x, heel_y = _xy(all_landmarks[f], _LEFT_HEEL)
        ankle_x, ankle_y = _xy(all_landmarks[f], _LEFT_ANKLE)
        dx = abs(heel_x - ankle_x)
        dy = abs(heel_y - ankle_y)
        if dy > 0:
            angles.append(math.degrees(math.atan2(dx, dy)))
    return float(np.mean(angles)) if angles else 0.0


def calculate_posterior_metrics(
    all_landmarks: list[list],
    fps: float = 50.0,
) -> BiomechanicalMetrics:
    """Calcule les 5 métriques postérieures depuis les keypoints bilatéraux.

    Utilise LEFT_HIP(23), RIGHT_HIP(24), LEFT_KNEE(25), RIGHT_KNEE(26),
    LEFT_ANKLE(27), RIGHT_ANKLE(28), LEFT_HEEL(29), RIGHT_HEEL(30).

    Args:
        all_landmarks: Frames valides de la vidéo postérieure.
        fps: Fréquence d'images en Hz.

    Returns:
        BiomechanicalMetrics avec uniquement les champs postérieurs renseignés
        et vue_posterieure_disponible=True.

    Raises:
        ValueError: Si les données sont insuffisantes.
    """
    if not all_landmarks:
        raise ValueError("Aucune frame valide pour la vue postérieure.")

    heel_y_left = np.array([float(frame[_LEFT_HEEL].y) for frame in all_landmarks])
    ic_frames = _detecter_ic(heel_y_left, fps)
    has_enough_ic = len(ic_frames) >= _MIN_IC

    drop = _pelvic_drop(all_landmarks)
    logger.info(f"Pelvic drop (95e pct) : {drop:.1f}°")

    valgus = _valgus_genou_moyen(all_landmarks, ic_frames) if has_enough_ic else None
    logger.info(f"Valgus genou moyen : {valgus}°")

    asym = _asymetrie_charge(all_landmarks, fps)
    logger.info(f"Asymétrie charge D/G : {asym:.1f}%")

    osc_lat = _oscillation_laterale(all_landmarks, fps)
    logger.info(f"Oscillation latérale hanche : {osc_lat} cm")

    pron = _pronation_pied(all_landmarks, ic_frames) if has_enough_ic else None
    logger.info(f"Pronation pied : {pron}°")

    return BiomechanicalMetrics(
        pelvic_drop=round(drop, 1),
        valgus_genou=round(valgus, 1) if valgus is not None else None,
        asymetrie_charge=round(asym, 1),
        oscillation_laterale_hanche=round(osc_lat, 1) if osc_lat is not None else None,
        pronation_pied=round(pron, 1) if pron is not None else None,
        vue_posterieure_disponible=True,
    )


def calculate_metrics(
    all_landmarks: list[list],
    fps: float = 50.0,
) -> BiomechanicalMetrics:
    """Calcule les 6 métriques biomécaniques ARIA depuis les keypoints sagittaux.

    Utilise les landmarks gauches (11,23,25,27,29,31) pour les métriques sagittales
    et le midpoint 23/24 pour l'oscillation verticale.

    Args:
        all_landmarks: Séquence de frames valides. Chaque frame est une liste de
            33 landmarks avec attributs .x, .y, .z, .visibility (coords normalisées 0-1,
            Y=0 en haut, Y=1 en bas). Ne doit contenir que des frames sans échec.
        fps: Fréquence d'images en Hz (défaut 50fps = FZ200 sagittale ARIA).

    Returns:
        BiomechanicalMetrics avec les 6 métriques.

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
    logger.info(
        f"Inclinaison tronc moyenne : {f'{inclin:.1f}' if inclin is not None else 'None'}°"
    )

    osc = _oscillation(all_landmarks, fps)
    logger.info(f"Oscillation verticale hanche : {osc} cm")

    ratio = _ratio_contact(cad)
    logger.info(f"Ratio contact/suspension : {ratio:.3f}")

    return BiomechanicalMetrics(
        cadence=round(cad, 1),
        angle_attaque_pied=round(angle_att, 1),
        flexion_genou_impact=round(flex_genou, 1),
        inclinaison_tronc=round(inclin, 1) if inclin is not None else None,
        oscillation_verticale=round(osc, 4) if osc is not None else None,
        ratio_contact_suspension=round(ratio, 3),
    )
