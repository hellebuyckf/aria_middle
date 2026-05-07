"""Annote les frames clés avec le squelette MediaPipe et les métriques biomécaniques.

Ce module est appelé par video_agent après le calcul des métriques. Il sélectionne
jusqu'à 4 frames représentatives dans la séquence sagittale, les annote avec les
segments squelettiques et les arcs d'angle pertinents, puis les encode en base64 PNG
pour intégration directe dans le rapport PDF via WeasyPrint (data URI).

Pipeline d'annotation par frame :
  1. Redimensionnement à _TARGET_W px de large (aspect ratio préservé)
  2. Tracé des segments osseux concernés (ombre noire + trait bronze)
  3. Arc d'angle cv2.ellipse au sommet du joint (rouge=anormal, vert=normal)
  4. Points landmarks (disque coloré + contour blanc)
  5. Bandeau texte en haut à gauche : nom métrique / valeur + norme / statut
"""

import base64
import math

import cv2
import numpy as np
from scipy.signal import find_peaks

from models.metrics import BiomechanicalMetrics
from services.pose.mediapipe_service import Landmark, PoseLandmarks

_TARGET_W = 480  # largeur cible des PNG générés (hauteur proportionnelle)

# ── BGR palette ───────────────────────────────────────────────────────────────
_BONE = (160, 130, 80)  # segments squelettiques
_NORMAL = (60, 180, 60)  # accent si métrique dans la norme
_ABNORMAL = (50, 50, 220)  # accent si métrique hors norme
_TEXT_BG = (20, 20, 20)  # fond du bandeau texte

# ── Seuils normatifs (lo, hi) — None = borne ouverte ─────────────────────────
# Miroir de pdf_service._is_abnormal ; doit rester synchronisé.
_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "cadence": (170.0, 180.0),
    "angle_attaque_pied": (None, 10.0),
    "flexion_genou_impact": (15.0, 25.0),
    "inclinaison_tronc": (5.0, 10.0),
    "oscillation_verticale": (None, 8.0),
    "ratio_contact_suspension": (0.35, 0.65),
    "pelvic_drop": (None, 5.0),
    "valgus_genou": (None, 8.0),
    "asymetrie_charge": (None, 10.0),
    "oscillation_laterale_hanche": (None, 3.0),
    "pronation_pied": (None, 8.0),
}

# ── Libellés affichés sur les frames (label, unité) ──────────────────────────
_LABELS: dict[str, tuple[str, str]] = {
    "flexion_genou_impact": ("Flexion genou (impact)", "°"),
    "angle_attaque_pied": ("Attaque pied", "°"),
    "inclinaison_tronc": ("Inclinaison tronc", "°"),
    "pelvic_drop": ("Pelvic drop", "°"),
    "valgus_genou": ("Valgus genou", "°"),
    "pronation_pied": ("Pronation pied", "°"),
    "oscillation_verticale": ("Oscillation verticale", "cm"),
}

# ── Overlays par métrique : (indices_lm, connexions, arc) ────────────────────
# arc = (A, vertex, B) où l'arc est dessiné au sommet V entre les branches VA et VB.
# None = pas d'arc (métrique sans angle articulaire direct).
_OVERLAYS: dict[
    str, tuple[list[int], list[tuple[int, int]], tuple[int, int, int] | None]
] = {
    "flexion_genou_impact": ([23, 25, 27], [(23, 25), (25, 27)], (23, 25, 27)),
    "angle_attaque_pied": ([27, 29, 31], [(27, 29), (29, 31)], None),
    "inclinaison_tronc": ([11, 23], [(11, 23)], None),
    "pelvic_drop": ([23, 24], [(23, 24)], None),
    "valgus_genou": ([23, 25, 27], [(23, 25), (25, 27)], (23, 25, 27)),
    "pronation_pied": ([27, 29], [(27, 29)], None),
    "oscillation_verticale": ([23, 24], [(23, 24)], None),
}

# Métriques dont la frame idéale est un Initial Contact (IC).
# Les autres utilisent la frame médiane de la séquence.
_IC_METRICS = {
    "flexion_genou_impact",
    "angle_attaque_pied",
    "valgus_genou",
    "pronation_pied",
}

# Ordre de priorité pour la sélection des 4 frames (les anormaux passent devant).
_PRIORITY = [
    "flexion_genou_impact",
    "angle_attaque_pied",
    "inclinaison_tronc",
    "pelvic_drop",
    "valgus_genou",
    "pronation_pied",
    "oscillation_verticale",
]


def _is_abnormal(field: str, value: float | None) -> bool:
    """Retourne True si value sort des bornes normatives de field."""
    if value is None:
        return False
    lo, hi = _THRESHOLDS.get(field, (None, None))
    return (lo is not None and value < lo) or (hi is not None and value > hi)


def _norm_str(metric: str, unit: str) -> str:
    """Formate la plage normative pour affichage : 'norme: 15–25°', 'norme: <10°', etc."""
    lo, hi = _THRESHOLDS.get(metric, (None, None))
    if lo is not None and hi is not None:
        return f"norme: {lo}–{hi}{unit}"
    if hi is not None:
        return f"norme: <{hi}{unit}"
    if lo is not None:
        return f"norme: >{lo}{unit}"
    return ""


def _px(lm: Landmark, w: int, h: int) -> tuple[int, int]:
    """Convertit les coordonnées normalisées MediaPipe (0–1) en pixels image."""
    return int(lm.x * w), int(lm.y * h)


def _draw_arc(
    img: np.ndarray,
    pts: dict[int, tuple[int, int]],
    a: int,
    v: int,
    b: int,
    color: tuple[int, int, int],
    r: int = 38,
) -> None:
    """Dessine l'arc angulaire au sommet V entre les branches VA et VB.

    Utilise cv2.ellipse dont les angles sont mesurés dans le sens horaire depuis
    l'est (3h), ce qui coïncide avec atan2 en repère image (Y vers le bas).
    La formule diff = ((ang2 - ang1 + 180) % 360) - 180 sélectionne toujours
    le plus court arc (< 180°) entre les deux directions.
    """
    vx, vy = pts[v]
    ang1 = math.degrees(math.atan2(pts[a][1] - vy, pts[a][0] - vx))
    ang2 = math.degrees(math.atan2(pts[b][1] - vy, pts[b][0] - vx))
    diff = ((ang2 - ang1) + 180) % 360 - 180
    start = ang1 if diff >= 0 else ang2
    end = start + abs(diff)
    cv2.ellipse(img, (vx, vy), (r, r), 0, start, end, color, 2, cv2.LINE_AA)


def _draw_label(img: np.ndarray, label: str, val_str: str, abnormal: bool) -> None:
    """Superpose un bandeau texte opaque en haut à gauche de l'image.

    Ligne 1 : nom de la métrique.
    Ligne 2 : valeur mesurée + plage normative + statut [ANORMAL/normal].
    Le fond noir semi-encadré assure la lisibilité quelle que soit la scène.
    """
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    status = "ANORMAL" if abnormal else "normal"
    lines = [label, f"{val_str}  [{status}]"]
    y = 8
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(img, (4, y - 2), (tw + 14, y + th + 4), _TEXT_BG, -1)
        cv2.putText(
            img, line, (8, y + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA
        )
        y += th + 8


def annotate_frame(
    frame: np.ndarray,
    landmarks: list[Landmark],
    metric: str,
    value: float | None,
) -> np.ndarray:
    """Retourne une copie redimensionnée et annotée du frame pour la métrique donnée.

    Args:
        frame: Frame BGR issue de extract_frames (visage déjà flouté, RGPD OK).
        landmarks: Les 33 keypoints BlazePose de ce frame.
        metric: Nom du champ BiomechanicalMetrics à visualiser (doit être dans _OVERLAYS).
        value: Valeur numérique de la métrique (None → affiche "N/A").

    Returns:
        Image BGR annotée, redimensionnée à _TARGET_W px de large.
        Retourne frame inchangé si metric n'est pas dans _OVERLAYS.
    """
    if metric not in _OVERLAYS:
        return frame

    h0, w0 = frame.shape[:2]
    nh = int(h0 * _TARGET_W / w0)
    img = cv2.resize(frame.copy(), (_TARGET_W, nh), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    lm_idxs, connections, arc = _OVERLAYS[metric]
    pts = {i: _px(landmarks[i], w, h) for i in lm_idxs}
    abnormal = _is_abnormal(metric, value)
    accent = _ABNORMAL if abnormal else _NORMAL

    # Segments : ombre noire (épaisseur 4) + trait bronze (épaisseur 2)
    for a_idx, b_idx in connections:
        cv2.line(img, pts[a_idx], pts[b_idx], (0, 0, 0), 4, cv2.LINE_AA)
        cv2.line(img, pts[a_idx], pts[b_idx], _BONE, 2, cv2.LINE_AA)

    if arc is not None:
        _draw_arc(img, pts, *arc, accent)

    # Points : disque coloré avec contour noir pour le détacher du fond
    for pt in pts.values():
        cv2.circle(img, pt, 6, (0, 0, 0), -1)
        cv2.circle(img, pt, 5, accent, -1, cv2.LINE_AA)

    label, unit = _LABELS.get(metric, (metric, ""))
    val_str = (
        f"{value:.1f}{unit}  {_norm_str(metric, unit)}" if value is not None else "N/A"
    )
    _draw_label(img, label, val_str, abnormal)

    return img


def _to_b64(frame: np.ndarray) -> str:
    """Encode un frame BGR en chaîne base64 PNG (compression niveau 6)."""
    ok, buf = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("Encodage PNG échoué")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _detect_ic(heel_y: np.ndarray, fps: float) -> np.ndarray:
    """Détecte les frames d'Initial Contact via les maxima du signal talon (Y vers le bas).

    Réimplémentation locale de metrics_calculator._detecter_ic pour éviter
    le couplage direct sur une fonction privée.
    """
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = find_peaks(heel_y, distance=min_dist)
    return peaks


def select_key_frames(
    frames_sag: list[np.ndarray],
    raw_sag: list,
    metrics: BiomechanicalMetrics,
    fps: float,
) -> list[str]:
    """Sélectionne et annote jusqu'à 4 frames clés pour le rapport PDF.

    Stratégie de sélection :
    - Les métriques sont triées : anormales d'abord, puis par ordre de _PRIORITY.
    - Les métriques à l'IC (flexion genou, attaque pied, valgus, pronation) reçoivent
      des frames différentes prélevées à 1/4, 1/2 et 3/4 de la séquence d'ICs,
      pour montrer la variabilité intra-séquence plutôt qu'un seul instant.
    - Les métriques globales (inclinaison tronc, pelvic drop, oscillation verticale)
      reçoivent la frame médiane de la séquence valide.

    Args:
        frames_sag: Frames BGR sagittales (avec visages floutés) issues de extract_frames.
        raw_sag: Résultats bruts de detect_pose — list[PoseLandmarks | None], un élément
            par frame dans frames_sag. Les None (échecs MediaPipe) sont ignorés.
        metrics: Métriques calculées par calculate_metrics / calculate_posterior_metrics.
        fps: Fréquence d'images utilisée lors de l'extraction (typiquement 25 Hz).

    Returns:
        Liste de 0 à 4 chaînes base64 PNG prêtes pour intégration dans le template
        Jinja2 via ``data:image/png;base64,…``.
        Liste vide si aucune frame valide n'est disponible.
    """
    valid_poses: list[PoseLandmarks] = [pl for pl in raw_sag if pl is not None]
    if not valid_poses:
        return []

    valid_lms = [pl.landmarks for pl in valid_poses]
    heel_y = np.array([float(lm[29].y) for lm in valid_lms])
    ic_arr = _detect_ic(heel_y, fps)

    n = len(ic_arr)
    if n >= 4:
        ic_pool = [ic_arr[n // 4], ic_arr[n // 2], ic_arr[3 * n // 4], ic_arr[1]]
    elif n >= 2:
        ic_pool = list(ic_arr[1:])
    else:
        ic_pool = []

    median_idx = len(valid_poses) // 2
    metric_vals = {f: getattr(metrics, f, None) for f in _PRIORITY}
    sorted_metrics = sorted(
        [m for m in _PRIORITY if metric_vals[m] is not None],
        key=lambda m: (not _is_abnormal(m, metric_vals[m]), _PRIORITY.index(m)),
    )

    result: list[str] = []
    ic_cursor = 0
    for m in sorted_metrics:
        if len(result) == 4:
            break
        if m in _IC_METRICS:
            if not ic_pool:
                continue
            valid_idx = ic_pool[ic_cursor % len(ic_pool)]
            ic_cursor += 1
        else:
            valid_idx = median_idx
        pose = valid_poses[valid_idx]
        frame = frames_sag[pose.frame_index]
        result.append(_to_b64(annotate_frame(frame, pose.landmarks, m, metric_vals[m])))

    return result
