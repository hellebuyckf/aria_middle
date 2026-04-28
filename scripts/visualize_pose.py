"""
visualize_pose.py — Visualisation MediaPipe Pose sur vidéo sagittale ARIA
Génère une vidéo annotée avec les landmarks et métriques en temps réel.

Usage :
    uv run python scripts/visualize_pose.py \
        --video chemin/vers/sagittale.MTS \
        --model pose_landmarker_full.task \
        --output output_pose.mp4 \
        --max-frames 300
"""

import argparse
import math
import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Indices landmarks (vue sagittale gauche)
# ---------------------------------------------------------------------------
IDX = {
    "shoulder": 11,
    "hip":       23,
    "knee":      25,
    "ankle":     27,
    "heel":      29,
    "foot":      31,
}

# Connexions à dessiner (paires d'indices)
CONNECTIONS = [
    (IDX["shoulder"], IDX["hip"]),
    (IDX["hip"],      IDX["knee"]),
    (IDX["knee"],     IDX["ankle"]),
    (IDX["ankle"],    IDX["heel"]),
    (IDX["heel"],     IDX["foot"]),
    (IDX["ankle"],    IDX["foot"]),
]

COULEUR_SQUELETTE = (0,   220, 100)   # vert ARIA
COULEUR_POINT     = (255, 255,   0)   # jaune
COULEUR_TEXTE     = (255, 255, 255)   # blanc
COULEUR_ALERTE    = (0,    60, 220)   # rouge BGR → orange


def angle_entre(a, b, c) -> float:
    """Angle ABC en degrés (b = sommet)."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1, 1)))


def angle_attaque(heel, foot) -> float:
    """Angle talon→pointe par rapport à l'horizontale (degrés)."""
    return math.degrees(math.atan2(foot[1] - heel[1], foot[0] - heel[0]))


def lm_to_px(lm, w, h):
    """Convertit un landmark normalisé en coordonnées pixel."""
    return int(lm.x * w), int(lm.y * h)


def dessiner_frame(frame, landmarks, frame_idx: int, fps: float):
    """Dessine squelette, points, métriques sur une frame OpenCV."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    lms = landmarks[0]  # première personne

    # — Connexions squelette —
    for i, j in CONNECTIONS:
        p1 = lm_to_px(lms[i], w, h)
        p2 = lm_to_px(lms[j], w, h)
        cv2.line(overlay, p1, p2, COULEUR_SQUELETTE, 3, cv2.LINE_AA)

    # — Points articulaires —
    for name, idx in IDX.items():
        px = lm_to_px(lms[idx], w, h)
        cv2.circle(overlay, px, 8, COULEUR_POINT, -1, cv2.LINE_AA)
        cv2.circle(overlay, px, 10, COULEUR_SQUELETTE, 2, cv2.LINE_AA)
        cv2.putText(overlay, name, (px[0] + 12, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COULEUR_TEXTE, 1, cv2.LINE_AA)

    # — Calcul métriques instantanées —
    hip_px   = lm_to_px(lms[IDX["hip"]],      w, h)
    knee_px  = lm_to_px(lms[IDX["knee"]],     w, h)
    ankle_px = lm_to_px(lms[IDX["ankle"]],    w, h)
    heel_px  = lm_to_px(lms[IDX["heel"]],     w, h)
    foot_px  = lm_to_px(lms[IDX["foot"]],     w, h)
    shld_px  = lm_to_px(lms[IDX["shoulder"]], w, h)

    flex_genou   = angle_entre(hip_px, knee_px, ankle_px)
    incl_tronc   = math.degrees(math.atan2(
        hip_px[0] - shld_px[0], hip_px[1] - shld_px[1]))
    att_pied     = angle_attaque(heel_px, foot_px)
    timestamp_s  = frame_idx / fps

    # — Arc angle genou —
    cv2.ellipse(overlay, knee_px, (30, 30), 0, -90, int(-90 + flex_genou),
                (100, 200, 255), 2, cv2.LINE_AA)

    # — Panneau métriques (coin supérieur gauche) —
    panel_x, panel_y = 20, 20
    lignes = [
        f"t = {timestamp_s:.1f}s  |  frame {frame_idx}",
        f"Flexion genou    : {flex_genou:.1f} deg  [norme 15-25]",
        f"Inclinaison tronc: {incl_tronc:.1f} deg  [norme 5-10]",
        f"Angle attaque    : {att_pied:.1f} deg  [<5=avant-pied]",
    ]
    for i, ligne in enumerate(lignes):
        y = panel_y + i * 26
        couleur = COULEUR_ALERTE if i > 0 and (
            (i == 1 and (flex_genou < 10 or flex_genou > 35)) or
            (i == 2 and (incl_tronc < 0  or incl_tronc > 20))
        ) else COULEUR_TEXTE
        cv2.putText(overlay, ligne, (panel_x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, ligne, (panel_x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, couleur, 1, cv2.LINE_AA)

    # — Blend transparent —
    return cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)


def main():
    parser = argparse.ArgumentParser(description="Visualisation pose ARIA")
    parser.add_argument("--video",      required=True, help="Chemin vidéo source (.MTS/.mp4)")
    parser.add_argument("--model",      default="pose_landmarker_full.task")
    parser.add_argument("--output",     default="output_pose.mp4")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Nombre max de frames à traiter (défaut 500 ≈ 10s)")
    args = parser.parse_args()

    # — Init MediaPipe Tasks API —
    BaseOptions        = mp.tasks.BaseOptions
    PoseLandmarker     = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode        = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    frame_idx = 0
    detectees = 0
    print(f"Traitement de {args.video} ({w}×{h} @ {fps}fps)...")

    with PoseLandmarker.create_from_options(options) as lm:
        while cap.isOpened() and frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = int(frame_idx * 1000 / fps)
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = lm.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                detectees += 1
                frame = dessiner_frame(frame, result.pose_landmarks, frame_idx, fps)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  {frame_idx} frames traitées ({detectees} détectées)...")

    cap.release()
    writer.release()
    taux = 100 * detectees / max(frame_idx, 1)
    print(f"\n✓ Vidéo annotée : {args.output}")
    print(f"  {detectees}/{frame_idx} frames détectées ({taux:.1f}%)")


if __name__ == "__main__":
    main()