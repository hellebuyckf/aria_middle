import cv2
import numpy as np
from loguru import logger


def extract_frames(video_path: str, fps: int = 25) -> list[np.ndarray]:
    """Extrait les frames d'une vidéo au fps cible.

    Pour une source à 50fps, step=2 → 25fps effectif (1 frame sur 2).

    TODO RGPD : appliquer le floutage visage (MediaPipe FaceMesh) avant
    tout write disque dans frame_extractor. Priorité avant mise en prod.

    Args:
        video_path: Chemin vers la vidéo source.
        fps: Fréquence d'échantillonnage cible en images/seconde.

    Returns:
        Liste de frames BGR sous forme de tableaux NumPy.

    Raises:
        ValueError: Si la vidéo ne peut pas être ouverte.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 50.0
    step = max(1, round(source_fps / fps))
    frames: list[np.ndarray] = []
    frame_idx = 0

    logger.debug(
        f"Extraction {video_path} | source={source_fps:.1f}fps step={step} → cible={fps}fps"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    logger.info(
        f"Extraction terminée : {len(frames)} frames retenues sur {frame_idx} totales"
    )
    return frames
