import cv2
import numpy as np
from loguru import logger

from core.config import settings

# Cascades Haar OpenCV — disponibles sans modèle externe (bundlées dans cv2)
# frontal : coureur face caméra ; profil : vue sagittale (cas le plus fréquent)
_FRONTAL: cv2.CascadeClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
)
_PROFILE: cv2.CascadeClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"  # type: ignore[attr-defined]
)


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """Floute tous les visages détectés dans une frame (RGPD).

    Teste les cascades frontale et profil pour couvrir la vue sagittale.
    Kernel proportionnel à la largeur du visage, minimum 51px (toujours impair).
    Retourne la frame originale si aucun visage n'est détecté.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = frame.copy()
    blurred = False

    for cascade in (_FRONTAL, _PROFILE):
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        for x, y, fw, fh in faces:
            k = max(51, fw | 1)  # impair, ≥ 51
            out[y : y + fh, x : x + fw] = cv2.GaussianBlur(
                out[y : y + fh, x : x + fw], (k, k), 0
            )
            blurred = True

    return out if blurred else frame


_MAX_WIDTH = 640


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= _MAX_WIDTH:
        return frame
    scale = _MAX_WIDTH / w
    return cv2.resize(frame, (_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def extract_specific_frames(
    video_path: str,
    indices: set[int],
    fps: int = 25,
) -> dict[int, np.ndarray]:
    """Extrait uniquement les frames dont l'index (dans la séquence sous-échantillonnée) est dans indices.

    Utilisé après la passe de détection de pose pour récupérer uniquement les frames
    nécessaires à l'annotation (typiquement ≤ 6), sans charger toute la vidéo.
    """
    if not indices:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 50.0
    step = max(1, round(source_fps / fps))
    frames: dict[int, np.ndarray] = {}
    frame_idx = 0
    pose_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            if pose_idx in indices:
                frame = _resize(frame)
                frames[pose_idx] = _blur_faces(frame) if settings.BLUR_FACES else frame
                if len(frames) == len(indices):
                    break
            pose_idx += 1
        frame_idx += 1

    cap.release()
    return frames


def extract_frames(video_path: str, fps: int = 25) -> list[np.ndarray]:
    """Extrait les frames d'une vidéo au fps cible avec floutage visage (RGPD).

    Pour une source à 50fps, step=2 → 25fps effectif (1 frame sur 2).
    Les visages sont floutés en mémoire avant tout usage des frames.
    Les frames sont redimensionnées à max 640px de large pour limiter la RAM.

    Args:
        video_path: Chemin vers la vidéo source.
        fps: Fréquence d'échantillonnage cible en images/seconde.

    Returns:
        Liste de frames BGR avec visages floutés.

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
            frame = _resize(frame)
            frames.append(_blur_faces(frame) if settings.BLUR_FACES else frame)
        frame_idx += 1

    cap.release()
    logger.info(
        f"Extraction terminée : {len(frames)} frames retenues sur {frame_idx} totales"
    )
    return frames
