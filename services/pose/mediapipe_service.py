import gc
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from core.config import settings

# Indices landmarks sagittaux utilisés pour la course (vue latérale gauche)
SAGITTAL_INDICES: frozenset[int] = frozenset(
    {
        11,  # LEFT_SHOULDER
        23,  # LEFT_HIP
        25,  # LEFT_KNEE
        27,  # LEFT_ANKLE
        29,  # LEFT_HEEL
        31,  # LEFT_FOOT_INDEX
    }
)


@dataclass
class Landmark:
    """Coordonnées normalisées d'un point clé BlazePose."""

    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseLandmarks:
    """Les 33 keypoints BlazePose GHUM pour une frame donnée."""

    frame_index: int
    landmarks: list[Landmark]  # longueur 33, indexé par numéro de landmark

    def get(self, index: int) -> Landmark:
        """Retourne le landmark par son index BlazePose."""
        return self.landmarks[index]


def detect_pose(
    frames: list[np.ndarray],
    fps: float = 50.0,
    model_path: str | None = None,
) -> list[PoseLandmarks | None]:
    """Détecte les 33 keypoints BlazePose pour chaque frame via Tasks API.

    Args:
        frames: Frames BGR issues de frame_extractor.
        fps: Fréquence d'images (pour les timestamps vidéo).
        model_path: Chemin vers le fichier .task ; utilise settings si None.

    Returns:
        Liste de PoseLandmarks ou None (une entrée par frame).
    """
    resolved_model = model_path or settings.MEDIAPIPE_MODEL_PATH

    BaseOptions = mp.tasks.BaseOptions  # type: ignore[attr-defined]
    PoseLandmarker = mp.tasks.vision.PoseLandmarker  # type: ignore[attr-defined]
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # type: ignore[attr-defined]
    RunningMode = mp.tasks.vision.RunningMode  # type: ignore[attr-defined]

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    results: list[PoseLandmarks | None] = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            ts_ms = int(i * 1000 / fps)
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(mp_img, ts_ms)

            if not result.pose_landmarks:
                results.append(None)
                continue

            raw_lms = result.pose_landmarks[0]
            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                for lm in raw_lms
            ]
            results.append(PoseLandmarks(frame_index=i, landmarks=landmarks))

    # Force la libération des ressources C++ internes de MediaPipe (sémaphores)
    # avant que le resource_tracker du GC ne signale des fuites au shutdown.
    gc.collect()

    nb_detectes = sum(1 for r in results if r is not None)
    logger.info(f"MediaPipe : {nb_detectes}/{len(frames)} frames détectées")
    return results
