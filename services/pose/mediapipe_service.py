from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

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


def detect_pose(frames: list[np.ndarray]) -> list[PoseLandmarks | None]:
    """Détecte les 33 keypoints BlazePose pour chaque frame.

    Utilise le backend Metal (MPS) automatiquement sur Apple Silicon M3.
    Retourne None pour les frames sans détection valide.

    Args:
        frames: Frames BGR issues de frame_extractor.

    Returns:
        Liste de PoseLandmarks ou None (une entrée par frame).
    """
    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
    results: list[PoseLandmarks | None] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as pose:
        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            if result.pose_landmarks is None:
                results.append(None)
                continue

            landmarks = [
                Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                for lm in result.pose_landmarks.landmark
            ]
            results.append(PoseLandmarks(frame_index=i, landmarks=landmarks))

    nb_detectes = sum(1 for r in results if r is not None)
    logger.info(f"MediaPipe : {nb_detectes}/{len(frames)} frames détectées")
    return results
