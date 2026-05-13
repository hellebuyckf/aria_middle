import gc
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from core.config import settings

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
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseLandmarks:
    frame_index: int
    landmarks: list[Landmark]

    def get(self, index: int) -> Landmark:
        return self.landmarks[index]


def _make_options(model_path: str | None) -> object:
    BaseOptions = mp.tasks.BaseOptions  # type: ignore[attr-defined]
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # type: ignore[attr-defined]
    RunningMode = mp.tasks.vision.RunningMode  # type: ignore[attr-defined]
    return PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path or settings.MEDIAPIPE_MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


def _parse_result(result: object, frame_index: int) -> PoseLandmarks | None:
    if not result.pose_landmarks:  # type: ignore[attr-defined]
        return None
    raw_lms = result.pose_landmarks[0]  # type: ignore[attr-defined]
    return PoseLandmarks(
        frame_index=frame_index,
        landmarks=[
            Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in raw_lms
        ],
    )


def detect_pose(
    frames: list[np.ndarray],
    fps: float = 50.0,
    model_path: str | None = None,
) -> list[PoseLandmarks | None]:
    """Détecte les keypoints BlazePose sur une liste de frames déjà chargées en mémoire."""
    PoseLandmarker = mp.tasks.vision.PoseLandmarker  # type: ignore[attr-defined]
    results: list[PoseLandmarks | None] = []

    with PoseLandmarker.create_from_options(_make_options(model_path)) as landmarker:
        for i, frame in enumerate(frames):
            ts_ms = int(i * 1000 / fps)
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results.append(_parse_result(landmarker.detect_for_video(mp_img, ts_ms), i))

    gc.collect()
    nb_detectes = sum(1 for r in results if r is not None)
    logger.info(f"MediaPipe : {nb_detectes}/{len(frames)} frames détectées")
    return results


def detect_pose_from_video(
    video_path: str,
    fps: int = 25,
    model_path: str | None = None,
) -> list[PoseLandmarks | None]:
    """Détecte les keypoints BlazePose en streaming directement depuis la vidéo.

    Aucune frame n'est conservée en mémoire après traitement.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 50.0
    step = max(1, round(source_fps / fps))

    PoseLandmarker = mp.tasks.vision.PoseLandmarker  # type: ignore[attr-defined]
    results: list[PoseLandmarks | None] = []
    frame_idx = 0
    pose_idx = 0

    with PoseLandmarker.create_from_options(_make_options(model_path)) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                ts_ms = int(pose_idx * 1000 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results.append(
                    _parse_result(landmarker.detect_for_video(mp_img, ts_ms), pose_idx)
                )
                del rgb, mp_img
                pose_idx += 1
            frame_idx += 1

    cap.release()
    gc.collect()

    nb_detectes = sum(1 for r in results if r is not None)
    logger.info(
        f"MediaPipe streaming : {nb_detectes}/{len(results)} frames détectées "
        f"({frame_idx} frames source)"
    )
    return results
