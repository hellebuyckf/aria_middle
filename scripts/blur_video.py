"""Outil standalone : floute les visages d'une vidéo (RGPD).

Utilise PoseLandmarker (MediaPipe Tasks) via le modèle déjà présent dans
le projet pour localiser la tête depuis les landmarks faciaux (0-10).
Aucun modèle supplémentaire à télécharger.

Usage :
    uv run python scripts/blur_video.py INPUT OUTPUT
    uv run python scripts/blur_video.py INPUT OUTPUT --ttl 20
    uv run python scripts/blur_video.py INPUT OUTPUT --model /chemin/pose_landmarker_full.task
"""

import argparse
import pathlib
import sys

import cv2
import mediapipe as mp
import numpy as np

# Landmarks BlazePose correspondant au visage / tête
_HEAD_LM_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # nez, yeux, oreilles, bouche

# Marge autour de la bbox calculée depuis les landmarks (% des dimensions)
_PADDING = 0.70

# Modèle par défaut : racine du projet (répertoire parent de scripts/)
_DEFAULT_MODEL = str(pathlib.Path(__file__).parents[1] / "pose_landmarker_full.task")


class _HeadTracker:
    """Localise la tête via les landmarks faciaux BlazePose et maintient la
    dernière bbox valide pendant `ttl` frames pour couvrir les occultations.
    """

    def __init__(self, landmarker: object, ttl: int) -> None:
        self._landmarker = landmarker
        self._ttl = ttl
        self._bbox: tuple[int, int, int, int] | None = None
        self._age = 0
        self._frame_idx = 0

    def _landmarks_to_bbox(
        self, lms: list, frame_h: int, frame_w: int
    ) -> tuple[int, int, int, int]:
        xs = [lms[i].x for i in _HEAD_LM_INDICES]
        ys = [lms[i].y for i in _HEAD_LM_INDICES]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        fw = (x_max - x_min) * frame_w
        fh = (y_max - y_min) * frame_h
        pad_x = fw * _PADDING
        pad_y = fh * _PADDING
        bx = max(0, int(x_min * frame_w - pad_x))
        by = max(0, int(y_min * frame_h - pad_y))
        bx2 = min(frame_w, int(x_max * frame_w + pad_x))
        by2 = min(frame_h, int(y_max * frame_h + pad_y))
        return bx, by, bx2 - bx, by2 - by

    def update(
        self, frame: np.ndarray, frame_h: int, frame_w: int, fps: float
    ) -> tuple[int, int, int, int] | None:
        ts_ms = int(self._frame_idx * 1000 / fps)
        self._frame_idx += 1

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self._landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            bbox = self._landmarks_to_bbox(lms, frame_h, frame_w)
            self._bbox = bbox
            self._age = 0
            return bbox

        if self._bbox is not None and self._age < self._ttl:
            self._age += 1
            return self._bbox

        self._bbox = None
        return None


def _apply_blur(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    out = frame.copy()
    k = max(51, w | 1)
    out[y : y + h, x : x + w] = cv2.GaussianBlur(out[y : y + h, x : x + w], (k, k), 0)
    return out


def blur_video(
    input_path: str,
    output_path: str,
    model_path: str = _DEFAULT_MODEL,
    codec: str = "mp4v",
    ttl: int = 15,
) -> None:
    if not pathlib.Path(model_path).exists():
        sys.exit(
            f"Modèle introuvable : {model_path}\n"
            "  → Télécharger avec : make download-model"
        )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Erreur : impossible d'ouvrir '{input_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"Erreur : impossible d'écrire '{output_path}' (codec {codec})")

    print(f"Floutage visages : {input_path}  →  {output_path}")
    print(f"  {w}×{h} @ {fps:.1f} fps  |  {total} frames  |  TTL={ttl}")

    BaseOptions = mp.tasks.BaseOptions  # type: ignore[attr-defined]
    PoseLandmarker = mp.tasks.vision.PoseLandmarker  # type: ignore[attr-defined]
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # type: ignore[attr-defined]
    RunningMode = mp.tasks.vision.RunningMode  # type: ignore[attr-defined]

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        tracker = _HeadTracker(landmarker, ttl)
        n = blurred_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bbox = tracker.update(frame, h, w, fps)
            out = _apply_blur(frame, bbox) if bbox else frame
            if bbox:
                blurred_count += 1
            writer.write(out)
            n += 1
            if n % 50 == 0 or n == total:
                pct = n / total * 100 if total else 0
                print(
                    f"\r  {n}/{total} ({pct:.0f}%) — floutées : {blurred_count}",
                    end="",
                    flush=True,
                )

    cap.release()
    writer.release()
    print(f"\nTerminé. {blurred_count}/{n} frames floutées → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Floute les visages d'une vidéo (RGPD) via PoseLandmarker."
    )
    parser.add_argument("input", help="Vidéo source")
    parser.add_argument("output", help="Vidéo de sortie")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Chemin vers pose_landmarker_full.task (défaut : {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="Codec FourCC (défaut : mp4v). Alternatives : avc1, XVID",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=15,
        help="Frames de persistance après perte de détection (défaut : 15 ≈ 0.6s à 25fps)",
    )
    args = parser.parse_args()
    blur_video(args.input, args.output, args.model, args.codec, args.ttl)


if __name__ == "__main__":
    main()
