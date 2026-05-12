"""Outil standalone : floute les visages d'une vidéo (RGPD).

Usage :
    uv run python scripts/blur_video.py INPUT OUTPUT
    uv run python scripts/blur_video.py INPUT OUTPUT --codec mp4v --ttl 20

Indépendant du serveur — aucune dépendance projet, uniquement OpenCV.
"""

import argparse
import sys

import cv2
import numpy as np

_FRONTAL: cv2.CascadeClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
)
_PROFILE: cv2.CascadeClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"  # type: ignore[attr-defined]
)

# Marge autour de la bbox détectée (% de la largeur du visage détecté).
_PADDING = 0.10

# Taille max de la zone floutée, exprimée en fraction de la hauteur de frame.
# Une tête humaine occupe rarement plus de 25 % de la hauteur sur une vue sagittale.
# Au-delà → fausse détection (main, bras…) → bbox rejetée, on garde la précédente.
_MAX_HEAD_RATIO = 0.25


class _FaceTracker:
    """Maintient la dernière bbox valide pour combler les gaps de détection.

    Quand la cascade échoue (main devant le visage, flou de mouvement…),
    on continue à flouter la zone mémorisée pendant `ttl` frames.
    Une bbox dont la hauteur dépasse _MAX_HEAD_RATIO * frame_h est considérée
    comme une anomalie (fausse détection) et ne met pas à jour le tracker.
    """

    def __init__(self, ttl: int) -> None:
        self._ttl = ttl
        self._bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h) paddée
        self._age = 0  # frames depuis la dernière détection valide

    def _raw_detections(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        found: list[tuple[int, int, int, int]] = []
        for cascade in (_FRONTAL, _PROFILE):
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )
            for x, y, fw, fh in faces:
                found.append((int(x), int(y), int(fw), int(fh)))
        return found

    def update(
        self, gray: np.ndarray, frame_h: int, frame_w: int
    ) -> list[tuple[int, int, int, int]]:
        """Détecte les visages et retourne les bboxes à flouter (avec persistance)."""
        detected = self._raw_detections(gray)

        if detected:
            # Bbox englobante de toutes les détections
            xs = [x for x, _, _, _ in detected]
            ys = [y for _, y, _, _ in detected]
            x2s = [x + fw for x, _, fw, _ in detected]
            y2s = [y + fh for _, y, _, fh in detected]
            x, y, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
            fw, fh = x2 - x, y2 - y

            # Rejet si la bbox est anormalement grande (fausse détection)
            if fh > _MAX_HEAD_RATIO * frame_h or fw > _MAX_HEAD_RATIO * frame_w:
                # On garde silencieusement la bbox précédente si elle existe
                if self._bbox is not None and self._age < self._ttl:
                    self._age += 1
                    return [self._bbox]
                return []

            pad_x = int(fw * _PADDING)
            pad_y = int(fh * _PADDING)
            bx = max(0, x - pad_x)
            by = max(0, y - pad_y)
            bw = min(frame_w, x2 + pad_x) - bx
            bh = min(frame_h, y2 + pad_y) - by
            self._bbox = (bx, by, bw, bh)
            self._age = 0
            return [self._bbox]

        if self._bbox is not None and self._age < self._ttl:
            self._age += 1
            return [self._bbox]

        self._bbox = None
        return []


def _apply_blur(
    frame: np.ndarray, bboxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    if not bboxes:
        return frame
    out = frame.copy()
    for x, y, fw, fh in bboxes:
        k = max(51, fw | 1)
        out[y : y + fh, x : x + fw] = cv2.GaussianBlur(
            out[y : y + fh, x : x + fw], (k, k), 0
        )
    return out


def blur_video(
    input_path: str, output_path: str, codec: str = "mp4v", ttl: int = 15
) -> None:
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
    print(f"  {w}×{h} @ {fps:.1f} fps  |  {total} frames  |  TTL persistance : {ttl}")

    tracker = _FaceTracker(ttl)
    n = 0
    blurred_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = tracker.update(gray, h, w)
        out = _apply_blur(frame, bboxes)
        if bboxes:
            blurred_count += 1
        writer.write(out)
        n += 1
        if n % 50 == 0 or n == total:
            pct = n / total * 100 if total else 0
            print(
                f"\r  {n}/{total} frames ({pct:.0f}%) — floutées : {blurred_count}",
                end="",
                flush=True,
            )

    cap.release()
    writer.release()
    print(f"\nTerminé. {blurred_count}/{n} frames floutées → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Floute les visages d'une vidéo (RGPD)."
    )
    parser.add_argument("input", help="Vidéo source")
    parser.add_argument("output", help="Vidéo de sortie")
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
    blur_video(args.input, args.output, args.codec, args.ttl)


if __name__ == "__main__":
    main()
