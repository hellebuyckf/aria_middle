"""Outil standalone : floute les visages d'une vidéo (RGPD).

Usage :
    uv run python scripts/blur_video.py INPUT OUTPUT
    uv run python scripts/blur_video.py INPUT OUTPUT --codec mp4v

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


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = frame.copy()
    blurred = False
    for cascade in (_FRONTAL, _PROFILE):
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        for x, y, fw, fh in faces:
            k = max(51, fw | 1)
            out[y : y + fh, x : x + fw] = cv2.GaussianBlur(
                out[y : y + fh, x : x + fw], (k, k), 0
            )
            blurred = True
    return out if blurred else frame


def blur_video(input_path: str, output_path: str, codec: str = "mp4v") -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"Erreur : impossible d'ouvrir '{input_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        sys.exit(f"Erreur : impossible d'écrire '{output_path}' (codec {codec})")

    print(f"Floutage visages : {input_path}  →  {output_path}")
    print(f"  {w}×{h} @ {fps:.1f} fps  |  {total} frames")

    n = 0
    faces_total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = _blur_faces(frame)
        if blurred is not frame:
            faces_total += 1
        out.write(blurred)
        n += 1
        if n % 50 == 0 or n == total:
            pct = n / total * 100 if total else 0
            print(
                f"\r  {n}/{total} frames ({pct:.0f}%) — visages floutés : {faces_total}",
                end="",
                flush=True,
            )

    cap.release()
    out.release()
    print(f"\nTerminé. {faces_total}/{n} frames avec visage flouté → {output_path}")


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
    args = parser.parse_args()
    blur_video(args.input, args.output, args.codec)


if __name__ == "__main__":
    main()
