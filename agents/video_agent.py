import asyncio
import json
import os
from functools import partial

from loguru import logger

import core.events as events
from core.state import ARIAState
from services.pose.frame_annotator import select_key_frames
from services.pose.frame_extractor import extract_frames
from services.pose.mediapipe_service import detect_pose
from services.pose.metrics_calculator import (
    calculate_metrics,
    calculate_posterior_metrics,
)

_SEUIL_ECHEC_FRAMES = 0.20


class ARIAVideoError(Exception):
    """Levée quand le pipeline vidéo ne peut pas produire de métriques fiables."""


def _validate_landmarks(landmarks: list, session_id: str, vue: str) -> list:
    """Vérifie le taux d'échec et retourne les landmarks valides."""
    nb_echec = sum(1 for lm in landmarks if lm is None)
    taux_echec = nb_echec / len(landmarks) if landmarks else 1.0
    if taux_echec > _SEUIL_ECHEC_FRAMES:
        raise ARIAVideoError(
            f"Taux d'échec MediaPipe trop élevé ({vue}) : {nb_echec}/{len(landmarks)} "
            f"({taux_echec:.0%} > seuil {_SEUIL_ECHEC_FRAMES:.0%})."
        )
    nb_ok = len(landmarks) - nb_echec
    logger.info(
        f"[{session_id}] Pose {vue} OK : {nb_ok}/{len(landmarks)} frames détectées"
    )
    return [pl.landmarks for pl in landmarks if pl is not None]


async def _run_pose(
    session_id: str,
    frames: list,
    fps: float,
    pct_start: int,
    pct_end: int,
    estimated_s: float,
) -> list:
    """Exécute detect_pose en executor avec un ticker de progression."""
    loop = asyncio.get_running_loop()
    ticker = events.tick(
        session_id, "video", pct_start, pct_end, estimated_s, "Analyse de la posture..."
    )
    try:
        result = await loop.run_in_executor(None, partial(detect_pose, frames, fps))
    finally:
        ticker.cancel()
    return result


async def video_agent(state: ARIAState) -> ARIAState:
    """Nœud LangGraph : video_path [+ video_path_posterior] → BiomechanicalMetrics."""
    session_id = state["session_id"]
    video_path = state["video_path"]
    video_path_posterior = state.get("video_path_posterior")

    logger.info(f"[{session_id}] video_agent démarré | sagittale={video_path}")

    if not os.path.exists(video_path):
        return {
            **state,
            "metrics": None,
            "statut": "erreur",
            "erreur": f"Fichier vidéo introuvable : {video_path}",
        }

    loop = asyncio.get_running_loop()

    try:
        # --- Extraction frames sagittales ---
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "video",
                "pct": 5,
                "message": "Extraction des frames...",
            },
        )
        frames_sag = await loop.run_in_executor(
            None, partial(extract_frames, video_path, 25)
        )
        logger.info(f"[{session_id}] {len(frames_sag)} frames sagittales extraites")

        # --- MediaPipe sagittale (étape la plus longue ~15s) ---
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "video",
                "pct": 15,
                "message": f"{len(frames_sag)} frames — analyse de la posture...",
            },
        )
        raw_sag = await _run_pose(session_id, frames_sag, 25.0, 15, 35, 14.0)
        valid_sag = _validate_landmarks(raw_sag, session_id, "sagittale")

        # --- Métriques sagittales ---
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "video",
                "pct": 36,
                "message": "Calcul des métriques biomécaniques...",
            },
        )
        metrics = await loop.run_in_executor(
            None, partial(calculate_metrics, valid_sag, 25.0)
        )
        logger.info(
            f"[{session_id}] Sagittale OK | cadence={metrics.cadence}spm "
            f"oscillation={metrics.oscillation_verticale}cm"
        )
        await events.emit(
            session_id,
            {
                "type": "metrics",
                "etape": "video",
                "pct": 38,
                "message": "Métriques sagittales calculées",
                "metrics": json.loads(metrics.model_dump_json()),
            },
        )

        # --- Vue postérieure (facultative) ---
        if video_path_posterior is not None:
            if not os.path.exists(video_path_posterior):
                logger.warning(
                    f"[{session_id}] Vue postérieure inexploitable : fichier introuvable ({video_path_posterior})"
                )
            else:
                logger.info(
                    f"[{session_id}] Vue postérieure détectée : {video_path_posterior}"
                )
                try:
                    await events.emit(
                        session_id,
                        {
                            "type": "progress",
                            "etape": "video",
                            "pct": 37,
                            "message": "Analyse vue postérieure...",
                        },
                    )
                    frames_post = await loop.run_in_executor(
                        None, partial(extract_frames, video_path_posterior, 25)
                    )
                    raw_post = await _run_pose(
                        session_id, frames_post, 25.0, 37, 55, 14.0
                    )
                    valid_post = _validate_landmarks(
                        raw_post, session_id, "postérieure"
                    )
                    post = await loop.run_in_executor(
                        None, partial(calculate_posterior_metrics, valid_post, 25.0)
                    )
                    metrics = metrics.model_copy(
                        update=post.model_dump(exclude_none=True)
                    )
                    logger.info(
                        f"[{session_id}] Postérieure OK | pelvic_drop={metrics.pelvic_drop}°"
                    )
                except ARIAVideoError as exc:
                    logger.warning(
                        f"[{session_id}] Vue postérieure inexploitable (taux d'échec MediaPipe trop élevé) : {exc}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{session_id}] Vue postérieure inexploitable (erreur inattendue) : {exc}"
                    )

        if metrics.vue_posterieure_disponible:
            await events.emit(
                session_id,
                {
                    "type": "metrics",
                    "etape": "video",
                    "pct": 40,
                    "message": "Métriques postérieures ajoutées",
                    "metrics": json.loads(metrics.model_dump_json()),
                },
            )
        else:
            await events.emit(
                session_id,
                {
                    "type": "progress",
                    "etape": "video",
                    "pct": 40,
                    "message": "Vidéo analysée",
                },
            )

        # --- Captures annotées pour le rapport PDF ---
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "video",
                "pct": 42,
                "message": "Génération des captures clés...",
            },
        )
        key_frames = await loop.run_in_executor(
            None, partial(select_key_frames, frames_sag, raw_sag, metrics, 25.0)
        )
        logger.info(f"[{session_id}] {len(key_frames)} capture(s) clé(s) générée(s)")

        return {**state, "key_frames": key_frames, "metrics": metrics, "statut": "rag"}

    except ARIAVideoError:
        raise
    except Exception as exc:
        logger.error(f"[{session_id}] video_agent erreur inattendue : {exc}")
        return {**state, "statut": "erreur", "erreur": str(exc)}
