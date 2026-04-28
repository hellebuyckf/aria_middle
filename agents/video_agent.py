from loguru import logger

from core.state import ARIAState
from services.pose.frame_extractor import extract_frames
from services.pose.mediapipe_service import detect_pose
from services.pose.metrics_calculator import (
    calculate_metrics,
    calculate_posterior_metrics,
)

_SEUIL_ECHEC_FRAMES = 0.20  # 20% de frames sans détection → erreur bloquante


class ARIAVideoError(Exception):
    """Levée quand le pipeline vidéo ne peut pas produire de métriques fiables."""


def _run_pose_pipeline(
    video_path: str,
    session_id: str,
    vue: str,
) -> list:
    """Extrait les frames, détecte la pose et vérifie le taux d'échec.

    Args:
        video_path: Chemin vers la vidéo.
        session_id: Identifiant de session (pour les logs uniquement).
        vue: Label de la vue pour les messages de log ("sagittale" | "postérieure").

    Returns:
        Liste de landmarks valides (frames None exclues).

    Raises:
        ARIAVideoError: Si le taux d'échec MediaPipe dépasse le seuil.
    """
    logger.info(f"[{session_id}] Extraction frames {vue} ({video_path})")
    frames = extract_frames(video_path, fps=25)
    logger.info(f"[{session_id}] {len(frames)} frames extraites ({vue})")

    landmarks = detect_pose(frames, fps=25.0)

    nb_echec = sum(1 for lm in landmarks if lm is None)
    taux_echec = nb_echec / len(landmarks) if landmarks else 1.0
    if taux_echec > _SEUIL_ECHEC_FRAMES:
        raise ARIAVideoError(
            f"Taux d'échec MediaPipe trop élevé ({vue}) : {nb_echec}/{len(landmarks)} frames "
            f"({taux_echec:.0%} > seuil {_SEUIL_ECHEC_FRAMES:.0%})."
        )

    return [pl.landmarks for pl in landmarks if pl is not None]


def video_agent(state: ARIAState) -> ARIAState:
    """Nœud LangGraph : video_path [+ video_path_posterior] → BiomechanicalMetrics.

    Traite obligatoirement la vidéo sagittale. Si video_path_posterior est
    présent, lance un second passage MediaPipe sur la vue postérieure et fusionne
    les métriques dans un seul BiomechanicalMetrics (vue_posterieure_disponible=True).

    Raises:
        ARIAVideoError: Si le taux d'échec MediaPipe dépasse 20% sur la vue sagittale.
    """
    session_id = state["session_id"]
    video_path = state["video_path"]
    video_path_posterior = state.get("video_path_posterior")

    logger.info(f"[{session_id}] video_agent démarré | sagittale={video_path}")

    try:
        valid_sagittal = _run_pose_pipeline(video_path, session_id, "sagittale")
        logger.info(f"[{session_id}] Calcul métriques sagittales")
        metrics = calculate_metrics(valid_sagittal, fps=25.0)
        logger.info(
            f"[{session_id}] Sagittale OK | cadence={metrics.cadence}spm "
            f"oscillation={metrics.oscillation_verticale}cm"
        )

        if video_path_posterior is not None:
            logger.info(
                f"[{session_id}] Vue postérieure détectée : {video_path_posterior}"
            )
            try:
                valid_posterior = _run_pose_pipeline(
                    video_path_posterior, session_id, "postérieure"
                )
                post = calculate_posterior_metrics(valid_posterior, fps=25.0)
                metrics = metrics.model_copy(update=post.model_dump(exclude_none=True))
                logger.info(
                    f"[{session_id}] Postérieure OK | pelvic_drop={metrics.pelvic_drop}°"
                )
            except ARIAVideoError as exc:
                logger.warning(
                    f"[{session_id}] Vue postérieure ignorée (échec pipeline) : {exc}"
                )

        return {**state, "metrics": metrics, "statut": "rag"}

    except ARIAVideoError:
        raise
    except Exception as exc:
        logger.error(f"[{session_id}] video_agent erreur inattendue : {exc}")
        return {**state, "statut": "erreur", "erreur": str(exc)}
