from loguru import logger

from core.state import ARIAState
from services.pose.frame_extractor import extract_frames
from services.pose.mediapipe_service import detect_pose
from services.pose.metrics_calculator import calculate_metrics

_SEUIL_ECHEC_FRAMES = 0.20  # 20% de frames sans détection → erreur bloquante


class ARIAVideoError(Exception):
    """Levée quand le pipeline vidéo ne peut pas produire de métriques fiables."""


def video_agent(state: ARIAState) -> ARIAState:
    """Nœud LangGraph : video_path → BiomechanicalMetrics.

    Orchestre l'extraction des frames, la détection de pose MediaPipe et le
    calcul des métriques biomécaniques sagittales. Ne lit jamais la vidéo brute
    directement : toutes les opérations passent par services/pose/.

    Met à jour statut → "rag" en cas de succès, "erreur" en cas d'exception
    non-ARIA. Lève ARIAVideoError si le taux d'échec MediaPipe dépasse 20%.

    Args:
        state: ARIAState avec video_path renseigné.

    Returns:
        ARIAState mis à jour avec metrics et statut.

    Raises:
        ARIAVideoError: Si plus de 20% des frames échouent à la détection.
    """
    session_id = state["session_id"]
    video_path = state["video_path"]

    logger.info(f"[{session_id}] video_agent démarré | vidéo={video_path}")

    try:
        logger.info(
            f"[{session_id}] Étape 1/3 : extraction des frames (25fps effectif)"
        )
        frames = extract_frames(video_path, fps=25)
        logger.info(f"[{session_id}] {len(frames)} frames extraites")

        logger.info(f"[{session_id}] Étape 2/3 : détection pose MediaPipe")
        landmarks = detect_pose(frames)

        nb_echec = sum(1 for lm in landmarks if lm is None)
        taux_echec = nb_echec / len(landmarks) if landmarks else 1.0

        if taux_echec > _SEUIL_ECHEC_FRAMES:
            raise ARIAVideoError(
                f"Taux d'échec MediaPipe trop élevé : {nb_echec}/{len(landmarks)} frames "
                f"({taux_echec:.0%} > seuil {_SEUIL_ECHEC_FRAMES:.0%}). "
                "Vérifier la qualité vidéo ou l'angle de prise de vue (vue sagittale requise)."
            )

        logger.info(f"[{session_id}] Étape 3/3 : calcul des métriques biomécaniques")
        metrics = calculate_metrics(landmarks)

        logger.info(
            f"[{session_id}] video_agent terminé | "
            f"cadence={metrics.cadence_spm}spm "
            f"oscillation={metrics.oscillation_verticale_cm}cm "
            f"frames_ok={metrics.nb_frames_analysees}/{len(frames)}"
        )

        return {**state, "metrics": metrics, "statut": "rag"}

    except ARIAVideoError:
        raise
    except Exception as exc:
        logger.error(f"[{session_id}] video_agent erreur inattendue : {exc}")
        return {**state, "statut": "erreur", "erreur": str(exc)}
