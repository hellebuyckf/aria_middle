from models.metrics import BiomechanicalMetrics, GaitCycle
from services.pose.mediapipe_service import PoseLandmarks


def calculate_metrics(landmarks: list[PoseLandmarks | None]) -> BiomechanicalMetrics:
    """Calcule les métriques biomécaniques depuis les keypoints sagittaux.

    Stub MVP — retourne des valeurs fixes représentatives dans les normes ARIA.
    Le calcul réel (détection de cycles via oscillation hanche, angles articulaires)
    sera implémenté dans l'itération suivante.

    Args:
        landmarks: Séquence de keypoints par frame (None = frame échouée).

    Returns:
        BiomechanicalMetrics avec les 6 métriques cibles du CLAUDE.md §6.
    """
    nb_total = len(landmarks)
    nb_echec = sum(1 for lm in landmarks if lm is None)

    # TODO: implémenter le calcul réel des métriques :
    # - cadence : détection pic/creux oscillation LEFT_HIP(23) sur axe Y
    # - angle_attaque_pied : vecteur LEFT_HEEL(29)→LEFT_FOOT_INDEX(31) à l'IC
    # - flexion_genou : angle LEFT_HIP(23)-LEFT_KNEE(25)-LEFT_ANKLE(27) à l'IC
    # - inclinaison_tronc : angle LEFT_SHOULDER(11)-LEFT_HIP(23) vs vertical
    # - oscillation_verticale : amplitude pic/creux LEFT_HIP(23) × px_to_cm
    # - ratio_contact_suspension : dérivé cadence + oscillation

    return BiomechanicalMetrics(
        cadence_spm=172.0,
        angle_attaque_pied_deg=8.5,
        flexion_genou_impact_deg=18.0,
        inclinaison_tronc_deg=7.0,
        oscillation_verticale_cm=6.5,
        ratio_contact_suspension=0.60,
        cycles=[GaitCycle(debut_frame=0, fin_frame=0, duree_ms=0.0)],
        nb_frames_analysees=nb_total - nb_echec,
        nb_frames_echec=nb_echec,
    )
