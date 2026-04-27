from dataclasses import dataclass, field


@dataclass
class GaitCycle:
    """Un cycle de foulée détecté dans la séquence vidéo."""

    debut_frame: int
    fin_frame: int
    duree_ms: float


@dataclass
class BiomechanicalMetrics:
    """Métriques biomécaniques calculées depuis la vidéo sagittale.

    Normes de référence documentées dans CLAUDE.md §6.
    """

    cadence_spm: float | None = None
    """Foulées/min — norme : 170–180 spm (Heiderscheit 2011)."""

    angle_attaque_pied_deg: float | None = None
    """Angle talon→pointe à l'impact — < 5° avant-pied, > 10° talon."""

    flexion_genou_impact_deg: float | None = None
    """Angle hanche-genou-cheville à l'initial contact — norme : 15–25°."""

    inclinaison_tronc_deg: float | None = None
    """Angle épaule-hanche vertical — optimal : 5–10° forward lean."""

    oscillation_verticale_cm: float | None = None
    """Amplitude verticale de la hanche sur un cycle — optimal : < 8 cm."""

    ratio_contact_suspension: float | None = None
    """Ratio temps contact sol / durée cycle."""

    cycles: list[GaitCycle] = field(default_factory=list)
    nb_frames_analysees: int = 0
    nb_frames_echec: int = 0
