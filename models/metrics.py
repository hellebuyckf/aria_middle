from dataclasses import asdict, dataclass, field


@dataclass
class GaitCycle:
    """Données d'un cycle de foulée individuel détecté."""

    frame_ic: int
    angle_attaque: float
    flexion_genou: float


@dataclass
class BiomechanicalMetrics:
    """Métriques biomécaniques calculées depuis la vidéo sagittale.

    Normes de référence documentées dans CLAUDE.md §6.
    """

    cadence_spm: float
    """Foulées/min — norme : 170–180 spm (Heiderscheit 2011)."""

    angle_attaque_pied_deg: float
    """Angle talon→pointe à l'impact — < 5° avant-pied, > 10° talon."""

    flexion_genou_impact_deg: float
    """Angle hanche-genou-cheville à l'initial contact — norme : 15–25°."""

    inclinaison_tronc_deg: float
    """Angle épaule-hanche vertical — optimal : 5–10° forward lean."""

    oscillation_verticale_cm: float | None
    """Amplitude verticale hanche sur un cycle. En cm si taille fournie,
    sinon valeur normalisée (0-1) avec approximatif=True."""

    ratio_contact_suspension: float
    """Ratio temps contact sol / durée cycle (approx. Morin 2011)."""

    nb_cycles_analyses: int
    """Nombre d'initial contacts détectés = cycles analysés."""

    cycles: list[GaitCycle] = field(default_factory=list)
    approximatif: bool = False
    """True si taille_patient absente — oscillation_verticale_cm non convertie."""

    confiance_detection: float = 1.0
    """Ratio frames avec landmarks valides / total frames."""

    def to_dict(self) -> dict:
        """Sérialise vers un dict JSON-compatible."""
        return asdict(self)
