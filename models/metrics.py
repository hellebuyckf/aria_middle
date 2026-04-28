from pydantic import BaseModel, Field


class BiomechanicalMetrics(BaseModel):
    """Métriques biomécaniques calculées depuis la vidéo sagittale.

    Normes de référence documentées dans CLAUDE.md §6.
    Tous les champs sont None si la métrique n'a pas pu être calculée.
    """

    cadence: float | None = Field(
        default=None,
        description="Cadence en foulées/min. Norme : 170–180 spm (Heiderscheit 2011).",
    )

    angle_attaque_pied: float | None = Field(
        default=None,
        description="Angle talon→pointe à l'initial contact, en degrés. "
        "< 5° = avant-pied, > 10° = attaque talon.",
    )

    flexion_genou_impact: float | None = Field(
        default=None,
        description="Angle hanche-genou-cheville à l'initial contact, en degrés. "
        "Norme : 15–25°.",
    )

    inclinaison_tronc: float | None = Field(
        default=None,
        description="Inclinaison du tronc par rapport à la verticale, en degrés. "
        "Optimal : 5–10° forward lean.",
    )

    oscillation_verticale: float | None = Field(
        default=None,
        description="Amplitude verticale de la hanche sur un cycle, en cm. "
        "Optimal : < 8 cm. None si la taille du patient est inconnue.",
    )

    ratio_contact_suspension: float | None = Field(
        default=None,
        description="Ratio temps contact sol / durée du cycle (sans unité). "
        "Approximation biomécanique (Morin 2011), clampé entre 0.35 et 0.65.",
    )

    # --- Vue postérieure ---

    pelvic_drop: float | None = Field(
        default=None,
        description="Chute du bassin côté oscillant, en degrés. Norme : < 5°.",
    )

    valgus_genou: float | None = Field(
        default=None,
        description="Effondrement médial du genou à l'impact, en degrés. Norme : < 5°.",
    )

    asymetrie_charge: float | None = Field(
        default=None,
        description="Différence de charge droite/gauche, en %. 0 % = symétrie parfaite.",
    )

    oscillation_laterale_hanche: float | None = Field(
        default=None,
        description="Amplitude latérale de la hanche sur un cycle, en cm. Optimal : < 5 cm.",
    )

    pronation_pied: float | None = Field(
        default=None,
        description="Angle de pronation du pied à l'impact, en degrés. Norme : 4–6°.",
    )

    vue_posterieure_disponible: bool = Field(
        default=False,
        description="True si une vidéo postérieure a été traitée et que les 5 champs ci-dessus sont renseignés.",
    )

    def to_dict(self) -> dict:
        """Sérialise vers un dict JSON-compatible."""
        return self.model_dump()
