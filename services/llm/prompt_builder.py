from core.state import ProfilChaussure
from models.metrics import BiomechanicalMetrics

# (champ, unité, norme textuelle)
_METRIC_NORMS: list[tuple[str, str, str]] = [
    ("cadence", "foulées/min", "norme : 170–180 spm"),
    ("angle_attaque_pied", "°", "norme : < 5° (avant-pied) ou > 10° (talon)"),
    ("flexion_genou_impact", "°", "norme : 15–25°"),
    ("inclinaison_tronc", "°", "norme : 5–10° forward lean"),
    ("oscillation_verticale", "cm", "norme : < 8 cm"),
    ("ratio_contact_suspension", "", "norme : < 0.5"),
    ("pelvic_drop", "°", "norme : < 5°"),
    ("valgus_genou", "°", "norme : < 8°"),
    ("asymetrie_charge", "%", "norme : < 10 %"),
    ("oscillation_laterale_hanche", "cm", "norme : < 3 cm"),
    ("pronation_pied", "°", "norme : < 8°"),
]

_MVP_PATHOLOGIES = (
    "Lombalgie",
    "SBIT (syndrome de la bandelette ilio-tibiale)",
    "Tendinite rotulienne",
    "Syndrome fémoro-patellaire",
    "Attaque talon excessive",
    "Asymétrie de charge",
)


def _section_contexte_patient(
    age: int | None,
    taille_cm: int | None,
    poids_kg: float | None,
    niveau_pratique: str | None,
    km_semaine: int | None,
) -> list[str]:
    champs = [
        ("Âge", f"{age} ans" if age is not None else None),
        ("Taille", f"{taille_cm} cm" if taille_cm is not None else None),
        ("Poids", f"{poids_kg} kg" if poids_kg is not None else None),
        ("Niveau", niveau_pratique),
        (
            "Volume hebdomadaire",
            f"{km_semaine} km/semaine" if km_semaine is not None else None,
        ),
    ]
    lignes = [f"- {label} : {valeur}" for label, valeur in champs if valeur is not None]
    if not lignes:
        return []
    return ["## CONTEXTE PATIENT", *lignes, ""]


def _section_profil_chaussure(profil: ProfilChaussure | None) -> list[str]:
    if not profil:
        return []
    champs = [
        ("Marque", profil.get("marque")),
        ("Modèle", profil.get("modele")),
        ("Drop", f"{profil['drop_mm']} mm" if "drop_mm" in profil else None),
        ("Stabilité", profil.get("stabilite")),
        ("Amorti", profil.get("amorti")),
    ]
    lignes = [f"- {label} : {valeur}" for label, valeur in champs if valeur is not None]
    if not lignes:
        return []
    return ["## PROFIL CHAUSSURE", *lignes, ""]


def _section_charge_entrainement(
    strava: dict | None,
    garmin: dict | None,
) -> list[str]:
    if not strava and not garmin:
        return []
    lines = ["## CHARGE ENTRAÎNEMENT"]
    if strava:
        lines.append("Source : Strava")
        for k, v in strava.items():
            lines.append(f"- {k} : {v}")
    if garmin:
        lines.append("Source : Garmin")
        for k, v in garmin.items():
            lines.append(f"- {k} : {v}")
    lines.append("")
    return lines


def build_diagnostic_prompt(
    metrics: BiomechanicalMetrics,
    pathologie_declaree: str | None,
    age: int | None = None,
    taille_cm: int | None = None,
    poids_kg: float | None = None,
    niveau_pratique: str | None = None,
    km_semaine: int | None = None,
    profil_chaussure: ProfilChaussure | None = None,
    strava_charge: dict | None = None,
    garmin_charge: dict | None = None,
) -> str:
    lines: list[str] = ["Tu es ARIA, assistant biomécanique clinique.", ""]

    lines += _section_contexte_patient(
        age, taille_cm, poids_kg, niveau_pratique, km_semaine
    )
    lines += _section_profil_chaussure(profil_chaussure)
    lines += _section_charge_entrainement(strava_charge, garmin_charge)

    lines.append("## MÉTRIQUES")
    data = metrics.model_dump()
    for field, unit, norm in _METRIC_NORMS:
        value = data.get(field)
        if value is None:
            continue
        unit_str = f" {unit}" if unit else ""
        lines.append(f'- "{field}": {value}{unit_str}  [{norm}]')

    lines += [
        "",
        "## PATHOLOGIE DÉCLARÉE PAR LE PRATICIEN",
        pathologie_declaree if pathologie_declaree else "aucune",
        "",
        "## INSTRUCTION",
        "Identifie la pathologie la plus probable parmi les suivantes :",
        ", ".join(f'"{p}"' for p in _MVP_PATHOLOGIES) + ".",
        "",
        "Réponds UNIQUEMENT en JSON avec les clés suivantes :",
        '{ "pathologie": "<nom>", "confiance": "élevée|modérée|faible", '
        '"justification": "<explication courte en référence aux métriques>" }',
    ]

    return "\n".join(lines)
