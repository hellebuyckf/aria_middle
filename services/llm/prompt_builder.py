from core.pathologies import NOMS
from core.state import ProfilChaussure
from models.metrics import BiomechanicalMetrics

# (champ, unité, lo, hi) — None = borne ouverte
_METRIC_NORMS: list[tuple[str, str, float | None, float | None]] = [
    ("cadence", "foulées/min", 170.0, 180.0),
    ("angle_attaque_pied", "°", None, 10.0),
    ("flexion_genou_impact", "°", 15.0, 25.0),
    ("inclinaison_tronc", "°", 5.0, 10.0),
    ("oscillation_verticale", "cm", None, 8.0),
    ("ratio_contact_suspension", "", 0.35, 0.65),
    ("pelvic_drop", "°", None, 5.0),
    ("valgus_genou", "°", None, 8.0),
    ("asymetrie_charge", "%", None, 10.0),
    ("oscillation_laterale_hanche", "cm", None, 3.0),
    ("pronation_pied", "°", None, 8.0),
]


def _norm_str(lo: float | None, hi: float | None, unit: str) -> str:
    if lo is not None and hi is not None:
        return f"norme : {lo}–{hi}{unit}"
    if hi is not None:
        return f"norme : < {hi}{unit}"
    if lo is not None:
        return f"norme : > {lo}{unit}"
    return ""


def _is_abnormal(value: float, lo: float | None, hi: float | None) -> bool:
    return (lo is not None and value < lo) or (hi is not None and value > hi)


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
        ("Drop actuel", f"{profil['drop_mm']} mm" if "drop_mm" in profil else None),
        ("Stabilité", profil.get("stabilite")),
        ("Amorti", profil.get("amorti")),
        ("Poids", profil.get("poids_type")),
        ("Dynamisme", profil.get("dynamisme")),
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

    data = metrics.model_dump()
    anormales = []
    normales = []
    for field, unit, lo, hi in _METRIC_NORMS:
        value = data.get(field)
        if value is None:
            continue
        unit_str = f" {unit}" if unit else ""
        norm = _norm_str(lo, hi, unit_str)
        if _is_abnormal(float(value), lo, hi):
            anormales.append(f'- [ANORMAL] "{field}": {value}{unit_str}  [{norm}]')
        else:
            normales.append(f'- [normal]  "{field}": {value}{unit_str}  [{norm}]')

    lines.append("## MÉTRIQUES BIOMÉCANIQUES OBJECTIVES")
    lines += anormales + normales

    lines += [
        "",
        "## INSTRUCTION",
        "En te basant UNIQUEMENT sur les métriques marquées [ANORMAL] ci-dessus,",
        "identifie la pathologie la plus probable parmi les suivantes :",
        ", ".join(f'"{p}"' for p in NOMS) + ".",
        "",
        "Réponds UNIQUEMENT en JSON valide avec exactement ces trois clés :",
        '{ "pathologie": "<nom exact choisi dans la liste>",',
        '  "confiance": "<élevée|modérée|faible>",',
        '  "justification": "<explication en 1-2 phrases citant les métriques anormales>" }',
    ]

    return "\n".join(lines)
