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


def build_diagnostic_prompt(
    metrics: BiomechanicalMetrics,
    pathologie_declaree: str | None,
) -> str:
    lines: list[str] = [
        "Tu es ARIA, assistant biomécanique clinique.",
        "",
        "## MÉTRIQUES",
    ]

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
