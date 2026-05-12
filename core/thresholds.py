from models.metrics import BiomechanicalMetrics

# Seuils normatifs (lo, hi) — None = borne ouverte.
# Source unique partagée par video_agent, report_agent et frame_annotator.
THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "cadence": (170.0, 180.0),
    "angle_attaque_pied": (None, 10.0),
    "flexion_genou_impact": (15.0, 25.0),
    "inclinaison_tronc": (5.0, 10.0),
    "oscillation_verticale": (None, 8.0),
    "ratio_contact_suspension": (0.35, 0.65),
    "pelvic_drop": (None, 5.0),
    "valgus_genou": (None, 8.0),
    "asymetrie_charge": (None, 10.0),
    "oscillation_laterale_hanche": (None, 3.0),
    "pronation_pied": (None, 8.0),
}


def is_abnormal(field: str, value: float) -> bool:
    lo, hi = THRESHOLDS.get(field, (None, None))
    return (lo is not None and value < lo) or (hi is not None and value > hi)


def compute_abnormal_metrics(metrics: BiomechanicalMetrics) -> list[str]:
    """Retourne les noms des champs hors norme (sagittaux + postérieurs)."""
    data = metrics.model_dump()
    return [
        field
        for field in THRESHOLDS
        if isinstance(data.get(field), (int, float))
        and is_abnormal(field, float(data[field]))
    ]
