from pathlib import Path

import jinja2

from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport

_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(Path(__file__).parent)),
    autoescape=True,
)

_METRICS_SAGITTAL: list[tuple[str, str, str, str]] = [
    ("cadence", "Cadence", "foulées/min", "170 – 180"),
    ("angle_attaque_pied", "Angle attaque pied", "°", "< 10"),
    ("flexion_genou_impact", "Flexion genou impact", "°", "15 – 25"),
    ("inclinaison_tronc", "Inclinaison tronc", "°", "5 – 10"),
    ("oscillation_verticale", "Oscillation verticale", "cm", "< 8"),
    ("ratio_contact_suspension", "Ratio contact/suspension", "", "0.35 – 0.65"),
]

_METRICS_POSTERIOR: list[tuple[str, str, str, str]] = [
    ("pelvic_drop", "Pelvic drop", "°", "< 5"),
    ("valgus_genou", "Valgus genou", "°", "< 8"),
    ("asymetrie_charge", "Asymétrie charge", "%", "< 10"),
    ("oscillation_laterale_hanche", "Oscillation latérale hanche", "cm", "< 3"),
    ("pronation_pied", "Pronation pied", "°", "< 8"),
]


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f} {unit}".strip()


def _is_abnormal(field: str, value: float | None) -> bool:
    if value is None:
        return False
    thresholds: dict[str, tuple[float | None, float | None]] = {
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
    lo, hi = thresholds.get(field, (None, None))
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


def render_pdf(state: ARIAState) -> bytes:
    report: ARIAReport = state["report"]  # type: ignore[assignment]
    metrics: BiomechanicalMetrics | None = state["metrics"]

    rows_sag = []
    rows_post = []
    if metrics:
        for field, label, unit, norm in _METRICS_SAGITTAL:
            val = getattr(metrics, field, None)
            rows_sag.append(
                {
                    "label": label,
                    "value": _fmt(val, unit),
                    "norm": norm,
                    "abnormal": _is_abnormal(field, val),
                }
            )
        if metrics.vue_posterieure_disponible:
            for field, label, unit, norm in _METRICS_POSTERIOR:
                val = getattr(metrics, field, None)
                rows_post.append(
                    {
                        "label": label,
                        "value": _fmt(val, unit),
                        "norm": norm,
                        "abnormal": _is_abnormal(field, val),
                    }
                )

    html = _env.get_template("report_template.html").render(
        report=report,
        rows_sag=rows_sag,
        rows_post=rows_post,
    )
    import weasyprint  # type: ignore[import-untyped]  # lazy : évite crash au démarrage si DYLD_LIBRARY_PATH absent

    result = weasyprint.HTML(string=html).write_pdf()
    if result is None:
        raise RuntimeError("WeasyPrint n'a pas pu générer le PDF")
    return result
