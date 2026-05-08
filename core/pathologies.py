import tomllib
from dataclasses import dataclass
from pathlib import Path

_TOML_PATH = Path(__file__).parent.parent / "pathologies.toml"


@dataclass(frozen=True)
class Pathologie:
    nom: str
    metriques: list[str]


def _load() -> list[Pathologie]:
    with open(_TOML_PATH, "rb") as f:
        data = tomllib.load(f)
    return [
        Pathologie(nom=p["nom"], metriques=p["metriques"]) for p in data["pathologies"]
    ]


# Chargé une seule fois au démarrage.
PATHOLOGIES: list[Pathologie] = _load()
NOMS: list[str] = [p.nom for p in PATHOLOGIES]
_BY_NOM: dict[str, Pathologie] = {p.nom: p for p in PATHOLOGIES}


def metriques_pour(pathologie: str) -> list[str] | None:
    """Retourne les métriques pertinentes pour une pathologie, ou None si inconnue."""
    p = _BY_NOM.get(pathologie)
    return p.metriques if p else None
