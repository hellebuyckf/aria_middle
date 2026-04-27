from dataclasses import dataclass, field


@dataclass
class PubMedReference:
    """Référence bibliographique issue de ChromaDB."""

    pmid: str
    titre: str
    resume: str
    score_similarite: float = 0.0


@dataclass
class RecommendationBlock:
    """Bloc de recommandation structurée du rapport ARIA."""

    categorie: str
    texte: str


@dataclass
class ARIAReport:
    """Rapport clinique généré par ARIA-ft."""

    session_id: str
    contenu: str
    recommandations: list[RecommendationBlock] = field(default_factory=list)
