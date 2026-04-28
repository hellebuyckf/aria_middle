"""Récupère des abstracts PubMed pour les 6 pathologies ARIA via NCBI Entrez.

Usage:
    uv run python scripts/fetch_pubmed.py

Sortie : data/corpus/{slug_pathologie}.json
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

NCBI_EMAIL = "pro.fhellebuyck@pm.me"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RETMAX = 100
YEARS = 10
DELAY = 0.34  # NCBI : max 3 req/s sans clé API

PATHOLOGIES: list[tuple[str, str]] = [
    ("low_back_pain", "low back pain running biomechanics"),
    ("patellar_tendinopathy", "patellar tendinopathy running"),
    ("iliotibial_band_syndrome", "iliotibial band syndrome running"),
    ("medial_tibial_stress_syndrome", "medial tibial stress syndrome running"),
    ("achilles_tendinopathy", "achilles tendinopathy running biomechanics"),
    ("plantar_fasciitis", "plantar fasciitis running gait"),
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "corpus"


def esearch(query: str) -> list[str]:
    """Retourne la liste des PMIDs pour la requête donnée."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": RETMAX,
        "retmode": "json",
        "reldate": YEARS * 365,
        "datetype": "pdat",
        "email": NCBI_EMAIL,
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    count = int(data["esearchresult"]["count"])
    ids: list[str] = data["esearchresult"]["idlist"]

    print(f"    count PubMed : {count} | retournés : {len(ids)}")
    if count == 0:
        print(f"    [DEBUG] réponse brute esearch : {resp.text[:500]}")

    return ids


def efetch(ids: list[str]) -> str:
    """Récupère le XML des articles pour la liste de PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_efetch_xml(xml_text: str, slug: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records: list[dict] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        titre = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(p.itertext()) for p in abstract_parts).strip()

        if abstract:
            records.append(
                {"pmid": pmid, "titre": titre, "abstract": abstract, "pathologie": slug}
            )

    return records


def fetch_pathologie(slug: str, query: str) -> list[dict]:
    print(f"  esearch…")
    ids = esearch(query)

    if not ids:
        print(f"  Aucun PMID retourné, pathologie ignorée.")
        return []

    time.sleep(DELAY)

    print(f"  efetch XML ({len(ids)} ids)…")
    xml_text = efetch(ids)

    time.sleep(DELAY)

    records = parse_efetch_xml(xml_text, slug)
    print(f"  {len(records)} abstracts avec texte")
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, (slug, query) in enumerate(PATHOLOGIES):
        print(f"\n[{i + 1}/{len(PATHOLOGIES)}] {slug}")
        start = time.monotonic()

        try:
            records = fetch_pathologie(slug, query)
        except Exception as exc:
            print(f"  ERREUR : {exc}")
            continue

        out_path = OUTPUT_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        elapsed = time.monotonic() - start
        print(f"  -> {out_path} ({elapsed:.1f}s)")

    total = sum(
        len(json.loads((OUTPUT_DIR / f"{slug}.json").read_text()))
        for slug, _ in PATHOLOGIES
        if (OUTPUT_DIR / f"{slug}.json").exists()
    )
    print(f"\nTerminé. {total} abstracts au total dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
