# scripts/test_video_agent.py  (ou dans un terminal uv run python -c "...")
# Usage :
#   uv run python scripts/test_video_agent.py                          ← sagittale seule
#   uv run python scripts/test_video_agent.py --posterior <chemin.mp4> ← bilatéral

import argparse
import asyncio

from agents.video_agent import video_agent
from core.state import ARIAState

parser = argparse.ArgumentParser()
parser.add_argument("--sagittale", default=None, help="Chemin vers la vidéo sagittale")
parser.add_argument(
    "--posterior", default=None, help="Chemin vers la vidéo postérieure"
)
args = parser.parse_args()

state: ARIAState = {
    "session_id": "SES-test-001",
    "patient_id": "PAT-042",
    "video_path": args.sagittale,
    "video_path_posterior": args.posterior,
    "pathologie_declaree": None,
    "age": None,
    "taille_cm": None,
    "poids_kg": None,
    "km_semaine": None,
    "niveau_pratique": None,
    "profil_chaussure": None,
    "strava_charge": None,
    "garmin_charge": None,
    "metrics": None,
    "diagnostic": None,
    "rag_refs": [],
    "prompt": None,
    "report": None,
    "statut": "idle",
    "erreur": None,
}

result = asyncio.run(video_agent(state))

print(f"statut : {result['statut']}")
print(
    f"vue_posterieure_disponible : {result['metrics'].vue_posterieure_disponible if result['metrics'] else 'N/A'}"
)
print(result["metrics"])
