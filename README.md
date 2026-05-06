# aria_middle

Couche middleware du système **ARIA** — orchestration FastAPI + LangGraph pour l'analyse biomécanique de la course à pied.

Tourne sur Mac M3 (port 8000). Ne fait aucune inférence LLM locale.

---

## Architecture

```
┌─────────────────────┐        REST / WebSocket         ┌──────────────────────────────────────────────────────┐
│   aria-frontend     │ ──────────────────────────────► │                   aria_middle                        │
│   Vue.js            │ ◄────────────────────────────── │                   FastAPI + LangGraph                │
│   port 5173         │   events WS (progress/done/err) │                   port 8000                          │
└─────────────────────┘                                  │                                                      │
                                                         │  POST /api/sessions (vidéo sagittale)                │
                                                         │          │                                           │
                                                         │          ├─ 1. Floutage visage (RGPD)                │
                                                         │          ├─ 2. FFmpeg → frames                       │
                                                         │          ├─ 3. MediaPipe Pose → 33 keypoints/frame   │
                                                         │          ├─ 4. metrics_calculator → BiomechanicalMetrics │
                                                         │          └─ 5. LangGraph pipeline                    │
                                                         │                    │                                 │
                                                         │              video_agent                             │
                                                         │                    │                                 │
                                                         │               rag_agent ──► ChromaDB (PubMed)        │
                                                         │                    │                                 │
                                                         │             report_agent                             │
                                                         │                    │                                 │
                                                         └────────────────────┼─────────────────────────────────┘
                                                                              │ HTTP (OpenAI-compatible)
                                                                              ▼
                                                         ┌──────────────────────────────────────────────────────┐
                                                         │                   aria_llm                           │
                                                         │                   vLLM + ARIA-ft                     │
                                                         │                   PC RTX 4060 Ti — port 8001          │
                                                         └──────────────────────────────────────────────────────┘
```

---

## Stack

| Composant | Technologie |
|---|---|
| API | FastAPI + Uvicorn |
| Orchestration | LangGraph |
| Pose estimation | MediaPipe Pose (BlazePose GHUM, 33 keypoints) |
| Vector store | ChromaDB local (corpus PubMed) |
| LLM client | `openai` SDK → vLLM distant |
| PDF | WeasyPrint |
| Config | Pydantic Settings + `.env` |

---

## Installation

```zsh
uv sync
```

## Lancement

```zsh
uv run uvicorn main:app --reload --port 8000
```

## Variables d'environnement

Créer un fichier `.env` à la racine :

```env
VLLM_BASE_URL=http://192.168.x.x:8001/v1
CHROMADB_PATH=./data/corpus
SESSIONS_DIR=./data/sessions
MAX_VIDEO_SIZE_GB=2
SESSION_RETENTION_DAYS=30
LOG_LEVEL=INFO
```

---

## API

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Santé du service + état connexion vLLM |
| `POST` | `/api/sessions` | Upload vidéo, lancement pipeline |
| `GET` | `/api/sessions/{id}` | Statut + résultats |
| `WS` | `/ws/session/{id}` | Streaming événements temps réel |

Format événement WebSocket :

```json
{ "type": "progress", "etape": "video", "pct": 45, "message": "Extraction keypoints..." }
{ "type": "completed", "etape": "rapport", "rapport_url": "/api/sessions/SES-xxx/report" }
{ "type": "error", "etape": "llm", "message": "vLLM timeout" }
```

---

## Développement

```zsh
# Lint + format
uv run ruff check . && uv run ruff format .

# Type checking
uv run pyright

# Tests
uv run pytest tests/ -v
```

Le CI GitHub Actions vérifie automatiquement : qualité de code (ruff + pyright), tests unitaires, et audit CVE des dépendances (pip-audit).

---

## Métriques biomécaniques calculées

### Vue sagittale

| Métrique | Norme référence |
|---|---|
| Cadence (foulées/min) | 170–180 spm |
| Angle attaque pied | < 5° avant-pied, > 10° talon |
| Flexion genou à l'impact | 15–25° |
| Inclinaison tronc | 5–10° forward lean |
| Oscillation verticale | < 8 cm |
| Ratio contact/suspension | Dérivé cadence (Morin 2011) |

### Vue postérieure (optionnelle)

| Métrique | Norme référence |
|---|---|
| Pelvic drop | < 5° |
| Valgus genou | < 8° |
| Asymétrie de charge | < 10% |
| Oscillation latérale hanche | < 3 cm |
| Pronation pied | < 8° |

---

## RGPD

- Floutage visage obligatoire avant tout write disque
- Fichiers nommés `{patient_id}_{date}_{session_id}` — jamais de nom complet
- Suppression automatique des vidéos à J+30
- Aucune donnée patient dans les logs — uniquement `session_id`

---

*ARIA MVP v2.0 · PFE IA & Santé 2025-2026 · François Hellebuyck*
