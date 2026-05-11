# CLAUDE.md — aria_middle
## ARIA · Couche Middleware · FastAPI + LangGraph

> Ce fichier est la référence de développement pour `aria_middle`.

---

## 1. Rôle de ce projet

`aria_middle` est la couche de contrôle et d'orchestration du système ARIA.
Il tourne sur **Linux** (port 8000) et fait le lien entre :

- `aria-frontend` (Vue.js, port 5173) — via REST + WebSocket
- `aria_llm` (vLLM sur RTX 4060 Ti, port 8001) — via HTTP REST

Il ne fait **aucune inférence LLM locale**. Toute génération de texte passe par vLLM.

---

## 2. Architecture interne

```
aria_middle/
├── main.py                        ← Point d'entrée FastAPI
├── api/
│   ├── routes/
│   │   ├── sessions.py            ← POST /api/sessions, GET /api/sessions/{id}
│   │   └── health.py              ← GET /health
│   └── websocket.py               ← WS /ws/session/{session_id}
├── agents/
│   ├── video_agent.py             ← Nœud LangGraph : vidéo → métriques JSON
│   ├── rag_agent.py               ← Nœud LangGraph : métriques → références PubMed
│   └── report_agent.py            ← Nœud LangGraph : prompt → rapport ARIA-ft
├── core/
│   ├── state.py                   ← ARIAState TypedDict + PubMedReference TypedDict (pmid, titre, extrait)
│   ├── graph.py                   ← Définition du graphe LangGraph
│   └── config.py                  ← Settings Pydantic depuis .env
├── services/
│   ├── pose/
│   │   ├── frame_extractor.py     ← FFmpeg/OpenCV : vidéo → frames
│   │   ├── mediapipe_service.py   ← MediaPipe Pose : frames → 33 keypoints
│   │   └── metrics_calculator.py ← Keypoints → métriques biomécaniques JSON
│   ├── rag/
│   │   └── chromadb_service.py    ← Retrieval PubMed depuis ChromaDB local
│   └── llm/
│       └── vllm_client.py         ← Client HTTP vers aria_llm (API OpenAI-compatible)
├── models/
│   ├── session.py                 ← SessionCreate, SessionResponse
│   ├── patient.py                 ← Patient, ProfilPatient
│   ├── metrics.py                 ← BiomechanicalMetrics, GaitCycle
│   ├── report.py                  ← ARIAReport, RecommendationBlock
│   └── diagnostic.py              ← DiagnosticLLM, sortie du diagnosis_agent
└── tests/
    ├── test_video_agent.py
    ├── test_rag_agent.py
    └── fixtures/                  ← Frames de test (ne pas versionner les vidéos)
```

---

## 3. Flux de données principal

```
POST /api/sessions  (patient_id + vidéo sagittale)
  │
  ├─ Couche 1 : floutage visage (RGPD) + pseudonymisation
  ├─ Couche 2 : frame_extractor → mediapipe_service → 33 keypoints/frame
  ├─ Couche 3 : metrics_calculator → BiomechanicalMetrics JSON
  ├─ Couche 4 : LangGraph
  │     video_agent → rag_agent → report_agent
  │     chaque nœud émet des événements → asyncio.Queue → WebSocket frontend
  └─ Couche 5 : ARIAReport JSON → WeasyPrint → PDF
```

Chaque étape publie son statut via `/ws/session/{id}` pour le streaming temps réel.

---

## 4. ARIAState — État partagé LangGraph

```python
# core/state.py
class ARIAState(TypedDict):
    session_id:  str
    patient_id:  str
    video_path:  str                    # chemin temporaire (supprimé J+30)
    video_path_posterior: str | None    # vue postérieure facultative (MVP)
    metrics:     BiomechanicalMetrics | None
    rag_refs:    list[PubMedReference]
    prompt:      str | None
    report:      ARIAReport | None
    statut:      Literal['idle', 'video', 'rag', 'llm', 'rapport', 'erreur']
    erreur:      str | None
```

**Règle** : aucun agent ne lit directement la vidéo brute. `video_agent` reçoit
`video_path` et retourne `metrics`. Les agents suivants ne voient jamais le chemin vidéo.

---

## 5. Pose Estimation — MediaPipe

- **Modèle** : MediaPipe Pose (33 keypoints, BlazePose GHUM)
- **Mode** : `STATIC_IMAGE_MODE = False` (vidéo, lissage temporel activé)
- **Landmarks utiles pour la course sagittale** :
  - `LEFT_HIP` (23) · `LEFT_KNEE` (25) · `LEFT_ANKLE` (27)
  - `LEFT_HEEL` (29) · `LEFT_FOOT_INDEX` (31)
  - `RIGHT_*` symétriques (24, 26, 28, 30, 32)
  - `LEFT_SHOULDER` (11) pour l'inclinaison tronc
- **Seuil de confiance** : `min_detection_confidence=0.7`, `min_tracking_confidence=0.5`
- **Backend** : MediaPipe utilise le CPU sur Linux (pas d'accélération GPU pour cette couche)

---

## 6. Métriques Biomécaniques cibles

### Vue sagittale — landmarks gauches : LEFT_SHOULDER(11), LEFT_HIP(23), LEFT_KNEE(25), LEFT_ANKLE(27), LEFT_HEEL(29), LEFT_FOOT_INDEX(31)

| Métrique | Calcul | Norme référence |
|---|---|---|
| Cadence (foulées/min) | Détection cycles via oscillation hanche | 170–180 spm (Heiderscheit 2011) |
| Angle attaque pied | Vecteur talon→pointe à l'impact | < 5° = avant-pied, > 10° = talon |
| Flexion genou impact | Angle hanche-genou-cheville à IC | 15–25° (normal) |
| Inclinaison tronc | Angle épaule-hanche vertical | 5–10° forward lean (optimal) |
| Oscillation verticale | Amplitude verticale hanche sur cycle | < 8 cm (optimal) |
| Ratio contact/suspension | Temps contact sol / temps cycle | Dérivé de la cadence + oscillation |

### Vue postérieure — landmarks bilatéraux : épaules(11,12), hanches(23,24), genoux(25,26), chevilles(27,28)

Disponibles uniquement si `video_path_posterior` est fourni (`vue_posterieure_disponible=True`).

| Métrique | Calcul | Norme référence |
|---|---|---|
| Pelvic drop (°) | 95e percentile de l'angle de la ligne des hanches par rapport à l'horizontale | < 5° |
| Valgus genou (°) | 180° − angle hanche-genou-cheville en plan frontal à l'IC | < 8° |
| Asymétrie charge (%) | Différence relative D/G du nombre de contacts sol | < 10% |
| Oscillation latérale hanche (cm) | Amplitude x du centre des hanches sur un cycle (ref. largeur inter-hanches = 32 cm) | < 3 cm |
| Pronation pied (°) | Inclinaison talon/cheville en plan frontal à l'IC | < 8° |

---

## 7. API REST — Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Santé du service + état connexion vLLM |
| `POST` | `/api/sessions` | Créer une session, upload vidéo, lancer pipeline |
| `GET` | `/api/sessions/{id}` | Statut + résultats d'une session |
| `WS` | `/ws/session/{id}` | Streaming événements temps réel |

**Format événement WebSocket** :
```json
{ "type": "progress", "etape": "video", "pct": 45, "message": "Extraction keypoints..." }
{ "type": "completed", "etape": "rapport", "rapport_url": "/api/sessions/SES-xxx/report" }
{ "type": "error", "etape": "llm", "message": "vLLM timeout" }
```

---

## 8. Connexion à aria_llm (vLLM)

`vllm_client.py` utilise le client `openai` pointé sur le PC RTX :

```python
client = AsyncOpenAI(
    base_url="http://<RTX_IP>:8001/v1",
    api_key="aria-local"          # vLLM n'authentifie pas en local
)
```

- Timeout : 30s (sinon `statut = 'erreur'`, rapport partiel sans LLM)
- Modèle cible : `aria-ft` (alias vLLM du checkpoint fine-tuné)
- Fallback MVP : si vLLM indisponible → rapport template sans génération LLM

---

## 9. RGPD — Règles immuables

1. **Floutage visage obligatoire** avant tout write disque (`services/pose/frame_extractor.py`)
2. **Nommage fichiers** : `{patient_id}_{date}_{session_id}` — jamais de nom complet
3. **Suppression vidéos J+30** — `aria_cleanup.py` (cron quotidien)
4. **Tokens Strava/Garmin** : jamais dans `ARIAState`, jamais loggués, utilisés puis détruits
5. **Logs** : pas de données patient dans les logs Loguru — uniquement `session_id`

---

## 10. Commandes de développement

```zsh
# Lancer le serveur en dev
uv run uvicorn main:app --reload --port 8000

# Linter + formatter
uv run ruff check . && uv run ruff format .

# Tests
uv run pytest tests/ -v

# Tester le pipeline vidéo isolément
uv run python -m services.pose.mediapipe_service --video data/sessions/test.mp4
```

---

## 11. Variables d'environnement (.env)

```env
VLLM_BASE_URL=http://192.168.x.x:8001/v1
CHROMADB_PATH=./data/corpus
SESSIONS_DIR=./data/sessions
MAX_VIDEO_SIZE_GB=2
SESSION_RETENTION_DAYS=30
LOG_LEVEL=INFO
```

---

## 12. Contraintes de performance

| Étape | Budget | Bloquant |
|---|---|---|
| Extraction frames (2 min @ 50fps) | ≤ 5s | Non |
| MediaPipe Pose (6000 frames) | ≤ 30s sur Linux CPU | Oui |
| Calcul métriques | ≤ 2s | Non |
| RAG ChromaDB top-5 | ≤ 3s | Non |
| Inférence ARIA-ft vLLM | ≤ 20s | Oui |
| **Total end-to-end** | **≤ 60s** | **Oui** |

---

*aria_middle · ARIA MVP v2.0 · Avril 2026*
*Auteur : François Hellebuyck — PFE IA & Santé 2025-2026*

