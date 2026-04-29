.PHONY: help fetch-corpus index-corpus build-corpus serve test test-video \
        test-diagnosis test-rag test-report test-graph lint format install download-model visualize clean

VIDEO  ?= data/sessions/test.mp4
OUTPUT ?= output_pose.mp4

# ────────────────────────────────────────────────────────────────────────────
help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage : make \033[36m<cible>\033[0m\n\nCibles :\n"} \
	      /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables surchargeables :"
	@echo "    VIDEO=<chemin>    vidéo source pour 'visualize'  (défaut : $(VIDEO))"
	@echo "    OUTPUT=<chemin>   vidéo annotée pour 'visualize' (défaut : $(OUTPUT))"
	@echo ""

# ── Corpus ───────────────────────────────────────────────────────────────────
fetch-corpus: ## Télécharge les abstracts PubMed (NCBI Entrez → data/corpus/*.json)
	uv run python scripts/fetch_pubmed.py

index-corpus: ## Indexe data/corpus/*.json dans ChromaDB (embeddings e5-multilingual)
	uv run python scripts/aria_index_corpus.py

build-corpus: fetch-corpus index-corpus ## Télécharge ET indexe le corpus en une commande

# ── Serveur ──────────────────────────────────────────────────────────────────
serve: ## Lance le serveur FastAPI en mode rechargement (port 8000)
	uv run uvicorn main:app --reload --port 8000

# ── Tests ────────────────────────────────────────────────────────────────────
test: ## Lance tous les tests pytest
	uv run pytest tests/ -v

test-video: ## Tests du video_agent uniquement (sortie détaillée)
	uv run pytest tests/test_video_agent.py -v -s

test-diagnosis: ## Tests du diagnosis_agent uniquement (sortie détaillée)
	uv run pytest tests/test_diagnosis_agent.py -v -s

test-rag: ## Tests du rag_agent uniquement (sortie détaillée)
	uv run pytest tests/test_rag_agent.py -v -s

test-report: ## Tests du report_agent uniquement (sortie détaillée)
	uv run pytest tests/test_report_agent.py -v -s

test-graph: ## Tests du graphe LangGraph bout-en-bout (sortie détaillée)
	uv run pytest tests/test_graph.py -v -s

# ── Qualité ──────────────────────────────────────────────────────────────────
lint: ## Lint + formatage automatique (ruff check & format)
	uv run ruff check . && uv run ruff format .

format: ## Vérification stricte sans correction (CI)
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

# ── Setup ────────────────────────────────────────────────────────────────────
install: ## Installe toutes les dépendances (uv sync)
	uv sync --all-groups

download-model: ## Télécharge le modèle MediaPipe PoseLandmarker
	mkdir -p models
	curl -L -o models/pose_landmarker_full.task \
	  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# ── Visualisation ────────────────────────────────────────────────────────────
visualize: ## Génère une vidéo annotée MediaPipe (VIDEO=... OUTPUT=...)
	uv run python scripts/visualize_pose.py --video $(VIDEO) --output $(OUTPUT)

# ── Nettoyage ────────────────────────────────────────────────────────────────
clean: ## Supprime __pycache__, .pytest_cache et fichiers .pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
