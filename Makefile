.PHONY: help fetch-corpus index-corpus build-corpus \
        build-protocols \
        serve debug health test-link test test-video \
        test-diagnosis test-rag test-report test-graph lint format install download-model visualize \
        blur-video \
        docs docs-serve clean docker-build docker-up docker-down docker-logs docker-debug

MIDDLE_URL ?= http://localhost:8000
DOCS_DIR   ?= docs

VIDEO   ?= data/sessions/test.mp4
OUTPUT  ?= output_pose.mp4
BLURRED ?= blurred.mp4

# ────────────────────────────────────────────────────────────────────────────
help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage : make \033[36m<cible>\033[0m\n\nCibles :\n"} \
	      /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables surchargeables :"
	@echo "    VIDEO=<chemin>    vidéo source pour 'visualize' et 'blur-video' (défaut : $(VIDEO))"
	@echo "    OUTPUT=<chemin>   vidéo annotée pour 'visualize'                (défaut : $(OUTPUT))"
	@echo "    BLURRED=<chemin>  vidéo floutée pour 'blur-video'               (défaut : $(BLURRED))"
	@echo ""

# ── Corpus ───────────────────────────────────────────────────────────────────
fetch-corpus: ## Télécharge les abstracts PubMed (NCBI Entrez → data/corpus/*.json)
	uv run python scripts/fetch_pubmed.py

index-corpus: ## Indexe data/corpus/*.json dans ChromaDB (embeddings e5-multilingual)
	uv run python scripts/aria_index_corpus.py

build-corpus: fetch-corpus index-corpus ## Télécharge ET indexe le corpus PubMed en une commande

build-protocols: ## Indexe les protocoles de rééducation dans ChromaDB (collection aria_protocols)
	uv run python scripts/build_protocol_corpus.py

# ── Santé ────────────────────────────────────────────────────────────────────
health: ## Vérifie la santé du middleware (port 8000) et du back LLM
	@BACK=$$(grep -m1 '^URL_BACK_LLM=' .env 2>/dev/null | cut -d= -f2- | tr -d ' '); \
	BACK=$${BACK:-http://localhost:8001}; \
	echo "── Middleware ($(MIDDLE_URL)/health) ──"; \
	BODY=$$(curl -sf --max-time 3 $(MIDDLE_URL)/health) && (echo "$$BODY" | python3 -m json.tool 2>/dev/null || echo "  ✓ ok") || echo "  ✗ middleware non joignable"; \
	echo "── Back LLM ($${BACK}/health) ──"; \
	HTTP=$$(curl -so /dev/null -w "%{http_code}" --max-time 3 $${BACK}/health); \
	[ "$$HTTP" = "200" ] && echo "  ✓ ok (HTTP $$HTTP)" || echo "  ✗ back LLM non joignable (HTTP $$HTTP)"

test-link: ## Envoie le payload tests/fixtures/test_llm_payload.json au back LLM et affiche la réponse
	uv run python scripts/test_llm_link.py

# ── Serveur ──────────────────────────────────────────────────────────────────
serve: ## Lance le serveur FastAPI en mode rechargement (port 8000)
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

debug: ## Lance le serveur en mode debug (LOG_LEVEL=DEBUG + tracebacks complets)
	LOG_LEVEL=DEBUG uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

docker-build: ## Construit l'image Docker aria_middle
	docker compose build

docker-up: ## Lance le conteneur en arrière-plan
	docker compose up -d

docker-down: ## Arrête et supprime le conteneur
	docker compose down

docker-logs: ## Suit les logs du conteneur en temps réel
	docker compose logs -f aria_middle

docker-debug: ## Lance le conteneur avec LOG_LEVEL=DEBUG
	LOG_LEVEL=DEBUG docker compose up

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

download-model: ## Télécharge le modèle MediaPipe PoseLandmarker (racine du projet)
	curl -L -o pose_landmarker_full.task \
	  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# ── Visualisation ────────────────────────────────────────────────────────────
visualize: ## Génère une vidéo annotée MediaPipe (VIDEO=... OUTPUT=...)
	uv run python scripts/visualize_pose.py --video $(VIDEO) --output $(OUTPUT)

blur-video: ## Floute les visages d'une vidéo RGPD (VIDEO=... BLURRED=...)
	uv run python scripts/blur_video.py $(VIDEO) $(BLURRED)

# ── Documentation ────────────────────────────────────────────────────────────
docs: ## Génère la documentation HTML depuis les docstrings (pdoc → $(DOCS_DIR)/)
	uv run --with pdoc pdoc --output-dir $(DOCS_DIR) \
		agents services core models api
	@echo "Documentation générée → $(DOCS_DIR)/index.html"

docs-serve: ## Sert la documentation en live avec rechargement automatique (port 8080)
	uv run --with pdoc pdoc --port 8080 agents services core models api

# ── Nettoyage ────────────────────────────────────────────────────────────────
clean: ## Supprime __pycache__, .pytest_cache, fichiers .pyc et la doc générée
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf $(DOCS_DIR)
