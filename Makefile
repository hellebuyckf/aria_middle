.PHONY: install lint format check test run

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn main:app --reload --port 8000

download-model:
	mkdir -p models
	curl -L -o models/pose_landmarker_full.task \
	  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
