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
