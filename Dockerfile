FROM python:3.12-slim-bookworm

# Libs système : WeasyPrint (Pango/Cairo) + OpenCV/MediaPipe (Mesa headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
    libgdk-pixbuf2.0-0 libffi8 shared-mime-info \
    libgl1 libgles2 libegl1 libglib2.0-0 \
    libglvnd0 libglx0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -u 1000 aria

WORKDIR /app

# Couche cache : dépendances seules (rebuild uniquement si lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Source
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

RUN chown -R aria:aria /app

USER aria

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
