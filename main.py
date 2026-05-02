import logging
import os
import sys
from contextlib import asynccontextmanager

# Sur macOS, WeasyPrint (cffi/pango) nécessite que les libs Homebrew soient accessibles
# via DYLD_LIBRARY_PATH. Ce bloc injecte le path et re-exec le process si nécessaire.
if sys.platform == "darwin":
    _BREW_LIB = "/opt/homebrew/lib"
    _dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if _BREW_LIB not in _dyld:
        os.environ["DYLD_LIBRARY_PATH"] = f"{_BREW_LIB}:{_dyld}".rstrip(":")
        os.execv(sys.executable, [sys.executable] + sys.argv)
from typing import AsyncIterator

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import health, sessions
from api.websocket import router as ws_router
from core.config import settings


class _InterceptHandler(logging.Handler):
    """Redirige le logging standard vers Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.LOG_LEVEL, enqueue=True)
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _setup_logging()
    try:
        client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        client.heartbeat()
        logger.info("ChromaDB connecté | path={}", settings.CHROMADB_PATH)
    except Exception as exc:
        logger.warning("ChromaDB indisponible au démarrage : {}", exc)
    yield
    logger.info("ARIA middleware arrêté")


app = FastAPI(title="ARIA Middleware", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sessions.router, prefix="/api")
app.include_router(ws_router)
