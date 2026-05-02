import chromadb
import httpx
from fastapi import APIRouter
from loguru import logger

from core.config import settings

router = APIRouter()


async def _check_chromadb() -> str:
    try:
        client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        client.list_collections()
        return "connected"
    except Exception as exc:
        logger.warning("ChromaDB health check failed: {}", exc)
        return "disconnected"


async def _check_vllm() -> str:
    url = settings.VLLM_BASE_URL.rstrip("/v1").rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code < 500:
                return "connected"
            return "disconnected"
    except Exception as exc:
        logger.warning("vLLM health check failed: {}", exc)
        return "disconnected"


@router.get("/health")
async def health() -> dict:
    chromadb_status, vllm_status = (
        await _check_chromadb(),
        await _check_vllm(),
    )
    return {
        "status": "ok",
        "version": "2.0",
        "chromadb": chromadb_status,
        "vllm": vllm_status,
    }
