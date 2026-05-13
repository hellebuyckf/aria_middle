import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

# session_id → queue d'événements (seul canal d'envoi)
queues: dict[str, asyncio.Queue] = {}

# Cache du dernier événement terminal pour les reconnexions tardives
_terminal_cache: dict[str, dict] = {}

_TERMINAL_TYPES = frozenset({"completed", "error", "ready"})
# Délai (s) avant fermeture après un événement terminal — laisse le client
# traiter le message et couper proprement sa connexion.
_CLOSE_GRACE_S = 1.5


def _get_queue(session_id: str) -> asyncio.Queue:
    if session_id not in queues:
        queues[session_id] = asyncio.Queue()
    return queues[session_id]


def reset_session(session_id: str) -> None:
    """Purge le cache terminal — à appeler au démarrage de chaque pipeline."""
    _terminal_cache.pop(session_id, None)


async def broadcast(session_id: str, event: dict) -> None:
    """Pousse un événement dans la queue de session."""
    if event.get("type") in _TERMINAL_TYPES:
        _terminal_cache[session_id] = event
    await _get_queue(session_id).put(event)


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    # Reconnexion après fin de session : rejouer l'événement terminal immédiatement
    # pour que le client sache s'arrêter sans épuiser ses tentatives.
    if session_id in _terminal_cache:
        logger.info(
            "[{}] WebSocket reconnecté sur session terminée — replay terminal",
            session_id,
        )
        await websocket.send_text(json.dumps(_terminal_cache[session_id]))
        await asyncio.sleep(_CLOSE_GRACE_S)
        await websocket.close()
        return

    queue = _get_queue(session_id)
    logger.info("[{}] WebSocket connecté", session_id)
    await websocket.send_text(
        json.dumps({"type": "connected", "session_id": session_id})
    )

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            try:
                await websocket.send_text(json.dumps(event))
            except TypeError as exc:
                logger.error(
                    "[{}] Event non sérialisable (type={}) : {}",
                    session_id,
                    event.get("type"),
                    exc,
                )
                queue.task_done()
                continue
            queue.task_done()

            if event.get("type") in _TERMINAL_TYPES:
                await asyncio.sleep(_CLOSE_GRACE_S)
                await websocket.close()
                break
    except WebSocketDisconnect:
        logger.info("[{}] WebSocket déconnecté", session_id)
    except Exception as exc:
        logger.warning("[{}] WebSocket fermé (erreur réseau) : {}", session_id, exc)
    finally:
        queues.pop(session_id, None)
