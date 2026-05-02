import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

# session_id → queue d'événements (seul canal d'envoi)
queues: dict[str, asyncio.Queue] = {}


def _get_queue(session_id: str) -> asyncio.Queue:
    if session_id not in queues:
        queues[session_id] = asyncio.Queue()
    return queues[session_id]


async def broadcast(session_id: str, event: dict) -> None:
    """Pousse un événement dans la queue de session."""
    await _get_queue(session_id).put(event)


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
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

            if event.get("type") in ("completed", "error", "ready"):
                await websocket.close()
                break
    except WebSocketDisconnect:
        logger.info("[{}] WebSocket déconnecté", session_id)
    except Exception as exc:
        logger.warning("[{}] WebSocket fermé (erreur réseau) : {}", session_id, exc)
    finally:
        queues.pop(session_id, None)
