import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()

# session_id → liste de WebSockets connectés
connections: dict[str, list[WebSocket]] = {}

# session_id → queue d'événements poussés par les agents
queues: dict[str, asyncio.Queue] = {}


def _get_queue(session_id: str) -> asyncio.Queue:
    if session_id not in queues:
        queues[session_id] = asyncio.Queue()
    return queues[session_id]


async def broadcast(session_id: str, event: dict) -> None:
    """Pousse un événement dans la queue et l'envoie à tous les clients connectés."""
    await _get_queue(session_id).put(event)
    sockets = connections.get(session_id, [])
    for ws in list(sockets):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            sockets.remove(ws)


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    connections.setdefault(session_id, []).append(websocket)
    queue = _get_queue(session_id)
    logger.info("[{}] WebSocket connecté", session_id)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # keepalive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            await websocket.send_text(json.dumps(event))
            queue.task_done()

            if event.get("type") in ("completed", "error"):
                break
    except WebSocketDisconnect:
        logger.info("[{}] WebSocket déconnecté", session_id)
    finally:
        conns = connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            connections.pop(session_id, None)
            queues.pop(session_id, None)
