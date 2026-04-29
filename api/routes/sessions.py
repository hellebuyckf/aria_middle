import json
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.params import File, Form
from loguru import logger

from core.config import settings
from core.graph import run_pipeline
from core.state import ARIAState, ProfilChaussure

router = APIRouter()

# session_id → ARIAState courant (mis à jour en fin de pipeline)
sessions_store: dict[str, ARIAState] = {}


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    dest.write_bytes(content)


async def _run_and_store(state: ARIAState) -> None:
    session_id = state["session_id"]
    try:
        result = await run_pipeline(state)
        sessions_store[session_id] = result
        logger.info("[{}] Pipeline terminé | statut={}", session_id, result["statut"])
    except Exception as exc:
        logger.error("[{}] Pipeline erreur inattendue : {}", session_id, exc)
        sessions_store[session_id] = {
            **state,
            "statut": "erreur",
            "erreur": str(exc),
        }


def _parse_profil_chaussure(raw: str | None) -> ProfilChaussure | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        return ProfilChaussure(
            **{k: v for k, v in data.items() if k in ProfilChaussure.__annotations__}
        )
    except Exception as exc:
        logger.warning("profil_chaussure JSON invalide, ignoré : {}", exc)
        return None


@router.post("/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    patient_id: Annotated[str, Form()],
    video_sagittale: Annotated[UploadFile, File()],
    pathologie_declaree: Annotated[str | None, Form()] = None,
    age: Annotated[int | None, Form()] = None,
    taille_cm: Annotated[int | None, Form()] = None,
    poids_kg: Annotated[float | None, Form()] = None,
    km_semaine: Annotated[int | None, Form()] = None,
    niveau_pratique: Annotated[str | None, Form()] = None,
    profil_chaussure: Annotated[str | None, Form()] = None,
    video_posterieure: Annotated[UploadFile | None, File()] = None,
) -> dict:
    session_id = f"SES-{uuid.uuid4().hex[:12].upper()}"
    today = date.today().isoformat()
    session_dir = Path(settings.SESSIONS_DIR) / session_id

    video_sagittale_path = session_dir / f"{patient_id}_{today}_{session_id}.mp4"
    await _save_upload(video_sagittale, video_sagittale_path)

    video_posterieure_path: str | None = None
    if video_posterieure is not None:
        post_path = session_dir / f"{patient_id}_{today}_{session_id}_post.mp4"
        await _save_upload(video_posterieure, post_path)
        video_posterieure_path = str(post_path)

    state = ARIAState(
        session_id=session_id,
        patient_id=patient_id,
        video_path=str(video_sagittale_path),
        video_path_posterior=video_posterieure_path,
        pathologie_declaree=pathologie_declaree,
        age=age,
        taille_cm=taille_cm,
        poids_kg=poids_kg,
        km_semaine=km_semaine,
        niveau_pratique=niveau_pratique,
        profil_chaussure=_parse_profil_chaussure(profil_chaussure),
        strava_charge=None,
        garmin_charge=None,
        metrics=None,
        diagnostic=None,
        rag_refs=[],
        prompt=None,
        report=None,
        statut="idle",
        erreur=None,
    )
    sessions_store[session_id] = state
    background_tasks.add_task(_run_and_store, state)

    logger.info("[{}] Session créée | patient={}", session_id, patient_id)
    return {
        "session_id": session_id,
        "statut": "idle",
        "ws_url": f"/ws/session/{session_id}",
    }


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    state = sessions_store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session introuvable")
    return {
        "session_id": state["session_id"],
        "statut": state["statut"],
        "erreur": state["erreur"],
        "metrics": state["metrics"],
        "report": state["report"],
    }
