import asyncio
import json
import uuid
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.params import File, Form
from fastapi.responses import Response
from loguru import logger

from api.websocket import broadcast
import core.events as events
from core.config import settings
from core.graph import run_analysis, run_report
from core.state import ARIAState, ProfilChaussure
from services.pdf.pdf_service import render_pdf

router = APIRouter()

sessions_store: dict[str, ARIAState] = {}


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    dest.write_bytes(content)


async def _run_pipeline(state: ARIAState) -> None:
    """Lance le pipeline complet : analyse vidéo → diagnostic → RAG → rapport."""
    session_id = state["session_id"]
    events.register(session_id, lambda evt: broadcast(session_id, evt))
    await broadcast(
        session_id,
        {"type": "progress", "etape": "video", "pct": 0, "message": "Analyse démarrée"},
    )
    try:
        # Phase 1 : vidéo → diagnostic → RAG
        analysis = await run_analysis(state)
        if analysis.get("statut") == "erreur":
            sessions_store[session_id] = analysis
            await broadcast(
                session_id,
                {
                    "type": "error",
                    "etape": "analyse",
                    "message": analysis.get("erreur", "Erreur inconnue"),
                },
            )
            return
        sessions_store[session_id] = ARIAState(**{**analysis, "statut": "pret"})
        logger.info("[{}] Analyse terminée", session_id)

        # Phase 2 : rapport LLM (automatique, pas d'action frontend requise)
        result = await run_report(sessions_store[session_id])
        sessions_store[session_id] = result
        logger.info("[{}] Rapport généré | statut={}", session_id, result["statut"])
        if result["statut"] == "erreur":
            await broadcast(
                session_id,
                {"type": "error", "etape": "rapport", "message": result["erreur"]},
            )
        else:
            # Génération et mise en cache PDF sur disque + libération key_frames
            loop = asyncio.get_running_loop()
            pdf_path = (
                Path(settings.SESSIONS_DIR) / session_id / f"aria_{session_id}.pdf"
            )
            try:
                pdf_bytes = await loop.run_in_executor(None, render_pdf, result)
                pdf_path.write_bytes(pdf_bytes)
                sessions_store[session_id] = ARIAState(**{**result, "key_frames": []})
                logger.info(
                    "[{}] PDF mis en cache ({} Ko)", session_id, len(pdf_bytes) // 1024
                )
            except Exception as pdf_exc:
                logger.warning(
                    "[{}] Mise en cache PDF échouée, key_frames conservées : {}",
                    session_id,
                    pdf_exc,
                )
            await broadcast(
                session_id,
                {
                    "type": "completed",
                    "etape": "rapport",
                    "rapport_url": f"/api/sessions/{session_id}/report",
                },
            )
    except Exception as exc:
        logger.exception("[{}] Pipeline erreur inattendue : {}", session_id, exc)
        sessions_store[session_id] = ARIAState(
            **{**state, "statut": "erreur", "erreur": str(exc)}
        )
        await broadcast(
            session_id, {"type": "error", "etape": "pipeline", "message": str(exc)}
        )
    finally:
        events.unregister(session_id)


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
        key_frames=[],
        metrics=None,
        diagnostic=None,
        rag_refs=[],
        prompt=None,
        report=None,
        statut="idle",
        erreur=None,
    )
    sessions_store[session_id] = state
    background_tasks.add_task(_run_pipeline, state)

    logger.info("[{}] Session créée | patient={}", session_id, patient_id)
    return {
        "session_id": session_id,
        "statut": "idle",
        "ws_url": f"/ws/session/{session_id}",
    }


@router.post("/sessions/{session_id}/generate")
async def get_report_data(session_id: str) -> dict:
    """Retourne les données du rapport (déjà généré par le pipeline automatique)."""
    state = sessions_store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session introuvable")
    if state["statut"] == "erreur":
        raise HTTPException(status_code=500, detail=state["erreur"])
    if state["statut"] != "rapport" or state["report"] is None:
        raise HTTPException(
            status_code=409, detail=f"Rapport non disponible (statut={state['statut']})"
        )
    return {
        "session_id": session_id,
        "statut": "rapport",
        "rapport_url": f"/api/sessions/{session_id}/report",
        "report": state["report"].model_dump(),
    }


@router.get("/sessions/{session_id}/report")
async def download_report(session_id: str) -> Response:
    state = sessions_store.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session introuvable")
    if state["statut"] != "rapport" or state["report"] is None:
        raise HTTPException(
            status_code=409, detail=f"Rapport non disponible (statut={state['statut']})"
        )
    pdf_path = Path(settings.SESSIONS_DIR) / session_id / f"aria_{session_id}.pdf"
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
    else:
        loop = asyncio.get_running_loop()
        pdf_bytes = await loop.run_in_executor(None, render_pdf, state)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="aria_{session_id}.pdf"'},
    )


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
        "diagnostic": state["diagnostic"].model_dump() if state["diagnostic"] else None,
        "rag_refs": state["rag_refs"],
        "report": state["report"],
    }
