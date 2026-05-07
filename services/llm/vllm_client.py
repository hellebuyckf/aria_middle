from openai import AsyncOpenAI

from core.config import settings
from models.diagnostic import DiagnosticLLM

_client = AsyncOpenAI(
    base_url=settings.VLLM_BASE_URL,
    api_key="aria-local",
)


async def generate_diagnostic(prompt: str) -> DiagnosticLLM:
    response = await _client.chat.completions.create(
        model="aria-ft",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30,
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("vllm_client: réponse vide du modèle")
    return DiagnosticLLM.model_validate_json(raw)


async def generate_report(
    prompt: str,
    session_id: str,
    patient_id: str,
    response_format: dict | None = None,
) -> str:
    response = await _client.chat.completions.create(
        model="aria-ft",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30,
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("vllm_client: réponse vide du modèle")
    return raw
