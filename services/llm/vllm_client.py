from openai import AsyncOpenAI

from models.diagnostic import DiagnosticLLM
from core.config import settings


_client = AsyncOpenAI(
    base_url=settings.VLLM_BASE_URL,
    api_key="aria-local",
)


async def generate_diagnostic(prompt: str) -> DiagnosticLLM:
    response = await _client.chat.completions.create(
        model="aria-ft",
        messages=[{"role": "user", "content": prompt}],
        timeout=30,
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("vllm_client: réponse vide du modèle")
    return DiagnosticLLM.model_validate_json(raw)
