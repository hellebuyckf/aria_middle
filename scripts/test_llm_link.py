"""Test du lien middleware → back LLM.

Lit le payload depuis tests/fixtures/test_llm_payload.json,
l'envoie au vLLM et affiche la réponse.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

from core.config import settings

PAYLOAD_FILE = Path(__file__).parent.parent / "tests" / "fixtures" / "test_llm_payload.json"


async def main() -> None:
    payload = json.loads(PAYLOAD_FILE.read_text())

    client = AsyncOpenAI(base_url=settings.VLLM_BASE_URL, api_key="aria-local")

    print(f"URL vLLM  : {settings.VLLM_BASE_URL}")
    print(f"Modèle    : {payload['model']}")
    print(f"Payload   : {PAYLOAD_FILE}")
    print()

    t0 = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            response_format=payload.get("response_format", {"type": "text"}),
            max_tokens=payload.get("max_tokens", 256),
            temperature=payload.get("temperature", 0.0),
            timeout=15,
        )
    except Exception as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    raw = response.choices[0].message.content or ""

    usage = response.usage
    tokens_info = (
        f"prompt={usage.prompt_tokens}  completion={usage.completion_tokens}"
        if usage else "n/a"
    )
    print(f"Statut    : OK  ({elapsed:.2f}s)")
    print(f"Tokens    : {tokens_info}")
    print()
    try:
        print(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(raw)


if __name__ == "__main__":
    asyncio.run(main())
