"""Validate-and-retry wrapper for structured LLM output.

Every LLM generation that must conform to a Pydantic schema goes through
``generate_structured``. On a ``ValidationError`` we re-prompt the model with the
error text appended, up to ``max_validation_retries``. If it still fails we raise,
so invalid output never reaches the dashboard.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from config import settings

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model response (handles ```json fences)."""
    text = text.strip()
    if "```" in text:
        # take content between the first pair of fences
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[len("json"):].strip()
            if p.startswith("{") or p.startswith("["):
                return p
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def generate_structured(
    chat_model,
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
) -> T:
    """Call the chat model and coerce its output into ``schema`` with retries."""
    from langchain_core.messages import HumanMessage, SystemMessage

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    base_system = (
        f"{system_prompt}\n\n"
        "Respond with ONLY a single JSON object that conforms to this JSON schema. "
        "No prose, no markdown fences.\n\n"
        f"JSON schema:\n{schema_json}"
    )

    messages = [SystemMessage(content=base_system), HumanMessage(content=user_prompt)]
    last_error: Exception | None = None

    # On the local Ollama backend, force JSON-formatted output for far more
    # reliable parsing. Claude doesn't take this kwarg, so only bind for ollama.
    model = chat_model
    if settings.backend == "ollama":
        try:
            model = chat_model.bind(format="json")
        except Exception:  # noqa: BLE001 - fall back to the unbound model
            model = chat_model

    for attempt in range(settings.agent.max_validation_retries + 1):
        response = model.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        try:
            payload = json.loads(_extract_json(raw))
            return schema.model_validate(payload)
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            from langchain_core.messages import AIMessage, HumanMessage as HM

            messages.append(AIMessage(content=raw))
            messages.append(HM(content=(
                f"That did not validate against the schema. Error:\n{exc}\n"
                "Return corrected JSON only."
            )))

    raise ValueError(
        f"Failed to produce valid {schema.__name__} after "
        f"{settings.agent.max_validation_retries + 1} attempts: {last_error}"
    )
