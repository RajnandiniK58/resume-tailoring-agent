"""One small wrapper around the LLM so the rest of the project never touches a vendor SDK.
To switch provider later, write another class with the same `structured()` method."""
import os
import time
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Free-tier-friendly defaults. Override with GEMINI_MODELS="modelA,modelB" in .env
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


class LLMError(RuntimeError):
    pass


class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None, models: Optional[List[str]] = None,
                 retries: int = 4, sleep=time.sleep):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set. Put it in a .env file (see .env.example).")
        self.client = genai.Client(api_key=key)
        env_models = os.getenv("GEMINI_MODELS", "")
        self.models = models or [m.strip() for m in env_models.split(",") if m.strip()] or DEFAULT_MODELS
        self.retries = retries
        self.sleep = sleep

    def structured(self, *, system: str, prompt: str, schema: Type[T], temperature: float = 0.2) -> T:
        """Ask for JSON matching `schema`. Retries on rate limits/server errors, then tries the next model."""
        from google.genai import errors, types

        last_error: Optional[Exception] = None
        for model in self.models:
            delay = 4
            for _ in range(self.retries):
                try:
                    resp = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            temperature=temperature,
                            response_mime_type="application/json",
                            response_schema=schema,
                        ),
                    )
                    return self._parse(resp, schema)
                except errors.APIError as e:
                    last_error = e
                    if e.code in (429, 500, 502, 503, 504):   # rate limit / temporary: wait, retry
                        self.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    if e.code == 404:                          # model name not available: next model
                        break
                    raise LLMError(f"Gemini API error {e.code}: {e}") from e
                except ValueError as e:                        # empty / malformed JSON: retry
                    last_error = e
                    continue
        raise LLMError(
            f"All models failed ({', '.join(self.models)}). Last error: {last_error}. "
            "If this is a quota error, wait a few minutes (per-minute limit) or until tomorrow (daily limit)."
        )

    @staticmethod
    def _parse(resp, schema: Type[T]) -> T:
        if isinstance(resp.parsed, schema):
            return resp.parsed
        if resp.text:
            return schema.model_validate_json(resp.text)    # raises ValueError subclasses if malformed
        raise ValueError("Empty response from model (possibly cut off or blocked).")