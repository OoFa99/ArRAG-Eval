"""
Ollama Client - Minimal compatible wrapper for the existing LLMJudge.
Uses Qwen2.5:3B as requested.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

import ollama
from ollama import RequestError, ResponseError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama generation fails after all retry attempts."""

class OllamaClient:
    """
    Thin wrapper around the official `ollama` Python client.

    Exposes the two operations every call site in this project needs:
    free-text generation, and JSON-constrained generation (replacing the
    old regex / markdown-fence JSON parsing the Gemini prompts relied on).
    
    Qwen3 is a hybrid reasoning model: unless told otherwise, it emits a
    <think>...</think> block before its real answer. On a modest
    `num_predict` budget (we default to 500, to stay VRAM-friendly), that
    reasoning can consume the entire budget and leave the actual answer
    empty. `think=False` (the default here) turns that off so num_predict
    goes straight to the answer/JSON we actually want.

    Example:
        >>> client = OllamaClient()  # qwen3:4b on localhost:11434
        >>> client.generate("What is the capital of Egypt?")
        'Cairo is the capital of Egypt.'

        >>> client.generate_json(
        ...     prompt='Decompose this question: ...',
        ...     system_instruction="Respond only with valid JSON."
        ... )
        {'sub_queries': ['...', '...']}
    """
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL, 
                 host: str = DEFAULT_HOST, 
                 default_temperature: float = 0.1, 
                 default_max_tokens: int = 500, 
                 num_ctx_tokens: int = 4096, 
                 low_vram: bool = True, 
                 timeout: float = 120.0, 
                 max_retries: int = 3, 
                 request_delay: float = 1.5,
                 think: bool = False):
        """
        Args:
            model: Ollama model tag. Default "qwen3:4b" (~2.5GB VRAM at Q4_K_M).
            host: Ollama server URL. Default local daemon.
            default_temperature: Used when a call doesn't override it.
            default_max_tokens: Maps to Ollama's `num_predict`.
            num_ctx_tokens: Context window size. Kept modest by default to stay
                     safely inside 4GB VRAM — raise it if you have more
                     headroom or need longer contexts.
            low_vram: Passes Ollama's low_vram option — trades a little
                      speed for a smaller memory footprint. Turn off if
                      you move to a bigger GPU.
            timeout: HTTP timeout (seconds) per request.
            max_retries: Retry attempts on transient errors.
            request_delay: Sleep between retries.
            think: Whether to let Qwen3 use its <think> reasoning block.
                   Defaults to False.
        """
        
        self.model = model
        self.host = host
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.num_ctx_tokens = num_ctx_tokens
        self.low_vram = low_vram
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.default_think = think
        
        # Lazily discovered: some older `ollama` package versions don't
        # accept a `think` kwarg on chat(). Once we learn that, stop
        # sending it rather than erroring on every call.
        self._think_supported = True
        
        self.client = ollama.Client(host=host, timeout=timeout)
        self._verify_model_available()
        
        logger.info(f"✅ OllamaClient initialized with {model} (think={think})")

    def _verify_model_available(self):
        """Warn (don't crash) if the target model hasn't been pulled yet."""
        try:
            local_models = {m.model for m in self.client.list().models}
        except Exception as e:
            logger.warning(
                f"Could not reach Ollama at {self.host} ({e}). "
                f"Is `ollama serve` running?"
            )
            return

        base_name = self.model.split(":")[0]
        if self.model not in local_models and not any(m.startswith(base_name) for m in local_models):
            logger.warning(
                f"Model '{self.model}' not found locally. Run: ollama pull {self.model}"
            )
            
    def _build_options(self, 
                       temperature: Optional[float], 
                       max_tokens: Optional[int]) -> Dict[str, Any]:
        return {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
            "num_ctx": self.num_ctx_tokens,
            "low_vram": self.low_vram,
        }
        
    def _chat(self, chat_kwargs: Dict[str, Any], think: Optional[bool]):
        """Call client.chat(), applying `think` if supported, falling back if not."""
        if think is not None and self._think_supported:
            chat_kwargs = {**chat_kwargs, "think": think}

        try:
            return self.client.chat(**chat_kwargs)
        except TypeError as e:
            if "think" in chat_kwargs and "think" in str(e):
                logger.warning(
                    "Installed `ollama` package doesn't support the `think` "
                    "kwarg — Qwen3's reasoning mode can't be disabled at the "
                    "request level here. Run `pip install -U ollama` to fix "
                    "this, or raise default_max_tokens to give thinking room "
                    "to finish. Retrying this call without `think`."
                )
                self._think_supported = False
                chat_kwargs = {k: v for k, v in chat_kwargs.items() if k != "think"}
                return self.client.chat(**chat_kwargs)
            raise
    
    def generate(self,
                 prompt: str,
                 system_instruction: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 json_mode: bool = False,
                 think: Optional[bool] = None) -> str:
        """
        Generate a text completion. Mirrors the old
        `model.generate_content(prompt).text` pattern from google.generativeai.

        Args:
            think: Override the client's default_think for this call only.

        Raises:
            OllamaGenerationError: If all retry attempts fail, or every
                attempt returns an empty response (most commonly: Qwen3
                spent the whole num_predict budget thinking).
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        chat_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": self._build_options(temperature, max_tokens),
        }
        if json_mode:
            chat_kwargs["format"] = "json"
            
        effective_think = self.default_think if think is None else think

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._chat(chat_kwargs, effective_think)
                content = response.message.content.strip()

                if not content:
                    thinking = getattr(response.message, "thinking", None)
                    hint = (
                        " (a non-empty `thinking` field was returned, so the "
                        "model likely used up num_predict reasoning instead of "
                        "answering — try think=False or a larger max_tokens)"
                        if thinking else ""
                    )
                    raise OllamaGenerationError(
                        f"Ollama returned an empty response{hint}"
                    )

                return content
            except (ResponseError, RequestError) as e:
                last_error = e
                logger.error(f"Ollama API error (attempt {attempt}/{self.max_retries}): {e}")
            except OllamaGenerationError as e:
                last_error = e
                logger.error(f"Empty/invalid response (attempt {attempt}/{self.max_retries}): {e}")
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error calling Ollama (attempt {attempt}/{self.max_retries}): {e}")

            if attempt < self.max_retries:
                time.sleep(self.request_delay)

        raise OllamaGenerationError(
            f"Ollama generation failed after {self.max_retries} attempts: {last_error}"
        )
        
    def generate_json(self,
                      prompt: str,
                      system_instruction: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      think: Optional[bool] = None,
                      retry_on_truncation: bool = True,
                      truncation_retry_multiplier: float = 1.8) -> Optional[Dict[str, Any]]:
        """
        Generate a response constrained to valid JSON and parse it.

        Uses Ollama's native `format="json"` mode, so we don't need 
        markdown-fence stripping or regex fallback parsing.
        This only returns None if the call fails outright or (rarely) 
        the constrained output doesn't match valid JSON.
        """
        attempted_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        max_attempts = 2 if retry_on_truncation else 1
        
        last_raw = ""
        for attempt in range(1, max_attempts + 1):
            try:
                text = self.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=attempted_tokens,
                    json_mode=True,
                    think=think
                )
            except OllamaGenerationError as e:
                logger.warning(f"generate_json: generation failed: {e}")
                return None
            
            last_raw = text
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                if attempt < max_attempts:
                    attempted_tokens = int(attempted_tokens * truncation_retry_multiplier)
                    logger.warning(
                        f"JSON parse failed (likely truncated at max_tokens="
                        f"{max_tokens if max_tokens is not None else self.default_max_tokens}): "
                        f"{e}. Retrying with max_tokens={attempted_tokens}. Raw: {text[:200]}"
                    )
                    continue
                logger.warning(
                    f"JSON parse failed despite json_mode=True and a retry: {e}. Raw: {last_raw[:200]}"
                )
                return None

# Shared client + module-level convenience functions
_default_client: Optional[OllamaClient] = None

def get_client(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> OllamaClient:
    """Get or lazily create a shared OllamaClient (avoids re-checking
    model availability on every single call)."""
    global _default_client
    if _default_client is None or _default_client.model != model or _default_client.host != host:
        _default_client = OllamaClient(model=model, host=host)
    return _default_client

def generate_text(prompt: str,
                   system_instruction: Optional[str] = None,
                   model: str = DEFAULT_MODEL,
                   host: str = DEFAULT_HOST,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   think: Optional[bool] = None) -> str:
    """One-off text generation using the shared client."""
    return get_client(model=model, host=host).generate(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think
    )
    
def generate_json(prompt: str,
                   system_instruction: Optional[str] = None,
                   model: str = DEFAULT_MODEL,
                   host: str = DEFAULT_HOST,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None,
                   think: Optional[bool] = None,
                   retry_on_truncation: bool = True) -> Optional[Dict[str, Any]]:
    """One-off JSON generation using the shared client."""
    return get_client(model=model, host=host).generate_json(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        retry_on_truncation=retry_on_truncation
    )