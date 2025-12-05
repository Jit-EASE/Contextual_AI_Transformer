# act_ie/llm_backend.py

import os
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

from .config import Config


class LLMBackend:
    """
    Thin wrapper so you can swap between OpenAI and local HuggingFace models.
    """

    def __init__(self, backend: str = Config.LLM_BACKEND):
        self.backend = backend
        self.openai_model = Config.OPENAI_MODEL
        self.local_model_name = Config.LOCAL_MODEL_NAME

        if backend == "openai":
            self._init_openai()
        elif backend == "local":
            self._init_local()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_openai(self) -> None:
        if openai is None:
            raise ImportError("openai package not installed. Install with: pip install openai")

        api_key = os.getenv(Config.OPENAI_API_KEY_ENV)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {Config.OPENAI_API_KEY_ENV} is not set. "
                f"Export your OpenAI key before running."
            )
        openai.api_key = api_key
        print(f"[LLMBackend] Using OpenAI backend with model: {self.openai_model}")

    def _init_local(self) -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            raise ImportError(
                "transformers and torch are required for local backend. "
                "Install with: pip install transformers torch"
            )
        print(f"[LLMBackend] Loading local HF model: {self.local_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.local_model_name)

    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        if self.backend == "openai":
            return self._generate_openai(prompt, max_tokens)
        elif self.backend == "local":
            return self._generate_local(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        resp = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ACT-IE, a specialised AI assistant for the Irish agri-food sector. "
                        "You must be precise, policy-aware, and transparent about uncertainty. "
                        "You do NOT provide legal or financial advice; you provide analytical, "
                        "scenario-based insight grounded in data and published sources."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message["content"]

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
