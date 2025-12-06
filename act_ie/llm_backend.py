# act_ie/llm_backend.py

from openai import OpenAI
from .config import Config

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None


class LLMBackend:

    def __init__(self, backend: str = Config.LLM_BACKEND):
        self.backend = backend
        self.openai_model = Config.OPENAI_MODEL

        if backend == "openai":
            self.client = OpenAI()   # NEW client init
        elif backend == "local":
            self._init_local()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_local(self):
        if AutoTokenizer is None:
            raise ImportError("Transformers not installed.")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LOCAL_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(Config.LOCAL_MODEL_NAME)

    def generate(self, prompt: str, max_tokens: int = 400) -> str:
        if self.backend == "openai":
            return self._generate_openai(prompt, max_tokens)
        else:
            return self._generate_local(prompt, max_tokens)

    def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ACT-IE, a specialised AI assistant for the Irish agri-food sector. "
                        "Be precise, policy-aware, transparent, and scenario-driven."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
