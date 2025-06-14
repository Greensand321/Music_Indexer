import os
from llama_cpp import Llama

class AssistantPlugin:
    """Simple chat assistant wrapper around a local Llama model."""

    def __init__(self, model_path="models/llama3_8b.gguf", n_threads=6, n_ctx=2048):
        """Load the Llama model from ``model_path`` or download it via Hugging Face."""
        if os.path.isfile(model_path):
            self.llm = Llama(model_path=model_path, n_threads=n_threads, n_ctx=n_ctx)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                filename="*.gguf",
                model_path=model_path,
                n_threads=n_threads,
            )

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        """Return the assistant's reply for ``prompt``."""
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>"]
        )
        return result["choices"][0]["text"].strip()
