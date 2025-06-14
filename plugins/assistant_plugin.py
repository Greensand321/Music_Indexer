import os
from llama_cpp import Llama

class AssistantPlugin:
    def __init__(self):
        """
        Always attempt to load the model from Hugging Face (or local cache).
        If it's already in models/, from_pretrained(local_files_only=True)
        will pick it up without hitting the internet.
        """
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        try:
            # this will auto-download on first run, then load from models_dir
            self.llm = Llama.from_pretrained(
                repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                filename="*.gguf",
                local_dir=models_dir,
                cache_dir=models_dir,
                local_files_only=False,
                n_threads=6,
                n_ctx=2048
            )
        except Exception as e:
            # rethrow so the GUI can catch it
            raise RuntimeError(f"Could not load LLM model: {e}") from e

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        """Return the assistant's reply for ``prompt``."""
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>"]
        )
        return result["choices"][0]["text"].strip()
