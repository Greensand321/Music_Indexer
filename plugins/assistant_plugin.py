import glob, os
from llama_cpp import Llama

class AssistantPlugin:
    def __init__(self):
        """
        Load the first GGUF model found in ./models/, or raise an error.
        """
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1) Discover any local .gguf file
        gguf_paths = glob.glob(os.path.join(models_dir, "*.gguf"))
        if gguf_paths:
            # pick the newest by modification time
            gguf_paths.sort(key=os.path.getmtime, reverse=True)
            model_path = gguf_paths[0]
            try:
                self.llm = Llama(model_path=model_path, n_threads=6, n_ctx=2048)
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load model '{model_path}': {e}") from e

        # 2) No local model found
        raise RuntimeError(
            "No GGUF model found in './models/'.\n"
            "Please download or copy your .gguf file into the 'models' folder and restart."
        )

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        """Return the assistant's reply for ``prompt``."""
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>"]
        )
        return result["choices"][0]["text"].strip()
