import glob, os
from ctransformers import LLM

class AssistantPlugin:
    def __init__(self):
        """
        Load the first GGUF model found in ./models/
        using ctransformers (bundled DLLs/no build).
        """
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1) Find any .gguf in models/
        ggufs = glob.glob(os.path.join(models_dir, "*.gguf"))
        if not ggufs:
            raise RuntimeError(
                "No GGUF model found in './models/'. "
                "Please drop your .gguf file into the models folder."
            )
        ggufs.sort(key=os.path.getmtime, reverse=True)
        model_path = ggufs[0]

        # 2) Initialize ctransformers LLM
        # Pass the path as the first argument instead of using the 'model' keyword
        self.llm = LLM(model_path, model_type="llama", n_ctx=2048, n_threads=6)

    def chat(self, prompt: str) -> str:
        """Send a prompt to the local model and return its response."""
        return self.llm(prompt, max_tokens=256)
