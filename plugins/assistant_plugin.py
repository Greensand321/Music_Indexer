import glob, os
from llama_cpp import Llama
from tkinter import messagebox

class AssistantPlugin:
    def __init__(self):
        """
        Auto-discovers any .gguf file in models/ or, if none, falls back
        to downloading a default model. Shows an error if both fail.
        """
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)

        # 1) Look for any .gguf file
        ggufs = glob.glob(os.path.join(models_dir, "*.gguf"))
        if ggufs:
            # Pick the newest file by modified time
            ggufs.sort(key=os.path.getmtime, reverse=True)
            model_path = ggufs[0]
            try:
                self.llm = Llama(model_path=model_path, n_threads=6, n_ctx=2048)
                return
            except Exception as e:
                messagebox.showwarning(
                    "Model Load Warning",
                    f"Failed to load {os.path.basename(model_path)}:\n{e}\n\n"
                    "Attempting automatic download instead."
                )

        # 2) Fallback: auto-download default via Hugging Face Hub
        try:
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
            messagebox.showerror(
                "Assistant Initialization Failed",
                f"Could not find or download any GGUF model:\n{e}"
            )
            raise

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        """Return the assistant's reply for ``prompt``."""
        result = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>"]
        )
        return result["choices"][0]["text"].strip()
