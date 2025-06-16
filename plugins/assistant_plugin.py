import os
import subprocess

class AssistantPlugin:
    def __init__(self):
        self.exe     = os.path.join("third_party", "llama", "llama-run.exe")
        self.model   = os.path.join("models",    "your-model.gguf")
        self.threads = max(1, os.cpu_count() - 1)

    def chat(self, prompt: str) -> str:
        args = [
            self.exe,
            "--threads",   str(self.threads),
            "--n_predict", "128",
            "--temp",      "0.7",
            "--repeat_penalty", "1.1",
            "--color=false",
            self.model,       # positional model path
            prompt            # positional prompt text
        ]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        out, _ = proc.communicate()

        # Drop any lines matching the prompt or empty lines
        lines = [l for l in out.splitlines()
                 if l.strip() and l.strip() != prompt]
        return "\n".join(lines)
