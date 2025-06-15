import llama_bindings

class AssistantPlugin:
    def __init__(self):
        self.model = llama_bindings.load_model("models/your-model.gguf")

    def chat(self, prompt: str) -> str:
        return llama_bindings.chat(self.model, prompt)
