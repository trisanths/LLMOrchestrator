import json
import random
from LLMOrchestrator.models import OpenAIModel, LocalModel
from LLMOrchestrator.generator import Generator
from LLMOrchestrator.verifier import Verifier

class OrchestrationManager:
    """
    Manages complex, multi-LLM orchestration strategies defined via a JSON config file.
    
    The configuration file specifies:
      - iterations: total number of iterations.
      - pattern: "json_defined", "random", or "controller_based".
      - steps: a list of steps (each step defines role, model, custom prompt, etc.).
      - selection_strategy: determines the order (deterministic, random, etc.).
    
    Models are looked up in a registry and instantiated accordingly.
    """
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.iterations = self.config.get("iterations", 1)
        self.steps = self.config.get("steps", [])
        self.strategy = self.config.get("selection_strategy", "deterministic")
        self.model_registry = {
            "OpenAI": lambda api_key: OpenAIModel(model_name="gpt-3.5-turbo", api_key=api_key),
            "Local": lambda _: LocalModel(model_name="facebook/opt-125m", device="cpu")
        }
    
    def get_generator(self, model_name: str, prompt: str = None, api_key: str = None):
        model_factory = self.model_registry.get(model_name)
        if model_factory:
            model = model_factory(api_key)
            return Generator(model)
        raise ValueError(f"Unknown model: {model_name}")
    
    def get_verifier(self, model_name: str, min_word_count: int = 5):
        # Custom verifier checking for minimum word count.
        def custom_verifier(text: str):
            if len(text.split()) >= min_word_count:
                return True, text
            return False, f"Output word count below {min_word_count}"
        return Verifier(custom_verifier)
    
    def execute(self, initial_prompt: str) -> str:
        current_output = initial_prompt
        for i in range(self.iterations):
            print(f"Iteration {i+1}:")
            for step in self.steps:
                role = step.get("role")
                model_name = step.get("model")
                api_key = step.get("api_key")
                if role == "generator":
                    prompt = step.get("prompt", current_output)
                    gen = self.get_generator(model_name, prompt, api_key)
                    current_output = gen.generate_output(prompt)
                    print(f"  Generator ({model_name}) produced: {current_output}")
                elif role == "verifier":
                    min_words = step.get("min_word_count", 5)
                    verifier = self.get_verifier(model_name, min_words)
                    valid, message = verifier.verify(current_output)
                    if not valid:
                        print(f"  Verifier ({model_name}) failed: {message}")
                        current_output = f"Retry: {current_output}"
                    else:
                        print(f"  Verifier ({model_name}) approved output.")
            print("Final output after iteration:", current_output)
        return current_output
