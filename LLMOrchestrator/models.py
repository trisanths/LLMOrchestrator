from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMModel:
    """
    Abstract base class for LLM models.
    Subclasses must implement generate(prompt: str) -> str.
    """
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

class OpenAILLMModel(LLMModel):
    """
    OpenAI API-based LLM.
    
    Parameters:
      model_name: Name of the OpenAI model (e.g., "text-davinci-003").
      api_key: Your OpenAI API key.
      kwargs: Additional parameters for the API call.
    """
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.kwargs = kwargs

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()

class ProprietaryLLMModel(LLMModel):
    """
    Placeholder for proprietary LLMs.
    
    Parameters:
      name: Identifier for the proprietary model.
    """
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str) -> str:
        # Replace this stub with actual integration.
        return f"[{self.name}]: {prompt} -> simulated proprietary response."

class LocalLLMModel(LLMModel):
    """
    Local model using Hugging Face Transformers (e.g., GPT-2, LLaMA).
    
    Parameters:
      model_name: Hugging Face model identifier.
      device: "cpu" or "cuda".
    """
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=150,
            pad_token_id=self.tokenizer.pad_token_id,
            attention_mask=inputs["attention_mask"]
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
