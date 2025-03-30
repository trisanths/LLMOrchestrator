# API Documentation

## Core Components

### Controller

The main orchestration class that manages LLM interactions.

```python
from LLMOrchestrator.controller import Controller

controller = Controller(
    generator,      # BaseModel instance for generating responses
    verifier,       # BaseModel instance for verifying responses
    max_iterations=3,
    max_verifications=2,
    prompt_template=None,
    monitoring_enabled=True
)
```

#### Methods

- `execute(prompt: str, stop_early: bool = False) -> str`
  - Executes a single prompt
  - Returns the best generated response

- `execute_parallel(prompts: List[str], max_workers: int = 2) -> List[str]`
  - Executes multiple prompts in parallel
  - Returns a list of responses

### Models

Base class and implementations for different LLM providers.

#### OpenAI Model

```python
from LLMOrchestrator.models.openai_model import OpenAIModel

model = OpenAIModel(
    model_name="gpt-3.5-turbo",
    api_key="your-api-key"
)
```

#### Local Model

```python
from LLMOrchestrator.models.local_model import LocalModel

model = LocalModel(
    model_name="facebook/opt-125m",
    device="cpu"
)
```

### Custom Models

Create custom models by inheriting from BaseModel:

```python
from LLMOrchestrator.models import BaseModel

class CustomModel(BaseModel):
    def generate_output(self, prompt: str) -> str:
        # Implement generation logic
        pass

    def verify(self, text: str, prompt: str = None) -> tuple[bool, str]:
        # Implement verification logic
        pass
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `LOG_LEVEL`: Logging level (default: INFO)

### Prompt Templates

```python
from LLMOrchestrator.controller import PromptTemplate

template = PromptTemplate("{prompt}")
```

## Error Handling

The framework provides comprehensive error handling:

```python
from LLMOrchestrator.exceptions import (
    ModelError,
    ValidationError,
    ConfigurationError
)

try:
    result = controller.execute(prompt)
except ModelError as e:
    # Handle model-related errors
    pass
except ValidationError as e:
    # Handle validation errors
    pass
except ConfigurationError as e:
    # Handle configuration errors
    pass
```

## Monitoring

Enable monitoring for detailed logs:

```python
import logging
logging.basicConfig(level=logging.INFO)

controller = Controller(
    generator=model,
    verifier=model,
    monitoring_enabled=True
)
```

## Best Practices

1. Always use environment variables for API keys
2. Enable monitoring in development
3. Implement proper error handling
4. Use appropriate model combinations for tasks
5. Configure timeouts for API calls
6. Cache results when appropriate
7. Leverage chain-of-thought reasoning for complex tasks
8. Use multiple models to enhance output diversity
9. Implement custom verification logic for specific use cases 