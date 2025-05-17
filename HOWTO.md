# Customizing LLM Providers and System Prompts

This guide explains how to customize LLM providers and system prompts in the Universal Knowledge Generator.

## Table of Contents
- [Provider Configuration](#provider-configuration)
  - [Available Providers](#available-providers)
  - [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [System Prompts](#system-prompts)
  - [Default Prompts](#default-prompts)
  - [Customizing Prompts](#customizing-prompts)
- [Advanced Configuration](#advanced-configuration)
  - [Custom Providers](#custom-providers)
  - [Rate Limiting](#rate-limiting)
  - [Model-Specific Settings](#model-specific-settings)
- [Examples](#examples)
  - [Basic Configuration](#basic-configuration)
  - [Multiple Providers](#multiple-providers)
  - [Custom System Prompt](#custom-system-prompt)

## Provider Configuration

### Available Providers
The system supports the following LLM providers out of the box:

| Provider    | Provider Name | Required Environment Variable | Notes |
|-------------|---------------|------------------------------|-------|
| OpenAI      | `openai`      | `OPENAI_API_KEY`             | Used for question generation |
| Anthropic   | `anthropic`   | `ANTHROPIC_API_KEY`          | - |
| Google Gemini | `gemini`    | `GOOGLE_API_KEY`             | - |
| Groq        | `groq`        | `GROQ_API_KEY`               | - |
| Perplexity  | `perplexity`  | `PERPLEXITY_API_KEY`         | - |
| Ollama      | `ollama`      | -                            | Local model server |

### Configuration

The main configuration is defined in Python in `modules/config/__init__.py`. This file contains:
- API endpoints and keys
- Model configurations
- System prompts
- Rate limiting settings
- Task type distributions

Key configuration values can be overridden using environment variables. The most important ones are:

```python
# Primary LLM Configuration (Novita)
PRIMARY_API_BASE_URL = os.getenv("PRIMARY_API_BASE_URL", "https://api.novita.ai/v3/openai")
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY", "<Your_API_Key_HERE>")
PRIMARY_MODEL_NAME = os.getenv("PRIMARY_MODEL_NAME", "qwen/qwen3-235b-a22b-fp8")

# Secondary LLM Configuration (Evaluator)
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "deepseek/deepseek-v3-0324")

# OpenAI Configuration (for Question Generation)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<Your_OpenAI_API_Key_HERE>")
OPENAI_QUESTION_MODEL = os.getenv("OPENAI_QUESTION_MODEL", "gpt-4.1-mini")
```

### Environment Variables

The following environment variables should be set before running the application:

```bash
# Required API keys
export PRIMARY_API_KEY='your-novita-api-key'
export OPENAI_API_KEY='your-openai-api-key'

# Optional overrides
export PRIMARY_MODEL_NAME='qwen/qwen3-235b-a22b-fp8'  # Default Solver
export SECONDARY_MODEL_NAME='deepseek/deepseek-v3-0324'  # Default Evaluator
export OPENAI_QUESTION_MODEL='gpt-4.1-mini'  # Default Proposer

# Rate limiting
export API_RPM_LIMIT=100  # Requests per minute
```

### Environment Variables
Sensitive information like API keys should be set as environment variables:

```bash
export OPENAI_API_KEY='your-api-key-here'
export ANTHROPIC_API_KEY='your-api-key-here'
# etc.
```

## System Prompts

### Default Prompts
The system comes with default prompts for different roles:
- `default`: General-purpose assistant
- `proposer`: For generating creative questions and tasks
- `solver`: For providing detailed answers
- `evaluator`: For evaluating responses

### Customizing Prompts
You can customize prompts in the `system_prompts` section of the config file:

```yaml
system_prompts:
  default: |
    You are a helpful AI assistant. Be concise and accurate in your responses.
  
  solver: |
    You are an expert problem solver. Break down complex problems into steps 
    and explain your reasoning clearly and thoroughly.
  
  evaluator: |
    You are a critical evaluator. Provide detailed feedback on the given response,
    pointing out strengths, weaknesses, and areas for improvement.
```

## Advanced Configuration

### Custom Providers
To add a custom provider:

1. Create a new Python file in `modules/llm/providers/`
2. Implement the `LLMProvider` interface
3. Add a `register()` function to register with the factory

Example:
```python
# modules/llm/providers/custom.py
from .base import LLMProvider, ProviderConfig, ProviderError
from ..factory import get_factory

class CustomProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Your initialization code
    
    @property
    def name(self) -> str:
        return "custom"
        
    async def generate(self, prompt: str, **kwargs) -> str:
        # Your generation logic
        return "Generated text"

def register():
    """Register this provider with the factory."""
    get_factory().register_provider("custom", CustomProvider)

# Register this provider when the module is imported
register()
```

### Rate Limiting

Rate limiting is configured globally in the main configuration:

```python
# API Throttling Configuration
API_RPM_LIMIT = int(os.getenv("API_RPM_LIMIT", "100"))  # Requests per minute
MIN_ITER_SLEEP = 0.2  # Minimum seconds between iterations
```

You can override these values using environment variables:

```bash
export API_RPM_LIMIT=100  # Adjust based on your API plan
export MIN_ITER_SLEEP=0.2  # Minimum seconds between requests
```

### Model-Specific Settings

Model-specific settings are configured in `modules/config/__init__.py`. The default configuration includes:

```python
# Default Models
PRIMARY_MODEL_NAME = "qwen/qwen3-235b-a22b-fp8"  # Solver
SECONDARY_MODEL_NAME = "deepseek/deepseek-v3-0324"  # Evaluator
OPENAI_QUESTION_MODEL = "gpt-4.1-mini"  # Proposer

# Model Parameters
PROPOSER_TEMPERATURE = 0.90  # Higher for more creative proposals
SOLVER_TEMPERATURE = 0.78    # Slightly higher for creative panel roles
CRITIQUE_TEMPERATURE = 0.5
REVISE_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.4

# Token Limits
MAX_TOKENS_PROPOSER = 1000
MAX_TOKENS_SOLVER = 5300
MAX_TOKENS_CRITIQUE_REVISE = 5300
MAX_TOKENS_EVALUATOR = 5300
```

You can override these values using environment variables:

```bash
export PRIMARY_MODEL_NAME="qwen/qwen3-235b-a22b-fp8"
export SECONDARY_MODEL_NAME="deepseek/deepseek-v3-0324"
export SOLVER_TEMPERATURE=0.78
# etc.
```

## Examples

### Basic Configuration
```yaml
# config/llm_config.yaml
default_provider: openai

providers:
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4-turbo-preview
```

### Multiple Providers
```yaml
providers:
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4-turbo-preview
    
  anthropic:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229
    
  groq:
    type: groq
    api_key: ${GROQ_API_KEY}
    default_model: mixtral-8x7b-32768
```

### Custom System Prompt
```yaml
system_prompts:
  solver: |
    You are an expert in multiple domains with deep knowledge of science, 
    technology, and philosophy. When answering questions:
    1. First analyze the question to understand its core
    2. Break down complex concepts into simpler parts
    3. Provide clear, accurate, and well-structured responses
    4. Include relevant examples and analogies
    5. Acknowledge any uncertainties or limitations in your knowledge
    
    Always maintain a helpful and professional tone.
```

## Best Practices

1. **Security**: Never commit API keys to version control. Use environment variables.
2. **Testing**: Test new configurations with simple prompts before production use.
3. **Monitoring**: Monitor API usage and adjust rate limits as needed.
4. **Fallbacks**: Configure fallback providers for critical applications.
5. **Versioning**: Keep your configuration files under version control.

## Troubleshooting

- **API Errors**: Check your API keys and internet connection
- **Rate Limiting**: Adjust `requests_per_minute` if hitting rate limits
- **Timeouts**: Increase `timeout` for slower models or complex queries
- **Model Availability**: Verify the model name and your access level

For additional help, please refer to the documentation or open an issue in the repository.
