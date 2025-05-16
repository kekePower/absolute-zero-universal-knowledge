# Customizing LLM Providers and System Prompts

This guide explains how to customize LLM providers and system prompts in the Universal Knowledge Generator.

## Table of Contents
- [Provider Configuration](#provider-configuration)
  - [Available Providers](#available-providers)
  - [Configuration File](#configuration-file)
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

| Provider    | Provider Name | Required Environment Variable |
|-------------|---------------|------------------------------|
| OpenAI      | `openai`      | `OPENAI_API_KEY`             |
| Anthropic   | `anthropic`   | `ANTHROPIC_API_KEY`          |
| Google Gemini | `gemini`    | `GOOGLE_API_KEY`             |
| Groq        | `groq`        | `GROQ_API_KEY`               |
| Perplexity  | `perplexity`  | `PERPLEXITY_API_KEY`         |
| Ollama      | `ollama`      | -                            |


### Configuration File
The main configuration is stored in `config/llm_config.yaml`. This file defines:
- Default provider
- Provider-specific settings
- System prompts
- Rate limiting
- Model-specific configurations

Example configuration:
```yaml
# config/llm_config.yaml
default_provider: openai

providers:
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-4-turbo-preview
    base_url: https://api.openai.com/v1
    timeout: 60
    rate_limit:
      requests_per_minute: 3500
      max_concurrent: 10

  anthropic:
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    default_model: claude-3-opus-20240229
    timeout: 60
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
3. Register it in `__init__.py`

Example:
```python
# modules/llm/providers/custom.py
from .base import LLMProvider, ProviderConfig

class CustomProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Your initialization code
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Your generation logic
        return "Generated text"

# Register the provider
from ..registry import register_provider
register_provider("custom", CustomProvider)
```

### Rate Limiting
Configure rate limits per provider:

```yaml
providers:
  openai:
    # ... other settings ...
    rate_limit:
      requests_per_minute: 3500  # 60 RPM for free tier, 3500 for pay-as-you-go
      max_concurrent: 10         # Max concurrent requests
      retry_delay: 5.0           # Seconds to wait before retrying after rate limit
      timeout: 30.0              # Request timeout in seconds
```

### Model-Specific Settings
Configure settings specific to each model:

```yaml
default_models:
  proposer: gpt-4.1-mini
  solver: qwen/qwen3-235b-a22b-fp8
  evaluator: deepseek/deepseek-v3-0324

model_settings:
  gpt-4.1-mini:
    temperature: 0.9
    max_tokens: 2000
    
  "qwen/qwen3-235b-a22b-fp8":
    temperature: 0.7
    max_tokens: 4000
    top_p: 0.95
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
