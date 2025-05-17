"""
Ollama Provider Implementation

This module provides an implementation of the LLMProvider interface for local Ollama models.
"""
from typing import Optional, Dict, Any
import httpx
import json
from urllib.parse import urljoin
from .base import LLMProvider, ProviderConfig, ProviderError
from ..factory import get_factory

class OllamaProvider(LLMProvider):
    """Provider for local Ollama models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._model = config.model or "llama2"
        self._base_url = config.base_url or "http://localhost:11434"
        self._timeout = config.timeout or 300  # Longer timeout for local models
    
    @property
    def name(self) -> str:
        return "ollama"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using a local Ollama model.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the Ollama API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API
        """
        url = urljoin(self._base_url, "/api/generate")
        
        # Prepare messages array for chat models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self._model,
            "prompt": prompt,
            "system": system_prompt or "",
            "options": {
                "temperature": max(0, min(temperature, 1)),  # Clamp to 0-1
                "num_predict": min(max_tokens, 4096),  # Reasonable max for local models
            },
            "stream": False,
            **kwargs
        }
        
        # Remove None values to avoid API errors
        payload = {k: v for k, v in payload.items() if v is not None}
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
                raise ProviderError(f"Ollama API error: {error_msg}") from e
            except (json.JSONDecodeError, KeyError) as e:
                raise ProviderError(f"Failed to parse Ollama API response: {str(e)}") from e
            except Exception as e:
                raise ProviderError(f"Error calling Ollama API: {str(e)}") from e

def register():
    """Register this provider with the factory."""
    from ..factory import get_factory
    get_factory().register_provider("ollama", OllamaProvider)

# Register this provider when the module is imported
register()
