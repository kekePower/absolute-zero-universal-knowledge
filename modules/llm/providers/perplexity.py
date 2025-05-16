"""
Perplexity Provider Implementation

This module provides an implementation of the LLMProvider interface for Perplexity's API.
"""
from typing import Optional, Dict, Any
import httpx
import json
from .. import LLMProvider, ProviderConfig, ProviderError

class PerplexityProvider(LLMProvider):
    """Provider for Perplexity's API."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key
        self._model = config.model or "sonar-medium-online"
        self._base_url = config.base_url or "https://api.perplexity.ai/chat/completions"
        self._timeout = config.timeout or 30
    
    @property
    def name(self) -> str:
        return "perplexity"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using Perplexity's API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the Perplexity API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": max(0, min(temperature, 1)),  # Clamp to 0-1
            "max_tokens": min(max_tokens, 4096),  # Perplexity's max is 4096
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    self._base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
                raise ProviderError(f"Perplexity API error: {error_msg}") from e
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                raise ProviderError(f"Failed to parse Perplexity API response: {str(e)}") from e
            except Exception as e:
                raise ProviderError(f"Error calling Perplexity API: {str(e)}") from e

# Register the provider with the factory
from .. import LLMFactory
LLMFactory.register_provider("perplexity", PerplexityProvider)
