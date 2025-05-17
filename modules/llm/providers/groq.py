"""
Groq Provider Implementation

This module provides an implementation of the LLMProvider interface for Groq's API.
"""
from typing import Optional, Dict, Any
import groq
from .base import LLMProvider, ProviderConfig, ProviderError
from ..factory import get_factory

class GroqProvider(LLMProvider):
    """Provider for Groq's API."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Only initialize the client if an API key is provided
        if config.api_key:
            self._client = groq.AsyncClient(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.groq.com/openai/v1"
            )
        else:
            self._client = None
    
    @property
    def name(self) -> str:
        return "groq"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using Groq's API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the Groq API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API or if no API key is configured
        """
        if not self._client:
            raise ProviderError(
                "Groq API key is not configured. "
                "Please set the GROQ_API_KEY environment variable or provide an API key in the config."
            )
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model or "mixtral-8x7b-32768",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ProviderError(f"Groq API error: {str(e)}") from e

def register():
    """Register this provider with the factory."""
    from ..factory import get_factory
    get_factory().register_provider("groq", GroqProvider)

# Register this provider when the module is imported
register()
