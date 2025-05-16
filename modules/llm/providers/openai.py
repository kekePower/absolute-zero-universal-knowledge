"""
OpenAI Provider Implementation

This module provides an implementation of the LLMProvider interface for OpenAI's API.
"""
from typing import Optional, Dict, Any
import openai
from openai import AsyncOpenAI
from .. import LLMProvider, ProviderConfig, ProviderError

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI's API."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.openai.com/v1"
        )
    
    @property
    def name(self) -> str:
        return "openai"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the OpenAI API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model or "gpt-4-turbo-preview",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}") from e

# Register the provider with the factory
from .. import LLMFactory
LLMFactory.register_provider("openai", OpenAIProvider)
