"""
Anthropic Provider Implementation

This module provides an implementation of the LLMProvider interface for Anthropic's API.
"""
from typing import Optional, Dict, Any
import anthropic
from anthropic import AsyncAnthropic
from .base import LLMProvider, ProviderConfig, ProviderError
from ..factory import get_factory

class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        # Only initialize the client if an API key is provided
        if config.api_key:
            self._client = AsyncAnthropic(api_key=config.api_key)
        else:
            self._client = None
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using Anthropic's API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the Anthropic API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API or if no API key is configured
        """
        if not self._client:
            raise ProviderError(
                "Anthropic API key is not configured. "
                "Please set the ANTHROPIC_API_KEY environment variable or provide an API key in the config."
            )
            
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._client.messages.create(
                model=self.config.model or "claude-3-opus-20240229",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {str(e)}") from e

def register():
    """Register this provider with the factory."""
    from ..factory import get_factory
    get_factory().register_provider("anthropic", AnthropicProvider)

# Register this provider when the module is imported
register()
