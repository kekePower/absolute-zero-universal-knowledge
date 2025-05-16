"""
Anthropic Provider Implementation

This module provides an implementation of the LLMProvider interface for Anthropic's API.
"""
from typing import Optional, Dict, Any
import anthropic
from anthropic import AsyncAnthropic
from .. import LLMProvider, ProviderConfig, ProviderError

class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = AsyncAnthropic(api_key=config.api_key)
    
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
            ProviderError: If there's an error calling the API
        """
        try:
            message = await self._client.messages.create(
                model=self.config.model or "claude-3-opus-20240229",
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return message.content[0].text
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {str(e)}") from e

# Register the provider with the factory
from .. import LLMFactory
LLMFactory.register_provider("anthropic", AnthropicProvider)
