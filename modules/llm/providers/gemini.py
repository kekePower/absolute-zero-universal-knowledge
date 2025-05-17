"""
Google Gemini Provider Implementation

This module provides an implementation of the LLMProvider interface for Google's Gemini API.
"""
from typing import Optional, Dict, Any
import google.generativeai as genai
from .base import LLMProvider, ProviderConfig, ProviderError
from ..factory import get_factory

class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self._model_name = config.model or "gemini-pro"
        self._model = genai.GenerativeModel(self._model_name)
    
    @property
    def name(self) -> str:
        return "gemini"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using Google's Gemini API.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the Gemini API
            
        Returns:
            Generated text from the model
            
        Raises:
            ProviderError: If there's an error calling the API
        """
        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Create generation config
            generation_config = {
                "temperature": min(max(temperature, 0), 1),  # Clamp to 0-1
                "max_output_tokens": min(max_tokens, 8192),  # Gemini's max is 8192
                **kwargs
            }
            
            # Generate content
            response = await self._model.generate_content_async(
                full_prompt,
                generation_config=generation_config
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return str(response)
                
        except Exception as e:
            raise ProviderError(f"Gemini API error: {str(e)}") from e

def register():
    """Register this provider with the factory."""
    from ..factory import get_factory
    get_factory().register_provider("gemini", GeminiProvider)

# Register this provider when the module is imported
register()
