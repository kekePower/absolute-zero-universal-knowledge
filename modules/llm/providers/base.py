"""
Base Provider Module

This module defines the base classes and interfaces for LLM providers.
"""
from typing import Optional, Dict, Any, Type
from abc import ABC, abstractmethod

class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass

class ProviderConfig:
    """Configuration for an LLM provider."""
    def __init__(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        self.provider_name = provider_name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.extra_params = kwargs

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate the provider configuration."""
        # API key validation is now handled by each provider's implementation
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The user's input prompt
            system_prompt: Optional system message to guide the model's behavior
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text from the model
        """
        pass
    
    async def __call__(self, *args, **kwargs):
        """Make the provider callable."""
        return await self.generate(*args, **kwargs)

class LLMFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, Type[LLMProvider]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a new provider type."""
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create a new provider instance.
        
        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            api_key: Provider API key
            model: Model name to use
            **kwargs: Additional provider-specific arguments
            
        Returns:
            An instance of the requested provider
        """
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
            
        config = ProviderConfig(
            provider_name=provider_name,
            api_key=api_key,
            model=model,
            **kwargs
        )
        
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers."""
        return list(cls._providers.keys())
