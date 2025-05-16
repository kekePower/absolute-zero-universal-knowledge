"""
LLM Provider Implementations

This package contains implementations of various LLM providers.
"""
from typing import Dict, Type, Optional, Any, List, Union
import importlib
import pkgutil
import os

from .base import (
    LLMProvider,
    ProviderConfig,
    ProviderError,
    LLMFactory,
    factory as default_factory
)
from . import utils

# Import all provider modules to ensure they're registered
from . import openai
from . import anthropic
from . import gemini
from . import groq
from . import perplexity
from . import ollama

# Re-export public API
__all__ = [
    # Base classes
    'LLMProvider',
    'ProviderConfig',
    'ProviderError',
    'LLMFactory',
    'default_factory',
    'get_available_providers',
    'create_provider',
    'create_provider_from_config',
    'get_provider_config',
    
    # Provider implementations
    'openai',
    'anthropic',
    'gemini',
    'groq',
    'perplexity',
    'ollama',
    
    # Utilities
    'utils',
]

def get_available_providers() -> List[str]:
    """Get a list of available provider names."""
    return list(default_factory.list_providers().keys())

def create_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 60,
    **kwargs
) -> LLMProvider:
    """
    Create a provider instance.
    
    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        api_key: API key for the provider
        model: Model name to use
        base_url: Base URL for the provider's API
        timeout: Request timeout in seconds
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An instance of the requested provider
        
    Raises:
        ValueError: If the provider is not found or configuration is invalid
    """
    return default_factory.get_provider(
        provider_name=provider_name,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        **kwargs
    )

def create_provider_from_config(
    config_name: str,
    config_path: Optional[Union[str, os.PathLike]] = None,
    **overrides
) -> LLMProvider:
    """
    Create a provider instance from a configuration file.
    
    Args:
        config_name: Name of the provider configuration to use
        config_path: Path to the configuration file. If None, uses default locations.
        **overrides: Configuration overrides
        
    Returns:
        An instance of the configured provider
        
    Raises:
        ValueError: If the configuration is invalid or provider cannot be created
    """
    from ...config.loader import get_config, load_config
    
    # Load configuration if path is provided, otherwise use global config
    if config_path:
        config = load_config(config_path)
    else:
        config = get_config()
    
    # Get provider configuration
    provider_config = config.get_provider_config(config_name)
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(provider_config, key):
            setattr(provider_config, key, value)
    
    # Create provider instance
    return create_provider(
        provider_name=provider_config.provider_type,
        api_key=provider_config.api_key,
        model=provider_config.default_model,
        base_url=provider_config.base_url,
        timeout=provider_config.timeout,
        **provider_config.extra_params
    )

def get_provider_config(provider_name: str) -> ProviderConfig:
    """
    Get the configuration for a provider.
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        The provider configuration
        
    Raises:
        ValueError: If the provider is not found
    """
    from ...config.loader import get_config
    return get_config().get_provider_config(provider_name)

# Import all modules in this package to ensure they're registered
for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    if name != 'base' and not name.startswith('_'):
        try:
            importlib.import_module(f'.{name}', __name__)
        except ImportError as e:
            import warnings
            warnings.warn(f"Failed to import provider {name}: {e}")

# Initialize default providers for easy access
try:
    openai_provider = create_provider('openai')
    anthropic_provider = create_provider('anthropic')
    gemini_provider = create_provider('gemini')
    groq_provider = create_provider('groq')
    perplexity_provider = create_provider('perplexity')
    ollama_provider = create_provider('ollama')
except (ValueError, ImportError) as e:
    # Silently handle missing providers
    pass
