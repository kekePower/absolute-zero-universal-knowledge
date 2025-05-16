"""
LLM Provider Registry

This module provides a registry for LLM providers and their configurations.
"""
from typing import Dict, Type, Optional, Any, List, Union
import importlib
import pkgutil
import os
import logging
from pathlib import Path

from .providers import (
    LLMProvider,
    ProviderConfig,
    ProviderError,
    get_available_providers,
    create_provider,
    create_provider_from_config,
)
from .providers.base import LLMFactory

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """Registry for managing LLM providers and their configurations."""
    
    def __init__(self, config_path: Optional[Union[str, os.PathLike]] = None):
        """Initialize the registry.
        
        Args:
            config_path: Path to the configuration file. If None, uses default locations.
        """
        self.config_path = Path(config_path) if config_path else None
        self._providers: Dict[str, LLMProvider] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        
        # Initialize with built-in providers
        self._register_builtin_providers()
    
    def _register_builtin_providers(self) -> None:
        """Register all built-in provider implementations."""
        # Import all provider modules to ensure they're registered
        providers_dir = Path(__file__).parent / 'providers'
        for _, name, _ in pkgutil.iter_modules([str(providers_dir)]):
            if name != 'base' and not name.startswith('_'):
                try:
                    importlib.import_module(f'.{name}', 'modules.llm.providers')
                except ImportError as e:
                    logger.warning(f"Failed to import provider {name}: {e}")
    
    def register_provider(
        self,
        name: str,
        provider_class: Type[LLMProvider],
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new provider type.
        
        Args:
            name: Name of the provider
            provider_class: Provider class (must be a subclass of LLMProvider)
            config: Optional default configuration for the provider
        """
        LLMFactory.register_provider(name, provider_class)
        
        if config:
            self._provider_configs[name] = ProviderConfig(
                provider_type=name,
                **config
            )
    
    def get_provider(
        self,
        name: str,
        use_config: bool = True,
        **overrides
    ) -> LLMProvider:
        """Get a provider instance.
        
        Args:
            name: Name of the provider
            use_config: Whether to use configuration from the registry
            **overrides: Configuration overrides
            
        Returns:
            An instance of the requested provider
            
        Raises:
            ValueError: If the provider is not found or configuration is invalid
        """
        if use_config:
            # Try to get config from registry first
            config = self._provider_configs.get(name)
            if config:
                # Apply overrides
                config_dict = {**config.to_dict(), **overrides}
                return create_provider(**config_dict)
            
            # Fall back to config file
            try:
                return create_provider_from_config(
                    name,
                    config_path=self.config_path,
                    **overrides
                )
            except Exception as e:
                logger.warning(f"Failed to create provider {name} from config: {e}")
        
        # Create with just the overrides
        return create_provider(provider_name=name, **overrides)
    
    def get_default_provider(self) -> LLMProvider:
        """Get the default provider.
        
        Returns:
            The default provider instance
            
        Raises:
            ValueError: If no providers are available
        """
        available = self.list_available_providers()
        if not available:
            raise ValueError("No LLM providers available")
        
        # Try to get the first available provider with valid config
        for name in available:
            try:
                return self.get_provider(name)
            except Exception as e:
                logger.warning(f"Failed to initialize provider {name}: {e}")
        
        # If all else fails, try to create without config
        return create_provider(available[0])
    
    def list_available_providers(self) -> List[str]:
        """List all available provider names.
        
        Returns:
            List of available provider names
        """
        return get_available_providers()
    
    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Get the configuration for a provider.
        
        Args:
            name: Name of the provider
            
        Returns:
            The provider configuration, or None if not found
        """
        if name in self._provider_configs:
            return self._provider_configs[name]
        
        try:
            from ...config.loader import get_config
            return get_config().get_provider_config(name)
        except Exception:
            return None
    
    def load_config(self, config_path: Union[str, os.PathLike]) -> None:
        """Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        # The actual loading happens when providers are requested

# Global registry instance
_registry_instance: Optional[ProviderRegistry] = None

def get_registry(config_path: Optional[Union[str, os.PathLike]] = None) -> ProviderRegistry:
    """Get the global provider registry.
    
    Args:
        config_path: Optional path to the configuration file
        
    Returns:
        The global provider registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry(config_path)
    return _registry_instance

def register_provider(
    name: str,
    provider_class: Type[LLMProvider],
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Register a provider with the global registry.
    
    Args:
        name: Name of the provider
        provider_class: Provider class (must be a subclass of LLMProvider)
        config: Optional default configuration for the provider
    """
    registry = get_registry()
    registry.register_provider(name, provider_class, config)

def get_provider(
    name: str,
    use_config: bool = True,
    **overrides
) -> LLMProvider:
    """Get a provider instance from the global registry.
    
    Args:
        name: Name of the provider
        use_config: Whether to use configuration from the registry
        **overrides: Configuration overrides
        
    Returns:
        An instance of the requested provider
    """
    registry = get_registry()
    return registry.get_provider(name, use_config=use_config, **overrides)

def get_default_provider() -> LLMProvider:
    """Get the default provider from the global registry.
    
    Returns:
        The default provider instance
    """
    registry = get_registry()
    return registry.get_default_provider()

def list_available_providers() -> List[str]:
    """List all available provider names.
    
    Returns:
        List of available provider names
    """
    registry = get_registry()
    return registry.list_available_providers()

# Register built-in providers when module is imported
# This ensures all providers are registered when the registry is first used
_registry = get_registry()
