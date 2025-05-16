"""
Configuration Management

This module handles configuration loading and management for the application.
"""

from .loader import (
    ConfigError,
    ProviderConfig,
    ModelConfig,
    LLMConfig,
    get_config,
    load_config
)

# Re-export public API
__all__ = [
    'ConfigError',
    'ProviderConfig',
    'ModelConfig',
    'LLMConfig',
    'get_config',
    'load_config',
]

# Initialize default config
config = get_config()
