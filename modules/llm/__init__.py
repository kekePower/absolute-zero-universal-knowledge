"""
LLM Provider Interface and Core Functionality

This module provides a unified interface for interacting with various LLM providers.
"""
from typing import Optional, Dict, Any, Type, List, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field

# Core components
from .providers import (
    LLMProvider,
    ProviderConfig,
    ProviderError,
    LLMFactory,
    default_factory,
    get_available_providers,
    create_provider as _create_provider
)

# Import all providers to ensure they're registered
from .providers import openai, anthropic, gemini, groq, perplexity, ollama

# Utility functions and classes
from .rate_limit import RateLimitConfig, RateLimiter, rate_limited
from .providers.utils import (
    snake_to_camel,
    camel_to_snake,
    filter_none,
    to_json_serializable,
    merge_dicts,
    get_required_params,
    validate_params,
    ProviderConfigMixin,
    APIResponse,
    APIError
)

# Re-export public API
__all__ = [
    # Core components
    'LLMProvider',
    'ProviderConfig',
    'ProviderError',
    'LLMFactory',
    'default_factory',
    'get_available_providers',
    
    # Provider implementations
    'openai',
    'anthropic',
    'gemini',
    'groq',
    'perplexity',
    'ollama',
    
    # Rate limiting
    'RateLimitConfig',
    'RateLimiter',
    'rate_limited',
    
    # Utilities
    'snake_to_camel',
    'camel_to_snake',
    'filter_none',
    'to_json_serializable',
    'merge_dicts',
    'get_required_params',
    'validate_params',
    'ProviderConfigMixin',
    'APIResponse',
    'APIError',
]

# Alias for backward compatibility
create_provider = _create_provider
