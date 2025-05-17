"""
Factory module for managing LLM provider instances.

This module provides a centralized way to manage and access the LLM factory instance.
"""
from .providers.base import LLMFactory

# Global factory instance
_factory = LLMFactory()

def get_factory() -> LLMFactory:
    """Get the global LLM factory instance.
    
    Returns:
        The global LLMFactory instance
    """
    return _factory

def set_factory(factory: LLMFactory) -> None:
    """Set the global LLM factory instance.
    
    Args:
        factory: The LLMFactory instance to use
    """
    global _factory
    _factory = factory
