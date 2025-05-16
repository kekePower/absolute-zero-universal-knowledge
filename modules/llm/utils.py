"""
LLM Utility Functions

This module provides utility functions for working with LLM providers.
"""
from typing import Dict, Any, Optional, List, Type, Union
import asyncio
from pathlib import Path
import yaml

from . import LLMProvider, LLMFactory, ProviderConfig
from ..config import LLMConfigManager, LLMProviderConfig as ProviderConfigModel

async def create_provider(
    provider_type: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Create an LLM provider instance.
    
    Args:
        provider_type: Type of provider (e.g., 'openai', 'anthropic')
        api_key: API key for the provider
        model: Model name to use
        base_url: Base URL for the API (if different from default)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An instance of the requested LLM provider
    """
    config = ProviderConfig(
        provider_name=provider_type,
        api_key=api_key,
        model=model,
        base_url=base_url,
        **kwargs
    )
    
    provider = LLMFactory.get_provider(provider_type, **{
        k: v for k, v in {
            'api_key': api_key,
            'model': model,
            'base_url': base_url,
            **kwargs
        }.items() if v is not None
    })
    
    return provider

async def create_provider_from_config(
    config_name: str,
    config_path: Optional[Union[str, Path]] = None,
    **overrides
) -> LLMProvider:
    """
    Create an LLM provider from a configuration file.
    
    Args:
        config_name: Name of the provider configuration to use
        config_path: Path to the configuration file
        **overrides: Values to override in the configuration
        
    Returns:
        An instance of the configured LLM provider
    """
    # Load the configuration
    config_manager = LLMConfigManager(str(config_path) if config_path else None)
    provider_config = config_manager.get_provider_config(config_name)
    
    if not provider_config:
        raise ValueError(f"No provider configuration found for '{config_name}'")
    
    # Apply overrides
    config_dict = {
        'api_key': provider_config.api_key,
        'model': provider_config.model,
        'base_url': provider_config.base_url,
        'timeout': provider_config.timeout,
        'temperature': provider_config.temperature,
        'max_tokens': provider_config.max_tokens,
        **provider_config.extra_params,
        **overrides
    }
    
    # Create and return the provider
    return await create_provider(
        provider_type=provider_config.provider_type,
        **config_dict
    )

class RateLimitedProvider(LLMProvider):
    """A wrapper that adds rate limiting to an LLM provider."""
    
    def __init__(self, provider: LLMProvider, rate_limit: RateLimitConfig):
        """Initialize with a provider and rate limit configuration."""
        self.provider = provider
        self.rate_limit = rate_limit
        self.limiter = RateLimiter(rate_limit)
        
        # Copy provider attributes
        self.__dict__.update(provider.__dict__)
    
    @property
    def name(self) -> str:
        """Get the provider name."""
        return f"rate_limited_{self.provider.name}"
    
    @property
    def config(self) -> ProviderConfig:
        """Get the provider configuration."""
        return self.provider.config
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text with rate limiting.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system message
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Generated text from the model
        """
        async def _generate():
            return await self.provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        return await self.limiter(_generate)

async def generate_with_provider(
    provider: Union[LLMProvider, str, Dict[str, Any]],
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    **kwargs
) -> GenerationResult:
    """
    Generate text using a provider with enhanced error handling and metadata.
    
    Args:
        provider: Provider instance, name, or config dict
        prompt: The input prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional generation parameters
        
    Returns:
        GenerationResult with the generated text and metadata
        
    Raises:
        ProviderError: If there's an error during generation
    """
    start_time = time.monotonic()
    
    try:
        if isinstance(provider, str):
            provider = await create_provider(provider)
        elif isinstance(provider, dict):
            provider = await create_provider_from_config(
                provider['config_name'],
                config_path=provider.get('config_path'),
                **provider.get('overrides', {})
            )
        
        if not isinstance(provider, LLMProvider):
            raise ValueError(f"Invalid provider type: {type(provider).__name__}")
        
        # Generate the response
        text = await provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Create result with metadata
        result = GenerationResult(
            text=text,
            model=getattr(provider, 'model', None) or getattr(provider.config, 'model', 'unknown'),
            provider=provider.name,
            metadata={
                'temperature': temperature,
                'max_tokens': max_tokens,
                'system_prompt': system_prompt,
                'generation_time_sec': time.monotonic() - start_time,
                **kwargs
            }
        )
        
        # Add usage information if available
        if hasattr(provider, 'last_usage'):
            result.usage = getattr(provider, 'last_usage')
        
        return result
        
    except Exception as e:
        error_msg = f"Error generating text with {getattr(provider, 'name', 'unknown')}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ProviderError(error_msg) from e

async def batch_generate(
    provider: Union[LLMProvider, str, Dict[str, Any]],
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    max_concurrent: int = 5,
    **kwargs
) -> List[GenerationResult]:
    """
    Generate text for multiple prompts in parallel with rate limiting.
    
    Args:
        provider: Provider instance, name, or config dict
        prompts: List of input prompts
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        max_concurrent: Maximum number of concurrent requests
        **kwargs: Additional generation parameters
        
    Returns:
        List of GenerationResult objects with generated texts and metadata
    """
    if isinstance(provider, str):
        provider = await create_provider(provider)
    elif isinstance(provider, dict):
        provider = await create_provider_from_config(
            provider['config_name'],
            config_path=provider.get('config_path'),
            **provider.get('overrides', {})
        )
    
    if not isinstance(provider, LLMProvider):
        raise ValueError(f"Invalid provider type: {type(provider).__name__}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_one(prompt: str) -> GenerationResult:
        async with semaphore:
            return await generate_with_provider(
                provider=provider,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
    
    # Run all generations concurrently
    tasks = [generate_one(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks, return_exceptions=False)

async def get_default_provider(
    provider_type: str = "openai",
    config_path: Optional[Union[str, Path]] = None,
    **overrides
) -> LLMProvider:
    """
    Get a default provider instance.
    
    Args:
        provider_type: Type of provider (e.g., 'openai', 'anthropic')
        config_path: Path to the configuration file
        **overrides: Values to override in the configuration
        
    Returns:
        An instance of the requested LLM provider
    """
    config_manager = LLMConfigManager(str(config_path) if config_path else None)
    
    # Try to find a matching provider config
    for name, config in config_manager.get_all_provider_configs().items():
        if config.provider_type == provider_type:
            return await create_provider_from_config(
                name,
                config_path=config_path,
                **overrides
            )
    
    # If no matching config found, create a basic provider
    return await create_provider(provider_type, **overrides)
