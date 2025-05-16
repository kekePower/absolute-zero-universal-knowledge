"""
Configuration Loader for LLM Providers

This module provides functionality to load and manage LLM provider configurations.
"""
import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import re
from dataclasses import dataclass, field
from ..llm.rate_limit import RateLimitConfig

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    provider_type: str
    api_key: Optional[str] = None
    default_model: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    rate_limit: Optional[RateLimitConfig] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'ProviderConfig':
        """Create a ProviderConfig from a dictionary."""
        # Resolve environment variables in string values
        def resolve_env_vars(value: Any) -> Any:
            if isinstance(value, str):
                # Handle ${VAR} or $VAR syntax
                def replace_var(match):
                    var_name = match.group(1) or match.group(2)
                    return os.getenv(var_name, '')
                return re.sub(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)', replace_var, value)
            elif isinstance(value, dict):
                return {k: resolve_env_vars(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_env_vars(v) for v in value]
            return value
        
        resolved_data = resolve_env_vars(data)
        
        # Extract rate limit config if present
        rate_limit_data = resolved_data.pop('rate_limit', {})
        rate_limit = RateLimitConfig(**rate_limit_data) if rate_limit_data else None
        
        return cls(
            name=name,
            provider_type=resolved_data.get('type', ''),
            api_key=resolved_data.get('api_key'),
            default_model=resolved_data.get('default_model'),
            base_url=resolved_data.get('base_url'),
            timeout=resolved_data.get('timeout', 60),
            rate_limit=rate_limit,
            extra_params=resolved_data.get('extra_params', {})
        )

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'ModelConfig':
        """Create a ModelConfig from a dictionary."""
        # Extract known fields, put the rest in extra_params
        known_fields = {
            'temperature', 'max_tokens', 'top_p', 
            'frequency_penalty', 'presence_penalty', 'stop'
        }
        
        extra_params = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            name=name,
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 2000),
            top_p=data.get('top_p'),
            frequency_penalty=data.get('frequency_penalty'),
            presence_penalty=data.get('presence_penalty'),
            stop=data.get('stop'),
            extra_params=extra_params
        )

class LLMConfig:
    """Main configuration class for LLM providers and models."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration.
        
        Args:
            config_path: Path to the configuration file. If None, looks for 'llm_config.yaml' 
                       in the config directory.
        """
        self.config_path = self._resolve_config_path(config_path)
        self.raw_config: Dict[str, Any] = {}
        self.providers: Dict[str, ProviderConfig] = {}
        self.model_settings: Dict[str, ModelConfig] = {}
        self.system_prompts: Dict[str, str] = {}
        self.default_provider: Optional[str] = None
        self.default_models: Dict[str, str] = {}
        
        self.load()
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the path to the configuration file."""
        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise ConfigError(f"Configuration file not found: {path}")
            return path
        
        # Look for config in default locations
        default_paths = [
            Path('llm_config.yaml'),
            Path('config/llm_config.yaml'),
            Path('~/.config/azr-ukg/llm_config.yaml').expanduser(),
        ]
        
        for path in default_paths:
            if path.exists():
                return path
        
        # If no config file is found, use an empty config
        return Path('config/llm_config.yaml')
    
    def load(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load the configuration from a file."""
        if config_path is not None:
            self.config_path = Path(config_path)
        
        try:
            with open(self.config_path, 'r') as f:
                self.raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to load configuration: {e}")
        
        self._parse_config()
    
    def _parse_config(self) -> None:
        """Parse the loaded configuration."""
        # Parse providers
        self.providers = {}
        for name, data in self.raw_config.get('providers', {}).items():
            try:
                self.providers[name] = ProviderConfig.from_dict(name, data)
            except Exception as e:
                raise ConfigError(f"Error parsing provider '{name}': {e}")
        
        # Parse default models
        self.default_models = self.raw_config.get('default_models', {})
        
        # Parse model settings
        self.model_settings = {}
        for name, data in self.raw_config.get('model_settings', {}).items():
            try:
                self.model_settings[name] = ModelConfig.from_dict(name, data)
            except Exception as e:
                raise ConfigError(f"Error parsing model settings for '{name}': {e}")
        
        # Parse system prompts
        self.system_prompts = self.raw_config.get('system_prompts', {})
        
        # Set default provider
        self.default_provider = self.raw_config.get('default_provider')
        if self.default_provider and self.default_provider not in self.providers:
            raise ConfigError(
                f"Default provider '{self.default_provider}' not found in providers"
            )
    
    def get_provider_config(self, name: Optional[str] = None) -> ProviderConfig:
        """Get a provider configuration by name."""
        if not name:
            if not self.default_provider:
                raise ConfigError("No default provider set and no provider specified")
            name = self.default_provider
        
        if name not in self.providers:
            raise ConfigError(f"Provider '{name}' not found in configuration")
        
        return self.providers[name]
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get the configuration for a specific model."""
        if model_name in self.model_settings:
            return self.model_settings[model_name]
        
        # Return default model config if not found
        return ModelConfig(name=model_name)
    
    def get_system_prompt(self, role: str = 'default') -> str:
        """Get a system prompt by role."""
        return self.system_prompts.get(role, self.system_prompts.get('default', ''))
    
    def get_default_model(self, task_type: str) -> Optional[str]:
        """Get the default model for a task type."""
        return self.default_models.get(task_type)
    
    def get_provider_for_model(self, model_name: str) -> Optional[ProviderConfig]:
        """Get the provider that provides a specific model."""
        for provider in self.providers.values():
            if provider.default_model == model_name:
                return provider
        return None

# Global configuration instance
_config_instance: Optional[LLMConfig] = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> LLMConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = LLMConfig(config_path)
    return _config_instance

def load_config(config_path: Union[str, Path]) -> LLMConfig:
    """Load a new configuration from a file."""
    global _config_instance
    _config_instance = LLMConfig(config_path)
    return _config_instance
