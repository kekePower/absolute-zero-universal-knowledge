"""
LLM Configuration Management

This module handles loading and managing configurations for different LLM providers.
"""
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from dataclasses import dataclass, field

@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    provider_type: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Handle environment variables in config values
        if self.api_key and self.api_key.startswith("$"):
            self.api_key = os.getenv(self.api_key[1:], "")
        
        # Set default base URLs for known providers if not specified
        if not self.base_url:
            self.base_url = self._get_default_base_url()
    
    def _get_default_base_url(self) -> Optional[str]:
        """Get default base URL based on provider type."""
        defaults = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "groq": "https://api.groq.com/openai/v1",
            "perplexity": "https://api.perplexity.ai/chat/completions",
            "ollama": "http://localhost:11434/api"
        }
        return defaults.get(self.provider_type.lower())

class LLMConfigManager:
    """Manages LLM provider configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.
        
        Args:
            config_path: Path to the YAML config file. If None, looks for 'llm_config.yaml' in the current directory.
        """
        self.config_path = config_path or "llm_config.yaml"
        self.providers: Dict[str, LLMProviderConfig] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from the YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Load providers
            for name, provider_data in config_data.get('providers', {}).items():
                self.providers[name] = LLMProviderConfig(
                    name=name,
                    provider_type=provider_data.get('type', '').lower(),
                    model=provider_data.get('model', ''),
                    api_key=provider_data.get('api_key_env', ''),
                    base_url=provider_data.get('base_url'),
                    timeout=provider_data.get('timeout', 30),
                    temperature=provider_data.get('temperature', 0.7),
                    max_tokens=provider_data.get('max_tokens', 2000),
                    system_prompt=provider_data.get('system_prompt'),
                    extra_params=provider_data.get('params', {})
                )
                
        except FileNotFoundError:
            # Create default config if it doesn't exist
            self._create_default_config()
        except Exception as e:
            raise ValueError(f"Error loading LLM config: {str(e)}")
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "providers": {
                "openai_default": {
                    "type": "openai",
                    "model": "gpt-4-turbo-preview",
                    "api_key_env": "OPENAI_API_KEY",
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "system_prompt": "You are a helpful AI assistant."
                },
                "ollama_default": {
                    "type": "ollama",
                    "model": "llama2",
                    "base_url": "http://localhost:11434",
                    "timeout": 300,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Write default config
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default LLM config at {self.config_path}")
        
        # Load the default config
        self._load_config()
    
    def get_provider_config(self, name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all configured provider names."""
        return list(self.providers.keys())
    
    def get_all_provider_configs(self) -> Dict[str, LLMProviderConfig]:
        """Get all provider configurations."""
        return self.providers.copy()
