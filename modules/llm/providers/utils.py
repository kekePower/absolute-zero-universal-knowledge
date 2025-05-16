"""
Provider Utilities

This module contains utility functions for working with LLM providers.
"""
from typing import Dict, Any, Optional, Type, TypeVar, Generic
import json
import re
from dataclasses import asdict

T = TypeVar('T')

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def filter_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary."""
    return {k: v for k, v in data.items() if v is not None}

def to_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return {k: to_json_serializable(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, 'isoformat'):  # Handle datetime objects
        return obj.isoformat()
    return obj

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries into one, with later dictionaries taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def get_required_params(params: Dict[str, Any], required: list) -> Dict[str, Any]:
    """Extract required parameters from a dictionary."""
    return {k: v for k, v in params.items() if k in required}

def validate_params(params: Dict[str, Any], required: list, optional: list = None) -> None:
    """Validate that all required parameters are present and no unknown parameters are provided."""
    optional = optional or []
    all_params = set(required + optional)
    
    # Check for missing required parameters
    missing = [p for p in required if p not in params or params[p] is None]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")
    
    # Check for unknown parameters
    unknown = [p for p in params if p not in all_params]
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(unknown)}")

class ProviderConfigMixin:
    """Mixin class for provider configuration."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderConfigMixin':
        """Create an instance from a dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary."""
        return asdict(self)
    
    def update(self, **kwargs) -> None:
        """Update the configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

class APIResponse(ProviderConfigMixin):
    """Base class for API responses."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        """Return a string representation of the response."""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def is_success(self) -> bool:
        """Check if the response indicates success."""
        return hasattr(self, 'success') and self.success

class APIError(Exception, ProviderConfigMixin):
    """Base class for API errors."""
    
    def __init__(self, message: str, code: Optional[int] = None, details: Any = None):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(message)
    
    def __str__(self) -> str:
        """Return a string representation of the error."""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary."""
        return {
            'message': self.message,
            'code': self.code,
            'details': self.details
        }
