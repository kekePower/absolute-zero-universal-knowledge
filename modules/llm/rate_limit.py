"""
Rate Limiting Utilities for LLM Providers

This module provides rate limiting functionality for LLM API calls.
"""
import asyncio
import time
from typing import Dict, Optional, Callable, Any, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field

T = TypeVar('T')

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60  # Default: 1 request per second
    max_concurrent: int = 5  # Maximum concurrent requests
    retry_delay: float = 5.0  # Seconds to wait before retrying after rate limit
    timeout: float = 30.0  # Default timeout in seconds

@dataclass
class RateLimiterState:
    """Internal state for rate limiting."""
    last_request_time: float = 0.0
    request_count: int = 0
    window_start: float = 0.0
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

class RateLimiter:
    """A rate limiter for LLM API calls."""
    
    _instances: Dict[str, 'RateLimiter'] = {}
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.state = RateLimiterState()
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_limiter(
        cls, 
        provider_name: str, 
        config: Optional[RateLimitConfig] = None
    ) -> 'RateLimiter':
        """Get or create a rate limiter for a provider."""
        if provider_name not in cls._instances:
            cls._instances[provider_name] = cls(config)
        return cls._instances[provider_name]
    
    async def _wait_for_capacity(self):
        """Wait until we have capacity for another request."""
        current_time = time.monotonic()
        
        async with self._lock:
            # Reset the window if needed
            if current_time - self.state.window_start >= 60.0:  # 1 minute window
                self.state.window_start = current_time
                self.state.request_count = 0
            
            # Calculate minimum time to wait
            min_interval = 60.0 / self.config.requests_per_minute
            time_since_last = current_time - self.state.last_request_time
            
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
                current_time = time.monotonic()
            
            # Update state
            self.state.last_request_time = current_time
            self.state.request_count += 1
    
    async def __call__(
        self, 
        coro: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """
        Execute a coroutine with rate limiting.
        
        Args:
            coro: The coroutine to execute
            *args: Positional arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            
        Returns:
            The result of the coroutine
            
        Raises:
            asyncio.TimeoutError: If the operation times out
            Exception: If the coroutine raises an exception
        """
        async with self.state.semaphore:
            await self._wait_for_capacity()
            try:
                return await asyncio.wait_for(coro(*args, **kwargs), timeout=self.config.timeout)
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("Operation timed out")
            except Exception as e:
                # Add rate limit information to the error if available
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if e.response.status_code == 429:  # Too Many Requests
                        retry_after = float(e.response.headers.get('Retry-After', self.config.retry_delay))
                        await asyncio.sleep(retry_after)
                        return await self(coro, *args, **kwargs)
                raise

def rate_limited(
    requests_per_minute: int = 60,
    max_concurrent: int = 5,
    retry_delay: float = 5.0,
    timeout: float = 30.0
):
    """
    Decorator to apply rate limiting to an async function.
    
    Args:
        requests_per_minute: Maximum requests per minute
        max_concurrent: Maximum concurrent requests
        retry_delay: Seconds to wait before retrying after rate limit
        timeout: Operation timeout in seconds
        
    Returns:
        A decorator that applies rate limiting to the wrapped function
    """
    def decorator(func):
        limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=requests_per_minute,
            max_concurrent=max_concurrent,
            retry_delay=retry_delay,
            timeout=timeout
        ))
        
        async def wrapper(*args, **kwargs):
            return await limiter(func, *args, **kwargs)
            
        return wrapper
    return decorator
