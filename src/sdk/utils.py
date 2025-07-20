"""
Utility functions and helpers for the Voice Agent SDK.

This module provides common utilities used throughout the SDK including
validation, type checking, async helpers, and component discovery.
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
from functools import wraps
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


def validate_type(value: Any, expected_type: Type[T], field_name: str) -> T:
    """
    Validate that a value is of the expected type.
    
    Args:
        value: The value to validate
        expected_type: The expected type
        field_name: Name of the field for error messages
        
    Returns:
        The validated value
        
    Raises:
        TypeError: If the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        raise TypeError(
            f"'{field_name}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value


def validate_enum(value: Any, enum_class: Type, field_name: str) -> Any:
    """Validate that a value is a valid enum member."""
    if not isinstance(value, enum_class):
        try:
            return enum_class(value)
        except ValueError:
            valid_values = [e.value for e in enum_class]
            raise ValueError(
                f"'{field_name}' must be one of {valid_values}, got '{value}'"
            )
    return value


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: str = "value"
) -> Union[int, float]:
    """Validate that a numeric value is within a specified range."""
    if min_value is not None and value < min_value:
        raise ValueError(f"'{field_name}' must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"'{field_name}' must be <= {max_value}, got {value}")
    return value


def ensure_async(func: Callable) -> Callable:
    """
    Decorator to ensure a function is async.
    
    If the decorated function is sync, it will be wrapped to run in an executor.
    """
    if asyncio.iscoroutinefunction(func):
        return func
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    
    return async_wrapper


def is_component_class(cls: Type) -> bool:
    """Check if a class is a valid component class."""
    # Check for common component base classes
    base_classes = [
        'BaseSTTProvider', 'BaseLLMProvider', 'BaseTTSProvider', 
        'BaseVADProvider', 'BasePipeline'
    ]
    
    for base in base_classes:
        if any(base in str(parent) for parent in inspect.getmro(cls)):
            return True
    
    return False


def discover_components(package_path: str) -> Dict[str, Type]:
    """
    Discover all component classes in a package.
    
    Args:
        package_path: Path to the package to scan
        
    Returns:
        Dictionary mapping component names to their classes
    """
    components = {}
    
    try:
        # Convert path to module name
        module_name = package_path.replace(os.sep, '.')
        if module_name.endswith('.py'):
            module_name = module_name[:-3]
            
        # Import the module
        module = importlib.import_module(module_name)
        
        # Scan for component classes
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and is_component_class(obj):
                components[name] = obj
                
    except Exception as e:
        logger.warning(f"Failed to discover components in {package_path}: {e}")
    
    return components


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {config_path.suffix}. "
                "Use .yaml, .yml, or .json"
            )


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configs override earlier ones. Nested dictionaries are merged recursively.
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


class AsyncContextManager:
    """Base class for async context managers in the SDK."""
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
    
    async def start(self):
        """Override in subclasses to implement startup logic."""
        pass
    
    async def stop(self):
        """Override in subclasses to implement cleanup logic."""
        pass


class SingletonMeta(type):
    """Metaclass for creating singleton classes."""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_component_info(component_class: Type) -> Dict[str, Any]:
    """Extract information about a component class."""
    info = {
        "name": component_class.__name__,
        "module": component_class.__module__,
        "docstring": inspect.getdoc(component_class),
        "methods": {},
        "config_class": None
    }
    
    # Find public methods
    for name, method in inspect.getmembers(component_class, inspect.isfunction):
        if not name.startswith('_'):
            info["methods"][name] = {
                "signature": str(inspect.signature(method)),
                "docstring": inspect.getdoc(method),
                "is_async": asyncio.iscoroutinefunction(method)
            }
    
    # Find config class
    type_hints = get_type_hints(component_class.__init__) if hasattr(component_class, '__init__') else {}
    for param_name, param_type in type_hints.items():
        if 'Config' in str(param_type):
            info["config_class"] = str(param_type)
            break
    
    return info


def create_task_with_logging(
    coro: Any,
    name: Optional[str] = None,
    error_handler: Optional[Callable[[Exception], None]] = None
) -> asyncio.Task:
    """
    Create an asyncio task with automatic error logging.
    
    Args:
        coro: The coroutine to run
        name: Optional name for the task
        error_handler: Optional callback for handling errors
        
    Returns:
        The created task
    """
    async def wrapped():
        try:
            return await coro
        except Exception as e:
            logger.error(f"Task '{name or 'unnamed'}' failed: {e}", exc_info=True)
            if error_handler:
                error_handler(e)
            raise
    
    task = asyncio.create_task(wrapped())
    if name:
        task.set_name(name)
    
    return task


def validate_api_key(api_key: Optional[str], provider_name: str) -> str:
    """
    Validate an API key.
    
    Args:
        api_key: The API key to validate
        provider_name: Name of the provider for error messages
        
    Returns:
        The validated API key
        
    Raises:
        ValueError: If the API key is invalid
    """
    if not api_key:
        raise ValueError(f"API key for {provider_name} is required")
    
    if not isinstance(api_key, str):
        raise ValueError(f"API key for {provider_name} must be a string")
    
    if len(api_key.strip()) < 10:  # Reasonable minimum length
        raise ValueError(f"API key for {provider_name} appears to be invalid (too short)")
    
    return api_key.strip()


class ComponentRegistry:
    """Registry for managing SDK components."""
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Type]] = {
            'stt': {},
            'llm': {},
            'tts': {},
            'vad': {},
            'pipeline': {}
        }
    
    def register(self, component_type: str, name: str, component_class: Type) -> None:
        """Register a component."""
        if component_type not in self._components:
            raise ValueError(f"Invalid component type: {component_type}")
        
        self._components[component_type][name] = component_class
        logger.debug(f"Registered {component_type} component: {name}")
    
    def get(self, component_type: str, name: str) -> Optional[Type]:
        """Get a registered component."""
        return self._components.get(component_type, {}).get(name)
    
    def list(self, component_type: Optional[str] = None) -> Dict[str, Dict[str, Type]]:
        """List all registered components."""
        if component_type:
            return {component_type: self._components.get(component_type, {})}
        return self._components.copy()
    
    def discover_and_register(self, base_path: str) -> None:
        """Discover and register all components in a directory."""
        for component_type in self._components.keys():
            component_path = os.path.join(base_path, 'components', component_type)
            if os.path.exists(component_path):
                for file in os.listdir(component_path):
                    if file.endswith('.py') and not file.startswith('_'):
                        module_path = f"components.{component_type}.{file[:-3]}"
                        components = discover_components(module_path)
                        for name, cls in components.items():
                            self.register(component_type, name.lower(), cls)