"""
Unit tests for SDK utilities module.

Tests utility functions, helpers, and component registry.
"""

import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from enum import Enum

from utils import (
    validate_type,
    validate_enum,
    validate_range,
    ensure_async,
    is_component_class,
    discover_components,
    load_config_file,
    merge_configs,
    AsyncContextManager,
    SingletonMeta,
    format_duration,
    get_component_info,
    create_task_with_logging,
    validate_api_key,
    ComponentRegistry
)


class TestTypeValidation:
    """Test type validation functions."""
    
    def test_validate_type_success(self):
        """Test successful type validation."""
        result = validate_type("test", str, "field")
        assert result == "test"
        
        result = validate_type(123, int, "number")
        assert result == 123
        
        result = validate_type([1, 2, 3], list, "items")
        assert result == [1, 2, 3]
    
    def test_validate_type_failure(self):
        """Test type validation failure."""
        with pytest.raises(TypeError) as exc:
            validate_type(123, str, "name")
        assert "'name' must be of type str" in str(exc.value)
        
        with pytest.raises(TypeError) as exc:
            validate_type("test", int, "count")
        assert "'count' must be of type int" in str(exc.value)
    
    def test_validate_enum_success(self):
        """Test successful enum validation."""
        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"
        
        # Already enum member
        result = validate_enum(Color.RED, Color, "color")
        assert result == Color.RED
        
        # String value
        result = validate_enum("green", Color, "color")
        assert result == Color.GREEN
    
    def test_validate_enum_failure(self):
        """Test enum validation failure."""
        class Size(Enum):
            SMALL = "small"
            LARGE = "large"
        
        with pytest.raises(ValueError) as exc:
            validate_enum("medium", Size, "size")
        assert "'size' must be one of" in str(exc.value)
        assert "['small', 'large']" in str(exc.value)
    
    def test_validate_range_success(self):
        """Test successful range validation."""
        result = validate_range(5, 0, 10, "value")
        assert result == 5
        
        result = validate_range(0.5, 0.0, 1.0, "ratio")
        assert result == 0.5
        
        # No min constraint
        result = validate_range(100, None, 200, "value")
        assert result == 100
        
        # No max constraint
        result = validate_range(-50, -100, None, "value")
        assert result == -50
    
    def test_validate_range_failure(self):
        """Test range validation failure."""
        with pytest.raises(ValueError) as exc:
            validate_range(-1, 0, 10, "count")
        assert "'count' must be >= 0" in str(exc.value)
        
        with pytest.raises(ValueError) as exc:
            validate_range(15, 0, 10, "level")
        assert "'level' must be <= 10" in str(exc.value)


class TestAsyncHelpers:
    """Test async helper functions."""
    
    @pytest.mark.asyncio
    async def test_ensure_async_with_async_func(self):
        """Test ensure_async with already async function."""
        async def async_func(x):
            return x * 2
        
        wrapped = ensure_async(async_func)
        assert wrapped == async_func  # Should return the same function
        result = await wrapped(5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_ensure_async_with_sync_func(self):
        """Test ensure_async with sync function."""
        def sync_func(x):
            return x * 2
        
        wrapped = ensure_async(sync_func)
        assert wrapped != sync_func  # Should return wrapper
        assert asyncio.iscoroutinefunction(wrapped)
        result = await wrapped(5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test AsyncContextManager base class."""
        class TestManager(AsyncContextManager):
            def __init__(self):
                self.started = False
                self.stopped = False
            
            async def start(self):
                self.started = True
            
            async def stop(self):
                self.stopped = True
        
        manager = TestManager()
        assert not manager.started
        assert not manager.stopped
        
        async with manager:
            assert manager.started
            assert not manager.stopped
        
        assert manager.started
        assert manager.stopped


class TestComponentDiscovery:
    """Test component discovery functions."""
    
    def test_is_component_class(self):
        """Test component class detection."""
        # Create mock component classes
        class MockSTT:
            pass
        
        class MockLLM:
            pass
        
        # Mock the MRO to simulate inheritance
        with patch('inspect.getmro') as mock_mro:
            # Simulate STT component
            mock_mro.return_value = [MockSTT, type('BaseSTTProvider', (), {}), object]
            assert is_component_class(MockSTT) is True
            
            # Simulate LLM component
            mock_mro.return_value = [MockLLM, type('BaseLLMProvider', (), {}), object]
            assert is_component_class(MockLLM) is True
            
            # Non-component class
            mock_mro.return_value = [str, object]
            assert is_component_class(str) is False
    
    @patch('importlib.import_module')
    @patch('inspect.getmembers')
    def test_discover_components(self, mock_getmembers, mock_import):
        """Test component discovery in a package."""
        # Mock module
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        
        # Mock component classes
        class MockComponent:
            pass
        
        mock_getmembers.return_value = [
            ('MockComponent', MockComponent),
            ('helper_func', lambda x: x),  # Non-class member
            ('CONSTANT', 42)  # Non-class member
        ]
        
        with patch('utils.is_component_class') as mock_is_component:
            mock_is_component.side_effect = lambda cls: cls == MockComponent
            
            components = discover_components('test.package')
            
            assert 'MockComponent' in components
            assert components['MockComponent'] == MockComponent
            assert len(components) == 1


class TestConfigFileHandling:
    """Test configuration file loading and merging."""
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
            api_key: test_key
            settings:
              temperature: 0.7
              max_tokens: 1000
            """
            f.write(yaml_content)
            f.flush()
            
            config = load_config_file(f.name)
            assert config['api_key'] == 'test_key'
            assert config['settings']['temperature'] == 0.7
            assert config['settings']['max_tokens'] == 1000
            
            Path(f.name).unlink()
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_content = {
                "api_key": "test_key",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
            json.dump(json_content, f)
            f.flush()
            
            config = load_config_file(f.name)
            assert config['api_key'] == 'test_key'
            assert config['settings']['temperature'] == 0.7
            
            Path(f.name).unlink()
    
    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config_file("non_existent_file.yaml")
    
    def test_load_config_unsupported_format(self):
        """Test loading unsupported config format."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            with pytest.raises(ValueError) as exc:
                load_config_file(f.name)
            assert "Unsupported config file format" in str(exc.value)
    
    def test_merge_configs_simple(self):
        """Test simple config merging."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}
        
        result = merge_configs(config1, config2)
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_configs_nested(self):
        """Test nested config merging."""
        config1 = {
            "api": {"key": "old_key", "url": "http://old.com"},
            "settings": {"debug": True}
        }
        config2 = {
            "api": {"key": "new_key"},
            "settings": {"verbose": True}
        }
        
        result = merge_configs(config1, config2)
        assert result["api"]["key"] == "new_key"
        assert result["api"]["url"] == "http://old.com"
        assert result["settings"]["debug"] is True
        assert result["settings"]["verbose"] is True


class TestSingletonMeta:
    """Test singleton metaclass."""
    
    def test_singleton_behavior(self):
        """Test that singleton returns same instance."""
        class SingletonClass(metaclass=SingletonMeta):
            def __init__(self):
                self.value = 42
        
        instance1 = SingletonClass()
        instance2 = SingletonClass()
        
        assert instance1 is instance2
        assert id(instance1) == id(instance2)
        
        # Modify one instance
        instance1.value = 100
        assert instance2.value == 100


class TestUtilityFunctions:
    """Test various utility functions."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(0.5) == "500ms"
        assert format_duration(1.5) == "1.5s"
        assert format_duration(65) == "1m 5s"
        assert format_duration(3665) == "1h 1m"
    
    def test_get_component_info(self):
        """Test extracting component information."""
        class TestComponent:
            """Test component for documentation."""
            
            def __init__(self, config: 'TestConfig'):
                self.config = config
            
            def process(self, data: str) -> str:
                """Process the data."""
                return data.upper()
            
            async def async_process(self, data: str) -> str:
                """Process data asynchronously."""
                return data.lower()
            
            def _private_method(self):
                """Private method."""
                pass
        
        info = get_component_info(TestComponent)
        
        assert info["name"] == "TestComponent"
        assert "Test component for documentation" in info["docstring"]
        assert "process" in info["methods"]
        assert "async_process" in info["methods"]
        assert "_private_method" not in info["methods"]
        assert info["methods"]["async_process"]["is_async"] is True
        assert info["methods"]["process"]["is_async"] is False
    
    @pytest.mark.asyncio
    async def test_create_task_with_logging(self):
        """Test task creation with logging."""
        # Successful task
        async def success_coro():
            await asyncio.sleep(0.01)
            return "success"
        
        task = create_task_with_logging(success_coro(), name="test_task")
        result = await task
        assert result == "success"
        assert task.get_name() == "test_task"
        
        # Failing task
        async def failing_coro():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        error_handled = False
        def error_handler(e):
            nonlocal error_handled
            error_handled = True
            assert isinstance(e, ValueError)
        
        task = create_task_with_logging(
            failing_coro(),
            name="failing_task",
            error_handler=error_handler
        )
        
        with pytest.raises(ValueError):
            await task
        
        assert error_handled
    
    def test_validate_api_key_success(self):
        """Test successful API key validation."""
        key = validate_api_key("sk-1234567890abcdef", "OpenAI")
        assert key == "sk-1234567890abcdef"
        
        # With whitespace
        key = validate_api_key("  sk-1234567890abcdef  ", "OpenAI")
        assert key == "sk-1234567890abcdef"
    
    def test_validate_api_key_failure(self):
        """Test API key validation failures."""
        # None
        with pytest.raises(ValueError) as exc:
            validate_api_key(None, "OpenAI")
        assert "API key for OpenAI is required" in str(exc.value)
        
        # Empty string
        with pytest.raises(ValueError) as exc:
            validate_api_key("", "OpenAI")
        assert "API key for OpenAI is required" in str(exc.value)
        
        # Wrong type
        with pytest.raises(ValueError) as exc:
            validate_api_key(123, "OpenAI")
        assert "API key for OpenAI must be a string" in str(exc.value)
        
        # Too short
        with pytest.raises(ValueError) as exc:
            validate_api_key("short", "OpenAI")
        assert "API key for OpenAI appears to be invalid" in str(exc.value)


class TestComponentRegistry:
    """Test the component registry."""
    
    def test_register_component(self):
        """Test registering components."""
        registry = ComponentRegistry()
        
        class MockSTT:
            pass
        
        registry.register('stt', 'mock', MockSTT)
        
        # Verify registration
        result = registry.get('stt', 'mock')
        assert result == MockSTT
    
    def test_register_invalid_type(self):
        """Test registering with invalid component type."""
        registry = ComponentRegistry()
        
        with pytest.raises(ValueError) as exc:
            registry.register('invalid_type', 'test', Mock)
        assert "Invalid component type" in str(exc.value)
    
    def test_get_nonexistent_component(self):
        """Test getting non-existent component."""
        registry = ComponentRegistry()
        result = registry.get('stt', 'nonexistent')
        assert result is None
    
    def test_list_components(self):
        """Test listing components."""
        registry = ComponentRegistry()
        
        # Register some components
        registry.register('stt', 'mock1', Mock)
        registry.register('llm', 'mock2', Mock)
        
        # List all
        all_components = registry.list()
        assert 'stt' in all_components
        assert 'mock1' in all_components['stt']
        assert 'llm' in all_components
        assert 'mock2' in all_components['llm']
        
        # List specific type
        stt_only = registry.list('stt')
        assert len(stt_only) == 1
        assert 'stt' in stt_only
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_discover_and_register(self, mock_listdir, mock_exists):
        """Test automatic discovery and registration."""
        registry = ComponentRegistry()
        
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ['openai_stt.py', 'azure_stt.py', '__init__.py', 'test.txt']
        
        with patch('utils.discover_components') as mock_discover:
            mock_discover.return_value = {'OpenAISTT': Mock()}
            
            registry.discover_and_register('/test/path')
            
            # Should have tried to discover components
            assert mock_discover.call_count > 0


class TestIntegrationPatterns:
    """Test common integration patterns with utilities."""
    
    def test_config_validation_pattern(self):
        """Test using validation utilities together."""
        class LogLevel(Enum):
            DEBUG = "debug"
            INFO = "info"
            ERROR = "error"
        
        def validate_config(config: Dict[str, Any]):
            # Validate types
            validate_type(config.get('name'), str, 'name')
            
            # Validate enum
            log_level = validate_enum(config.get('log_level', 'info'), LogLevel, 'log_level')
            
            # Validate range
            if 'timeout' in config:
                validate_range(config['timeout'], 0, 300, 'timeout')
            
            return True
        
        # Valid config
        valid_config = {
            'name': 'test',
            'log_level': 'debug',
            'timeout': 30
        }
        assert validate_config(valid_config) is True
        
        # Invalid configs
        with pytest.raises(TypeError):
            validate_config({'name': 123})
        
        with pytest.raises(ValueError):
            validate_config({'name': 'test', 'log_level': 'invalid'})
        
        with pytest.raises(ValueError):
            validate_config({'name': 'test', 'timeout': -1})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])