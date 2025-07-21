"""
Unit tests for the configuration management module.

Tests configuration loading, validation, and management functionality.
"""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from config_manager import (
    ConfigManager,
    SDKConfig,
    LogLevel,
    Environment,
    ProviderConfig,
    OpenAIConfig,
    AnthropicConfig,
    AzureConfig,
    ElevenLabsConfig,
    PipelineConfig,
    ServerConfig
)
from exceptions import ConfigurationError, ValidationError


class TestEnums:
    """Test configuration enums."""
    
    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    
    def test_environment_enum(self):
        """Test Environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TEST.value == "test"


class TestProviderConfigs:
    """Test provider configuration classes."""
    
    def test_provider_config_base(self):
        """Test base provider configuration."""
        config = ProviderConfig(api_key="test_key", timeout=60.0)
        assert config.api_key == "test_key"
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
    
    @patch.dict(os.environ, {"TEST_API_KEY": "env_key"})
    def test_provider_config_env_loading(self):
        """Test loading API key from environment."""
        # Mock the class name for env var lookup
        class TestConfig(ProviderConfig):
            pass
        
        config = TestConfig()
        # Base class doesn't auto-load, but sets up the pattern
        assert config.api_key is None
    
    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = OpenAIConfig(
            api_key="sk-test",
            model="gpt-4",
            temperature=0.5,
            organization="test-org"
        )
        assert config.api_key == "sk-test"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.organization == "test-org"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_openai_key"})
    def test_openai_config_env(self):
        """Test OpenAI config loading from environment."""
        config = OpenAIConfig()
        assert config.api_key == "env_openai_key"
    
    def test_anthropic_config(self):
        """Test Anthropic configuration."""
        config = AnthropicConfig(
            api_key="ant-test",
            model="claude-3",
            temperature=0.8
        )
        assert config.api_key == "ant-test"
        assert config.model == "claude-3"
        assert config.temperature == 0.8
    
    def test_azure_config(self):
        """Test Azure configuration."""
        config = AzureConfig(
            subscription_key="sub-key",
            region="westus",
            endpoint="https://test.azure.com"
        )
        assert config.subscription_key == "sub-key"
        assert config.region == "westus"
        assert config.endpoint == "https://test.azure.com"
        assert config.api_key == "sub-key"  # Should mirror subscription_key
    
    def test_elevenlabs_config(self):
        """Test ElevenLabs configuration."""
        config = ElevenLabsConfig(
            api_key="el-test",
            voice_id="custom_voice",
            model_id="eleven_turbo_v2"
        )
        assert config.api_key == "el-test"
        assert config.voice_id == "custom_voice"
        assert config.model_id == "eleven_turbo_v2"


class TestPipelineConfig:
    """Test pipeline configuration."""
    
    def test_pipeline_config_defaults(self):
        """Test pipeline config default values."""
        config = PipelineConfig()
        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.buffer_size == 4096
        assert config.enable_vad is True
        assert config.latency_mode == "low"
    
    def test_pipeline_config_validation(self):
        """Test pipeline config validation."""
        # Valid config
        config = PipelineConfig(sample_rate=44100, chunk_size=2048)
        assert config.sample_rate == 44100
        assert config.chunk_size == 2048
        
        # Invalid sample rate
        with pytest.raises(ValueError):
            PipelineConfig(sample_rate=4000)  # Too low
        
        with pytest.raises(ValueError):
            PipelineConfig(sample_rate=96000)  # Too high
        
        # Invalid chunk size
        with pytest.raises(ValueError):
            PipelineConfig(chunk_size=64)  # Too small
        
        with pytest.raises(ValueError):
            PipelineConfig(chunk_size=8192)  # Too large


class TestServerConfig:
    """Test server configuration."""
    
    def test_server_config_defaults(self):
        """Test server config defaults."""
        config = ServerConfig()
        assert config.host == "localhost"
        assert config.port == 8080
        assert config.enable_hot_reload is True
        assert config.enable_cors is True
        assert config.cors_origins == ["*"]
        assert config.ssl_cert is None
        assert config.ssl_key is None
    
    def test_server_config_ssl(self):
        """Test SSL configuration."""
        # No SSL
        config = ServerConfig()
        assert config.is_ssl_enabled is False
        
        # With SSL
        config = ServerConfig(
            ssl_cert="/path/to/cert.pem",
            ssl_key="/path/to/key.pem"
        )
        assert config.is_ssl_enabled is True


class TestSDKConfig:
    """Test main SDK configuration."""
    
    def test_sdk_config_defaults(self):
        """Test SDK config defaults."""
        config = SDKConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        assert config.project_name == "voice-agent"
        assert config.enable_auto_discovery is True
        assert config.component_timeout == 30.0
    
    def test_sdk_config_paths(self):
        """Test SDK config path creation."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            config = SDKConfig()
            
            # Verify paths are set
            assert config.config_dir == Path.home() / ".voice-agent"
            assert config.cache_dir == Path.home() / ".voice-agent" / "cache"
            assert config.log_dir == Path.home() / ".voice-agent" / "logs"
            
            # Verify mkdir was called for each path
            assert mock_mkdir.call_count >= 3
    
    def test_sdk_config_provider_initialization(self):
        """Test that provider configs are initialized."""
        config = SDKConfig()
        assert isinstance(config.openai, OpenAIConfig)
        assert isinstance(config.anthropic, AnthropicConfig)
        assert isinstance(config.azure, AzureConfig)
        assert isinstance(config.elevenlabs, ElevenLabsConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.server, ServerConfig)


class TestConfigManager:
    """Test the configuration manager."""
    
    def test_init_with_defaults(self):
        """Test initialization with default config."""
        manager = ConfigManager()
        assert isinstance(manager.config, SDKConfig)
        assert manager.config.environment == Environment.DEVELOPMENT
    
    def test_init_with_sdk_config(self):
        """Test initialization with SDKConfig instance."""
        sdk_config = SDKConfig(
            project_name="test-project",
            environment=Environment.PRODUCTION
        )
        manager = ConfigManager(sdk_config)
        assert manager.config.project_name == "test-project"
        assert manager.config.environment == Environment.PRODUCTION
    
    def test_init_with_dict(self):
        """Test initialization with dictionary."""
        config_dict = {
            "project_name": "dict-project",
            "log_level": "DEBUG",
            "pipeline": {
                "sample_rate": 22050,
                "chunk_size": 512
            }
        }
        manager = ConfigManager(config_dict)
        assert manager.config.project_name == "dict-project"
        assert manager.config.log_level == LogLevel.DEBUG
        assert manager.config.pipeline.sample_rate == 22050
    
    def test_init_with_yaml_file(self):
        """Test initialization with YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
            project_name: yaml-project
            environment: staging
            openai:
              api_key: test-key
              model: gpt-3.5-turbo
            """
            f.write(yaml_content)
            f.flush()
            
            manager = ConfigManager(f.name)
            assert manager.config.project_name == "yaml-project"
            assert manager.config.environment == Environment.STAGING
            assert manager.config.openai.api_key == "test-key"
            assert manager.config.openai.model == "gpt-3.5-turbo"
            
            Path(f.name).unlink()
    
    def test_init_with_invalid_type(self):
        """Test initialization with invalid type."""
        with pytest.raises(ConfigurationError) as exc:
            ConfigManager(123)
        assert "Invalid configuration type" in str(exc.value)
    
    def test_get_config_value(self):
        """Test getting configuration values."""
        manager = ConfigManager({"project_name": "test", "pipeline": {"sample_rate": 48000}})
        
        # Direct attribute
        assert manager.get("project_name") == "test"
        
        # Nested attribute
        assert manager.get("pipeline.sample_rate") == 48000
        
        # Non-existent with default
        assert manager.get("non.existent", "default") == "default"
    
    @patch.dict(os.environ, {"VOICE_AGENT_PROJECT_NAME": "env-project"})
    def test_get_with_env_override(self):
        """Test getting value with environment override."""
        manager = ConfigManager({"project_name": "config-project"})
        assert manager.get("project_name") == "env-project"
    
    def test_set_runtime_override(self):
        """Test setting runtime configuration override."""
        manager = ConfigManager({"project_name": "original"})
        
        # Set override
        manager.set("project_name", "overridden")
        assert manager.get("project_name") == "overridden"
        
        # Original config unchanged
        assert manager.config.project_name == "original"
    
    def test_update_multiple_values(self):
        """Test updating multiple values."""
        manager = ConfigManager()
        updates = {
            "project_name": "updated",
            "log_level": "DEBUG",
            "pipeline.sample_rate": 44100
        }
        manager.update(updates)
        
        assert manager.get("project_name") == "updated"
        assert manager.get("log_level") == "DEBUG"
        assert manager.get("pipeline.sample_rate") == 44100
    
    def test_reset_overrides(self):
        """Test resetting runtime overrides."""
        manager = ConfigManager({"project_name": "original"})
        manager.set("project_name", "overridden")
        assert manager.get("project_name") == "overridden"
        
        manager.reset()
        assert manager.get("project_name") == "original"
    
    def test_validate_success(self):
        """Test successful configuration validation."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "ELEVENLABS_API_KEY": "test-key"
        }):
            manager = ConfigManager()
            errors = manager.validate()
            assert len(errors) == 0
    
    def test_validate_missing_api_keys(self):
        """Test validation with missing API keys."""
        manager = ConfigManager()
        manager.config.environment = Environment.PRODUCTION
        errors = manager.validate()
        
        # Should have errors for missing API keys
        assert any("OpenAI API key" in error for error in errors)
        assert any("ElevenLabs API key" in error for error in errors)
    
    def test_validate_test_environment(self):
        """Test validation skips API keys in test environment."""
        manager = ConfigManager({"environment": "test"})
        errors = manager.validate()
        
        # Should not complain about API keys in test env
        assert not any("API key" in error for error in errors)
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        manager = ConfigManager({
            "project_name": "test",
            "environment": "production",
            "pipeline": {"sample_rate": 44100}
        })
        
        config_dict = manager.to_dict()
        assert config_dict["project_name"] == "test"
        assert config_dict["environment"] == "production"
        assert config_dict["pipeline"]["sample_rate"] == 44100
    
    def test_to_dict_with_overrides(self):
        """Test to_dict includes runtime overrides."""
        manager = ConfigManager({"project_name": "original"})
        manager.set("project_name", "overridden")
        manager.set("new_key", "new_value")
        
        config_dict = manager.to_dict()
        assert config_dict["project_name"] == "overridden"
        assert config_dict["new_key"] == "new_value"
    
    def test_save_yaml(self):
        """Test saving configuration to YAML."""
        manager = ConfigManager({"project_name": "save-test"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config.yaml"
            manager.save(save_path)
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path) as f:
                loaded = yaml.safe_load(f)
                assert loaded["project_name"] == "save-test"
    
    def test_save_json(self):
        """Test saving configuration to JSON."""
        manager = ConfigManager({"project_name": "save-test"})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config.json"
            manager.save(save_path)
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path) as f:
                loaded = json.load(f)
                assert loaded["project_name"] == "save-test"
    
    def test_save_default_location(self):
        """Test saving to default location."""
        manager = ConfigManager()
        
        with patch('builtins.open', MagicMock()) as mock_open:
            with patch('pathlib.Path.exists', return_value=True):
                manager.save()
                
                # Should save to config_dir / "config.yaml"
                expected_path = manager.config.config_dir / "config.yaml"
                mock_open.assert_called()


class TestEnvironmentValueParsing:
    """Test parsing environment variable values."""
    
    def test_parse_env_json(self):
        """Test parsing JSON from environment."""
        assert ConfigManager._parse_env_value('{"key": "value"}') == {"key": "value"}
        assert ConfigManager._parse_env_value('[1, 2, 3]') == [1, 2, 3]
    
    def test_parse_env_boolean(self):
        """Test parsing boolean values."""
        # True values
        assert ConfigManager._parse_env_value('true') is True
        assert ConfigManager._parse_env_value('True') is True
        assert ConfigManager._parse_env_value('yes') is True
        assert ConfigManager._parse_env_value('1') is True
        
        # False values
        assert ConfigManager._parse_env_value('false') is False
        assert ConfigManager._parse_env_value('False') is False
        assert ConfigManager._parse_env_value('no') is False
        assert ConfigManager._parse_env_value('0') is False
    
    def test_parse_env_numbers(self):
        """Test parsing numeric values."""
        assert ConfigManager._parse_env_value('42') == 42
        assert ConfigManager._parse_env_value('3.14') == 3.14
        assert ConfigManager._parse_env_value('-10') == -10
    
    def test_parse_env_string(self):
        """Test parsing string values."""
        assert ConfigManager._parse_env_value('hello world') == 'hello world'
        assert ConfigManager._parse_env_value('not-a-number') == 'not-a-number'


class TestConfigurationIntegration:
    """Test configuration integration scenarios."""
    
    def test_full_configuration_workflow(self):
        """Test complete configuration workflow."""
        # Create config with multiple sources
        base_config = {
            "project_name": "base",
            "log_level": "INFO",
            "pipeline": {"sample_rate": 16000}
        }
        
        # Initialize manager
        manager = ConfigManager(base_config)
        
        # Apply runtime overrides
        manager.set("log_level", "DEBUG")
        manager.set("pipeline.sample_rate", 44100)
        
        # Validate
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test",
            "ELEVENLABS_API_KEY": "test"
        }):
            errors = manager.validate()
            assert len(errors) == 0
        
        # Export config
        exported = manager.to_dict()
        assert exported["project_name"] == "base"
        assert exported["log_level"] == "DEBUG"
        assert exported["pipeline"]["sample_rate"] == 44100
    
    def test_provider_specific_configuration(self):
        """Test configuring specific providers."""
        config_dict = {
            "openai": {
                "api_key": "sk-test",
                "model": "gpt-4",
                "temperature": 0.5
            },
            "anthropic": {
                "api_key": "ant-test",
                "model": "claude-3"
            }
        }
        
        manager = ConfigManager(config_dict)
        
        # Verify provider configs
        assert manager.config.openai.api_key == "sk-test"
        assert manager.config.openai.model == "gpt-4"
        assert manager.config.openai.temperature == 0.5
        
        assert manager.config.anthropic.api_key == "ant-test"
        assert manager.config.anthropic.model == "claude-3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])