"""
Configuration management for the Voice Agent SDK.

This module provides a flexible configuration system that supports:
- Multiple configuration sources (files, environment, code)
- Configuration validation and defaults
- Dynamic configuration updates
- Provider-specific configurations
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import yaml
from enum import Enum

from .exceptions import ConfigurationError, ValidationError
from .utils import validate_type, validate_enum, validate_range, merge_configs, load_config_file

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Logging levels for the SDK."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class ProviderConfig:
    """Base configuration for AI providers."""
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            provider_name = self.__class__.__name__.replace('Config', '').upper()
            env_key = f"{provider_name}_API_KEY"
            self.api_key = os.getenv(env_key)


@dataclass
class OpenAIConfig(ProviderConfig):
    """Configuration for OpenAI providers."""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    organization: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.organization:
            self.organization = os.getenv("OPENAI_ORGANIZATION")


@dataclass
class AnthropicConfig(ProviderConfig):
    """Configuration for Anthropic providers."""
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class AzureConfig(ProviderConfig):
    """Configuration for Azure providers."""
    subscription_key: Optional[str] = None
    region: str = "eastus"
    endpoint: Optional[str] = None
    
    def __post_init__(self):
        if not self.subscription_key:
            self.subscription_key = os.getenv("AZURE_SUBSCRIPTION_KEY")
        if not self.endpoint:
            self.endpoint = os.getenv("AZURE_ENDPOINT")
        # For Azure, api_key is the subscription key
        self.api_key = self.subscription_key


@dataclass
class ElevenLabsConfig(ProviderConfig):
    """Configuration for ElevenLabs TTS."""
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default voice
    model_id: str = "eleven_monolingual_v1"
    optimize_streaming_latency: int = 2
    
    def __post_init__(self):
        super().__post_init__()
        if not self.api_key:
            self.api_key = os.getenv("ELEVENLABS_API_KEY")


@dataclass
class PipelineConfig:
    """Configuration for the audio pipeline."""
    sample_rate: int = 16000
    chunk_size: int = 1024
    buffer_size: int = 4096
    enable_vad: bool = True
    enable_echo_cancellation: bool = True
    enable_noise_suppression: bool = True
    enable_auto_gain_control: bool = True
    latency_mode: str = "low"  # low, balanced, quality
    
    def __post_init__(self):
        validate_range(self.sample_rate, 8000, 48000, "sample_rate")
        validate_range(self.chunk_size, 128, 4096, "chunk_size")


@dataclass
class ServerConfig:
    """Configuration for the development server."""
    host: str = "localhost"
    port: int = 8080
    enable_hot_reload: bool = True
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    
    @property
    def is_ssl_enabled(self) -> bool:
        return bool(self.ssl_cert and self.ssl_key)


@dataclass
class SDKConfig:
    """Main SDK configuration."""
    # General settings
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    project_name: str = "voice-agent"
    
    # Component settings
    enable_auto_discovery: bool = True
    component_timeout: float = 30.0
    
    # Pipeline settings
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Provider configurations
    openai: Optional[OpenAIConfig] = None
    anthropic: Optional[AnthropicConfig] = None
    azure: Optional[AzureConfig] = None
    elevenlabs: Optional[ElevenLabsConfig] = None
    
    # Server settings
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # Paths
    config_dir: Path = Path.home() / ".voice-agent"
    cache_dir: Path = Path.home() / ".voice-agent" / "cache"
    log_dir: Path = Path.home() / ".voice-agent" / "logs"
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize provider configs if not provided
        if self.openai is None:
            self.openai = OpenAIConfig()
        if self.anthropic is None:
            self.anthropic = AnthropicConfig()
        if self.azure is None:
            self.azure = AzureConfig()
        if self.elevenlabs is None:
            self.elevenlabs = ElevenLabsConfig()


class ConfigManager:
    """
    Manages SDK configuration with support for multiple sources.
    
    Configuration precedence (highest to lowest):
    1. Runtime overrides
    2. Environment variables
    3. Configuration files
    4. Default values
    """
    
    def __init__(self, config: Optional[Union[SDKConfig, Dict[str, Any], str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config: Can be:
                - SDKConfig instance
                - Dictionary of configuration values
                - Path to configuration file (str or Path)
                - None (uses defaults)
        """
        self._config = self._load_config(config)
        self._runtime_overrides: Dict[str, Any] = {}
        self._setup_logging()
    
    def _load_config(self, config: Optional[Union[SDKConfig, Dict[str, Any], str, Path]]) -> SDKConfig:
        """Load configuration from various sources."""
        if config is None:
            # Use defaults
            return SDKConfig()
        
        elif isinstance(config, SDKConfig):
            return config
        
        elif isinstance(config, dict):
            return self._create_config_from_dict(config)
        
        elif isinstance(config, (str, Path)):
            config_dict = load_config_file(config)
            return self._create_config_from_dict(config_dict)
        
        else:
            raise ConfigurationError(
                f"Invalid configuration type: {type(config)}. "
                "Expected SDKConfig, dict, str, Path, or None"
            )
    
    def _create_config_from_dict(self, config_dict: Dict[str, Any]) -> SDKConfig:
        """Create SDKConfig from dictionary."""
        try:
            # Handle nested configurations
            if 'pipeline' in config_dict and isinstance(config_dict['pipeline'], dict):
                config_dict['pipeline'] = PipelineConfig(**config_dict['pipeline'])
            
            if 'server' in config_dict and isinstance(config_dict['server'], dict):
                config_dict['server'] = ServerConfig(**config_dict['server'])
            
            # Handle provider configurations
            for provider in ['openai', 'anthropic', 'azure', 'elevenlabs']:
                if provider in config_dict and isinstance(config_dict[provider], dict):
                    provider_class = globals()[f"{provider.capitalize()}Config"]
                    config_dict[provider] = provider_class(**config_dict[provider])
            
            # Handle enums
            if 'environment' in config_dict:
                config_dict['environment'] = validate_enum(
                    config_dict['environment'], Environment, 'environment'
                )
            
            if 'log_level' in config_dict:
                config_dict['log_level'] = validate_enum(
                    config_dict['log_level'], LogLevel, 'log_level'
                )
            
            return SDKConfig(**config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {e}")
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        logging.basicConfig(
            level=self._config.log_level.value,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self._config.log_dir / f"{self._config.project_name}.log"
                )
            ]
        )
    
    @property
    def config(self) -> SDKConfig:
        """Get the current configuration."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Supports nested keys using dot notation (e.g., 'pipeline.sample_rate').
        """
        # Check runtime overrides first
        if key in self._runtime_overrides:
            return self._runtime_overrides[key]
        
        # Check environment variables
        env_key = f"VOICE_AGENT_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)
        
        # Get from config object
        parts = key.split('.')
        value = self._config
        
        try:
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration override."""
        self._runtime_overrides[key] = value
        logger.debug(f"Set runtime override: {key} = {value}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        for key, value in updates.items():
            self.set(key, value)
    
    def reset(self) -> None:
        """Reset all runtime overrides."""
        self._runtime_overrides.clear()
        logger.debug("Reset all runtime overrides")
    
    def validate(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required API keys based on environment
        if self._config.environment != Environment.TEST:
            if not self._config.openai.api_key:
                errors.append("OpenAI API key is required")
            
            if not self._config.elevenlabs.api_key:
                errors.append("ElevenLabs API key is required")
        
        # Validate paths exist and are writable
        for path_name in ['config_dir', 'cache_dir', 'log_dir']:
            path = getattr(self._config, path_name)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create {path_name}: {e}")
            elif not os.access(path, os.W_OK):
                errors.append(f"{path_name} is not writable: {path}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def convert(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        result[key] = convert(value)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
        
        config_dict = convert(self._config)
        # Apply runtime overrides
        for key, value in self._runtime_overrides.items():
            parts = key.split('.')
            target = config_dict
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        
        return config_dict
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self._config.config_dir / "config.yaml"
        else:
            path = Path(path)
        
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {path}")
    
    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Check for boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value