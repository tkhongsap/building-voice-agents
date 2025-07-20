"""
Fluent API for building voice agents.

This module provides an intuitive builder pattern for creating and configuring
voice agents with a clean, chainable API.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

from .exceptions import ConfigurationError, ComponentNotFoundError, ValidationError
from .config_manager import SDKConfig, PipelineConfig
from .utils import validate_type, ComponentRegistry

if TYPE_CHECKING:
    from ..components.stt.base_stt import BaseSTTProvider, STTConfig
    from ..components.llm.base_llm import BaseLLMProvider, LLMConfig
    from ..components.tts.base_tts import BaseTTSProvider, TTSConfig
    from ..components.vad.base_vad import BaseVADProvider, VADConfig
    from ..pipeline.audio_pipeline import AudioPipeline
    from ..conversation.context_manager import ConversationContextManager
    from ..conversation.turn_detector import TurnDetector
    from ..conversation.interruption_handler import InterruptionHandler

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Capabilities that can be enabled for a voice agent."""
    TURN_DETECTION = "turn_detection"
    INTERRUPTION_HANDLING = "interruption_handling"
    CONTEXT_MANAGEMENT = "context_management"
    CONVERSATION_STATE = "conversation_state"
    MULTI_LANGUAGE = "multi_language"
    EMOTION_DETECTION = "emotion_detection"
    CUSTOM_TOOLS = "custom_tools"
    STREAMING = "streaming"


@dataclass
class AgentMetadata:
    """Metadata for a voice agent."""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class VoiceAgentBuilder:
    """
    Fluent builder for creating voice agents.
    
    Example:
        agent = (VoiceAgentBuilder()
            .with_name("Customer Service Agent")
            .with_stt("openai", language="en")
            .with_llm("gpt-4", temperature=0.7)
            .with_tts("elevenlabs", voice_id="custom_voice")
            .with_capability(AgentCapability.INTERRUPTION_HANDLING)
            .build())
    """
    
    def __init__(self, config: Optional[SDKConfig] = None, registry: Optional[ComponentRegistry] = None):
        """
        Initialize the agent builder.
        
        Args:
            config: SDK configuration
            registry: Component registry
        """
        self._config = config or SDKConfig()
        self._registry = registry or ComponentRegistry()
        
        # Agent configuration
        self._metadata = AgentMetadata(name="VoiceAgent")
        self._capabilities: List[AgentCapability] = []
        
        # Component configurations
        self._stt_provider: Optional[str] = None
        self._stt_config: Optional[Dict[str, Any]] = None
        self._llm_provider: Optional[str] = None
        self._llm_config: Optional[Dict[str, Any]] = None
        self._tts_provider: Optional[str] = None
        self._tts_config: Optional[Dict[str, Any]] = None
        self._vad_provider: Optional[str] = None
        self._vad_config: Optional[Dict[str, Any]] = None
        
        # Pipeline configuration
        self._pipeline_config: Optional[PipelineConfig] = None
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_start': [],
            'on_stop': [],
            'on_error': [],
            'on_user_speech': [],
            'on_agent_speech': [],
            'on_turn_end': [],
            'on_interruption': []
        }
        
        # Custom components
        self._custom_components: Dict[str, Any] = {}
    
    def with_name(self, name: str, description: Optional[str] = None) -> 'VoiceAgentBuilder':
        """Set the agent's name and description."""
        self._metadata.name = name
        if description:
            self._metadata.description = description
        return self
    
    def with_metadata(
        self,
        version: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'VoiceAgentBuilder':
        """Set additional agent metadata."""
        if version:
            self._metadata.version = version
        if author:
            self._metadata.author = author
        if tags:
            self._metadata.tags = tags
        return self
    
    def with_stt(
        self,
        provider: str,
        **config
    ) -> 'VoiceAgentBuilder':
        """
        Configure the Speech-to-Text provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'azure', 'google')
            **config: Provider-specific configuration
        """
        self._stt_provider = provider
        self._stt_config = config
        logger.debug(f"Configured STT provider: {provider}")
        return self
    
    def with_llm(
        self,
        provider: str,
        model: Optional[str] = None,
        **config
    ) -> 'VoiceAgentBuilder':
        """
        Configure the Language Model provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'local')
            model: Model name (e.g., 'gpt-4', 'claude-3')
            **config: Provider-specific configuration
        """
        self._llm_provider = provider
        self._llm_config = config
        if model:
            self._llm_config['model'] = model
        logger.debug(f"Configured LLM provider: {provider}")
        return self
    
    def with_tts(
        self,
        provider: str,
        voice_id: Optional[str] = None,
        **config
    ) -> 'VoiceAgentBuilder':
        """
        Configure the Text-to-Speech provider.
        
        Args:
            provider: Provider name (e.g., 'elevenlabs', 'openai', 'azure')
            voice_id: Voice identifier
            **config: Provider-specific configuration
        """
        self._tts_provider = provider
        self._tts_config = config
        if voice_id:
            self._tts_config['voice_id'] = voice_id
        logger.debug(f"Configured TTS provider: {provider}")
        return self
    
    def with_vad(
        self,
        provider: str = "silero",
        **config
    ) -> 'VoiceAgentBuilder':
        """
        Configure the Voice Activity Detection provider.
        
        Args:
            provider: Provider name (e.g., 'silero', 'webrtc')
            **config: Provider-specific configuration
        """
        self._vad_provider = provider
        self._vad_config = config
        logger.debug(f"Configured VAD provider: {provider}")
        return self
    
    def with_pipeline_config(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        latency_mode: Optional[str] = None,
        **config
    ) -> 'VoiceAgentBuilder':
        """
        Configure the audio pipeline.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_size: Audio chunk size in samples
            latency_mode: Latency mode ('low', 'balanced', 'quality')
            **config: Additional pipeline configuration
        """
        pipeline_dict = {}
        if sample_rate:
            pipeline_dict['sample_rate'] = sample_rate
        if chunk_size:
            pipeline_dict['chunk_size'] = chunk_size
        if latency_mode:
            pipeline_dict['latency_mode'] = latency_mode
        pipeline_dict.update(config)
        
        self._pipeline_config = PipelineConfig(**pipeline_dict)
        return self
    
    def with_capability(self, capability: AgentCapability) -> 'VoiceAgentBuilder':
        """Enable a specific capability for the agent."""
        if capability not in self._capabilities:
            self._capabilities.append(capability)
            logger.debug(f"Enabled capability: {capability.value}")
        return self
    
    def with_capabilities(self, *capabilities: AgentCapability) -> 'VoiceAgentBuilder':
        """Enable multiple capabilities at once."""
        for capability in capabilities:
            self.with_capability(capability)
        return self
    
    def with_system_prompt(self, prompt: str) -> 'VoiceAgentBuilder':
        """Set the system prompt for the LLM."""
        if self._llm_config is None:
            self._llm_config = {}
        self._llm_config['system_prompt'] = prompt
        return self
    
    def with_custom_vocabulary(self, vocabulary: List[str]) -> 'VoiceAgentBuilder':
        """Add custom vocabulary for STT recognition."""
        if self._stt_config is None:
            self._stt_config = {}
        self._stt_config['custom_vocabulary'] = vocabulary
        return self
    
    def on_start(self, callback: Callable) -> 'VoiceAgentBuilder':
        """Register a callback for agent start event."""
        self._callbacks['on_start'].append(callback)
        return self
    
    def on_stop(self, callback: Callable) -> 'VoiceAgentBuilder':
        """Register a callback for agent stop event."""
        self._callbacks['on_stop'].append(callback)
        return self
    
    def on_error(self, callback: Callable[[Exception], None]) -> 'VoiceAgentBuilder':
        """Register a callback for error events."""
        self._callbacks['on_error'].append(callback)
        return self
    
    def on_user_speech(self, callback: Callable[[str], None]) -> 'VoiceAgentBuilder':
        """Register a callback for user speech events."""
        self._callbacks['on_user_speech'].append(callback)
        return self
    
    def on_agent_speech(self, callback: Callable[[str], None]) -> 'VoiceAgentBuilder':
        """Register a callback for agent speech events."""
        self._callbacks['on_agent_speech'].append(callback)
        return self
    
    def on_turn_end(self, callback: Callable) -> 'VoiceAgentBuilder':
        """Register a callback for turn end events."""
        self._callbacks['on_turn_end'].append(callback)
        return self
    
    def on_interruption(self, callback: Callable) -> 'VoiceAgentBuilder':
        """Register a callback for interruption events."""
        self._callbacks['on_interruption'].append(callback)
        return self
    
    def with_custom_component(self, name: str, component: Any) -> 'VoiceAgentBuilder':
        """Add a custom component to the agent."""
        self._custom_components[name] = component
        logger.debug(f"Added custom component: {name}")
        return self
    
    def validate(self) -> List[str]:
        """
        Validate the agent configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required components
        if not self._stt_provider:
            errors.append("STT provider is required")
        if not self._llm_provider:
            errors.append("LLM provider is required")
        if not self._tts_provider:
            errors.append("TTS provider is required")
        
        # Check if providers are registered
        if self._stt_provider and not self._registry.get('stt', self._stt_provider):
            errors.append(f"STT provider '{self._stt_provider}' not found")
        if self._llm_provider and not self._registry.get('llm', self._llm_provider):
            errors.append(f"LLM provider '{self._llm_provider}' not found")
        if self._tts_provider and not self._registry.get('tts', self._tts_provider):
            errors.append(f"TTS provider '{self._tts_provider}' not found")
        
        # Validate capability dependencies
        if AgentCapability.INTERRUPTION_HANDLING in self._capabilities:
            if not self._vad_provider:
                errors.append("VAD provider is required for interruption handling")
        
        return errors
    
    def build(self) -> 'VoiceAgent':
        """
        Build and return the configured voice agent.
        
        Returns:
            Configured VoiceAgent instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate configuration
        errors = self.validate()
        if errors:
            raise ConfigurationError(
                f"Invalid agent configuration: {'; '.join(errors)}"
            )
        
        # Create agent instance
        from .voice_agent import VoiceAgent
        
        agent = VoiceAgent(
            metadata=self._metadata,
            config=self._config,
            registry=self._registry
        )
        
        # Configure components
        agent._configure_stt(self._stt_provider, self._stt_config)
        agent._configure_llm(self._llm_provider, self._llm_config)
        agent._configure_tts(self._tts_provider, self._tts_config)
        
        if self._vad_provider:
            agent._configure_vad(self._vad_provider, self._vad_config)
        
        if self._pipeline_config:
            agent._configure_pipeline(self._pipeline_config)
        
        # Enable capabilities
        for capability in self._capabilities:
            agent._enable_capability(capability)
        
        # Register callbacks
        for event, callbacks in self._callbacks.items():
            for callback in callbacks:
                agent._register_callback(event, callback)
        
        # Add custom components
        for name, component in self._custom_components.items():
            agent._add_custom_component(name, component)
        
        logger.info(f"Built voice agent: {self._metadata.name}")
        return agent
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert builder configuration to dictionary."""
        return {
            'metadata': {
                'name': self._metadata.name,
                'description': self._metadata.description,
                'version': self._metadata.version,
                'author': self._metadata.author,
                'tags': self._metadata.tags
            },
            'providers': {
                'stt': {
                    'provider': self._stt_provider,
                    'config': self._stt_config
                },
                'llm': {
                    'provider': self._llm_provider,
                    'config': self._llm_config
                },
                'tts': {
                    'provider': self._tts_provider,
                    'config': self._tts_config
                },
                'vad': {
                    'provider': self._vad_provider,
                    'config': self._vad_config
                }
            },
            'capabilities': [cap.value for cap in self._capabilities],
            'pipeline': self._pipeline_config.__dict__ if self._pipeline_config else None,
            'custom_components': list(self._custom_components.keys())
        }
    
    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        sdk_config: Optional[SDKConfig] = None,
        registry: Optional[ComponentRegistry] = None
    ) -> 'VoiceAgentBuilder':
        """Create builder from configuration dictionary."""
        builder = cls(config=sdk_config, registry=registry)
        
        # Set metadata
        metadata = config_dict.get('metadata', {})
        builder.with_name(
            metadata.get('name', 'VoiceAgent'),
            metadata.get('description')
        )
        builder.with_metadata(
            version=metadata.get('version'),
            author=metadata.get('author'),
            tags=metadata.get('tags')
        )
        
        # Configure providers
        providers = config_dict.get('providers', {})
        
        if 'stt' in providers and providers['stt']['provider']:
            builder.with_stt(
                providers['stt']['provider'],
                **(providers['stt'].get('config', {}) or {})
            )
        
        if 'llm' in providers and providers['llm']['provider']:
            builder.with_llm(
                providers['llm']['provider'],
                **(providers['llm'].get('config', {}) or {})
            )
        
        if 'tts' in providers and providers['tts']['provider']:
            builder.with_tts(
                providers['tts']['provider'],
                **(providers['tts'].get('config', {}) or {})
            )
        
        if 'vad' in providers and providers['vad']['provider']:
            builder.with_vad(
                providers['vad']['provider'],
                **(providers['vad'].get('config', {}) or {})
            )
        
        # Enable capabilities
        for capability_str in config_dict.get('capabilities', []):
            try:
                capability = AgentCapability(capability_str)
                builder.with_capability(capability)
            except ValueError:
                logger.warning(f"Unknown capability: {capability_str}")
        
        # Configure pipeline
        if 'pipeline' in config_dict and config_dict['pipeline']:
            builder.with_pipeline_config(**config_dict['pipeline'])
        
        return builder


class QuickBuilder:
    """
    Simplified builder for common agent configurations.
    
    Provides pre-configured templates for quick setup.
    """
    
    @staticmethod
    def customer_service_agent(
        name: str = "Customer Service Agent",
        language: str = "en"
    ) -> VoiceAgentBuilder:
        """Create a pre-configured customer service agent."""
        return (VoiceAgentBuilder()
            .with_name(name, "AI-powered customer service representative")
            .with_stt("openai", language=language)
            .with_llm("openai", model="gpt-4", temperature=0.7)
            .with_tts("elevenlabs", voice_id="21m00Tcm4TlvDq8ikWAM")
            .with_vad("silero")
            .with_capabilities(
                AgentCapability.TURN_DETECTION,
                AgentCapability.INTERRUPTION_HANDLING,
                AgentCapability.CONTEXT_MANAGEMENT
            )
            .with_system_prompt(
                "You are a helpful and professional customer service representative. "
                "Be polite, patient, and aim to resolve customer issues efficiently."
            ))
    
    @staticmethod
    def telehealth_agent(
        name: str = "Telehealth Assistant",
        language: str = "en"
    ) -> VoiceAgentBuilder:
        """Create a pre-configured telehealth agent."""
        return (VoiceAgentBuilder()
            .with_name(name, "Medical consultation voice assistant")
            .with_stt("azure", language=language, enable_medical_vocabulary=True)
            .with_llm("openai", model="gpt-4", temperature=0.3)
            .with_tts("azure", voice_name="en-US-JennyNeural")
            .with_vad("silero")
            .with_capabilities(
                AgentCapability.TURN_DETECTION,
                AgentCapability.INTERRUPTION_HANDLING,
                AgentCapability.CONTEXT_MANAGEMENT,
                AgentCapability.CONVERSATION_STATE
            )
            .with_system_prompt(
                "You are a medical consultation assistant. Be professional, "
                "empathetic, and always remind patients to consult with their "
                "healthcare provider for medical advice."
            ))
    
    @staticmethod
    def translator_agent(
        name: str = "Real-time Translator",
        source_language: str = "en",
        target_language: str = "es"
    ) -> VoiceAgentBuilder:
        """Create a pre-configured translation agent."""
        return (VoiceAgentBuilder()
            .with_name(name, f"Translates from {source_language} to {target_language}")
            .with_stt("google", language=source_language)
            .with_llm("openai", model="gpt-4", temperature=0.1)
            .with_tts("google", language=target_language)
            .with_vad("webrtc")
            .with_capability(AgentCapability.MULTI_LANGUAGE)
            .with_system_prompt(
                f"You are a real-time translator. Translate everything from "
                f"{source_language} to {target_language}. Maintain the tone "
                "and context of the original message."
            ))