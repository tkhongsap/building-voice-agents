"""
LiveKit Voice Agents Python SDK

This is the main entry point for the Voice Agents SDK, providing a high-level
interface for building and managing voice-enabled AI agents.
"""

import asyncio
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union
from pathlib import Path
import signal

from .exceptions import SDKNotInitializedError, ConfigurationError
from .config_manager import ConfigManager, SDKConfig
from .agent_builder import VoiceAgentBuilder, QuickBuilder, AgentCapability
from .utils import ComponentRegistry, SingletonMeta, create_task_with_logging

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    'VoiceAgentSDK',
    'VoiceAgent',
    'VoiceAgentBuilder',
    'QuickBuilder',
    'AgentCapability',
    'SDKConfig',
    'ConfigManager'
]


class VoiceAgent:
    """
    Represents a configured voice agent instance.
    
    This class manages the lifecycle of a voice agent including
    initialization, execution, and cleanup.
    """
    
    def __init__(
        self,
        metadata: 'AgentMetadata',
        config: SDKConfig,
        registry: ComponentRegistry
    ):
        self.metadata = metadata
        self._config = config
        self._registry = registry
        
        # Component instances
        self._stt_provider = None
        self._llm_provider = None
        self._tts_provider = None
        self._vad_provider = None
        self._pipeline = None
        
        # Capability handlers
        self._turn_detector = None
        self._interruption_handler = None
        self._context_manager = None
        self._state_manager = None
        
        # State
        self._is_initialized = False
        self._is_running = False
        self._tasks: List[asyncio.Task] = []
        
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
    
    async def initialize(self) -> None:
        """Initialize the voice agent and all its components."""
        if self._is_initialized:
            return
        
        try:
            logger.info(f"Initializing voice agent: {self.metadata.name}")
            
            # Initialize providers
            if self._stt_provider:
                await self._stt_provider.initialize()
            if self._llm_provider:
                await self._llm_provider.initialize()
            if self._tts_provider:
                await self._tts_provider.initialize()
            if self._vad_provider:
                await self._vad_provider.initialize()
            
            # Initialize pipeline
            if self._pipeline:
                await self._pipeline.initialize()
            
            # Initialize capability handlers
            if self._turn_detector:
                await self._turn_detector.initialize()
            if self._interruption_handler:
                await self._interruption_handler.initialize()
            if self._context_manager:
                await self._context_manager.initialize()
            if self._state_manager:
                await self._state_manager.initialize()
            
            self._is_initialized = True
            logger.info(f"Voice agent '{self.metadata.name}' initialized successfully")
            
            # Call on_start callbacks
            await self._trigger_callbacks('on_start')
            
        except Exception as e:
            logger.error(f"Failed to initialize voice agent: {e}")
            await self._trigger_callbacks('on_error', e)
            raise
    
    async def start(self) -> None:
        """Start the voice agent."""
        if not self._is_initialized:
            await self.initialize()
        
        if self._is_running:
            logger.warning("Voice agent is already running")
            return
        
        try:
            logger.info(f"Starting voice agent: {self.metadata.name}")
            self._is_running = True
            
            # Start pipeline
            if self._pipeline:
                pipeline_task = create_task_with_logging(
                    self._pipeline.run(),
                    name=f"{self.metadata.name}_pipeline",
                    error_handler=lambda e: self._trigger_callbacks('on_error', e)
                )
                self._tasks.append(pipeline_task)
            
            # Start capability handlers
            if self._turn_detector:
                td_task = create_task_with_logging(
                    self._turn_detector.start_monitoring(),
                    name=f"{self.metadata.name}_turn_detector"
                )
                self._tasks.append(td_task)
            
            if self._interruption_handler:
                ih_task = create_task_with_logging(
                    self._interruption_handler.start_monitoring(),
                    name=f"{self.metadata.name}_interruption_handler"
                )
                self._tasks.append(ih_task)
            
            if self._context_manager:
                cm_task = create_task_with_logging(
                    self._context_manager.start_monitoring(),
                    name=f"{self.metadata.name}_context_manager"
                )
                self._tasks.append(cm_task)
            
            logger.info(f"Voice agent '{self.metadata.name}' started")
            
        except Exception as e:
            logger.error(f"Failed to start voice agent: {e}")
            self._is_running = False
            await self._trigger_callbacks('on_error', e)
            raise
    
    async def stop(self) -> None:
        """Stop the voice agent."""
        if not self._is_running:
            return
        
        try:
            logger.info(f"Stopping voice agent: {self.metadata.name}")
            self._is_running = False
            
            # Cancel all tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            
            # Stop components
            if self._pipeline:
                await self._pipeline.stop()
            
            if self._turn_detector:
                await self._turn_detector.stop_monitoring()
            
            if self._interruption_handler:
                await self._interruption_handler.stop_monitoring()
            
            if self._context_manager:
                await self._context_manager.stop_monitoring()
            
            # Call on_stop callbacks
            await self._trigger_callbacks('on_stop')
            
            logger.info(f"Voice agent '{self.metadata.name}' stopped")
            
        except Exception as e:
            logger.error(f"Error stopping voice agent: {e}")
            await self._trigger_callbacks('on_error', e)
    
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        if self._is_running:
            await self.stop()
        
        # Cleanup providers
        if self._stt_provider:
            await self._stt_provider.cleanup()
        if self._llm_provider:
            await self._llm_provider.cleanup()
        if self._tts_provider:
            await self._tts_provider.cleanup()
        if self._vad_provider:
            await self._vad_provider.cleanup()
        
        # Cleanup capability handlers
        if self._turn_detector:
            await self._turn_detector.cleanup()
        if self._interruption_handler:
            await self._interruption_handler.cleanup()
        if self._context_manager:
            await self._context_manager.cleanup()
        
        self._is_initialized = False
        logger.info(f"Voice agent '{self.metadata.name}' cleaned up")
    
    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._is_running
    
    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._is_initialized
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name."""
        components = {
            'stt': self._stt_provider,
            'llm': self._llm_provider,
            'tts': self._tts_provider,
            'vad': self._vad_provider,
            'pipeline': self._pipeline,
            'turn_detector': self._turn_detector,
            'interruption_handler': self._interruption_handler,
            'context_manager': self._context_manager,
            'state_manager': self._state_manager
        }
        
        # Check built-in components
        if name in components:
            return components[name]
        
        # Check custom components
        return self._custom_components.get(name)
    
    async def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for an event."""
        callbacks = self._callbacks.get(event, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {event} callback: {e}", exc_info=True)
    
    # Internal configuration methods (called by builder)
    def _configure_stt(self, provider: str, config: Optional[Dict[str, Any]]) -> None:
        """Configure STT provider."""
        provider_class = self._registry.get('stt', provider)
        if not provider_class:
            raise ConfigurationError(f"STT provider '{provider}' not found")
        
        # Get provider-specific config class
        from ..components.stt.base_stt import STTConfig
        stt_config = STTConfig(**(config or {}))
        
        self._stt_provider = provider_class(stt_config)
        logger.debug(f"Configured STT provider: {provider}")
    
    def _configure_llm(self, provider: str, config: Optional[Dict[str, Any]]) -> None:
        """Configure LLM provider."""
        provider_class = self._registry.get('llm', provider)
        if not provider_class:
            raise ConfigurationError(f"LLM provider '{provider}' not found")
        
        # Provider-specific configuration
        if provider == 'openai':
            from ..components.llm.openai_llm import OpenAIConfig
            llm_config = OpenAIConfig(**(config or {}))
        elif provider == 'anthropic':
            from ..components.llm.anthropic_llm import AnthropicConfig
            llm_config = AnthropicConfig(**(config or {}))
        else:
            # Default config
            llm_config = config or {}
        
        self._llm_provider = provider_class(llm_config)
        logger.debug(f"Configured LLM provider: {provider}")
    
    def _configure_tts(self, provider: str, config: Optional[Dict[str, Any]]) -> None:
        """Configure TTS provider."""
        provider_class = self._registry.get('tts', provider)
        if not provider_class:
            raise ConfigurationError(f"TTS provider '{provider}' not found")
        
        from ..components.tts.base_tts import TTSConfig
        tts_config = TTSConfig(**(config or {}))
        
        self._tts_provider = provider_class(tts_config)
        logger.debug(f"Configured TTS provider: {provider}")
    
    def _configure_vad(self, provider: str, config: Optional[Dict[str, Any]]) -> None:
        """Configure VAD provider."""
        provider_class = self._registry.get('vad', provider)
        if not provider_class:
            raise ConfigurationError(f"VAD provider '{provider}' not found")
        
        from ..components.vad.base_vad import VADConfig
        vad_config = VADConfig(**(config or {}))
        
        self._vad_provider = provider_class(vad_config)
        logger.debug(f"Configured VAD provider: {provider}")
    
    def _configure_pipeline(self, config: 'PipelineConfig') -> None:
        """Configure audio pipeline."""
        from ..pipeline.audio_pipeline import AudioPipeline
        
        self._pipeline = AudioPipeline(
            config=config,
            stt_provider=self._stt_provider,
            llm_provider=self._llm_provider,
            tts_provider=self._tts_provider,
            vad_provider=self._vad_provider
        )
        logger.debug("Configured audio pipeline")
    
    def _enable_capability(self, capability: AgentCapability) -> None:
        """Enable a specific capability."""
        if capability == AgentCapability.TURN_DETECTION:
            from ..conversation.turn_detector import TurnDetector, TurnDetectionConfig
            self._turn_detector = TurnDetector(
                config=TurnDetectionConfig(),
                vad_provider=self._vad_provider,
                stt_provider=self._stt_provider
            )
        
        elif capability == AgentCapability.INTERRUPTION_HANDLING:
            from ..conversation.interruption_handler import InterruptionHandler, InterruptionHandlerConfig
            self._interruption_handler = InterruptionHandler(
                config=InterruptionHandlerConfig()
            )
        
        elif capability == AgentCapability.CONTEXT_MANAGEMENT:
            from ..conversation.context_manager import ConversationContextManager, ContextManagerConfig
            self._context_manager = ConversationContextManager(
                conversation_id=self.metadata.name,
                config=ContextManagerConfig()
            )
        
        elif capability == AgentCapability.CONVERSATION_STATE:
            from ..conversation.conversation_state_manager import ConversationStateManager, ConversationStateConfig
            self._state_manager = ConversationStateManager(
                conversation_id=self.metadata.name,
                config=ConversationStateConfig()
            )
        
        logger.debug(f"Enabled capability: {capability.value}")
    
    def _register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for event: {event}")
    
    def _add_custom_component(self, name: str, component: Any) -> None:
        """Add a custom component."""
        self._custom_components[name] = component
        logger.debug(f"Added custom component: {name}")


class VoiceAgentSDK(metaclass=SingletonMeta):
    """
    Main SDK class for managing voice agents.
    
    This is a singleton that provides global SDK functionality including
    component registration, configuration management, and agent lifecycle.
    """
    
    def __init__(self):
        self._initialized = False
        self._config_manager: Optional[ConfigManager] = None
        self._registry: Optional[ComponentRegistry] = None
        self._agents: Dict[str, VoiceAgent] = {}
        self._shutdown_handler_registered = False
    
    async def initialize(
        self,
        config: Optional[Union[SDKConfig, Dict[str, Any], str, Path]] = None,
        auto_discover: bool = True
    ) -> None:
        """
        Initialize the SDK.
        
        Args:
            config: Configuration (SDKConfig, dict, file path, or None for defaults)
            auto_discover: Whether to auto-discover available components
        """
        if self._initialized:
            logger.warning("SDK already initialized")
            return
        
        try:
            # Initialize configuration
            self._config_manager = ConfigManager(config)
            
            # Validate configuration
            errors = self._config_manager.validate()
            if errors:
                raise ConfigurationError(
                    f"Configuration validation failed: {'; '.join(errors)}"
                )
            
            # Initialize component registry
            self._registry = ComponentRegistry()
            
            # Auto-discover components
            if auto_discover and self._config_manager.config.enable_auto_discovery:
                await self._discover_components()
            
            # Register shutdown handler
            if not self._shutdown_handler_registered:
                self._register_shutdown_handler()
            
            self._initialized = True
            logger.info(f"Voice Agent SDK v{__version__} initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SDK: {e}")
            raise
    
    @property
    def is_initialized(self) -> bool:
        """Check if SDK is initialized."""
        return self._initialized
    
    @property
    def config(self) -> SDKConfig:
        """Get SDK configuration."""
        if not self._config_manager:
            raise SDKNotInitializedError()
        return self._config_manager.config
    
    @property
    def config_manager(self) -> ConfigManager:
        """Get configuration manager."""
        if not self._config_manager:
            raise SDKNotInitializedError()
        return self._config_manager
    
    @property
    def registry(self) -> ComponentRegistry:
        """Get component registry."""
        if not self._registry:
            raise SDKNotInitializedError()
        return self._registry
    
    def create_builder(self) -> VoiceAgentBuilder:
        """Create a new agent builder."""
        if not self._initialized:
            raise SDKNotInitializedError()
        
        return VoiceAgentBuilder(
            config=self.config,
            registry=self.registry
        )
    
    def get_agent(self, name: str) -> Optional[VoiceAgent]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())
    
    def register_agent(self, agent: VoiceAgent) -> None:
        """Register an agent with the SDK."""
        self._agents[agent.metadata.name] = agent
        logger.info(f"Registered agent: {agent.metadata.name}")
    
    def unregister_agent(self, name: str) -> None:
        """Unregister an agent."""
        if name in self._agents:
            del self._agents[name]
            logger.info(f"Unregistered agent: {name}")
    
    def register_component(
        self,
        component_type: str,
        name: str,
        component_class: Type
    ) -> None:
        """
        Register a custom component.
        
        Args:
            component_type: Type of component ('stt', 'llm', 'tts', 'vad')
            name: Component name
            component_class: Component class
        """
        if not self._initialized:
            raise SDKNotInitializedError()
        
        self.registry.register(component_type, name, component_class)
    
    async def _discover_components(self) -> None:
        """Discover and register available components."""
        logger.info("Discovering available components...")
        
        # Get base path
        base_path = Path(__file__).parent.parent
        
        # Discover built-in components
        self.registry.discover_and_register(str(base_path))
        
        # Log discovered components
        components = self.registry.list()
        for comp_type, providers in components.items():
            if providers:
                logger.info(f"Found {comp_type} providers: {list(providers.keys())}")
    
    def _register_shutdown_handler(self) -> None:
        """Register shutdown signal handlers."""
        async def shutdown():
            logger.info("Shutting down Voice Agent SDK...")
            
            # Stop all agents
            for agent in self._agents.values():
                if agent.is_running:
                    await agent.stop()
            
            # Cleanup
            for agent in self._agents.values():
                await agent.cleanup()
            
            logger.info("SDK shutdown complete")
        
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._shutdown_handler_registered = True
    
    async def cleanup(self) -> None:
        """Clean up SDK resources."""
        # Stop and cleanup all agents
        for agent in self._agents.values():
            if agent.is_running:
                await agent.stop()
            await agent.cleanup()
        
        self._agents.clear()
        self._initialized = False
        logger.info("SDK cleanup complete")


# Global SDK instance
sdk = VoiceAgentSDK()


async def initialize_sdk(
    config: Optional[Union[SDKConfig, Dict[str, Any], str, Path]] = None,
    auto_discover: bool = True
) -> VoiceAgentSDK:
    """
    Initialize and return the global SDK instance.
    
    This is a convenience function for quick SDK initialization.
    """
    await sdk.initialize(config, auto_discover)
    return sdk