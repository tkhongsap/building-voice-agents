#!/usr/bin/env python3
"""
Local Development Tools with Hot Reloading

This module provides development tools for building and testing voice agents
locally with hot reloading, live debugging, and rapid iteration capabilities.

Features:
- Hot reloading of agent configurations
- Live code updates without restart
- Real-time conversation debugging
- Configuration validation
- Performance monitoring
- Local testing environment

Usage:
    # Start development server
    python cli/dev_tools.py dev --config agent_config.yaml --port 8080
    
    # Watch for changes
    python cli/dev_tools.py watch --directory ./src/agents
    
    # Validate configuration
    python cli/dev_tools.py validate --config agent_config.yaml
"""

import asyncio
import json
import os
import sys
import time
import argparse
import logging
import importlib
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk.python_sdk import VoiceAgentSDK, initialize_sdk
from sdk.agent_builder import VoiceAgentBuilder
from sdk.config_manager import ConfigManager, SDKConfig
from sdk.exceptions import ConfigurationError, VoiceAgentError


@dataclass
class DevServerConfig:
    """Development server configuration."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = True
    hot_reload: bool = True
    auto_restart: bool = True
    watch_directories: List[str] = field(default_factory=lambda: ["src", "configs"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*.pyc", "__pycache__", ".git"])
    reload_delay: float = 1.0  # seconds


@dataclass
class WatchEvent:
    """File system watch event."""
    event_type: str  # created, modified, deleted, moved
    file_path: str
    timestamp: datetime
    is_directory: bool


class ConfigWatcher(FileSystemEventHandler):
    """
    File system watcher for configuration changes.
    
    Monitors configuration files and triggers hot reloads when changes are detected.
    """
    
    def __init__(self, callback: Callable[[WatchEvent], None], exclude_patterns: List[str] = None):
        super().__init__()
        self.callback = callback
        self.exclude_patterns = exclude_patterns or []
        self.last_modified = {}
        self.debounce_delay = 0.5  # seconds
    
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored based on patterns."""
        for pattern in self.exclude_patterns:
            if pattern.replace("*", "") in file_path:
                return True
        return False
    
    def on_modified(self, event):
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        # Debounce rapid file changes
        now = time.time()
        if event.src_path in self.last_modified:
            if now - self.last_modified[event.src_path] < self.debounce_delay:
                return
        
        self.last_modified[event.src_path] = now
        
        watch_event = WatchEvent(
            event_type="modified",
            file_path=event.src_path,
            timestamp=datetime.now(),
            is_directory=event.is_directory
        )
        
        self.callback(watch_event)
    
    def on_created(self, event):
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        watch_event = WatchEvent(
            event_type="created",
            file_path=event.src_path,
            timestamp=datetime.now(),
            is_directory=event.is_directory
        )
        
        self.callback(watch_event)


class HotReloader:
    """
    Hot reloading system for voice agents.
    
    Automatically reloads agent configurations and code without
    stopping the development server.
    """
    
    def __init__(self, dev_server: 'DevServer'):
        self.dev_server = dev_server
        self.reload_queue: asyncio.Queue = asyncio.Queue()
        self.is_reloading = False
        self.reload_count = 0
    
    async def handle_reload(self, event: WatchEvent) -> None:
        """Handle a reload request."""
        if self.is_reloading:
            print(f"⏳ Reload in progress, queuing: {event.file_path}")
            await self.reload_queue.put(event)
            return
        
        self.is_reloading = True
        
        try:
            print(f"🔄 Hot reloading: {event.file_path}")
            
            # Determine reload type
            if event.file_path.endswith(('.yaml', '.yml', '.json')):
                await self._reload_config(event.file_path)
            elif event.file_path.endswith('.py'):
                await self._reload_code(event.file_path)
            else:
                print(f"⚠️ Unknown file type for reload: {event.file_path}")
            
            self.reload_count += 1
            print(f"✅ Hot reload #{self.reload_count} completed")
            
        except Exception as e:
            print(f"❌ Hot reload failed: {e}")
            logging.exception("Hot reload error")
        
        finally:
            self.is_reloading = False
            
            # Process queued reloads
            if not self.reload_queue.empty():
                next_event = await self.reload_queue.get()
                await self.handle_reload(next_event)
    
    async def _reload_config(self, config_path: str) -> None:
        """Reload agent configuration."""
        try:
            # Validate configuration first
            validator = ConfigValidator()
            is_valid, errors = await validator.validate_file(config_path)
            
            if not is_valid:
                print(f"❌ Invalid configuration: {errors}")
                return
            
            # Load new configuration
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    new_config = json.load(f)
                else:
                    new_config = yaml.safe_load(f)
            
            # Apply to dev server
            await self.dev_server.update_config(new_config)
            print(f"✅ Configuration reloaded: {config_path}")
            
        except Exception as e:
            print(f"❌ Config reload failed: {e}")
            raise
    
    async def _reload_code(self, code_path: str) -> None:
        """Reload Python code modules."""
        try:
            # Convert file path to module name
            relative_path = Path(code_path).relative_to(Path.cwd())
            module_parts = relative_path.with_suffix('').parts
            
            # Skip if not in src directory
            if not module_parts or module_parts[0] != 'src':
                return
            
            module_name = '.'.join(module_parts[1:])  # Remove 'src' prefix
            
            # Reload module
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                print(f"✅ Module reloaded: {module_name}")
            
            # Restart agent if needed
            if self.dev_server.auto_restart:
                await self.dev_server.restart_agent()
            
        except Exception as e:
            print(f"❌ Code reload failed: {e}")
            raise


class ConfigValidator:
    """
    Configuration validator for voice agent configs.
    
    Validates agent configurations before applying them to catch
    errors early in development.
    """
    
    def __init__(self):
        self.required_fields = {
            "agent": ["name"],
            "providers": {
                "stt": ["provider"],
                "llm": ["provider", "model"],
                "tts": ["provider"],
                "vad": ["provider"]
            }
        }
    
    async def validate_file(self, config_path: str) -> tuple[bool, List[str]]:
        """Validate a configuration file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            return await self.validate_config(config)
            
        except Exception as e:
            return False, [f"Failed to load config file: {e}"]
    
    async def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a configuration dictionary."""
        errors = []
        
        # Check required top-level fields
        for field in self.required_fields.get("agent", []):
            if field not in config.get("agent", {}):
                errors.append(f"Missing required field: agent.{field}")
        
        # Check provider configurations
        providers = config.get("providers", {})
        for provider_type, requirements in self.required_fields["providers"].items():
            if provider_type in providers:
                provider_config = providers[provider_type]
                for req_field in requirements:
                    if req_field not in provider_config:
                        errors.append(f"Missing required field: providers.{provider_type}.{req_field}")
        
        # Validate provider availability
        available_providers = await self._get_available_providers()
        for provider_type, provider_config in providers.items():
            provider_name = provider_config.get("provider")
            if provider_name and provider_name not in available_providers.get(provider_type, []):
                errors.append(f"Unknown {provider_type} provider: {provider_name}")
        
        # Validate model names
        if "llm" in providers:
            llm_provider = providers["llm"].get("provider")
            llm_model = providers["llm"].get("model")
            if llm_provider and llm_model:
                valid_models = await self._get_valid_models(llm_provider)
                if valid_models and llm_model not in valid_models:
                    errors.append(f"Invalid model '{llm_model}' for provider '{llm_provider}'")
        
        return len(errors) == 0, errors
    
    async def _get_available_providers(self) -> Dict[str, List[str]]:
        """Get list of available providers by type."""
        # In a real implementation, this would query the registry
        return {
            "stt": ["openai", "azure", "google"],
            "llm": ["openai", "anthropic", "google", "local"],
            "tts": ["openai", "elevenlabs", "azure"],
            "vad": ["silero", "webrtc"]
        }
    
    async def _get_valid_models(self, provider: str) -> Optional[List[str]]:
        """Get valid models for a provider."""
        model_mappings = {
            "openai": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "google": ["gemini-pro", "gemini-pro-vision"]
        }
        return model_mappings.get(provider)


class ConversationDebugger:
    """
    Real-time conversation debugging tools.
    
    Provides detailed insights into conversation flow, timing,
    and component performance during development.
    """
    
    def __init__(self):
        self.conversation_log: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "stt_latency": [],
            "llm_latency": [],
            "tts_latency": [],
            "total_latency": []
        }
        self.debug_callbacks: List[Callable] = []
    
    def log_conversation_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a conversation event for debugging."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        self.conversation_log.append(event)
        
        # Call debug callbacks
        for callback in self.debug_callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.error(f"Debug callback error: {e}")
        
        # Print real-time debug info
        self._print_debug_event(event)
    
    def _print_debug_event(self, event: Dict[str, Any]) -> None:
        """Print debug event to console."""
        timestamp = event["timestamp"]
        event_type = event["type"]
        data = event["data"]
        
        if event_type == "user_speech":
            print(f"🎤 [{timestamp}] User: {data.get('text', '')}")
            if "confidence" in data:
                print(f"   Confidence: {data['confidence']:.2f}")
        
        elif event_type == "agent_response":
            print(f"🤖 [{timestamp}] Agent: {data.get('text', '')}")
            if "latency" in data:
                print(f"   Latency: {data['latency']:.2f}s")
        
        elif event_type == "component_timing":
            component = data.get("component", "unknown")
            latency = data.get("latency", 0)
            print(f"⏱️ [{timestamp}] {component}: {latency:.2f}s")
            
            # Track performance metrics
            if f"{component}_latency" in self.performance_metrics:
                self.performance_metrics[f"{component}_latency"].append(latency)
        
        elif event_type == "error":
            error_msg = data.get("error", "Unknown error")
            print(f"❌ [{timestamp}] Error: {error_msg}")
        
        elif event_type == "turn_detected":
            print(f"🔄 [{timestamp}] Turn detected")
            if "confidence" in data:
                print(f"   Turn confidence: {data['confidence']:.2f}")
    
    def add_debug_callback(self, callback: Callable) -> None:
        """Add a debug callback function."""
        self.debug_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics summary."""
        summary = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
            else:
                summary[metric_name] = {"avg": 0, "min": 0, "max": 0, "count": 0}
        
        return summary
    
    def export_conversation_log(self, file_path: str) -> None:
        """Export conversation log to file."""
        with open(file_path, 'w') as f:
            json.dump(self.conversation_log, f, indent=2)
        
        print(f"📄 Conversation log exported: {file_path}")


class DevServer:
    """
    Development server for voice agents.
    
    Provides a local development environment with hot reloading,
    debugging tools, and rapid iteration capabilities.
    """
    
    def __init__(self, config: DevServerConfig):
        self.config = config
        self.agent = None
        self.sdk = None
        self.is_running = False
        self.auto_restart = config.auto_restart
        
        # Development tools
        self.hot_reloader = HotReloader(self) if config.hot_reload else None
        self.config_validator = ConfigValidator()
        self.debugger = ConversationDebugger()
        
        # File watching
        self.observer = Observer()
        self.watchers: List[ConfigWatcher] = []
        
        # Current configuration
        self.current_config: Optional[Dict[str, Any]] = None
    
    async def start(self, agent_config_path: Optional[str] = None) -> None:
        """Start the development server."""
        print(f"🚀 Starting Voice Agent Development Server")
        print(f"   Host: {self.config.host}:{self.config.port}")
        print(f"   Hot Reload: {'Enabled' if self.config.hot_reload else 'Disabled'}")
        print(f"   Auto Restart: {'Enabled' if self.auto_restart else 'Disabled'}")
        
        try:
            # Initialize SDK
            self.sdk = await initialize_sdk({
                "project_name": "dev_server",
                "environment": "development",
                "enable_debug_logging": True
            })
            
            # Load initial configuration
            if agent_config_path:
                await self.load_config(agent_config_path)
            
            # Start file watching
            if self.config.hot_reload:
                self._start_file_watching()
            
            # Start agent if config is loaded
            if self.current_config:
                await self.start_agent()
            
            self.is_running = True
            print(f"✅ Development server started successfully")
            
            # Keep server running
            await self._run_server_loop()
            
        except Exception as e:
            print(f"❌ Failed to start development server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the development server."""
        print("🛑 Stopping development server...")
        
        self.is_running = False
        
        # Stop agent
        if self.agent:
            await self.agent.stop()
            await self.agent.cleanup()
        
        # Stop file watching
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        
        print("✅ Development server stopped")
    
    async def load_config(self, config_path: str) -> None:
        """Load agent configuration from file."""
        print(f"📋 Loading configuration: {config_path}")
        
        # Validate configuration
        is_valid, errors = await self.config_validator.validate_file(config_path)
        if not is_valid:
            raise ConfigurationError(f"Invalid configuration: {errors}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                self.current_config = json.load(f)
            else:
                self.current_config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded successfully")
    
    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update agent configuration with hot reload."""
        print("🔄 Updating configuration...")
        
        # Validate new configuration
        is_valid, errors = await self.config_validator.validate_config(new_config)
        if not is_valid:
            raise ConfigurationError(f"Invalid configuration: {errors}")
        
        self.current_config = new_config
        
        # Restart agent with new config
        if self.agent and self.auto_restart:
            await self.restart_agent()
        
        print("✅ Configuration updated")
    
    async def start_agent(self) -> None:
        """Start the voice agent with current configuration."""
        if not self.current_config:
            raise ValueError("No configuration loaded")
        
        print("🤖 Starting voice agent...")
        
        try:
            # Build agent from configuration
            builder = self.sdk.create_builder()
            
            # Apply configuration
            agent_config = self.current_config.get("agent", {})
            providers_config = self.current_config.get("providers", {})
            
            # Set agent name
            if "name" in agent_config:
                builder = builder.with_name(agent_config["name"])
            
            # Configure providers
            if "stt" in providers_config:
                stt_config = providers_config["stt"]
                builder = builder.with_stt(
                    stt_config["provider"],
                    **{k: v for k, v in stt_config.items() if k != "provider"}
                )
            
            if "llm" in providers_config:
                llm_config = providers_config["llm"]
                builder = builder.with_llm(
                    llm_config["provider"],
                    **{k: v for k, v in llm_config.items() if k != "provider"}
                )
            
            if "tts" in providers_config:
                tts_config = providers_config["tts"]
                builder = builder.with_tts(
                    tts_config["provider"],
                    **{k: v for k, v in tts_config.items() if k != "provider"}
                )
            
            if "vad" in providers_config:
                vad_config = providers_config["vad"]
                builder = builder.with_vad(
                    vad_config["provider"],
                    **{k: v for k, v in vad_config.items() if k != "provider"}
                )
            
            # Add debug callbacks
            builder = (builder
                .with_callback("on_user_speech", self._on_user_speech)
                .with_callback("on_agent_speech", self._on_agent_speech)
                .with_callback("on_error", self._on_error)
            )
            
            # Build and start agent
            self.agent = builder.build()
            await self.agent.start()
            
            print("✅ Voice agent started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start voice agent: {e}")
            raise
    
    async def restart_agent(self) -> None:
        """Restart the voice agent."""
        print("🔄 Restarting voice agent...")
        
        # Stop current agent
        if self.agent:
            await self.agent.stop()
            await self.agent.cleanup()
            self.agent = None
        
        # Start new agent
        await self.start_agent()
        
        print("✅ Voice agent restarted")
    
    def _start_file_watching(self) -> None:
        """Start file system watching for hot reload."""
        print("👀 Starting file watching...")
        
        for directory in self.config.watch_directories:
            if os.path.exists(directory):
                watcher = ConfigWatcher(
                    callback=self._on_file_change,
                    exclude_patterns=self.config.exclude_patterns
                )
                self.watchers.append(watcher)
                self.observer.schedule(watcher, directory, recursive=True)
                print(f"   Watching: {directory}")
        
        self.observer.start()
        print("✅ File watching started")
    
    def _on_file_change(self, event: WatchEvent) -> None:
        """Handle file system changes."""
        if self.hot_reloader:
            # Schedule reload in the event loop
            asyncio.create_task(self.hot_reloader.handle_reload(event))
    
    async def _run_server_loop(self) -> None:
        """Main server loop."""
        try:
            while self.is_running:
                # Print status every 30 seconds
                await asyncio.sleep(30)
                if self.is_running:
                    self._print_status()
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        finally:
            await self.stop()
    
    def _print_status(self) -> None:
        """Print current server status."""
        status = "🟢 Running" if self.agent and self.agent.is_running else "🔴 Stopped"
        reload_count = self.hot_reloader.reload_count if self.hot_reloader else 0
        
        print(f"📊 Status: {status} | Hot reloads: {reload_count}")
        
        # Print performance summary
        if self.debugger:
            perf_summary = self.debugger.get_performance_summary()
            if perf_summary.get("total_latency", {}).get("count", 0) > 0:
                avg_latency = perf_summary["total_latency"]["avg"]
                print(f"   Average latency: {avg_latency:.2f}s")
    
    async def _on_user_speech(self, text: str, **kwargs) -> None:
        """Handle user speech event."""
        self.debugger.log_conversation_event("user_speech", {
            "text": text,
            "confidence": kwargs.get("confidence"),
            "duration": kwargs.get("duration")
        })
    
    async def _on_agent_speech(self, text: str, **kwargs) -> None:
        """Handle agent speech event."""
        self.debugger.log_conversation_event("agent_response", {
            "text": text,
            "latency": kwargs.get("latency"),
            "voice": kwargs.get("voice")
        })
    
    async def _on_error(self, error: Exception, **kwargs) -> None:
        """Handle error event."""
        self.debugger.log_conversation_event("error", {
            "error": str(error),
            "type": type(error).__name__,
            "component": kwargs.get("component")
        })


# CLI Commands

async def cmd_dev(args) -> None:
    """Start development server with hot reloading."""
    config = DevServerConfig(
        host=args.host,
        port=args.port,
        hot_reload=not args.no_hot_reload,
        auto_restart=not args.no_auto_restart,
        watch_directories=args.watch_dirs,
        reload_delay=args.reload_delay
    )
    
    server = DevServer(config)
    await server.start(args.config)


async def cmd_validate(args) -> None:
    """Validate agent configuration."""
    print(f"🔍 Validating configuration: {args.config}")
    
    validator = ConfigValidator()
    is_valid, errors = await validator.validate_file(args.config)
    
    if is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)


async def cmd_watch(args) -> None:
    """Watch directories for changes."""
    print(f"👀 Watching directories: {args.directories}")
    
    def on_change(event: WatchEvent):
        print(f"📝 {event.event_type}: {event.file_path}")
    
    observer = Observer()
    
    for directory in args.directories:
        if os.path.exists(directory):
            watcher = ConfigWatcher(callback=on_change)
            observer.schedule(watcher, directory, recursive=True)
            print(f"   Watching: {directory}")
        else:
            print(f"⚠️ Directory not found: {directory}")
    
    observer.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...")
        observer.stop()
    
    observer.join()
    print("✅ File watcher stopped")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Voice Agent Development Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dev server command
    dev_parser = subparsers.add_parser("dev", help="Start development server")
    dev_parser.add_argument("--config", type=str, help="Agent configuration file")
    dev_parser.add_argument("--host", default="localhost", help="Server host")
    dev_parser.add_argument("--port", type=int, default=8080, help="Server port")
    dev_parser.add_argument("--no-hot-reload", action="store_true", help="Disable hot reloading")
    dev_parser.add_argument("--no-auto-restart", action="store_true", help="Disable auto restart")
    dev_parser.add_argument("--watch-dirs", nargs="+", default=["src", "configs"], help="Directories to watch")
    dev_parser.add_argument("--reload-delay", type=float, default=1.0, help="Reload delay in seconds")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config", help="Configuration file to validate")
    
    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch directories for changes")
    watch_parser.add_argument("directories", nargs="+", help="Directories to watch")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run command
    try:
        if args.command == "dev":
            asyncio.run(cmd_dev(args))
        elif args.command == "validate":
            asyncio.run(cmd_validate(args))
        elif args.command == "watch":
            asyncio.run(cmd_watch(args))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()