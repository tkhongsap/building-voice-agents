"""
Logging Configuration and Setup Utilities

This module provides configuration management and setup utilities for the
structured logging system across the voice agents platform.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
from dataclasses import dataclass

from .structured_logging import StructuredJSONFormatter, CorrelationIdManager


class LogFormat(Enum):
    """Available log formats."""
    JSON = "json"
    PLAIN = "plain"
    COLORED = "colored"


class LogOutput(Enum):
    """Available log outputs."""
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"
    SYSLOG = "syslog"


@dataclass
class LoggingConfig:
    """Configuration for logging setup."""
    level: str = "INFO"
    format_type: LogFormat = LogFormat.JSON
    output: LogOutput = LogOutput.CONSOLE
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_traceback: bool = True
    include_extra: bool = True
    correlation_tracking: bool = True
    performance_tracking: bool = True
    console_colors: bool = True
    filter_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    structured_fields: Optional[Dict[str, Any]] = None
    syslog_address: str = "localhost:514"
    syslog_facility: str = "local0"


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        if not hasattr(record, 'color'):
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            
            # Format the message
            formatted = super().format(record)
            
            # Add color to level name
            formatted = formatted.replace(
                record.levelname,
                f"{color}{record.levelname}{reset}"
            )
            
            return formatted
        
        return super().format(record)


class StructuredLoggingSetup:
    """Setup utility for structured logging configuration."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self._setup_trace_level()
    
    def _setup_trace_level(self):
        """Add TRACE level to logging."""
        logging.TRACE = 5
        logging.addLevelName(logging.TRACE, "TRACE")
        
        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.TRACE):
                self._log(logging.TRACE, message, args, **kwargs)
        
        logging.Logger.trace = trace
    
    def setup_logging(self) -> Dict[str, Any]:
        """Setup logging configuration."""
        # Base configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {},
            'handlers': {},
            'loggers': {},
            'root': {
                'level': self.config.level,
                'handlers': []
            }
        }
        
        # Setup formatters
        self._setup_formatters(config)
        
        # Setup handlers
        self._setup_handlers(config)
        
        # Setup filters
        self._setup_filters(config)
        
        # Setup loggers
        self._setup_loggers(config)
        
        # Apply configuration
        logging.config.dictConfig(config)
        
        # Set correlation tracking
        if self.config.correlation_tracking:
            self._enable_correlation_tracking()
        
        return config
    
    def _setup_formatters(self, config: Dict[str, Any]):
        """Setup log formatters."""
        if self.config.format_type == LogFormat.JSON:
            config['formatters']['json'] = {
                '()': StructuredJSONFormatter,
                'include_extra': self.config.include_extra,
                'include_traceback': self.config.include_traceback,
                'sort_keys': True
            }
        
        if self.config.format_type == LogFormat.COLORED and self.config.console_colors:
            config['formatters']['colored'] = {
                '()': ColoredFormatter,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        
        config['formatters']['plain'] = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    
    def _setup_handlers(self, config: Dict[str, Any]):
        """Setup log handlers."""
        handlers = []
        
        # Console handler
        if self.config.output in [LogOutput.CONSOLE, LogOutput.BOTH]:
            formatter = self._get_console_formatter()
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': self.config.level,
                'formatter': formatter,
                'stream': 'ext://sys.stdout'
            }
            handlers.append('console')
        
        # File handler
        if self.config.output in [LogOutput.FILE, LogOutput.BOTH]:
            if not self.config.log_file:
                # Default log file path
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                self.config.log_file = str(log_dir / "voice_agents.log")
            
            # Ensure log directory exists
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': self.config.level,
                'formatter': 'json' if self.config.format_type == LogFormat.JSON else 'plain',
                'filename': self.config.log_file,
                'maxBytes': self.config.max_file_size,
                'backupCount': self.config.backup_count,
                'encoding': 'utf-8'
            }
            handlers.append('file')
        
        # Syslog handler
        if self.config.output == LogOutput.SYSLOG:
            config['handlers']['syslog'] = {
                'class': 'logging.handlers.SysLogHandler',
                'level': self.config.level,
                'formatter': 'json' if self.config.format_type == LogFormat.JSON else 'plain',
                'address': self._parse_syslog_address(),
                'facility': self.config.syslog_facility
            }
            handlers.append('syslog')
        
        config['root']['handlers'] = handlers
    
    def _get_console_formatter(self) -> str:
        """Get appropriate console formatter."""
        if self.config.format_type == LogFormat.JSON:
            return 'json'
        elif self.config.format_type == LogFormat.COLORED and self.config.console_colors:
            return 'colored'
        else:
            return 'plain'
    
    def _parse_syslog_address(self) -> Union[str, tuple]:
        """Parse syslog address."""
        if ':' in self.config.syslog_address:
            host, port = self.config.syslog_address.split(':', 1)
            return (host, int(port))
        return self.config.syslog_address
    
    def _setup_filters(self, config: Dict[str, Any]):
        """Setup log filters."""
        if self.config.filter_modules or self.config.exclude_modules:
            config['filters'] = {}
            
            if self.config.filter_modules:
                config['filters']['module_filter'] = {
                    '()': ModuleFilter,
                    'modules': self.config.filter_modules,
                    'include': True
                }
            
            if self.config.exclude_modules:
                config['filters']['module_exclude'] = {
                    '()': ModuleFilter,
                    'modules': self.config.exclude_modules,
                    'include': False
                }
            
            # Apply filters to handlers
            for handler_name in config.get('handlers', {}):
                handler = config['handlers'][handler_name]
                if 'filters' not in handler:
                    handler['filters'] = []
                
                if self.config.filter_modules:
                    handler['filters'].append('module_filter')
                if self.config.exclude_modules:
                    handler['filters'].append('module_exclude')
    
    def _setup_loggers(self, config: Dict[str, Any]):
        """Setup specific loggers."""
        # Voice agent component loggers
        component_loggers = {
            'voice_agents.stt': {'level': self.config.level},
            'voice_agents.llm': {'level': self.config.level},
            'voice_agents.tts': {'level': self.config.level},
            'voice_agents.vad': {'level': self.config.level},
            'voice_agents.pipeline': {'level': self.config.level},
            'voice_agents.api': {'level': self.config.level},
            'voice_agents.monitoring': {'level': self.config.level},
        }
        
        config['loggers'].update(component_loggers)
        
        # Third-party library loggers (reduce noise)
        third_party_loggers = {
            'urllib3': {'level': 'WARNING'},
            'requests': {'level': 'WARNING'},
            'websockets': {'level': 'INFO'},
            'asyncio': {'level': 'WARNING'},
            'livekit': {'level': 'INFO'},
        }
        
        config['loggers'].update(third_party_loggers)
    
    def _enable_correlation_tracking(self):
        """Enable correlation ID tracking."""
        # This would be handled by the StructuredLogger and decorators
        pass


class ModuleFilter(logging.Filter):
    """Filter to include/exclude specific modules."""
    
    def __init__(self, modules: List[str], include: bool = True):
        super().__init__()
        self.modules = modules
        self.include = include
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on module."""
        module_match = any(
            record.name.startswith(module) for module in self.modules
        )
        
        return module_match if self.include else not module_match


class LoggingManager:
    """Manager for logging configuration and lifecycle."""
    
    def __init__(self):
        self.config: Optional[LoggingConfig] = None
        self.setup: Optional[StructuredLoggingSetup] = None
        self._is_configured = False
    
    def configure(
        self, 
        config: Optional[LoggingConfig] = None,
        **kwargs
    ) -> LoggingConfig:
        """Configure logging with provided or default configuration."""
        if config is None:
            config = self._create_default_config(**kwargs)
        
        self.config = config
        self.setup = StructuredLoggingSetup(config)
        
        # Apply configuration
        log_config = self.setup.setup_logging()
        self._is_configured = True
        
        # Log configuration success
        logger = logging.getLogger(__name__)
        logger.info(
            "Structured logging configured successfully",
            extra={
                'extra_data': {
                    'format': config.format_type.value,
                    'output': config.output.value,
                    'level': config.level,
                    'correlation_tracking': config.correlation_tracking
                }
            }
        )
        
        return config
    
    def _create_default_config(self, **kwargs) -> LoggingConfig:
        """Create default logging configuration."""
        # Get configuration from environment variables
        env_config = {
            'level': os.getenv('LOG_LEVEL', 'INFO').upper(),
            'format_type': LogFormat(os.getenv('LOG_FORMAT', 'json').lower()),
            'output': LogOutput(os.getenv('LOG_OUTPUT', 'console').lower()),
            'log_file': os.getenv('LOG_FILE'),
            'correlation_tracking': os.getenv('LOG_CORRELATION_TRACKING', 'true').lower() == 'true',
            'performance_tracking': os.getenv('LOG_PERFORMANCE_TRACKING', 'true').lower() == 'true',
            'console_colors': os.getenv('LOG_CONSOLE_COLORS', 'true').lower() == 'true',
        }
        
        # Override with provided kwargs
        env_config.update(kwargs)
        
        return LoggingConfig(**env_config)
    
    def configure_from_file(self, config_file: Union[str, Path]) -> LoggingConfig:
        """Configure logging from JSON configuration file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Convert format and output strings to enums
        if 'format_type' in config_data:
            config_data['format_type'] = LogFormat(config_data['format_type'])
        if 'output' in config_data:
            config_data['output'] = LogOutput(config_data['output'])
        
        config = LoggingConfig(**config_data)
        return self.configure(config)
    
    def reconfigure(self, **kwargs) -> LoggingConfig:
        """Reconfigure logging with new parameters."""
        if not self._is_configured or not self.config:
            return self.configure(**kwargs)
        
        # Update existing configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        return self.configure(self.config)
    
    def get_logger(
        self, 
        name: str, 
        component: Optional[str] = None
    ) -> logging.Logger:
        """Get a structured logger instance."""
        if not self._is_configured:
            self.configure()
        
        from .structured_logging import StructuredLogger
        return StructuredLogger(name, component)
    
    def set_correlation_context(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> Dict[str, str]:
        """Set logging correlation context."""
        context = {}
        
        if correlation_id:
            context['correlation_id'] = CorrelationIdManager.set_correlation_id(correlation_id)
        if request_id:
            context['request_id'] = CorrelationIdManager.set_request_id(request_id)
        if session_id:
            context['session_id'] = CorrelationIdManager.set_session_id(session_id)
        if user_id:
            CorrelationIdManager.set_user_id(user_id)
            context['user_id'] = user_id
        if component:
            CorrelationIdManager.set_component_context(component)
            context['component'] = component
        
        return context
    
    def clear_correlation_context(self):
        """Clear all correlation context."""
        CorrelationIdManager.clear_correlation_id()
        # Clear other context variables
        from .structured_logging import request_id_var, session_id_var, user_id_var, component_var
        request_id_var.set(None)
        session_id_var.set(None)
        user_id_var.set(None)
        component_var.set(None)
    
    def is_configured(self) -> bool:
        """Check if logging is configured."""
        return self._is_configured
    
    def get_configuration(self) -> Optional[LoggingConfig]:
        """Get current logging configuration."""
        return self.config


# Global logging manager instance
logging_manager = LoggingManager()


def setup_logging(
    level: str = "INFO",
    format_type: Union[str, LogFormat] = LogFormat.JSON,
    output: Union[str, LogOutput] = LogOutput.CONSOLE,
    log_file: Optional[str] = None,
    **kwargs
) -> LoggingConfig:
    """Convenience function to setup structured logging."""
    if isinstance(format_type, str):
        format_type = LogFormat(format_type.lower())
    if isinstance(output, str):
        output = LogOutput(output.lower())
    
    config = LoggingConfig(
        level=level.upper(),
        format_type=format_type,
        output=output,
        log_file=log_file,
        **kwargs
    )
    
    return logging_manager.configure(config)


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """Get a structured logger instance."""
    return logging_manager.get_logger(name, component)


def set_correlation_context(**kwargs) -> Dict[str, str]:
    """Set logging correlation context."""
    return logging_manager.set_correlation_context(**kwargs)


def clear_correlation_context():
    """Clear logging correlation context."""
    logging_manager.clear_correlation_context()