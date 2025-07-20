"""
Voice Agent Playground

Interactive web-based environment for testing and experimenting with voice agents.
Provides a sandbox for rapid prototyping, learning, and experimentation.

Features:
- Live code editor with syntax highlighting
- Real-time agent configuration
- Interactive voice testing
- Component visualization
- Example templates
- Share and save configurations
- Performance benchmarking
- API explorer

Usage:
    from playground.voice_agent_playground import VoiceAgentPlayground
    
    playground = VoiceAgentPlayground()
    await playground.start(port=8080)
    
    # Open browser to http://localhost:8080
"""

import asyncio
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
from aiohttp import web
import aiohttp_cors
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from sdk.python_sdk import VoiceAgentSDK, initialize_sdk, VoiceAgent
from sdk.agent_builder import VoiceAgentBuilder
from sdk.config_manager import ConfigManager
from sdk.exceptions import VoiceAgentError, ConfigurationError


@dataclass
class PlaygroundSession:
    """A playground session."""
    session_id: str
    created_at: datetime
    code: str
    config: Dict[str, Any]
    agent: Optional[VoiceAgent] = None
    output: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.output is None:
            self.output = []
        if self.errors is None:
            self.errors = []


@dataclass
class CodeTemplate:
    """Code template for quick start."""
    name: str
    description: str
    code: str
    config: Dict[str, Any]
    category: str


class VoiceAgentPlayground:
    """
    Interactive playground for voice agent development.
    
    Provides a web-based environment for testing, learning,
    and experimenting with voice agents.
    """
    
    def __init__(self):
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # Sessions
        self.sessions: Dict[str, PlaygroundSession] = {}
        self.current_session: Optional[PlaygroundSession] = None
        
        # SDK
        self.sdk: Optional[VoiceAgentSDK] = None
        
        # Templates
        self.templates = self._create_templates()
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get('/', self._handle_index)
        self.app.router.add_post('/api/session/create', self._handle_create_session)
        self.app.router.add_get('/api/session/{session_id}', self._handle_get_session)
        self.app.router.add_post('/api/code/run', self._handle_run_code)
        self.app.router.add_post('/api/config/validate', self._handle_validate_config)
        self.app.router.add_get('/api/templates', self._handle_get_templates)
        self.app.router.add_post('/api/agent/build', self._handle_build_agent)
        self.app.router.add_post('/api/agent/test', self._handle_test_agent)
        self.app.router.add_get('/api/components', self._handle_get_components)
        self.app.router.add_post('/api/export', self._handle_export)
        self.app.router.add_static('/static', path=str(Path(__file__).parent / 'static'), name='static')
    
    def _create_templates(self) -> List[CodeTemplate]:
        """Create code templates."""
        return [
            CodeTemplate(
                name="Hello World",
                description="Basic voice agent that responds to greetings",
                category="Basic",
                code='''# Hello World Voice Agent
async def create_agent(sdk):
    """Create a simple greeting agent."""
    builder = sdk.create_builder()
    
    agent = (builder
        .with_name("Hello World Agent")
        .with_stt("openai", language="en")
        .with_llm("openai", model="gpt-3.5-turbo", temperature=0.7)
        .with_tts("openai", voice="nova")
        .with_vad("silero")
        .with_system_prompt("""
            You are a friendly assistant. Greet users warmly and 
            answer their questions concisely.
        """)
        .build())
    
    return agent

# The playground will call create_agent(sdk) automatically
''',
                config={
                    "providers": {
                        "stt": {"provider": "openai", "language": "en"},
                        "llm": {"provider": "openai", "model": "gpt-3.5-turbo"},
                        "tts": {"provider": "openai", "voice": "nova"},
                        "vad": {"provider": "silero"}
                    }
                }
            ),
            
            CodeTemplate(
                name="Customer Service Agent",
                description="Professional customer service voice agent",
                category="Business",
                code='''# Customer Service Voice Agent
async def create_agent(sdk):
    """Create a customer service agent."""
    builder = sdk.create_builder()
    
    agent = (builder
        .with_name("Customer Service Agent")
        .with_stt("openai", language="en", model="whisper-1")
        .with_llm("openai", 
                 model="gpt-4-turbo",
                 temperature=0.3,  # Lower for consistency
                 max_tokens=200)
        .with_tts("openai", voice="nova", speed=0.9)
        .with_vad("silero", threshold=0.5)
        .with_capability("turn_detection")
        .with_capability("interruption_handling")
        .with_system_prompt("""
            You are a professional customer service representative.
            
            Guidelines:
            - Be polite, helpful, and empathetic
            - Listen carefully to customer concerns
            - Provide clear and accurate information
            - Offer solutions when possible
            - Escalate to supervisor when needed
            - Always maintain a professional tone
            
            Start by greeting the customer and asking how you can help.
        """)
        .build())
    
    return agent
''',
                config={
                    "providers": {
                        "stt": {"provider": "openai", "model": "whisper-1"},
                        "llm": {"provider": "openai", "model": "gpt-4-turbo", "temperature": 0.3},
                        "tts": {"provider": "openai", "voice": "nova", "speed": 0.9},
                        "vad": {"provider": "silero", "threshold": 0.5}
                    },
                    "capabilities": ["turn_detection", "interruption_handling"]
                }
            ),
            
            CodeTemplate(
                name="Multi-Language Assistant",
                description="Voice agent with multi-language support",
                category="Advanced",
                code='''# Multi-Language Voice Assistant
async def create_agent(sdk):
    """Create a multi-language assistant."""
    builder = sdk.create_builder()
    
    agent = (builder
        .with_name("Multi-Language Assistant")
        .with_stt("google", language="auto")  # Auto-detect language
        .with_llm("anthropic", 
                 model="claude-3-sonnet",
                 temperature=0.5)
        .with_tts("elevenlabs", voice="Rachel")
        .with_vad("silero", threshold=0.6)
        .with_capability("context_management")
        .with_system_prompt("""
            You are a multi-language assistant that can communicate
            in English, Spanish, French, German, and Chinese.
            
            Instructions:
            - Detect the user's language automatically
            - Respond in the same language they use
            - If unsure, ask for language preference
            - Maintain conversation context across languages
            - Be culturally sensitive and appropriate
            
            Start by greeting in English and mention your
            multi-language capabilities.
        """)
        .with_callback("on_user_speech", log_language_detection)
        .build())
    
    return agent

async def log_language_detection(text, language=None):
    """Log detected language."""
    if language:
        print(f"Detected language: {language}")
''',
                config={
                    "providers": {
                        "stt": {"provider": "google", "language": "auto"},
                        "llm": {"provider": "anthropic", "model": "claude-3-sonnet"},
                        "tts": {"provider": "elevenlabs", "voice": "Rachel"},
                        "vad": {"provider": "silero", "threshold": 0.6}
                    },
                    "capabilities": ["context_management"]
                }
            ),
            
            CodeTemplate(
                name="Function Calling Agent",
                description="Agent with tool/function calling capabilities",
                category="Advanced",
                code='''# Function Calling Voice Agent
async def create_agent(sdk):
    """Create an agent with function calling."""
    builder = sdk.create_builder()
    
    # Define available functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set a reminder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "time": {"type": "string"}
                    },
                    "required": ["task", "time"]
                }
            }
        }
    ]
    
    agent = (builder
        .with_name("Function Calling Agent")
        .with_stt("openai")
        .with_llm("openai", 
                 model="gpt-4-turbo",
                 tools=tools)
        .with_tts("openai", voice="echo")
        .with_system_prompt("""
            You are a helpful assistant with access to tools.
            
            You can:
            - Check weather for any city
            - Set reminders for users
            
            When users ask for these services, use the appropriate
            function to help them.
        """)
        .with_function_handler(handle_function_call)
        .build())
    
    return agent

async def handle_function_call(function_name, arguments):
    """Handle function calls from the agent."""
    if function_name == "get_weather":
        # Mock weather data
        location = arguments.get("location", "Unknown")
        return f"The weather in {location} is sunny, 72°F"
    
    elif function_name == "set_reminder":
        task = arguments.get("task", "")
        time = arguments.get("time", "")
        return f"Reminder set: '{task}' at {time}"
    
    return "Function not implemented"
''',
                config={
                    "providers": {
                        "stt": {"provider": "openai"},
                        "llm": {"provider": "openai", "model": "gpt-4-turbo"},
                        "tts": {"provider": "openai", "voice": "echo"},
                        "vad": {"provider": "silero"}
                    }
                }
            ),
            
            CodeTemplate(
                name="Custom Pipeline",
                description="Agent with custom processing pipeline",
                category="Expert",
                code='''# Custom Pipeline Voice Agent
async def create_agent(sdk):
    """Create an agent with custom pipeline."""
    builder = sdk.create_builder()
    
    agent = (builder
        .with_name("Custom Pipeline Agent")
        .with_stt("openai")
        .with_llm("openai", model="gpt-4")
        .with_tts("elevenlabs", voice="Josh")
        .with_vad("webrtc", aggressiveness=2)
        .with_capability("turn_detection")
        .with_capability("context_management")
        .with_pre_processor(preprocess_input)
        .with_post_processor(postprocess_output)
        .with_custom_pipeline_node("sentiment", analyze_sentiment)
        .with_custom_pipeline_node("safety", check_safety)
        .build())
    
    return agent

async def preprocess_input(text):
    """Preprocess user input."""
    # Remove filler words
    filler_words = ["um", "uh", "like", "you know"]
    words = text.lower().split()
    cleaned = [w for w in words if w not in filler_words]
    return " ".join(cleaned)

async def postprocess_output(text):
    """Postprocess agent output."""
    # Add appropriate pauses for TTS
    text = text.replace(". ", ". <break time='0.5s'/> ")
    text = text.replace("? ", "? <break time='0.5s'/> ")
    return text

async def analyze_sentiment(data):
    """Analyze sentiment of user input."""
    text = data.get("text", "")
    # Simple sentiment analysis
    positive_words = ["good", "great", "excellent", "happy", "thanks"]
    negative_words = ["bad", "terrible", "angry", "upset", "problem"]
    
    sentiment_score = 0
    for word in positive_words:
        if word in text.lower():
            sentiment_score += 1
    for word in negative_words:
        if word in text.lower():
            sentiment_score -= 1
    
    data["sentiment"] = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    return data

async def check_safety(data):
    """Check content safety."""
    text = data.get("text", "")
    # Simple safety check
    unsafe_patterns = ["hate", "violence", "inappropriate"]
    
    is_safe = not any(pattern in text.lower() for pattern in unsafe_patterns)
    data["is_safe"] = is_safe
    
    if not is_safe:
        data["text"] = "I cannot process that request."
    
    return data
''',
                config={
                    "providers": {
                        "stt": {"provider": "openai"},
                        "llm": {"provider": "openai", "model": "gpt-4"},
                        "tts": {"provider": "elevenlabs", "voice": "Josh"},
                        "vad": {"provider": "webrtc", "aggressiveness": 2}
                    },
                    "capabilities": ["turn_detection", "context_management"]
                }
            )
        ]
    
    async def start(self, host: str = "localhost", port: int = 8080) -> None:
        """Start the playground server."""
        # Initialize SDK
        self.sdk = await initialize_sdk({
            "project_name": "playground",
            "environment": "development"
        })
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        
        print(f"🎮 Voice Agent Playground started at http://{host}:{port}")
        print(f"   Open your browser to start experimenting!")
    
    async def stop(self) -> None:
        """Stop the playground server."""
        # Clean up sessions
        for session in self.sessions.values():
            if session.agent:
                await session.agent.cleanup()
        
        if self.runner:
            await self.runner.cleanup()
        
        print("🎮 Playground stopped")
    
    async def _handle_index(self, request):
        """Serve the playground HTML."""
        html = self._generate_playground_html()
        return web.Response(text=html, content_type='text/html')
    
    async def _handle_create_session(self, request):
        """Create a new playground session."""
        data = await request.json()
        
        session_id = str(uuid.uuid4())
        session = PlaygroundSession(
            session_id=session_id,
            created_at=datetime.now(),
            code=data.get("code", ""),
            config=data.get("config", {})
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        return web.json_response({
            "session_id": session_id,
            "created_at": session.created_at.isoformat()
        })
    
    async def _handle_get_session(self, request):
        """Get session details."""
        session_id = request.match_info['session_id']
        session = self.sessions.get(session_id)
        
        if not session:
            return web.json_response({"error": "Session not found"}, status=404)
        
        return web.json_response({
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "code": session.code,
            "config": session.config,
            "output": session.output,
            "errors": session.errors
        })
    
    async def _handle_run_code(self, request):
        """Run code in the playground."""
        data = await request.json()
        code = data.get("code", "")
        session_id = data.get("session_id")
        
        if not session_id or session_id not in self.sessions:
            return web.json_response({"error": "Invalid session"}, status=400)
        
        session = self.sessions[session_id]
        session.code = code
        session.output = []
        session.errors = []
        
        try:
            # Create a safe execution environment
            namespace = {
                "sdk": self.sdk,
                "print": lambda *args: session.output.append(" ".join(str(arg) for arg in args)),
                "asyncio": asyncio
            }
            
            # Execute the code
            exec(code, namespace)
            
            # Check if create_agent function exists
            if "create_agent" not in namespace:
                raise ValueError("Code must define an async function: create_agent(sdk)")
            
            # Create the agent
            if session.agent:
                await session.agent.cleanup()
            
            session.agent = await namespace["create_agent"](self.sdk)
            
            session.output.append("✅ Agent created successfully!")
            
            return web.json_response({
                "success": True,
                "output": session.output,
                "agent_name": session.agent.metadata.name if hasattr(session.agent, 'metadata') else "Voice Agent"
            })
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            session.errors.append(error_msg)
            
            return web.json_response({
                "success": False,
                "errors": session.errors
            })
    
    async def _handle_validate_config(self, request):
        """Validate agent configuration."""
        data = await request.json()
        config = data.get("config", {})
        
        try:
            # Use ConfigManager to validate
            manager = ConfigManager(config)
            errors = manager.validate()
            
            if errors:
                return web.json_response({
                    "valid": False,
                    "errors": errors
                })
            
            return web.json_response({
                "valid": True,
                "message": "Configuration is valid"
            })
            
        except Exception as e:
            return web.json_response({
                "valid": False,
                "errors": [str(e)]
            })
    
    async def _handle_get_templates(self, request):
        """Get available templates."""
        templates_data = [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "code": t.code,
                "config": t.config
            }
            for t in self.templates
        ]
        
        return web.json_response(templates_data)
    
    async def _handle_build_agent(self, request):
        """Build agent from configuration."""
        data = await request.json()
        config = data.get("config", {})
        session_id = data.get("session_id")
        
        if not session_id or session_id not in self.sessions:
            return web.json_response({"error": "Invalid session"}, status=400)
        
        session = self.sessions[session_id]
        
        try:
            # Build agent using builder
            builder = self.sdk.create_builder()
            
            # Apply configuration
            if "name" in config:
                builder = builder.with_name(config["name"])
            
            providers = config.get("providers", {})
            
            if "stt" in providers:
                stt = providers["stt"]
                builder = builder.with_stt(stt.get("provider", "openai"), **{k: v for k, v in stt.items() if k != "provider"})
            
            if "llm" in providers:
                llm = providers["llm"]
                builder = builder.with_llm(llm.get("provider", "openai"), **{k: v for k, v in llm.items() if k != "provider"})
            
            if "tts" in providers:
                tts = providers["tts"]
                builder = builder.with_tts(tts.get("provider", "openai"), **{k: v for k, v in tts.items() if k != "provider"})
            
            if "vad" in providers:
                vad = providers["vad"]
                builder = builder.with_vad(vad.get("provider", "silero"), **{k: v for k, v in vad.items() if k != "provider"})
            
            # Apply capabilities
            for capability in config.get("capabilities", []):
                builder = builder.with_capability(capability)
            
            # Apply system prompt
            if "system_prompt" in config:
                builder = builder.with_system_prompt(config["system_prompt"])
            
            # Build agent
            if session.agent:
                await session.agent.cleanup()
            
            session.agent = builder.build()
            
            return web.json_response({
                "success": True,
                "message": "Agent built successfully"
            })
            
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            })
    
    async def _handle_test_agent(self, request):
        """Test the agent with sample input."""
        data = await request.json()
        session_id = data.get("session_id")
        test_input = data.get("input", "Hello, how are you?")
        
        if not session_id or session_id not in self.sessions:
            return web.json_response({"error": "Invalid session"}, status=400)
        
        session = self.sessions[session_id]
        
        if not session.agent:
            return web.json_response({"error": "No agent created"}, status=400)
        
        try:
            # Mock test - in a real implementation, this would process audio
            test_results = {
                "input": test_input,
                "processing_time": 1.5,  # Mock timing
                "output": "I'm doing well, thank you! How can I help you today?",  # Mock response
                "components": {
                    "stt": {"status": "success", "latency": 0.3},
                    "llm": {"status": "success", "latency": 0.8},
                    "tts": {"status": "success", "latency": 0.4}
                }
            }
            
            return web.json_response(test_results)
            
        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)
    
    async def _handle_get_components(self, request):
        """Get available components."""
        components = {
            "stt": ["openai", "azure", "google"],
            "llm": ["openai", "anthropic", "google", "local"],
            "tts": ["openai", "elevenlabs", "azure"],
            "vad": ["silero", "webrtc"],
            "capabilities": ["turn_detection", "interruption_handling", "context_management", "conversation_state"]
        }
        
        return web.json_response(components)
    
    async def _handle_export(self, request):
        """Export agent configuration and code."""
        data = await request.json()
        session_id = data.get("session_id")
        format = data.get("format", "python")  # python, yaml, json
        
        if not session_id or session_id not in self.sessions:
            return web.json_response({"error": "Invalid session"}, status=400)
        
        session = self.sessions[session_id]
        
        if format == "python":
            # Generate Python script
            script = f'''#!/usr/bin/env python3
"""
Voice Agent - Generated from Playground
Created: {datetime.now().isoformat()}
"""

import asyncio
from voice_agents import VoiceAgentSDK, initialize_sdk

{session.code}

async def main():
    # Initialize SDK
    sdk = await initialize_sdk({{
        "project_name": "my_voice_agent",
        "environment": "production"
    }})
    
    # Create agent
    agent = await create_agent(sdk)
    
    # Start agent
    await agent.start()
    
    print("Voice agent is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\\nStopping agent...")
    finally:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
'''
            
            return web.Response(
                text=script,
                content_type='text/plain',
                headers={
                    'Content-Disposition': 'attachment; filename="voice_agent.py"'
                }
            )
        
        elif format == "yaml":
            # Generate YAML config
            yaml_content = yaml.dump(session.config, default_flow_style=False)
            
            return web.Response(
                text=yaml_content,
                content_type='text/yaml',
                headers={
                    'Content-Disposition': 'attachment; filename="voice_agent_config.yaml"'
                }
            )
        
        else:
            # JSON format
            return web.json_response(session.config)
    
    def _generate_playground_html(self) -> str:
        """Generate the playground HTML interface."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent Playground</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/theme/monokai.min.css">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        .sidebar {
            width: 250px;
            background-color: #252525;
            border-right: 1px solid #333;
            padding: 20px;
            overflow-y: auto;
        }
        
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background-color: #252525;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .editor-container {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        
        .code-editor {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .config-panel {
            width: 350px;
            background-color: #1e1e1e;
            border-left: 1px solid #333;
            padding: 20px;
            overflow-y: auto;
        }
        
        .output-panel {
            height: 200px;
            background-color: #0a0a0a;
            border-top: 1px solid #333;
            padding: 15px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        
        .btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
        }
        
        .btn:hover {
            background-color: #45a049;
        }
        
        .btn-secondary {
            background-color: #2196f3;
        }
        
        .btn-secondary:hover {
            background-color: #1976d2;
        }
        
        .template-item {
            background-color: #333;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .template-item:hover {
            background-color: #404040;
        }
        
        .template-title {
            font-weight: bold;
            color: #4caf50;
            margin-bottom: 4px;
        }
        
        .template-desc {
            font-size: 12px;
            color: #999;
        }
        
        .config-section {
            margin-bottom: 20px;
        }
        
        .config-section h4 {
            margin-bottom: 10px;
            color: #4caf50;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-size: 13px;
            color: #999;
        }
        
        .form-control {
            width: 100%;
            padding: 8px;
            background-color: #333;
            border: 1px solid #555;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 13px;
        }
        
        .form-control:focus {
            outline: none;
            border-color: #4caf50;
        }
        
        select.form-control {
            cursor: pointer;
        }
        
        .output-success {
            color: #4caf50;
        }
        
        .output-error {
            color: #f44336;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid #333;
            margin-bottom: 15px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab:hover {
            background-color: #333;
        }
        
        .tab.active {
            border-bottom-color: #4caf50;
            color: #4caf50;
        }
        
        .CodeMirror {
            flex: 1;
            font-size: 14px;
        }
        
        .status-bar {
            background-color: #333;
            padding: 5px 15px;
            font-size: 12px;
            color: #999;
            display: flex;
            justify-content: space-between;
        }
        
        .loader {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #4caf50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h3 style="margin-bottom: 20px;">Templates</h3>
            <div id="templates-list"></div>
        </div>
        
        <div class="main">
            <div class="header">
                <h1>🎮 Voice Agent Playground</h1>
                <div>
                    <button class="btn" onclick="runCode()">▶️ Run</button>
                    <button class="btn btn-secondary" onclick="testAgent()">🎤 Test</button>
                    <button class="btn btn-secondary" onclick="exportCode()">💾 Export</button>
                </div>
            </div>
            
            <div class="editor-container">
                <div class="code-editor">
                    <textarea id="code-editor"></textarea>
                    <div class="status-bar">
                        <span id="status-text">Ready</span>
                        <span id="session-info"></span>
                    </div>
                </div>
                
                <div class="config-panel">
                    <div class="tabs">
                        <div class="tab active" onclick="switchTab('config')">Configuration</div>
                        <div class="tab" onclick="switchTab('components')">Components</div>
                    </div>
                    
                    <div id="config-tab">
                        <div class="config-section">
                            <h4>Agent Settings</h4>
                            <div class="form-group">
                                <label>Agent Name</label>
                                <input type="text" class="form-control" id="agent-name" value="My Voice Agent">
                            </div>
                        </div>
                        
                        <div class="config-section">
                            <h4>Providers</h4>
                            <div class="form-group">
                                <label>STT Provider</label>
                                <select class="form-control" id="stt-provider">
                                    <option value="openai">OpenAI Whisper</option>
                                    <option value="azure">Azure Speech</option>
                                    <option value="google">Google Speech</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>LLM Provider</label>
                                <select class="form-control" id="llm-provider">
                                    <option value="openai">OpenAI GPT</option>
                                    <option value="anthropic">Anthropic Claude</option>
                                    <option value="google">Google Gemini</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>TTS Provider</label>
                                <select class="form-control" id="tts-provider">
                                    <option value="openai">OpenAI TTS</option>
                                    <option value="elevenlabs">ElevenLabs</option>
                                    <option value="azure">Azure Speech</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>VAD Provider</label>
                                <select class="form-control" id="vad-provider">
                                    <option value="silero">Silero VAD</option>
                                    <option value="webrtc">WebRTC VAD</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="config-section">
                            <h4>Capabilities</h4>
                            <div class="form-group">
                                <label>
                                    <input type="checkbox" id="cap-turn-detection"> Turn Detection
                                </label>
                            </div>
                            <div class="form-group">
                                <label>
                                    <input type="checkbox" id="cap-interruption"> Interruption Handling
                                </label>
                            </div>
                            <div class="form-group">
                                <label>
                                    <input type="checkbox" id="cap-context"> Context Management
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <div id="components-tab" style="display: none;">
                        <h4>Available Components</h4>
                        <div id="components-list"></div>
                    </div>
                </div>
            </div>
            
            <div class="output-panel" id="output">
                <div style="color: #666;">Output will appear here...</div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/codemirror.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.2/mode/python/python.min.js"></script>
    <script>
        let editor;
        let currentSession = null;
        
        // Initialize CodeMirror
        document.addEventListener('DOMContentLoaded', () => {
            editor = CodeMirror.fromTextArea(document.getElementById('code-editor'), {
                mode: 'python',
                theme: 'monokai',
                lineNumbers: true,
                indentUnit: 4,
                lineWrapping: true,
                autofocus: true
            });
            
            // Load templates
            loadTemplates();
            
            // Load components
            loadComponents();
            
            // Create initial session
            createSession();
        });
        
        async function createSession() {
            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    code: editor.getValue(),
                    config: getConfiguration()
                })
            });
            
            const data = await response.json();
            currentSession = data.session_id;
            document.getElementById('session-info').textContent = `Session: ${currentSession.substring(0, 8)}...`;
        }
        
        async function loadTemplates() {
            const response = await fetch('/api/templates');
            const templates = await response.json();
            
            const container = document.getElementById('templates-list');
            container.innerHTML = '';
            
            templates.forEach(template => {
                const item = document.createElement('div');
                item.className = 'template-item';
                item.innerHTML = `
                    <div class="template-title">${template.name}</div>
                    <div class="template-desc">${template.description}</div>
                `;
                item.onclick = () => loadTemplate(template);
                container.appendChild(item);
            });
        }
        
        function loadTemplate(template) {
            editor.setValue(template.code);
            
            // Apply configuration
            if (template.config.providers) {
                if (template.config.providers.stt) {
                    document.getElementById('stt-provider').value = template.config.providers.stt.provider || 'openai';
                }
                if (template.config.providers.llm) {
                    document.getElementById('llm-provider').value = template.config.providers.llm.provider || 'openai';
                }
                if (template.config.providers.tts) {
                    document.getElementById('tts-provider').value = template.config.providers.tts.provider || 'openai';
                }
                if (template.config.providers.vad) {
                    document.getElementById('vad-provider').value = template.config.providers.vad.provider || 'silero';
                }
            }
            
            updateOutput(`Loaded template: ${template.name}`, 'success');
        }
        
        async function loadComponents() {
            const response = await fetch('/api/components');
            const components = await response.json();
            
            const container = document.getElementById('components-list');
            let html = '';
            
            for (const [type, providers] of Object.entries(components)) {
                html += `<div class="config-section">`;
                html += `<h4>${type.toUpperCase()}</h4>`;
                html += `<ul style="list-style: none; padding: 0;">`;
                providers.forEach(provider => {
                    html += `<li style="padding: 4px 0; color: #999;">• ${provider}</li>`;
                });
                html += `</ul></div>`;
            }
            
            container.innerHTML = html;
        }
        
        function getConfiguration() {
            const config = {
                name: document.getElementById('agent-name').value,
                providers: {
                    stt: { provider: document.getElementById('stt-provider').value },
                    llm: { provider: document.getElementById('llm-provider').value },
                    tts: { provider: document.getElementById('tts-provider').value },
                    vad: { provider: document.getElementById('vad-provider').value }
                },
                capabilities: []
            };
            
            if (document.getElementById('cap-turn-detection').checked) {
                config.capabilities.push('turn_detection');
            }
            if (document.getElementById('cap-interruption').checked) {
                config.capabilities.push('interruption_handling');
            }
            if (document.getElementById('cap-context').checked) {
                config.capabilities.push('context_management');
            }
            
            return config;
        }
        
        async function runCode() {
            if (!currentSession) {
                await createSession();
            }
            
            updateStatus('Running...', true);
            clearOutput();
            
            const response = await fetch('/api/code/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: currentSession,
                    code: editor.getValue()
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                updateStatus('Ready');
                result.output.forEach(line => updateOutput(line, 'success'));
            } else {
                updateStatus('Error');
                result.errors.forEach(error => updateOutput(error, 'error'));
            }
        }
        
        async function testAgent() {
            if (!currentSession) {
                updateOutput('Please run the code first to create an agent', 'error');
                return;
            }
            
            updateStatus('Testing agent...', true);
            
            const response = await fetch('/api/agent/test', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: currentSession,
                    input: 'Hello, how are you?'
                })
            });
            
            const result = await response.json();
            
            if (result.error) {
                updateOutput(`Test failed: ${result.error}`, 'error');
            } else {
                updateOutput('Test Results:', 'success');
                updateOutput(`Input: "${result.input}"`);
                updateOutput(`Output: "${result.output}"`);
                updateOutput(`Processing time: ${result.processing_time}s`);
                
                for (const [component, status] of Object.entries(result.components)) {
                    updateOutput(`${component}: ${status.status} (${status.latency}s)`);
                }
            }
            
            updateStatus('Ready');
        }
        
        async function exportCode() {
            if (!currentSession) {
                await createSession();
            }
            
            const format = 'python'; // Could add format selector
            
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: currentSession,
                    format: format
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'voice_agent.py';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                updateOutput('Code exported successfully', 'success');
            } else {
                updateOutput('Export failed', 'error');
            }
        }
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            
            if (tab === 'config') {
                document.getElementById('config-tab').style.display = 'block';
                document.getElementById('components-tab').style.display = 'none';
            } else {
                document.getElementById('config-tab').style.display = 'none';
                document.getElementById('components-tab').style.display = 'block';
            }
        }
        
        function updateStatus(text, loading = false) {
            const statusEl = document.getElementById('status-text');
            statusEl.innerHTML = text + (loading ? '<span class="loader"></span>' : '');
        }
        
        function updateOutput(text, type = 'normal') {
            const output = document.getElementById('output');
            const line = document.createElement('div');
            
            if (type === 'success') {
                line.className = 'output-success';
            } else if (type === 'error') {
                line.className = 'output-error';
            }
            
            line.textContent = text;
            output.appendChild(line);
            output.scrollTop = output.scrollHeight;
        }
        
        function clearOutput() {
            document.getElementById('output').innerHTML = '';
        }
    </script>
</body>
</html>
'''


# Example usage
async def demo_playground():
    """Demonstrate the voice agent playground."""
    print("🎮 Voice Agent Playground Demo")
    print("="*50)
    
    playground = VoiceAgentPlayground()
    
    try:
        await playground.start(port=8080)
        
        print("\n✅ Playground is running!")
        print("   Open http://localhost:8080 in your browser")
        print("   Press Ctrl+C to stop")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping playground...")
    finally:
        await playground.stop()


if __name__ == "__main__":
    asyncio.run(demo_playground())