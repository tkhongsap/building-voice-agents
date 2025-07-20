#!/usr/bin/env python3
"""
Basic Voice Agent Example

This example demonstrates core voice agent functionality including:
- Speech-to-text recognition
- Language model conversation  
- Text-to-speech synthesis
- Basic conversation management
- Error handling and logging

This is perfect for understanding the fundamentals before moving to advanced features.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add SDK to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdk.python_sdk import initialize_sdk, VoiceAgentBuilder, AgentCapability
from sdk.exceptions import VoiceAgentError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasicVoiceAgent:
    """A basic voice agent with essential functionality."""
    
    def __init__(self):
        self.agent = None
        self.conversation_count = 0
        self.is_running = False
    
    async def initialize(self):
        """Initialize the voice agent."""
        logger.info("Initializing Basic Voice Agent...")
        
        # Initialize SDK
        await initialize_sdk()
        
        # Build the agent with basic capabilities
        self.agent = (VoiceAgentBuilder()
            .with_name("Basic Voice Assistant")
            .with_stt("openai", 
                language="en",
                enable_interim_results=True,
                enable_automatic_punctuation=True
            )
            .with_llm("openai",
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=200
            )
            .with_tts("elevenlabs",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                optimize_streaming_latency=2
            )
            .with_vad("silero")
            .with_capability(AgentCapability.TURN_DETECTION)
            .with_system_prompt(
                "You are a helpful and friendly AI assistant named Alex. "
                "Keep your responses conversational and helpful. "
                "Be concise but informative. "
                "If you're unsure about something, say so honestly."
            )
            .build())
        
        # Register event callbacks
        self._setup_callbacks()
        
        logger.info("Basic Voice Agent initialized successfully!")
    
    def _setup_callbacks(self):
        """Setup event callbacks for the agent."""
        
        @self.agent.on_start
        async def on_start():
            self.is_running = True
            logger.info("🚀 Agent started and ready for conversation!")
            print("\n" + "="*50)
            print("🎤 Basic Voice Agent is Ready!")
            print("="*50)
            print("💬 Start speaking to begin a conversation")
            print("⏹️  Press Ctrl+C to stop")
            print("="*50)
        
        @self.agent.on_stop
        async def on_stop():
            self.is_running = False
            logger.info("⏹️ Agent stopped")
        
        @self.agent.on_user_speech
        async def on_user_speech(text: str):
            self.conversation_count += 1
            logger.info(f"User input #{self.conversation_count}: {text}")
            print(f"\n👤 You said: {text}")
        
        @self.agent.on_agent_speech
        async def on_agent_speech(text: str):
            logger.info(f"Agent response #{self.conversation_count}: {text}")
            print(f"🤖 Alex: {text}")
        
        @self.agent.on_error
        async def on_error(error: Exception):
            logger.error(f"Agent error: {error}")
            print(f"❌ Error occurred: {error}")
            
            # Handle specific error types
            if isinstance(error, VoiceAgentError):
                print(f"   Details: {error.details}")
    
    async def start(self):
        """Start the voice agent."""
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        await self.agent.start()
    
    async def stop(self):
        """Stop the voice agent."""
        if self.agent and self.agent.is_running:
            await self.agent.stop()
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.agent:
            await self.agent.cleanup()
    
    def get_stats(self):
        """Get conversation statistics."""
        return {
            "conversation_count": self.conversation_count,
            "is_running": self.is_running,
            "agent_name": self.agent.metadata.name if self.agent else None
        }


async def demo_basic_conversation():
    """Demonstrate basic voice agent conversation."""
    agent_wrapper = BasicVoiceAgent()
    
    try:
        # Initialize the agent
        await agent_wrapper.initialize()
        
        # Start the agent
        await agent_wrapper.start()
        
        # Run conversation loop
        while agent_wrapper.is_running:
            await asyncio.sleep(1)
            
            # Print stats every 30 seconds
            if agent_wrapper.conversation_count > 0 and agent_wrapper.conversation_count % 5 == 0:
                stats = agent_wrapper.get_stats()
                print(f"\n📊 Stats: {stats['conversation_count']} conversations so far")
    
    except KeyboardInterrupt:
        print("\n👋 Stopping agent...")
    
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"❌ Demo error: {e}")
    
    finally:
        # Always cleanup
        await agent_wrapper.stop()
        await agent_wrapper.cleanup()
        print("✅ Cleanup completed")


async def demo_with_custom_prompts():
    """Demonstrate agent with different personality prompts."""
    print("\n🎭 Demo: Different AI Personalities")
    print("="*50)
    
    personalities = [
        {
            "name": "Professional Assistant",
            "prompt": "You are a professional business assistant. Be formal, efficient, and helpful.",
            "duration": 30
        },
        {
            "name": "Casual Friend", 
            "prompt": "You are a casual, friendly AI buddy. Use informal language and be relaxed.",
            "duration": 30
        },
        {
            "name": "Educational Tutor",
            "prompt": "You are an educational tutor. Explain concepts clearly and encourage learning.",
            "duration": 30
        }
    ]
    
    for personality in personalities:
        print(f"\n🎯 Switching to: {personality['name']}")
        print(f"Duration: {personality['duration']} seconds")
        
        # Create agent with this personality
        agent = (VoiceAgentBuilder()
            .with_name(personality['name'])
            .with_stt("openai", language="en")
            .with_llm("openai", model="gpt-3.5-turbo", temperature=0.7)
            .with_tts("elevenlabs")
            .with_system_prompt(personality['prompt'])
            .build())
        
        try:
            await agent.start()
            
            # Run for specified duration
            await asyncio.sleep(personality['duration'])
            
        except Exception as e:
            logger.error(f"Error with personality {personality['name']}: {e}")
        
        finally:
            await agent.stop()
            await agent.cleanup()
        
        print(f"✅ {personality['name']} demo completed")


async def main():
    """Main function to run the basic voice agent demo."""
    print("🎤 Basic Voice Agent Demo")
    print("="*50)
    
    # Choose demo mode
    print("Choose demo mode:")
    print("1. Basic conversation (default)")
    print("2. Different personalities demo")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    try:
        if choice == "2":
            await demo_with_custom_prompts()
        else:
            await demo_basic_conversation()
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    # Check environment setup
    import os
    
    required_keys = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("❌ Missing required environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n💡 Set these in your .env file or environment")
        sys.exit(1)
    
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)