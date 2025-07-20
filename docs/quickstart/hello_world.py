#!/usr/bin/env python3
"""
Hello World Voice Agent

The simplest possible voice agent - perfect for getting started!
This example shows the minimal code needed to create a working voice agent.

Requirements:
- OpenAI API key (for STT and LLM)
- ElevenLabs API key (for TTS)
- Microphone and speakers

Usage:
    python hello_world.py

Features:
- Speech-to-text recognition
- LLM conversation
- Text-to-speech output
- Basic error handling
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add SDK to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sdk.python_sdk import initialize_sdk, VoiceAgentBuilder
from sdk.exceptions import VoiceAgentError, ConfigurationError, AuthenticationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the hello world voice agent."""
    
    print("🎤 Hello World Voice Agent")
    print("=" * 50)
    print("Starting up your first voice agent...")
    
    try:
        # Step 1: Initialize the SDK
        print("📦 Initializing SDK...")
        await initialize_sdk()
        print("✅ SDK initialized successfully!")
        
        # Step 2: Build the voice agent
        print("🔧 Building voice agent...")
        agent = (VoiceAgentBuilder()
            .with_name("Hello World Agent")
            .with_stt("openai", language="en")
            .with_llm("openai", model="gpt-3.5-turbo", temperature=0.7)
            .with_tts("elevenlabs")
            .with_system_prompt(
                "You are a friendly AI assistant named Hello. "
                "Keep your responses short and conversational. "
                "You're excited to be someone's first voice agent!"
            )
            .build())
        
        print("✅ Agent built successfully!")
        
        # Step 3: Add some callbacks for feedback
        agent.on_user_speech(lambda text: print(f"👤 You said: {text}"))
        agent.on_agent_speech(lambda text: print(f"🤖 Agent: {text}"))
        agent.on_error(lambda error: logger.error(f"Agent error: {error}"))
        
        # Step 4: Start the agent
        print("🚀 Starting voice agent...")
        await agent.start()
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS! Your voice agent is now running!")
        print("=" * 50)
        print("👋 Say 'Hello' to start a conversation")
        print("💬 Try asking questions or having a chat")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        
        # Step 5: Keep the agent running
        try:
            while agent.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Stopping agent...")
            
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your .env file has the required API keys")
        print("   2. Verify API key format (OpenAI: sk-..., ElevenLabs: ...)")
        print("   3. Make sure your API keys have sufficient credits")
        
    except AuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify your API keys are correct")
        print("   2. Check if your API keys have expired")
        print("   3. Ensure your account has active credits")
        
    except VoiceAgentError as e:
        print(f"\n❌ Voice Agent Error: {e}")
        print(f"   Details: {e.details}")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        logger.exception("Unexpected error occurred")
        
    finally:
        # Always cleanup
        try:
            if 'agent' in locals() and agent.is_initialized:
                print("🧹 Cleaning up...")
                await agent.stop()
                await agent.cleanup()
                print("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def check_requirements():
    """Check if basic requirements are met."""
    import os
    
    required_vars = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\n💡 Create a .env file with:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        print("   ELEVENLABS_API_KEY=your-key-here")
        return False
    
    return True


if __name__ == "__main__":
    print("Hello World Voice Agent - Quick Start Example")
    print("=" * 60)
    
    # Check requirements first
    if not check_requirements():
        print("\n⚠️  Please set up your environment variables and try again.")
        sys.exit(1)
    
    # Run the agent
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)