"""
Basic Voice Bot Example

A simple voice assistant that demonstrates the core functionality of the platform.
Perfect for beginners to understand the basic concepts.

Features:
- Basic speech-to-text and text-to-speech
- Simple conversation flow
- Weather information lookup
- Time queries
- Basic math calculations

Usage:
    python basic_voice_bot.py
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from src.sdk.python_sdk import VoiceAgentSDK
from src.agents.voice_assistant import VoiceAssistant
from src.components.stt.openai_stt import OpenAISTT
from src.components.llm.openai_llm import OpenAILLM
from src.components.tts.openai_tts import OpenAITTS
from src.components.vad.webrtc_vad import WebRTCVAD


class BasicVoiceBot:
    """
    A simple voice bot that can handle basic conversations.
    
    This example demonstrates:
    - Setting up a basic voice agent
    - Handling simple queries
    - Implementing custom functions
    - Basic conversation management
    """
    
    def __init__(self):
        self.sdk = VoiceAgentSDK()
        self.agent = None
    
    async def setup(self):
        """Setup the voice bot with basic configuration."""
        # Configure components
        stt_config = {
            "model": "whisper-1",
            "language": "en",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        llm_config = {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "system_prompt": """You are a helpful voice assistant named Bobby. 
            Keep your responses concise and conversational. 
            You can help with weather, time, basic math, and general questions."""
        }
        
        tts_config = {
            "model": "tts-1",
            "voice": "nova",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        vad_config = {
            "sensitivity": 0.6,
            "min_speech_duration": 0.5
        }
        
        # Create agent
        self.agent = await self.sdk.create_agent(
            stt_provider="openai",
            stt_config=stt_config,
            llm_provider="openai", 
            llm_config=llm_config,
            tts_provider="openai",
            tts_config=tts_config,
            vad_provider="webrtc",
            vad_config=vad_config
        )
        
        # Add custom functions
        self._register_functions()
        
        print("🤖 Bobby the Voice Bot is ready!")
        print("   Say 'hello' to start a conversation")
        print("   Try: 'What time is it?', 'What's 15 + 27?', 'Tell me a joke'")
    
    def _register_functions(self):
        """Register custom functions for the bot."""
        
        @self.agent.function(
            name="get_current_time",
            description="Get the current date and time"
        )
        async def get_current_time() -> str:
            """Get the current time."""
            now = datetime.now()
            return f"The current time is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}"
        
        @self.agent.function(
            name="calculate_math",
            description="Perform basic math calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Math expression to calculate (e.g., '15 + 27', '100 / 4')"
                }
            }
        )
        async def calculate_math(expression: str) -> str:
            """Calculate basic math expressions."""
            try:
                # Simple eval for basic operations (in production, use a safer parser)
                allowed_chars = set('0123456789+-*/()., ')
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    return f"{expression} = {result}"
                else:
                    return "I can only do basic math with numbers and +, -, *, / operations."
            except Exception:
                return "Sorry, I couldn't calculate that. Please try a simpler expression."
        
        @self.agent.function(
            name="get_weather_info",
            description="Get weather information (mock implementation)",
            parameters={
                "location": {
                    "type": "string", 
                    "description": "Location to get weather for"
                }
            }
        )
        async def get_weather_info(location: str) -> str:
            """Get weather information (mock)."""
            # In a real implementation, this would call a weather API
            return f"The weather in {location} is sunny with a temperature of 72°F. Perfect day to be outside!"
        
        @self.agent.function(
            name="tell_joke",
            description="Tell a random joke"
        )
        async def tell_joke() -> str:
            """Tell a joke."""
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "What do you call a fake noodle? An impasta!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "What do you call a dinosaur that crashes his car? Tyrannosaurus Wrecks!",
                "Why don't eggs tell jokes? They'd crack each other up!"
            ]
            import random
            return random.choice(jokes)
    
    async def start_conversation(self):
        """Start the voice conversation."""
        if not self.agent:
            await self.setup()
        
        print("\n🎤 Starting voice conversation...")
        print("   Speak naturally - the bot will respond")
        print("   Press Ctrl+C to stop\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        finally:
            if self.agent:
                await self.agent.stop_conversation()
    
    async def demo_text_mode(self):
        """Run a text-based demo for testing without audio."""
        if not self.agent:
            await self.setup()
        
        print("\n💬 Text Demo Mode")
        print("   Type your messages instead of speaking")
        print("   Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if user_input:
                    response = await self.agent.process_text(user_input)
                    print(f"Bobby: {response}")
                    
            except KeyboardInterrupt:
                break
        
        print("\n👋 Demo completed!")


# Conversation examples and scenarios
class ConversationScenarios:
    """Example conversation scenarios to demonstrate capabilities."""
    
    @staticmethod
    def get_sample_conversations() -> Dict[str, list]:
        """Get sample conversation flows."""
        return {
            "greeting": [
                "User: Hello there!",
                "Bobby: Hello! I'm Bobby, your voice assistant. How can I help you today?",
                "User: What can you do?",
                "Bobby: I can help you with the time, basic math, weather information, and tell jokes. What would you like to know?"
            ],
            
            "time_query": [
                "User: What time is it?",
                "Bobby: The current time is 2:30 PM on Wednesday, March 15, 2024",
                "User: Thank you!",
                "Bobby: You're welcome! Is there anything else I can help you with?"
            ],
            
            "math_calculation": [
                "User: What's 25 times 4?",
                "Bobby: 25 * 4 = 100",
                "User: How about 150 divided by 6?",
                "Bobby: 150 / 6 = 25.0"
            ],
            
            "weather_inquiry": [
                "User: What's the weather like in San Francisco?",
                "Bobby: The weather in San Francisco is sunny with a temperature of 72°F. Perfect day to be outside!",
                "User: Sounds nice!",
                "Bobby: It really does! Would you like to know anything else?"
            ],
            
            "joke_request": [
                "User: Tell me a joke",
                "Bobby: Why don't scientists trust atoms? Because they make up everything!",
                "User: That's funny!",
                "Bobby: I'm glad you enjoyed it! I have more jokes if you'd like to hear another one."
            ]
        }


async def main():
    """Main entry point."""
    print("🎙️ Basic Voice Bot Example")
    print("=" * 50)
    
    bot = BasicVoiceBot()
    
    # Show sample conversations
    scenarios = ConversationScenarios.get_sample_conversations()
    print("\n📝 Sample Conversations:")
    for scenario_name, conversation in scenarios.items():
        print(f"\n{scenario_name.replace('_', ' ').title()}:")
        for line in conversation:
            print(f"  {line}")
    
    print("\n" + "=" * 50)
    
    # Choose mode
    print("Choose a mode:")
    print("1. Voice conversation (requires microphone)")
    print("2. Text demo (type instead of speak)")
    print("3. Show examples only")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            await bot.start_conversation()
        elif choice == "2":
            await bot.demo_text_mode()
        elif choice == "3":
            print("\n✅ Examples shown above!")
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("   Set it with: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    asyncio.run(main())