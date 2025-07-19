"""
Basic Voice Agent Implementation
This module implements the core voice agent with STT, LLM, TTS, and VAD components.
"""

import logging
from typing import Optional
from livekit.agents import Agent, llm, stt, tts, vad
from livekit.plugins import openai, elevenlabs, silero
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class VoiceAssistant(Agent):
    """
    A voice assistant that can listen, understand, and respond to user speech.
    
    This agent uses:
    - OpenAI Whisper for speech-to-text
    - OpenAI GPT for language understanding and response generation
    - ElevenLabs for text-to-speech
    - Silero VAD for voice activity detection
    """
    
    def __init__(
        self,
        *,
        instructions: str = None,
        model: str = None,
        voice_id: str = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the voice assistant with configurable components.
        
        Args:
            instructions: Custom instructions for the assistant's behavior
            model: LLM model to use (defaults to env var or gpt-4o-mini)
            voice_id: ElevenLabs voice ID (defaults to env var)
            temperature: LLM temperature for response generation
        """
        # Set default instructions if none provided
        if instructions is None:
            instructions = """You are a helpful and friendly AI voice assistant. 
            You engage in natural, conversational dialogue with users.
            Keep your responses concise and conversational.
            If you don't understand something, ask for clarification.
            Be warm, professional, and helpful."""
        
        # Get configuration from environment or use defaults
        model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        voice_id = voice_id or os.getenv("ELEVEN_LABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        # Initialize components
        logger.info(f"Initializing VoiceAssistant with model={model}, voice_id={voice_id}")
        
        # Speech-to-Text component
        stt_component = openai.STT(
            model="whisper-1",
            language=None,  # Auto-detect language
        )
        
        # Large Language Model component
        llm_component = openai.LLM(
            model=model,
            temperature=temperature,
        )
        
        # Text-to-Speech component
        tts_component = elevenlabs.TTS(
            voice_id=voice_id,
            model="eleven_monolingual_v1",
            encoding="mp3_44100_128",
        )
        
        # Voice Activity Detection component
        vad_component = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            speech_threshold=0.5,
        )
        
        # Initialize the parent Agent class
        super().__init__(
            instructions=instructions,
            stt=stt_component,
            llm=llm_component,
            tts=tts_component,
            vad=vad_component,
            interrupt_speech=True,  # Allow users to interrupt the assistant
        )
        
        logger.info("VoiceAssistant initialized successfully")
    
    async def on_participant_connected(self, participant):
        """Handle when a participant joins the conversation."""
        logger.info(f"Participant connected: {participant.identity}")
        # You can add custom greeting logic here if needed
    
    async def on_participant_disconnected(self, participant):
        """Handle when a participant leaves the conversation."""
        logger.info(f"Participant disconnected: {participant.identity}")


class CustomVoiceAssistant(VoiceAssistant):
    """
    Extended voice assistant with custom capabilities.
    You can add domain-specific logic, tools, or behaviors here.
    """
    
    def __init__(self, **kwargs):
        # Add any custom initialization
        super().__init__(**kwargs)
        
        # Example: Add custom context or tools
        self.custom_context = []
        
    async def before_llm_response(self, user_input: str) -> Optional[str]:
        """
        Pre-process user input before sending to LLM.
        Return None to continue normal processing, or a string to override the response.
        """
        # Example: Handle specific commands
        if user_input.lower().strip() == "what is your name":
            return "I'm your friendly AI voice assistant! You can call me Assistant."
        
        return None
    
    async def after_llm_response(self, response: str) -> str:
        """
        Post-process LLM response before sending to TTS.
        """
        # Example: Add any response filtering or modification
        return response