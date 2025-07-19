"""
Advanced Voice Agent with enhanced turn detection and interruption handling.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from livekit.agents import Agent, llm, stt, tts, vad
from livekit.plugins import openai, elevenlabs, silero
from livekit.agents.metrics import AgentMetrics, LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedVoiceAssistant(Agent):
    """
    Advanced voice assistant with enhanced turn detection, interruption handling,
    and performance monitoring.
    """
    
    def __init__(
        self,
        *,
        instructions: str = None,
        model: str = None,
        voice_id: str = None,
        temperature: float = 0.7,
        enable_metrics: bool = True,
    ) -> None:
        """
        Initialize the advanced voice assistant.
        
        Args:
            instructions: Custom instructions for the assistant's behavior
            model: LLM model to use (defaults to env var or gpt-4o-mini)
            voice_id: ElevenLabs voice ID (defaults to env var)
            temperature: LLM temperature for response generation
            enable_metrics: Whether to enable performance metrics collection
        """
        # Set default instructions if none provided
        if instructions is None:
            instructions = """You are a helpful and friendly AI voice assistant. 
            You engage in natural, conversational dialogue with users.
            Keep your responses concise and conversational.
            If you don't understand something, ask for clarification.
            Be warm, professional, and helpful.
            Remember this is a voice conversation, so avoid long monologues."""
        
        # Get configuration from environment or use defaults
        model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        voice_id = voice_id or os.getenv("ELEVEN_LABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        # Initialize components with metrics wrappers if enabled
        logger.info(f"Initializing AdvancedVoiceAssistant with model={model}, voice_id={voice_id}")
        
        # Speech-to-Text component
        stt_component = openai.STT(
            model="whisper-1",
            language=None,  # Auto-detect language
        )
        
        # Wrap with metrics if enabled
        if enable_metrics:
            stt_component = STTMetrics(
                stt=stt_component,
                callback=self._on_stt_metrics,
            )
        
        # Large Language Model component
        llm_component = openai.LLM(
            model=model,
            temperature=temperature,
        )
        
        # Wrap with metrics if enabled
        if enable_metrics:
            llm_component = LLMMetrics(
                llm=llm_component,
                callback=self._on_llm_metrics,
            )
        
        # Text-to-Speech component
        tts_component = elevenlabs.TTS(
            voice_id=voice_id,
            model="eleven_monolingual_v1",
            encoding="mp3_44100_128",
        )
        
        # Wrap with metrics if enabled
        if enable_metrics:
            tts_component = TTSMetrics(
                tts=tts_component,
                callback=self._on_tts_metrics,
            )
        
        # Voice Activity Detection component with fine-tuned settings
        vad_component = silero.VAD.load(
            min_speech_duration=0.1,      # Minimum 100ms of speech to trigger
            min_silence_duration=0.5,     # Wait 500ms of silence before ending turn
            speech_threshold=0.5,         # Confidence threshold for speech detection
        )
        
        # Wrap with metrics if enabled
        if enable_metrics:
            # Create EOU (End of Utterance) metrics wrapper
            self.eou_metrics = EOUMetrics(callback=self._on_eou_metrics)
        
        # Initialize the parent Agent class
        super().__init__(
            instructions=instructions,
            stt=stt_component,
            llm=llm_component,
            tts=tts_component,
            vad=vad_component,
            interrupt_speech=True,  # Allow users to interrupt the assistant
            interrupt_threshold=0.6,  # Threshold for interruption detection
        )
        
        # Store metrics
        self.enable_metrics = enable_metrics
        self.metrics_data = {
            "llm": [],
            "stt": [],
            "tts": [],
            "eou": []
        }
        
        logger.info("AdvancedVoiceAssistant initialized successfully")
    
    async def _on_llm_metrics(self, metrics: LLMMetrics):
        """Handle LLM metrics."""
        if self.enable_metrics:
            metrics_data = {
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "tokens_per_second": metrics.tokens_per_second,
                "time_to_first_token": metrics.time_to_first_token,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.metrics_data["llm"].append(metrics_data)
            
            logger.info(
                f"LLM Metrics - Prompt tokens: {metrics.prompt_tokens}, "
                f"Completion tokens: {metrics.completion_tokens}, "
                f"Tokens/sec: {metrics.tokens_per_second:.2f}, "
                f"TTFT: {metrics.time_to_first_token:.3f}s"
            )
    
    async def _on_stt_metrics(self, metrics: STTMetrics):
        """Handle STT metrics."""
        if self.enable_metrics:
            metrics_data = {
                "duration": metrics.duration,
                "audio_duration": metrics.audio_duration,
                "is_streaming": metrics.is_streaming,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.metrics_data["stt"].append(metrics_data)
            
            logger.info(
                f"STT Metrics - Duration: {metrics.duration:.3f}s, "
                f"Audio duration: {metrics.audio_duration:.3f}s, "
                f"Streaming: {metrics.is_streaming}"
            )
    
    async def _on_tts_metrics(self, metrics: TTSMetrics):
        """Handle TTS metrics."""
        if self.enable_metrics:
            metrics_data = {
                "time_to_first_byte": metrics.time_to_first_byte,
                "duration": metrics.duration,
                "audio_duration": metrics.audio_duration,
                "is_streaming": metrics.is_streaming,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.metrics_data["tts"].append(metrics_data)
            
            logger.info(
                f"TTS Metrics - TTFB: {metrics.time_to_first_byte:.3f}s, "
                f"Duration: {metrics.duration:.3f}s, "
                f"Audio duration: {metrics.audio_duration:.3f}s"
            )
    
    async def _on_eou_metrics(self, metrics: EOUMetrics):
        """Handle End of Utterance metrics."""
        if self.enable_metrics:
            metrics_data = {
                "end_of_utterance_delay": metrics.end_of_utterance_delay,
                "transcription_delay": metrics.transcription_delay,
                "timestamp": asyncio.get_event_loop().time()
            }
            self.metrics_data["eou"].append(metrics_data)
            
            logger.info(
                f"EOU Metrics - EOU delay: {metrics.end_of_utterance_delay:.3f}s, "
                f"Transcription delay: {metrics.transcription_delay:.3f}s"
            )
    
    async def on_participant_connected(self, participant):
        """Handle when a participant joins the conversation."""
        logger.info(f"Participant connected: {participant.identity}")
        
        # Send a welcome message
        await self.say("Hello! I'm your AI voice assistant. How can I help you today?")
    
    async def on_participant_disconnected(self, participant):
        """Handle when a participant leaves the conversation."""
        logger.info(f"Participant disconnected: {participant.identity}")
    
    async def on_interruption(self):
        """Handle when the user interrupts the assistant."""
        logger.info("User interruption detected")
        # You can add custom logic here, like stopping ongoing processes
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        if not self.enable_metrics:
            return {"error": "Metrics not enabled"}
        
        summary = {}
        
        # LLM metrics summary
        if self.metrics_data["llm"]:
            llm_metrics = self.metrics_data["llm"]
            summary["llm"] = {
                "total_requests": len(llm_metrics),
                "avg_tokens_per_second": sum(m["tokens_per_second"] for m in llm_metrics) / len(llm_metrics),
                "avg_time_to_first_token": sum(m["time_to_first_token"] for m in llm_metrics) / len(llm_metrics),
                "total_tokens": sum(m["completion_tokens"] for m in llm_metrics),
            }
        
        # STT metrics summary
        if self.metrics_data["stt"]:
            stt_metrics = self.metrics_data["stt"]
            summary["stt"] = {
                "total_requests": len(stt_metrics),
                "avg_duration": sum(m["duration"] for m in stt_metrics) / len(stt_metrics),
                "streaming_percentage": sum(1 for m in stt_metrics if m["is_streaming"]) / len(stt_metrics) * 100,
            }
        
        # TTS metrics summary
        if self.metrics_data["tts"]:
            tts_metrics = self.metrics_data["tts"]
            summary["tts"] = {
                "total_requests": len(tts_metrics),
                "avg_time_to_first_byte": sum(m["time_to_first_byte"] for m in tts_metrics) / len(tts_metrics),
                "avg_duration": sum(m["duration"] for m in tts_metrics) / len(tts_metrics),
            }
        
        return summary
    
    async def reset_metrics(self):
        """Reset all collected metrics."""
        if self.enable_metrics:
            self.metrics_data = {
                "llm": [],
                "stt": [],
                "tts": [],
                "eou": []
            }
            logger.info("Metrics reset")


class InterruptibleVoiceAssistant(AdvancedVoiceAssistant):
    """
    Voice assistant optimized for natural interruption handling.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Configure for better interruption handling
        self.interrupt_threshold = 0.5  # Lower threshold for more sensitive interruption
        self.vad.min_silence_duration = 0.3  # Shorter silence before turn detection
        
    async def on_interruption(self):
        """Enhanced interruption handling."""
        logger.info("User interruption detected - stopping current response")
        
        # Stop any ongoing TTS generation
        await self.stop_speaking()
        
        # You could add logic here to:
        # - Save the conversation state
        # - Prepare for the new user input
        # - Send an acknowledgment like "Yes?" or "Go ahead"