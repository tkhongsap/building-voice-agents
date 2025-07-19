"""
Main entry point for the Voice Agent application.
This module sets up the LiveKit agent and handles room connections.
"""

import asyncio
import logging
import os
from typing import Callable
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import JobContext, JobRequest, WorkerOptions

from agents.voice_agent import VoiceAssistant, CustomVoiceAssistant

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    """
    Main entry point for each agent session.
    This function is called when a new room is created or a participant joins.
    
    Args:
        ctx: JobContext containing room information and connection details
    """
    logger.info(f"Agent entrypoint called for room: {ctx.room.name}")
    
    # Connect to the LiveKit room
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create the voice assistant
    assistant = CustomVoiceAssistant(
        instructions="""You are a helpful AI assistant in a voice conversation.
        Be friendly, concise, and natural in your responses.
        Remember this is a voice conversation, so avoid long monologues."""
    )
    
    # Start the agent session
    session = agents.AgentSession(
        agent=assistant,
        track_publish_options=rtc.TrackPublishOptions(
            audio=rtc.AudioTrackPublishOptions(
                source=rtc.AudioSource.MICROPHONE,
                red=True,  # Redundancy encoding for better quality
            ),
        ),
    )
    
    # Start the session with the room
    await session.start(ctx.room)
    logger.info("Agent session started")
    
    # Keep the session running
    await session.wait()
    logger.info("Agent session ended")


def request_handler(req: JobRequest) -> Callable:
    """
    Handle incoming job requests.
    This can be used to route different types of requests to different agents.
    
    Args:
        req: The job request containing room and participant information
        
    Returns:
        The entrypoint function to handle this request
    """
    logger.info(f"Received job request for room: {req.room.name}")
    
    # You can add logic here to select different agents based on the request
    # For example, based on room name, metadata, or participant info
    
    return entrypoint


def main():
    """
    Main function to start the LiveKit agent worker.
    """
    logger.info("Starting Voice Agent application...")
    
    # Create worker options
    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        ws_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
    )
    
    # Run the agent worker
    agents.run_app(opts)


if __name__ == "__main__":
    main()