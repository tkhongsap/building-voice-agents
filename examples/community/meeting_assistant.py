"""
Meeting Assistant Example

An advanced voice assistant designed for meeting support and productivity.
Demonstrates real-time meeting transcription, action item tracking, 
and participant management.

Features:
- Real-time meeting transcription
- Action item detection and tracking
- Speaker identification
- Meeting summarization
- Calendar integration
- Note-taking capabilities

Usage:
    python meeting_assistant.py --mode [live|demo|replay]
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from src.sdk.python_sdk import VoiceAgentSDK
from src.agents.voice_assistant import VoiceAssistant
from src.components.stt.azure_stt import AzureSTT
from src.components.llm.openai_llm import OpenAILLM
from src.components.tts.elevenlabs_tts import ElevenLabsTTS
from src.components.vad.silero_vad import SileroVAD


@dataclass
class MeetingParticipant:
    """Meeting participant information."""
    name: str
    role: Optional[str] = None
    email: Optional[str] = None
    speaking_time: float = 0.0
    contributions: int = 0


@dataclass
class ActionItem:
    """Action item from meeting."""
    description: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "medium"
    status: str = "pending"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MeetingSession:
    """Meeting session data."""
    meeting_id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    participants: List[MeetingParticipant] = None
    transcript: List[Dict[str, Any]] = None
    action_items: List[ActionItem] = None
    summary: Optional[str] = None
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.transcript is None:
            self.transcript = []
        if self.action_items is None:
            self.action_items = []


class MeetingAssistant:
    """
    Advanced meeting assistant with transcription and productivity features.
    
    Provides comprehensive meeting support including:
    - Real-time transcription with speaker identification
    - Automatic action item detection
    - Meeting summarization
    - Participant analytics
    - Integration with calendar systems
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.sdk = VoiceAgentSDK()
        self.agent = None
        self.current_session: Optional[MeetingSession] = None
        self.is_recording = False
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Meeting data storage
        self.meetings_dir = Path("meetings")
        self.meetings_dir.mkdir(exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "stt": {
                "provider": "azure",
                "language": "en-US",
                "enable_speaker_recognition": True,
                "real_time_transcription": True
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 2000
            },
            "tts": {
                "provider": "elevenlabs",
                "voice_id": "professional",
                "stability": 0.7,
                "clarity": 0.8
            },
            "meeting": {
                "auto_detect_action_items": True,
                "auto_summarize": True,
                "min_action_item_confidence": 0.8,
                "speaker_identification_threshold": 0.7
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
        
        return default_config
    
    async def setup(self):
        """Setup the meeting assistant."""
        print("🔧 Setting up Meeting Assistant...")
        
        # Configure components based on config
        stt_config = {
            **self.config["stt"],
            "api_key": os.getenv("AZURE_SPEECH_KEY"),
            "region": os.getenv("AZURE_SPEECH_REGION", "eastus")
        }
        
        llm_config = {
            **self.config["llm"],
            "api_key": os.getenv("OPENAI_API_KEY"),
            "system_prompt": """You are an AI meeting assistant. Your role is to:
            1. Help identify and extract action items from conversations
            2. Summarize key points and decisions
            3. Track meeting participants and their contributions
            4. Provide meeting insights and analytics
            
            Always be professional, concise, and accurate. Focus on actionable information."""
        }
        
        tts_config = {
            **self.config["tts"],
            "api_key": os.getenv("ELEVENLABS_API_KEY")
        }
        
        vad_config = {
            "sensitivity": 0.7,
            "min_speech_duration": 0.3,
            "max_silence_duration": 2.0
        }
        
        # Create agent
        self.agent = await self.sdk.create_agent(
            stt_provider=self.config["stt"]["provider"],
            stt_config=stt_config,
            llm_provider=self.config["llm"]["provider"],
            llm_config=llm_config,
            tts_provider=self.config["tts"]["provider"],
            tts_config=tts_config,
            vad_provider="silero",
            vad_config=vad_config
        )
        
        # Register meeting-specific functions
        self._register_meeting_functions()
        
        print("✅ Meeting Assistant ready!")
    
    def _register_meeting_functions(self):
        """Register meeting-specific functions."""
        
        @self.agent.function(
            name="start_meeting",
            description="Start a new meeting session",
            parameters={
                "title": {"type": "string", "description": "Meeting title"},
                "participants": {"type": "array", "items": {"type": "string"}, "description": "List of participant names"}
            }
        )
        async def start_meeting(title: str, participants: List[str] = None) -> str:
            """Start a new meeting session."""
            meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = MeetingSession(
                meeting_id=meeting_id,
                title=title,
                start_time=datetime.now()
            )
            
            if participants:
                for name in participants:
                    self.current_session.participants.append(
                        MeetingParticipant(name=name)
                    )
            
            self.is_recording = True
            
            print(f"📝 Meeting started: {title}")
            print(f"   ID: {meeting_id}")
            print(f"   Participants: {', '.join(participants) if participants else 'None specified'}")
            
            return f"Meeting '{title}' started successfully. Recording and transcription active."
        
        @self.agent.function(
            name="add_action_item",
            description="Add an action item to the current meeting",
            parameters={
                "description": {"type": "string", "description": "Action item description"},
                "assignee": {"type": "string", "description": "Person assigned to the action"},
                "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD format)"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Priority level"}
            }
        )
        async def add_action_item(description: str, assignee: str = None, due_date: str = None, priority: str = "medium") -> str:
            """Add an action item to the current meeting."""
            if not self.current_session:
                return "No active meeting session. Please start a meeting first."
            
            action_item = ActionItem(
                description=description,
                assignee=assignee,
                due_date=due_date,
                priority=priority
            )
            
            self.current_session.action_items.append(action_item)
            
            print(f"📋 Action item added: {description}")
            if assignee:
                print(f"   Assigned to: {assignee}")
            if due_date:
                print(f"   Due: {due_date}")
            
            return f"Action item added successfully. Total action items: {len(self.current_session.action_items)}"
        
        @self.agent.function(
            name="get_meeting_summary",
            description="Generate a summary of the current meeting"
        )
        async def get_meeting_summary() -> str:
            """Generate a meeting summary."""
            if not self.current_session:
                return "No active meeting session."
            
            summary = await self._generate_meeting_summary()
            return summary
        
        @self.agent.function(
            name="end_meeting",
            description="End the current meeting and save the session"
        )
        async def end_meeting() -> str:
            """End the current meeting session."""
            if not self.current_session:
                return "No active meeting session."
            
            self.current_session.end_time = datetime.now()
            self.is_recording = False
            
            # Generate final summary
            self.current_session.summary = await self._generate_meeting_summary()
            
            # Save meeting data
            await self._save_meeting_session()
            
            duration = self.current_session.end_time - self.current_session.start_time
            
            print(f"🏁 Meeting ended: {self.current_session.title}")
            print(f"   Duration: {duration}")
            print(f"   Action items: {len(self.current_session.action_items)}")
            print(f"   Participants: {len(self.current_session.participants)}")
            
            result = f"Meeting ended successfully. Duration: {duration}. Summary and action items saved."
            self.current_session = None
            
            return result
        
        @self.agent.function(
            name="get_action_items",
            description="List all action items from the current meeting"
        )
        async def get_action_items() -> str:
            """Get all action items from the current meeting."""
            if not self.current_session or not self.current_session.action_items:
                return "No action items in the current meeting."
            
            items_text = "Current Action Items:\n"
            for i, item in enumerate(self.current_session.action_items, 1):
                items_text += f"{i}. {item.description}"
                if item.assignee:
                    items_text += f" (Assigned to: {item.assignee})"
                if item.due_date:
                    items_text += f" (Due: {item.due_date})"
                items_text += f" [Priority: {item.priority}]\n"
            
            return items_text
    
    async def _generate_meeting_summary(self) -> str:
        """Generate an AI-powered meeting summary."""
        if not self.current_session or not self.current_session.transcript:
            return "No transcript available for summary."
        
        # Prepare transcript text
        transcript_text = "\n".join([
            f"{entry.get('speaker', 'Unknown')}: {entry.get('text', '')}"
            for entry in self.current_session.transcript
        ])
        
        # Generate summary using LLM
        summary_prompt = f"""
        Please provide a concise summary of this meeting transcript:
        
        Meeting: {self.current_session.title}
        Duration: {datetime.now() - self.current_session.start_time}
        Participants: {len(self.current_session.participants)}
        
        Transcript:
        {transcript_text}
        
        Please include:
        1. Key discussion points
        2. Decisions made
        3. Next steps
        4. Important outcomes
        
        Keep the summary professional and concise.
        """
        
        try:
            summary = await self.agent.llm.generate(summary_prompt)
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    async def _save_meeting_session(self):
        """Save the meeting session to file."""
        if not self.current_session:
            return
        
        session_file = self.meetings_dir / f"{self.current_session.meeting_id}.json"
        
        # Convert to serializable format
        session_data = {
            "meeting_id": self.current_session.meeting_id,
            "title": self.current_session.title,
            "start_time": self.current_session.start_time.isoformat(),
            "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            "participants": [asdict(p) for p in self.current_session.participants],
            "transcript": self.current_session.transcript,
            "action_items": [asdict(item) for item in self.current_session.action_items],
            "summary": self.current_session.summary
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"💾 Meeting saved to: {session_file}")
    
    async def load_meeting_session(self, meeting_id: str) -> bool:
        """Load a previous meeting session."""
        session_file = self.meetings_dir / f"{meeting_id}.json"
        
        if not session_file.exists():
            print(f"❌ Meeting {meeting_id} not found")
            return False
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Reconstruct session object
        self.current_session = MeetingSession(
            meeting_id=session_data["meeting_id"],
            title=session_data["title"],
            start_time=datetime.fromisoformat(session_data["start_time"]),
            end_time=datetime.fromisoformat(session_data["end_time"]) if session_data.get("end_time") else None
        )
        
        # Load participants
        for p_data in session_data.get("participants", []):
            self.current_session.participants.append(MeetingParticipant(**p_data))
        
        # Load action items
        for item_data in session_data.get("action_items", []):
            if "created_at" in item_data:
                item_data["created_at"] = datetime.fromisoformat(item_data["created_at"])
            self.current_session.action_items.append(ActionItem(**item_data))
        
        self.current_session.transcript = session_data.get("transcript", [])
        self.current_session.summary = session_data.get("summary")
        
        print(f"📂 Loaded meeting: {self.current_session.title}")
        return True
    
    def list_meetings(self) -> List[Dict[str, Any]]:
        """List all saved meetings."""
        meetings = []
        
        for session_file in self.meetings_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                
                meetings.append({
                    "meeting_id": session_data["meeting_id"],
                    "title": session_data["title"],
                    "start_time": session_data["start_time"],
                    "duration": session_data.get("end_time"),
                    "participants": len(session_data.get("participants", [])),
                    "action_items": len(session_data.get("action_items", []))
                })
            except Exception as e:
                print(f"Error reading {session_file}: {e}")
        
        return sorted(meetings, key=lambda x: x["start_time"], reverse=True)
    
    async def start_live_meeting(self):
        """Start a live meeting session."""
        if not self.agent:
            await self.setup()
        
        print("\n🎤 Live Meeting Mode")
        print("   Start by saying: 'Start meeting titled [meeting name]'")
        print("   During the meeting, you can:")
        print("   - Add action items: 'Add action item: [description]'")
        print("   - Get summary: 'Generate meeting summary'")
        print("   - End meeting: 'End the meeting'")
        print("   Press Ctrl+C to stop\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            print("\n🛑 Meeting session ended")
        finally:
            if self.current_session and self.is_recording:
                await self.agent.process_text("End the meeting")
    
    async def demo_mode(self):
        """Run a demonstration of meeting capabilities."""
        if not self.agent:
            await self.setup()
        
        print("\n📊 Meeting Assistant Demo")
        print("=" * 50)
        
        # Start demo meeting
        await self.agent.process_text("Start meeting titled 'Product Roadmap Review' with participants Alice, Bob, Charlie")
        
        # Simulate some conversation and action items
        demo_actions = [
            "Add action item: Review user feedback on mobile app - assigned to Alice - due 2024-03-20 - priority high",
            "Add action item: Update API documentation - assigned to Bob - due 2024-03-18 - priority medium",
            "Add action item: Schedule user testing sessions - assigned to Charlie - due 2024-03-25 - priority high"
        ]
        
        for action in demo_actions:
            print(f"\n> Processing: {action}")
            response = await self.agent.process_text(action)
            print(f"Assistant: {response}")
        
        # Get action items
        print(f"\n> Getting action items...")
        response = await self.agent.process_text("Get action items")
        print(f"Assistant: {response}")
        
        # Generate summary
        print(f"\n> Generating summary...")
        response = await self.agent.process_text("Generate meeting summary")
        print(f"Assistant: {response}")
        
        # End meeting
        print(f"\n> Ending meeting...")
        response = await self.agent.process_text("End the meeting")
        print(f"Assistant: {response}")
        
        print("\n✅ Demo completed!")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Assistant Example")
    parser.add_argument("--mode", choices=["live", "demo", "list"], default="demo",
                       help="Run mode: live meeting, demo, or list meetings")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--load", help="Load existing meeting by ID")
    
    args = parser.parse_args()
    
    print("📅 Meeting Assistant Example")
    print("=" * 50)
    
    assistant = MeetingAssistant(args.config)
    
    if args.mode == "list":
        meetings = assistant.list_meetings()
        if meetings:
            print("\n📋 Saved Meetings:")
            for meeting in meetings:
                print(f"  {meeting['meeting_id']}: {meeting['title']}")
                print(f"    Started: {meeting['start_time']}")
                print(f"    Participants: {meeting['participants']}, Action Items: {meeting['action_items']}")
        else:
            print("\n📋 No saved meetings found")
    
    elif args.load:
        await assistant.setup()
        if await assistant.load_meeting_session(args.load):
            print(f"📂 Meeting loaded: {assistant.current_session.title}")
            # Interactive mode with loaded meeting
            while True:
                try:
                    user_input = input("\nCommand: ").strip()
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    response = await assistant.agent.process_text(user_input)
                    print(f"Assistant: {response}")
                except KeyboardInterrupt:
                    break
    
    elif args.mode == "live":
        await assistant.start_live_meeting()
    
    else:  # demo mode
        await assistant.demo_mode()


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM",
        "AZURE_SPEECH_KEY": "Azure Speech Service key",
        "ELEVENLABS_API_KEY": "ElevenLabs API key"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nSet them with: export VARIABLE_NAME='your-key-here'")
        exit(1)
    
    asyncio.run(main())