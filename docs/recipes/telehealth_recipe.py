#!/usr/bin/env python3
"""
Telehealth Voice Agent Recipe

A comprehensive telehealth voice assistant designed for medical consultations,
patient intake, symptom assessment, and healthcare support conversations.

Features:
- Medical terminology optimization
- HIPAA-compliant conversation handling
- Symptom collection and assessment
- Appointment scheduling support
- Emergency situation detection
- Professional medical tone
- Patient confidentiality reminders

⚠️  DISCLAIMER: This is a demo implementation. For production medical use,
   ensure compliance with healthcare regulations (HIPAA, etc.) and have
   medical professionals review all interactions.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add SDK to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sdk.python_sdk import initialize_sdk, VoiceAgentBuilder, AgentCapability
from sdk.exceptions import VoiceAgentError
from sdk.config_manager import SDKConfig

logger = logging.getLogger(__name__)


class TelehealthAgent:
    """
    Specialized voice agent for telehealth applications.
    
    Provides medical consultation support with appropriate safeguards,
    professional tone, and healthcare-specific functionality.
    """
    
    def __init__(self, provider_name: str = "Healthcare Assistant", specialization: Optional[str] = None):
        self.provider_name = provider_name
        self.specialization = specialization
        self.agent = None
        self.session_data = {
            "start_time": None,
            "patient_concerns": [],
            "symptoms_mentioned": [],
            "medications_mentioned": [],
            "emergency_flags": [],
            "appointment_requests": []
        }
    
    async def initialize(self):
        """Initialize the telehealth agent with medical-specific configuration."""
        logger.info("Initializing Telehealth Voice Agent...")
        
        # Initialize SDK with healthcare-optimized config
        config = SDKConfig(
            project_name="telehealth-agent",
            log_level="INFO"
        )
        await initialize_sdk(config)
        
        # Build specialized agent
        system_prompt = self._build_medical_system_prompt()
        
        self.agent = (VoiceAgentBuilder()
            .with_name(f"{self.provider_name} - Telehealth Assistant")
            .with_stt("azure",  # Azure often better for medical terminology
                language="en-US",
                enable_medical_vocabulary=True,
                enable_automatic_punctuation=True,
                enable_profanity_filter=False  # Medical terms might be flagged
            )
            .with_llm("openai",
                model="gpt-4",  # More capable for medical conversations
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=300   # Longer responses for detailed explanations
            )
            .with_tts("azure",
                voice_name="en-US-JennyNeural",  # Professional, clear voice
                language="en-US"
            )
            .with_vad("silero")
            .with_capabilities(
                AgentCapability.TURN_DETECTION,
                AgentCapability.INTERRUPTION_HANDLING,
                AgentCapability.CONTEXT_MANAGEMENT,
                AgentCapability.CONVERSATION_STATE
            )
            .with_system_prompt(system_prompt)
            .with_custom_vocabulary([
                # Common medical terms that might be misheard
                "hypertension", "diabetes", "myocardial", "infarction",
                "prescription", "medication", "dosage", "milligrams",
                "symptoms", "diagnosis", "prognosis", "chronic",
                "acute", "inflammation", "infection", "allergy"
            ])
            .build())
        
        # Setup medical-specific callbacks
        self._setup_medical_callbacks()
        
        logger.info("Telehealth agent initialized successfully!")
    
    def _build_medical_system_prompt(self) -> str:
        """Build a comprehensive system prompt for medical consultations."""
        specialization_text = f" specializing in {self.specialization}" if self.specialization else ""
        
        return f"""You are a professional healthcare voice assistant{specialization_text} named {self.provider_name}.

CRITICAL GUIDELINES:
1. You are NOT a doctor and cannot provide medical diagnoses or treatment advice
2. Always remind patients to consult with their healthcare provider for medical decisions
3. You can help collect symptoms, schedule appointments, and provide general health information
4. If someone mentions emergency symptoms, immediately advise them to call 911 or go to ER
5. Maintain patient confidentiality and never share personal health information
6. Be empathetic, professional, and reassuring
7. Ask clarifying questions to better understand patient concerns
8. Speak clearly and avoid medical jargon unless necessary

EMERGENCY KEYWORDS to watch for:
- Chest pain, difficulty breathing, severe bleeding, unconsciousness
- Stroke symptoms: sudden confusion, trouble speaking, severe headache
- Allergic reactions: swelling, difficulty breathing
- Severe injuries, poisoning, overdose

Your role is to:
- Collect patient information and symptoms
- Schedule appointments and reminders
- Provide general health education
- Escalate urgent situations appropriately
- Offer emotional support and reassurance

Always be compassionate and professional. Remember that patients may be anxious or in distress."""
    
    def _setup_medical_callbacks(self):
        """Setup callbacks specific to medical consultations."""
        
        @self.agent.on_start
        async def on_start():
            self.session_data["start_time"] = datetime.now()
            logger.info("🏥 Telehealth session started")
            print("\n" + "="*60)
            print("🏥 Telehealth Voice Assistant Ready")
            print("="*60)
            print(f"Provider: {self.provider_name}")
            if self.specialization:
                print(f"Specialization: {self.specialization}")
            print("⚠️  This is a healthcare assistant, not a replacement for medical care")
            print("🚨 For emergencies, call 911 immediately")
            print("="*60)
        
        @self.agent.on_user_speech
        async def on_user_speech(text: str):
            # Analyze patient input for medical content
            await self._analyze_patient_input(text)
            logger.info(f"Patient: {text}")
            print(f"\n👤 Patient: {text}")
        
        @self.agent.on_agent_speech
        async def on_agent_speech(text: str):
            logger.info(f"Healthcare Assistant: {text}")
            print(f"🏥 {self.provider_name}: {text}")
        
        @self.agent.on_error
        async def on_error(error: Exception):
            logger.error(f"Healthcare agent error: {error}")
            print(f"❌ System error: {error}")
            # In production, this should alert technical support
    
    async def _analyze_patient_input(self, text: str):
        """Analyze patient input for medical keywords and concerns."""
        text_lower = text.lower()
        
        # Check for emergency keywords
        emergency_keywords = [
            "chest pain", "can't breathe", "difficulty breathing", "severe bleeding",
            "unconscious", "overdose", "poisoning", "severe headache",
            "can't speak", "can't move", "severe allergic reaction"
        ]
        
        for keyword in emergency_keywords:
            if keyword in text_lower:
                self.session_data["emergency_flags"].append({
                    "keyword": keyword,
                    "timestamp": datetime.now(),
                    "context": text
                })
                logger.warning(f"EMERGENCY KEYWORD DETECTED: {keyword}")
                # In production, this would trigger immediate escalation
        
        # Collect symptoms
        symptom_keywords = [
            "pain", "ache", "fever", "nausea", "vomiting", "dizziness",
            "headache", "cough", "fatigue", "weakness", "rash", "swelling"
        ]
        
        for symptom in symptom_keywords:
            if symptom in text_lower:
                self.session_data["symptoms_mentioned"].append({
                    "symptom": symptom,
                    "timestamp": datetime.now(),
                    "context": text
                })
        
        # Check for medication mentions
        medication_keywords = [
            "medication", "medicine", "prescription", "pills", "dosage",
            "mg", "milligrams", "taking", "prescribed"
        ]
        
        for med_keyword in medication_keywords:
            if med_keyword in text_lower:
                self.session_data["medications_mentioned"].append({
                    "keyword": med_keyword,
                    "timestamp": datetime.now(),
                    "context": text
                })
        
        # Check for appointment requests
        appointment_keywords = [
            "appointment", "schedule", "book", "see doctor", "visit",
            "consultation", "check-up", "follow-up"
        ]
        
        for appt_keyword in appointment_keywords:
            if appt_keyword in text_lower:
                self.session_data["appointment_requests"].append({
                    "type": appt_keyword,
                    "timestamp": datetime.now(),
                    "context": text
                })
    
    async def start(self):
        """Start the telehealth agent."""
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        await self.agent.start()
    
    async def stop(self):
        """Stop the telehealth agent and generate session summary."""
        if self.agent and self.agent.is_running:
            await self.agent.stop()
        
        # Generate session summary
        await self._generate_session_summary()
    
    async def cleanup(self):
        """Clean up agent resources."""
        if self.agent:
            await self.agent.cleanup()
    
    async def _generate_session_summary(self):
        """Generate a summary of the telehealth session."""
        if not self.session_data["start_time"]:
            return
        
        duration = datetime.now() - self.session_data["start_time"]
        
        print("\n" + "="*60)
        print("📋 Telehealth Session Summary")
        print("="*60)
        print(f"Duration: {duration}")
        print(f"Provider: {self.provider_name}")
        
        if self.session_data["emergency_flags"]:
            print(f"\n🚨 EMERGENCY FLAGS: {len(self.session_data['emergency_flags'])}")
            for flag in self.session_data["emergency_flags"]:
                print(f"   - {flag['keyword']} at {flag['timestamp'].strftime('%H:%M:%S')}")
        
        if self.session_data["symptoms_mentioned"]:
            print(f"\n🩺 Symptoms Mentioned: {len(self.session_data['symptoms_mentioned'])}")
            unique_symptoms = set(s["symptom"] for s in self.session_data["symptoms_mentioned"])
            for symptom in unique_symptoms:
                print(f"   - {symptom}")
        
        if self.session_data["medications_mentioned"]:
            print(f"\n💊 Medication References: {len(self.session_data['medications_mentioned'])}")
        
        if self.session_data["appointment_requests"]:
            print(f"\n📅 Appointment Requests: {len(self.session_data['appointment_requests'])}")
        
        print("="*60)
        
        # In production, this summary would be saved to patient records
        logger.info("Session summary generated")
    
    def get_session_data(self) -> Dict[str, Any]:
        """Get current session data for integration with EMR systems."""
        return self.session_data.copy()


# Specialized telehealth agents for different medical areas

class PrimaryCareTelehealthAgent(TelehealthAgent):
    """Primary care focused telehealth agent."""
    
    def __init__(self):
        super().__init__(
            provider_name="Primary Care Assistant",
            specialization="Primary Care Medicine"
        )


class MentalHealthTelehealthAgent(TelehealthAgent):
    """Mental health focused telehealth agent."""
    
    def __init__(self):
        super().__init__(
            provider_name="Mental Health Assistant", 
            specialization="Mental Health Support"
        )
    
    def _build_medical_system_prompt(self) -> str:
        """Override with mental health specific prompt."""
        return """You are a compassionate mental health support voice assistant.

CRITICAL GUIDELINES:
1. You are NOT a licensed therapist or psychiatrist
2. You cannot provide therapy or prescribe treatments
3. If someone mentions self-harm or suicide, immediately provide crisis resources
4. Maintain absolute confidentiality and create a safe space
5. Be non-judgmental, empathetic, and supportive
6. Help schedule appointments with mental health professionals
7. Provide general wellness information and coping strategies

CRISIS KEYWORDS to watch for:
- Suicide, self-harm, wanting to die, hopeless
- Harm to others, violence
- Severe depression, panic attacks
- Substance abuse crisis

CRISIS RESOURCES:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- Emergency: 911

Your role is to:
- Provide emotional support and active listening
- Help schedule mental health appointments
- Offer general wellness and self-care information
- Connect users with appropriate crisis resources
- Encourage professional mental health care

Always be compassionate, patient, and understanding. Mental health conversations require extra sensitivity."""


class PediatricTelehealthAgent(TelehealthAgent):
    """Pediatric focused telehealth agent."""
    
    def __init__(self):
        super().__init__(
            provider_name="Pediatric Assistant",
            specialization="Pediatric Medicine"
        )
    
    def _build_medical_system_prompt(self) -> str:
        """Override with pediatric specific prompt."""
        return """You are a friendly pediatric healthcare voice assistant.

CRITICAL GUIDELINES:
1. You work with children and their parents/caregivers
2. Use age-appropriate language that children can understand
3. Always involve parents/guardians in medical discussions
4. Be extra vigilant about pediatric emergency symptoms
5. Make the experience less scary for children
6. Encourage questions from both children and parents

PEDIATRIC EMERGENCY SYMPTOMS:
- High fever (over 102°F in infants under 3 months)
- Difficulty breathing, severe cough
- Dehydration signs, excessive vomiting
- Severe allergic reactions
- Head injuries, loss of consciousness

Your role is to:
- Help parents describe their child's symptoms
- Provide child-friendly explanations
- Schedule pediatric appointments
- Offer general child health information
- Calm anxious parents and children

Be warm, friendly, and reassuring. Remember that parents may be very worried about their children."""


async def demo_telehealth_agent():
    """Demonstrate the telehealth agent functionality."""
    print("🏥 Telehealth Voice Agent Demo")
    print("="*50)
    
    # Choose agent type
    print("Choose telehealth specialization:")
    print("1. Primary Care (default)")
    print("2. Mental Health")
    print("3. Pediatrics")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled")
        return
    
    # Create appropriate agent
    if choice == "2":
        agent = MentalHealthTelehealthAgent()
    elif choice == "3":
        agent = PediatricTelehealthAgent()
    else:
        agent = PrimaryCareTelehealthAgent()
    
    try:
        # Initialize and start
        await agent.initialize()
        await agent.start()
        
        # Run conversation
        while agent.agent.is_running:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Ending telehealth session...")
    
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo error: {e}")
    
    finally:
        await agent.stop()
        await agent.cleanup()
        print("✅ Telehealth session ended")


if __name__ == "__main__":
    # Environment check
    import os
    
    required_keys = ["OPENAI_API_KEY", "AZURE_SUBSCRIPTION_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("❌ Missing required environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n💡 For telehealth, Azure Speech is recommended for medical terminology")
        sys.exit(1)
    
    # Medical disclaimer
    print("⚠️  MEDICAL DISCLAIMER")
    print("="*50)
    print("This is a demonstration telehealth voice assistant.")
    print("It is NOT a substitute for professional medical advice.")
    print("For medical emergencies, call 911 immediately.")
    print("Always consult with healthcare professionals for medical decisions.")
    print("="*50)
    
    try:
        confirmation = input("Do you understand and agree? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Demo cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        sys.exit(0)
    
    # Run demo
    try:
        asyncio.run(demo_telehealth_agent())
    except KeyboardInterrupt:
        print("\n👋 Demo ended by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)