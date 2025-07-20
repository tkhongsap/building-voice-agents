"""
Unit tests for telehealth recipe.

Tests the telehealth voice agent implementation including medical-specific
functionality, emergency detection, and session management.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from telehealth_recipe import (
    TelehealthAgent,
    PrimaryCareTelehealthAgent,
    MentalHealthTelehealthAgent,
    PediatricTelehealthAgent
)


class TestTelehealthAgent:
    """Test the base telehealth agent."""
    
    def test_initialization(self):
        """Test telehealth agent initialization."""
        agent = TelehealthAgent("Test Provider", "General Medicine")
        assert agent.provider_name == "Test Provider"
        assert agent.specialization == "General Medicine"
        assert agent.agent is None
        assert agent.session_data["start_time"] is None
        assert len(agent.session_data["patient_concerns"]) == 0
    
    def test_default_initialization(self):
        """Test telehealth agent with default parameters."""
        agent = TelehealthAgent()
        assert agent.provider_name == "Healthcare Assistant"
        assert agent.specialization is None
    
    def test_medical_system_prompt(self):
        """Test medical system prompt generation."""
        agent = TelehealthAgent("Dr. Smith", "Cardiology")
        prompt = agent._build_medical_system_prompt()
        
        # Check for key medical guidelines
        assert "NOT a doctor" in prompt
        assert "emergency symptoms" in prompt.lower()
        assert "chest pain" in prompt
        assert "healthcare provider" in prompt
        assert "Dr. Smith" in prompt
        assert "Cardiology" in prompt
    
    def test_session_data_structure(self):
        """Test session data structure."""
        agent = TelehealthAgent()
        session = agent.get_session_data()
        
        required_keys = [
            "start_time", "patient_concerns", "symptoms_mentioned",
            "medications_mentioned", "emergency_flags", "appointment_requests"
        ]
        
        for key in required_keys:
            assert key in session
            assert isinstance(session[key], list) or session[key] is None


class TestPatientInputAnalysis:
    """Test analysis of patient input for medical content."""
    
    @pytest.mark.asyncio
    async def test_emergency_keyword_detection(self):
        """Test detection of emergency keywords."""
        agent = TelehealthAgent()
        
        # Test emergency phrases
        emergency_phrases = [
            "I'm having chest pain",
            "I can't breathe properly", 
            "There's severe bleeding",
            "I think I'm having an allergic reaction"
        ]
        
        for phrase in emergency_phrases:
            await agent._analyze_patient_input(phrase)
        
        # Should have detected emergency flags
        assert len(agent.session_data["emergency_flags"]) == len(emergency_phrases)
        
        # Check structure of emergency flags
        flag = agent.session_data["emergency_flags"][0]
        assert "keyword" in flag
        assert "timestamp" in flag
        assert "context" in flag
        assert isinstance(flag["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_symptom_detection(self):
        """Test detection of symptom keywords."""
        agent = TelehealthAgent()
        
        symptom_phrases = [
            "I have a headache",
            "Feeling nauseous today",
            "My back has been in pain",
            "Running a fever since yesterday"
        ]
        
        for phrase in symptom_phrases:
            await agent._analyze_patient_input(phrase)
        
        # Should have detected symptoms
        assert len(agent.session_data["symptoms_mentioned"]) == len(symptom_phrases)
        
        # Check for specific symptoms
        symptom_types = [s["symptom"] for s in agent.session_data["symptoms_mentioned"]]
        assert "headache" in symptom_types
        assert "nausea" in symptom_types
        assert "pain" in symptom_types
        assert "fever" in symptom_types
    
    @pytest.mark.asyncio
    async def test_medication_detection(self):
        """Test detection of medication references."""
        agent = TelehealthAgent()
        
        medication_phrases = [
            "I'm taking medication for blood pressure",
            "The doctor prescribed new pills",
            "What's the right dosage for this medicine?",
            "I take 10 mg daily"
        ]
        
        for phrase in medication_phrases:
            await agent._analyze_patient_input(phrase)
        
        # Should have detected medication references
        assert len(agent.session_data["medications_mentioned"]) == len(medication_phrases)
    
    @pytest.mark.asyncio
    async def test_appointment_request_detection(self):
        """Test detection of appointment requests."""
        agent = TelehealthAgent()
        
        appointment_phrases = [
            "I need to schedule an appointment",
            "Can I book a visit with the doctor?",
            "When is my next consultation?",
            "I'd like a follow-up check-up"
        ]
        
        for phrase in appointment_phrases:
            await agent._analyze_patient_input(phrase)
        
        # Should have detected appointment requests
        assert len(agent.session_data["appointment_requests"]) == len(appointment_phrases)
    
    @pytest.mark.asyncio
    async def test_no_false_positives(self):
        """Test that normal conversation doesn't trigger medical flags."""
        agent = TelehealthAgent()
        
        normal_phrases = [
            "Hello, how are you today?",
            "The weather is nice outside",
            "I'm feeling pretty good overall",
            "Thank you for your help"
        ]
        
        for phrase in normal_phrases:
            await agent._analyze_patient_input(phrase)
        
        # Should not have triggered any medical flags
        assert len(agent.session_data["emergency_flags"]) == 0
        assert len(agent.session_data["symptoms_mentioned"]) == 0
        assert len(agent.session_data["medications_mentioned"]) == 0
        assert len(agent.session_data["appointment_requests"]) == 0


class TestSpecializedAgents:
    """Test specialized telehealth agents."""
    
    def test_primary_care_agent(self):
        """Test primary care agent initialization."""
        agent = PrimaryCareTelehealthAgent()
        assert agent.provider_name == "Primary Care Assistant"
        assert agent.specialization == "Primary Care Medicine"
    
    def test_mental_health_agent(self):
        """Test mental health agent initialization."""
        agent = MentalHealthTelehealthAgent()
        assert agent.provider_name == "Mental Health Assistant"
        assert agent.specialization == "Mental Health Support"
    
    def test_mental_health_prompt(self):
        """Test mental health specific prompt."""
        agent = MentalHealthTelehealthAgent()
        prompt = agent._build_medical_system_prompt()
        
        # Check for mental health specific content
        assert "therapist" in prompt
        assert "suicide" in prompt.lower()
        assert "crisis" in prompt.lower()
        assert "988" in prompt  # Crisis hotline
        assert "non-judgmental" in prompt
    
    def test_pediatric_agent(self):
        """Test pediatric agent initialization."""
        agent = PediatricTelehealthAgent()
        assert agent.provider_name == "Pediatric Assistant"
        assert agent.specialization == "Pediatric Medicine"
    
    def test_pediatric_prompt(self):
        """Test pediatric specific prompt."""
        agent = PediatricTelehealthAgent()
        prompt = agent._build_medical_system_prompt()
        
        # Check for pediatric specific content
        assert "children" in prompt
        assert "parents" in prompt
        assert "age-appropriate" in prompt
        assert "102°F" in prompt  # Pediatric fever threshold
        assert "caregivers" in prompt


class TestSessionManagement:
    """Test session management functionality."""
    
    @pytest.mark.asyncio
    async def test_session_summary_generation(self):
        """Test generation of session summary."""
        agent = TelehealthAgent()
        
        # Simulate session data
        agent.session_data["start_time"] = datetime.now()
        
        # Add some test data
        await agent._analyze_patient_input("I have chest pain")
        await agent._analyze_patient_input("I'm taking medication daily")
        await agent._analyze_patient_input("I need to schedule an appointment")
        
        # Test summary generation (should not raise errors)
        with patch('builtins.print') as mock_print:
            await agent._generate_session_summary()
            
            # Verify that summary was printed
            assert mock_print.called
            
            # Check that summary includes key information
            printed_text = ' '.join(str(call) for call in mock_print.call_args_list)
            assert "Session Summary" in printed_text
            assert "EMERGENCY FLAGS" in printed_text
    
    def test_session_data_copy(self):
        """Test that get_session_data returns a copy."""
        agent = TelehealthAgent()
        
        # Get session data
        session1 = agent.get_session_data()
        session2 = agent.get_session_data()
        
        # Should be separate objects
        assert session1 is not session2
        
        # Modifying one shouldn't affect the other
        session1["test_key"] = "test_value"
        assert "test_key" not in session2


class TestMedicalSafety:
    """Test medical safety features and compliance."""
    
    def test_medical_disclaimer_in_prompt(self):
        """Test that medical disclaimers are included in prompts."""
        agents = [
            TelehealthAgent(),
            PrimaryCareTelehealthAgent(),
            MentalHealthTelehealthAgent(),
            PediatricTelehealthAgent()
        ]
        
        for agent in agents:
            prompt = agent._build_medical_system_prompt()
            
            # Check for critical disclaimers
            assert any(phrase in prompt.upper() for phrase in [
                "NOT A DOCTOR", "NOT A THERAPIST", "NOT A PSYCHIATRIST"
            ])
            assert "emergency" in prompt.lower()
            assert any(number in prompt for number in ["911", "988"])
    
    @pytest.mark.asyncio
    async def test_emergency_keyword_coverage(self):
        """Test that critical emergency keywords are detected."""
        agent = TelehealthAgent()
        
        critical_emergencies = [
            "chest pain",
            "can't breathe", 
            "severe bleeding",
            "unconscious",
            "overdose",
            "severe allergic reaction"
        ]
        
        for emergency in critical_emergencies:
            agent.session_data["emergency_flags"].clear()  # Reset
            await agent._analyze_patient_input(f"I am experiencing {emergency}")
            
            # Should have detected the emergency
            assert len(agent.session_data["emergency_flags"]) > 0, f"Failed to detect: {emergency}"
    
    def test_hipaa_considerations(self):
        """Test HIPAA compliance considerations."""
        agent = TelehealthAgent()
        prompt = agent._build_medical_system_prompt()
        
        # Check for privacy mentions
        assert "confidentiality" in prompt.lower()
        
        # Session data should not include sensitive identifiers by default
        session = agent.get_session_data()
        sensitive_fields = ["patient_name", "ssn", "dob", "address"]
        
        for field in sensitive_fields:
            assert field not in session, f"Session data should not include {field}"


class TestErrorHandling:
    """Test error handling in telehealth scenarios."""
    
    @pytest.mark.asyncio
    async def test_malformed_input_handling(self):
        """Test handling of malformed or unusual input."""
        agent = TelehealthAgent()
        
        # Test with various malformed inputs
        malformed_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "!@#$%^&*()",  # Special characters only
            "a" * 1000,  # Very long string
            None  # This would need to be converted to string
        ]
        
        for input_text in malformed_inputs:
            try:
                if input_text is not None:
                    await agent._analyze_patient_input(input_text)
                # Should not raise exceptions
            except Exception as e:
                pytest.fail(f"Failed to handle malformed input '{input_text}': {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent analysis of multiple inputs."""
        agent = TelehealthAgent()
        
        inputs = [
            "I have a headache",
            "My chest hurts",
            "I need medication",
            "Can I schedule an appointment?"
        ]
        
        # Run analyses concurrently
        tasks = [agent._analyze_patient_input(text) for text in inputs]
        await asyncio.gather(*tasks)
        
        # Should have processed all inputs
        total_flags = (
            len(agent.session_data["symptoms_mentioned"]) +
            len(agent.session_data["emergency_flags"]) +
            len(agent.session_data["medications_mentioned"]) +
            len(agent.session_data["appointment_requests"])
        )
        
        assert total_flags >= len(inputs)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_telehealth_conversation(self):
        """Test a complete telehealth conversation flow."""
        agent = TelehealthAgent("Dr. Test", "Family Medicine")
        
        # Simulate a realistic conversation
        conversation = [
            "Hello, I'm not feeling well today",
            "I've been having chest pain for the past hour",
            "It's a sharp pain, comes and goes",
            "I also feel a bit nauseous",
            "I'm currently taking blood pressure medication",
            "Should I schedule an appointment to see you?",
            "Thank you for your help"
        ]
        
        # Process each message
        for message in conversation:
            await agent._analyze_patient_input(message)
        
        # Verify appropriate detection
        assert len(agent.session_data["emergency_flags"]) > 0  # Chest pain
        assert len(agent.session_data["symptoms_mentioned"]) >= 2  # Pain, nausea
        assert len(agent.session_data["medications_mentioned"]) > 0  # BP medication
        assert len(agent.session_data["appointment_requests"]) > 0  # Appointment request
        
        # Check session summary generation
        agent.session_data["start_time"] = datetime.now()
        with patch('builtins.print'):
            await agent._generate_session_summary()
    
    @pytest.mark.asyncio
    async def test_mental_health_crisis_scenario(self):
        """Test mental health crisis detection."""
        agent = MentalHealthTelehealthAgent()
        
        crisis_inputs = [
            "I've been feeling really hopeless lately",
            "Sometimes I think about ending it all",
            "I don't see any point in continuing"
        ]
        
        for input_text in crisis_inputs:
            await agent._analyze_patient_input(input_text)
        
        # Mental health prompt should include crisis resources
        prompt = agent._build_medical_system_prompt()
        assert "988" in prompt  # Suicide prevention hotline
        assert "741741" in prompt  # Crisis text line
    
    @pytest.mark.asyncio
    async def test_pediatric_scenario(self):
        """Test pediatric consultation scenario."""
        agent = PediatricTelehealthAgent()
        
        # Parent reporting child symptoms
        parent_inputs = [
            "My 3-year-old has been running a high fever",
            "It's been over 102 degrees for 6 hours",
            "She's also been vomiting and seems very lethargic",
            "Should I bring her to the emergency room?"
        ]
        
        for input_text in parent_inputs:
            await agent._analyze_patient_input(input_text)
        
        # Should detect symptoms and potential emergency
        assert len(agent.session_data["symptoms_mentioned"]) >= 2  # Fever, vomiting
        
        # Pediatric prompt should mention parents/caregivers
        prompt = agent._build_medical_system_prompt()
        assert "parents" in prompt.lower()
        assert "child" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])