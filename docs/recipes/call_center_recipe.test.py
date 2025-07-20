"""
Unit tests for call center recipe.

Tests the call center voice agent implementation including customer service,
technical support, and sales functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from call_center_recipe import (
    CallCenterAgent,
    CustomerServiceAgent,
    TechnicalSupportAgent,
    SalesAgent,
    CallPriority,
    CallCategory,
    CallMetrics,
    CustomerInfo
)


class TestCallCenterAgent:
    """Test the base call center agent."""
    
    def test_initialization(self):
        """Test call center agent initialization."""
        agent = CallCenterAgent("Test Agent", "Test Corp", "Support")
        assert agent.agent_name == "Test Agent"
        assert agent.company_name == "Test Corp"
        assert agent.department == "Support"
        assert agent.is_available is True
        assert agent.current_call is None
        assert len(agent.call_history) == 0
    
    def test_default_initialization(self):
        """Test call center agent with default parameters."""
        agent = CallCenterAgent()
        assert agent.agent_name == "Call Center Assistant"
        assert agent.company_name == "Your Company"
        assert agent.department == "Customer Service"
    
    def test_call_center_prompt_generation(self):
        """Test call center system prompt generation."""
        agent = CallCenterAgent("Agent Smith", "TechCorp", "Support")
        prompt = agent._build_call_center_prompt()
        
        # Check for key call center elements
        assert "Agent Smith" in prompt
        assert "TechCorp" in prompt
        assert "professional" in prompt.lower()
        assert "escalate" in prompt.lower()
        assert "confidentiality" in prompt.lower()
        assert "opening" in prompt.lower()
        assert "authentication" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_call_lifecycle(self):
        """Test complete call lifecycle."""
        agent = CallCenterAgent()
        
        # Mock agent initialization
        agent.agent = Mock()
        agent.agent.start = AsyncMock()
        agent.agent.stop = AsyncMock()
        
        # Start call
        call_id = await agent.start_call("+1234567890")
        assert agent.current_call is not None
        assert agent.current_call.call_id == call_id
        assert agent.customer_info is not None
        assert agent.customer_info.phone == "+1234567890"
        
        # End call
        metrics = await agent.end_call("Test resolution")
        assert metrics.end_time is not None
        assert metrics.resolution_time > timedelta(0)
        assert agent.current_call is None
        assert len(agent.call_history) == 1


class TestCallAnalysis:
    """Test call analysis and categorization."""
    
    @pytest.mark.asyncio
    async def test_escalation_detection(self):
        """Test detection of escalation triggers."""
        agent = CallCenterAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        escalation_phrases = [
            "I want to speak to your supervisor",
            "This is terrible service",
            "I'm going to sue you",
            "Cancel my account immediately"
        ]
        
        for phrase in escalation_phrases:
            agent.current_call.escalated = False  # Reset
            await agent._on_customer_speech(phrase)
        
        # Should have flagged for escalation
        assert agent.current_call.escalated is True
        assert agent.current_call.priority == CallPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_call_categorization(self):
        """Test automatic call categorization."""
        agent = CallCenterAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        category_tests = [
            ("I have a question about my bill", CallCategory.BILLING),
            ("The software is not working", CallCategory.TECHNICAL_SUPPORT),
            ("I want to buy your product", CallCategory.SALES),
            ("I want to complain about service", CallCategory.COMPLAINTS),
            ("I need to cancel my account", CallCategory.CANCELLATION),
            ("I forgot my password", CallCategory.ACCOUNT_MANAGEMENT)
        ]
        
        for phrase, expected_category in category_tests:
            agent.current_call.category = None  # Reset
            await agent._categorize_call_content(phrase)
            assert agent.current_call.category == expected_category
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self):
        """Test basic sentiment analysis."""
        agent = CallCenterAgent()
        
        # Test positive sentiment
        positive_sentiment = await agent._analyze_sentiment("Thank you, this is excellent service")
        assert positive_sentiment > 0
        
        # Test negative sentiment
        negative_sentiment = await agent._analyze_sentiment("This is terrible and awful")
        assert negative_sentiment < 0
        
        # Test neutral sentiment
        neutral_sentiment = await agent._analyze_sentiment("I need help with my account")
        assert neutral_sentiment is None  # No sentiment words
    
    def test_call_metrics_structure(self):
        """Test call metrics data structure."""
        metrics = CallMetrics("test_call", datetime.now())
        
        # Check required fields
        assert metrics.call_id == "test_call"
        assert isinstance(metrics.start_time, datetime)
        assert metrics.end_time is None
        assert metrics.priority == CallPriority.NORMAL
        assert metrics.resolved is False
        assert metrics.escalated is False
    
    def test_customer_info_structure(self):
        """Test customer info data structure."""
        customer = CustomerInfo()
        
        # Check default values
        assert customer.customer_id is None
        assert customer.account_status == "active"
        assert customer.tier == "standard"
        assert isinstance(customer.previous_calls, list)
        assert isinstance(customer.preferences, dict)


class TestCustomerServiceAgent:
    """Test customer service specific functionality."""
    
    def test_initialization(self):
        """Test customer service agent initialization."""
        agent = CustomerServiceAgent("TechCorp")
        assert agent.agent_name == "Customer Service Representative"
        assert agent.company_name == "TechCorp"
        assert agent.department == "Customer Service"
        assert agent.can_process_refunds is True
        assert agent.refund_limit == 500.0
    
    def test_customer_service_prompt(self):
        """Test customer service specific prompt."""
        agent = CustomerServiceAgent("TechCorp")
        prompt = agent._build_call_center_prompt()
        
        # Check for customer service specific content
        assert "customer satisfaction" in prompt.lower()
        assert "refund" in prompt.lower()
        assert "empathetic" in prompt.lower()
        assert f"${agent.refund_limit}" in prompt
    
    @pytest.mark.asyncio
    async def test_refund_processing(self):
        """Test refund processing functionality."""
        agent = CustomerServiceAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        # Test valid refund
        result = await agent.process_refund(100.0, "Billing error")
        assert result is True
        assert agent.current_call.resolved is True
        
        # Test refund over limit
        agent.current_call.resolved = False
        result = await agent.process_refund(1000.0, "Large refund")
        assert result is False
        assert agent.current_call.escalated is True
    
    @pytest.mark.asyncio
    async def test_refund_permission_check(self):
        """Test refund processing when not permitted."""
        agent = CustomerServiceAgent()
        agent.can_process_refunds = False
        
        result = await agent.process_refund(50.0, "Test refund")
        assert result is False


class TestTechnicalSupportAgent:
    """Test technical support specific functionality."""
    
    def test_initialization(self):
        """Test technical support agent initialization."""
        agent = TechnicalSupportAgent("TechCorp", "Level 2", "https://kb.techcorp.com")
        assert agent.agent_name == "Technical Support - Level 2"
        assert agent.company_name == "TechCorp"
        assert agent.department == "Technical Support"
        assert agent.support_level == "Level 2"
        assert agent.knowledge_base_url == "https://kb.techcorp.com"
        assert agent.can_remote_access is True  # Level 2 can remote access
    
    def test_level1_permissions(self):
        """Test Level 1 support permissions."""
        agent = TechnicalSupportAgent("TechCorp", "Level 1")
        assert agent.can_remote_access is False
        assert agent.can_create_tickets is True
    
    def test_technical_support_prompt(self):
        """Test technical support specific prompt."""
        agent = TechnicalSupportAgent("TechCorp", "Level 2")
        prompt = agent._build_call_center_prompt()
        
        # Check for technical support specific content
        assert "troubleshooting" in prompt.lower()
        assert "technical" in prompt.lower()
        assert "level 2" in prompt.lower()
        assert "reproduce the issue" in prompt.lower()
        assert "support ticket" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_support_ticket_creation(self):
        """Test support ticket creation."""
        agent = TechnicalSupportAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        agent.customer_info = CustomerInfo(customer_id="CUST123")
        
        ticket_id = await agent.create_support_ticket("Software crash issue", CallPriority.HIGH)
        
        assert ticket_id.startswith("TECH-")
        assert len(ticket_id) > 10  # Should have timestamp
    
    @pytest.mark.asyncio
    async def test_ticket_creation_permission_check(self):
        """Test ticket creation when not permitted."""
        agent = TechnicalSupportAgent()
        agent.can_create_tickets = False
        
        ticket_id = await agent.create_support_ticket("Test issue")
        assert ticket_id == ""
    
    @pytest.mark.asyncio
    async def test_technical_escalation_triggers(self):
        """Test technical-specific escalation triggers."""
        agent = TechnicalSupportAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        technical_escalations = [
            "We lost all our data",
            "There's been a security breach",
            "The server is completely down",
            "This is a critical error"
        ]
        
        for phrase in technical_escalations:
            agent.current_call.escalated = False  # Reset
            await agent._on_customer_speech(phrase)
        
        assert agent.current_call.escalated is True


class TestSalesAgent:
    """Test sales agent specific functionality."""
    
    def test_initialization(self):
        """Test sales agent initialization."""
        agent = SalesAgent("TechCorp", sales_territory="West Coast")
        assert agent.agent_name == "Sales Representative"
        assert agent.company_name == "TechCorp"
        assert agent.department == "Sales"
        assert agent.sales_territory == "West Coast"
        assert agent.can_generate_quotes is True
        assert agent.max_discount_percent == 15.0
    
    def test_sales_prompt(self):
        """Test sales specific prompt."""
        agent = SalesAgent("TechCorp")
        prompt = agent._build_call_center_prompt()
        
        # Check for sales specific content
        assert "bant" in prompt.lower()
        assert "qualify" in prompt.lower()
        assert "budget" in prompt.lower()
        assert "authority" in prompt.lower()
        assert "objections" in prompt.lower()
        assert "commitment" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_lead_qualification_high_score(self):
        """Test lead qualification with high BANT score."""
        agent = SalesAgent()
        
        customer_responses = {
            "budget": "We have $50,000 budget allocated for this project",
            "authority": "I'm the decision maker and can approve this purchase",
            "need": "We have a critical business problem that needs solving",
            "timeline": "We need to implement this solution within 3 months"
        }
        
        qualification = await agent.qualify_lead(customer_responses)
        
        assert qualification["qualification_score"] == 100
        assert qualification["recommendation"] == "Hot lead"
        assert qualification["details"]["budget"] == "Qualified"
        assert qualification["details"]["authority"] == "Qualified"
        assert qualification["details"]["need"] == "Qualified"
        assert qualification["details"]["timeline"] == "Qualified"
    
    @pytest.mark.asyncio
    async def test_lead_qualification_low_score(self):
        """Test lead qualification with low BANT score."""
        agent = SalesAgent()
        
        customer_responses = {
            "budget": "We're just looking at options",
            "authority": "I need to check with my manager",
            "need": "Not sure if we really need this",
            "timeline": "Maybe sometime next year"
        }
        
        qualification = await agent.qualify_lead(customer_responses)
        
        assert qualification["qualification_score"] == 0
        assert qualification["recommendation"] == "Cold lead"
        assert all(detail == "Needs follow-up" for detail in qualification["details"].values())
    
    @pytest.mark.asyncio
    async def test_lead_qualification_partial_score(self):
        """Test lead qualification with partial BANT score."""
        agent = SalesAgent()
        
        customer_responses = {
            "budget": "We have budget for the right solution",
            "authority": "I can make recommendations but need approval",
            "need": "We definitely have problems we need to solve",
            "timeline": "No specific timeline yet"
        }
        
        qualification = await agent.qualify_lead(customer_responses)
        
        assert qualification["qualification_score"] == 50
        assert qualification["recommendation"] == "Warm lead"
    
    @pytest.mark.asyncio
    async def test_sales_metrics_tracking(self):
        """Test sales metrics tracking."""
        agent = SalesAgent()
        
        initial_leads = agent.leads_qualified
        
        await agent.qualify_lead({
            "budget": "We have budget",
            "authority": "I'm the decision maker",
            "need": "We need this solution",
            "timeline": "Soon"
        })
        
        assert agent.leads_qualified == initial_leads + 1


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_customer_service_escalation_flow(self):
        """Test customer service escalation workflow."""
        agent = CustomerServiceAgent("TechCorp")
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        # Mock agent
        agent.agent = Mock()
        agent.agent.start = AsyncMock()
        agent.agent.stop = AsyncMock()
        
        # Start call
        await agent.start_call()
        
        # Customer escalates
        await agent._on_customer_speech("I want to speak to your supervisor immediately")
        
        # Try to process large refund (should escalate)
        await agent.process_refund(1000.0, "Customer demands refund")
        
        # End call
        metrics = await agent.end_call("Escalated to supervisor")
        
        assert metrics.escalated is True
        assert metrics.priority == CallPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_technical_support_ticket_flow(self):
        """Test technical support ticket creation workflow."""
        agent = TechnicalSupportAgent("TechCorp", "Level 1")
        agent.current_call = CallMetrics("test_call", datetime.now())
        agent.customer_info = CustomerInfo(customer_id="CUST123")
        
        # Mock agent
        agent.agent = Mock()
        agent.agent.start = AsyncMock()
        agent.agent.stop = AsyncMock()
        
        # Start call
        await agent.start_call()
        
        # Customer reports technical issue
        await agent._on_customer_speech("My application keeps crashing every time I try to save")
        
        # Create support ticket
        ticket_id = await agent.create_support_ticket("Application crash on save operation")
        
        # End call
        metrics = await agent.end_call(f"Created ticket {ticket_id}")
        
        assert ticket_id.startswith("TECH-")
        assert metrics.category == CallCategory.TECHNICAL_SUPPORT
    
    @pytest.mark.asyncio
    async def test_sales_qualification_flow(self):
        """Test sales lead qualification workflow."""
        agent = SalesAgent("TechCorp")
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        # Mock agent
        agent.agent = Mock()
        agent.agent.start = AsyncMock()
        agent.agent.stop = AsyncMock()
        
        # Start call
        await agent.start_call()
        
        # Customer shows interest
        await agent._on_customer_speech("I'm interested in purchasing your enterprise solution")
        
        # Qualify lead
        responses = {
            "budget": "We have $100k budget approved",
            "authority": "I'm the VP of Engineering, I can approve this",
            "need": "We need to solve our scaling problems",
            "timeline": "We want to implement within 6 months"
        }
        
        qualification = await agent.qualify_lead(responses)
        
        # End call
        metrics = await agent.end_call("Qualified hot lead - scheduling demo")
        
        assert qualification["recommendation"] == "Hot lead"
        assert metrics.category == CallCategory.SALES
        assert agent.leads_qualified > 0


class TestErrorHandling:
    """Test error handling in call center scenarios."""
    
    @pytest.mark.asyncio
    async def test_malformed_input_handling(self):
        """Test handling of malformed customer input."""
        agent = CallCenterAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        # Test with various malformed inputs
        malformed_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "!@#$%^&*()",  # Special characters only
            "a" * 1000,  # Very long string
        ]
        
        for input_text in malformed_inputs:
            try:
                await agent._on_customer_speech(input_text)
                # Should not raise exceptions
            except Exception as e:
                pytest.fail(f"Failed to handle malformed input '{input_text}': {e}")
    
    @pytest.mark.asyncio
    async def test_concurrent_call_handling(self):
        """Test handling of concurrent customer inputs."""
        agent = CallCenterAgent()
        agent.current_call = CallMetrics("test_call", datetime.now())
        
        inputs = [
            "I need help with billing",
            "The system is not working",
            "I want to buy something",
            "This service is terrible"
        ]
        
        # Process inputs concurrently
        tasks = [agent._on_customer_speech(text) for text in inputs]
        await asyncio.gather(*tasks)
        
        # Should have processed all inputs without errors
        assert agent.current_call.category is not None
    
    @pytest.mark.asyncio
    async def test_missing_call_context(self):
        """Test operations without active call context."""
        agent = CallCenterAgent()
        
        # Try to end call without starting one
        with pytest.raises(ValueError):
            await agent.end_call()
        
        # Try to get metrics without active call
        metrics = agent.get_call_metrics()
        assert metrics == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])