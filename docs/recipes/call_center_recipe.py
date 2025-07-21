"""
Call Center Voice Agent Recipe

This recipe provides pre-built voice agents optimized for call center operations,
including customer service, technical support, and sales scenarios.

Key Features:
- Call routing and queue management
- Customer authentication and verification
- Issue categorization and escalation
- CRM integration capabilities
- Call analytics and sentiment tracking
- Multi-language support
- Compliance with call center regulations

Usage:
    from call_center_recipe import CustomerServiceAgent, TechnicalSupportAgent
    
    # Create customer service agent
    agent = CustomerServiceAgent("TechCorp Support")
    
    # Create technical support agent
    tech_agent = TechnicalSupportAgent("Level 1 Support", knowledge_base_url="...")
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sdk.python_sdk import VoiceAgentSDK, initialize_sdk
from sdk.agent_builder import VoiceAgentBuilder, AgentCapability
from sdk.config_manager import SDKConfig


class CallPriority(Enum):
    """Call priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class CallCategory(Enum):
    """Call categorization types."""
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    SALES = "sales"
    COMPLAINTS = "complaints"
    GENERAL_INQUIRY = "general_inquiry"
    ACCOUNT_MANAGEMENT = "account_management"
    CANCELLATION = "cancellation"
    PRODUCT_INFO = "product_info"


@dataclass
class CallMetrics:
    """Call performance metrics."""
    call_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    wait_time: timedelta = field(default_factory=lambda: timedelta(0))
    talk_time: timedelta = field(default_factory=lambda: timedelta(0))
    hold_time: timedelta = field(default_factory=lambda: timedelta(0))
    resolution_time: timedelta = field(default_factory=lambda: timedelta(0))
    category: Optional[CallCategory] = None
    priority: CallPriority = CallPriority.NORMAL
    customer_satisfaction: Optional[int] = None  # 1-10 scale
    resolved: bool = False
    escalated: bool = False
    sentiment_score: Optional[float] = None  # -1 to 1


@dataclass
class CustomerInfo:
    """Customer information structure."""
    customer_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    account_status: str = "active"
    tier: str = "standard"  # standard, premium, enterprise
    previous_calls: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


class CallCenterAgent:
    """
    Base class for call center voice agents.
    
    Provides common functionality for customer service, technical support,
    and sales agents with call center specific features.
    """
    
    def __init__(
        self,
        agent_name: str = "Call Center Assistant",
        company_name: str = "Your Company",
        department: str = "Customer Service",
        agent_id: Optional[str] = None
    ):
        self.agent_name = agent_name
        self.company_name = company_name
        self.department = department
        self.agent_id = agent_id or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Call state
        self.current_call: Optional[CallMetrics] = None
        self.customer_info: Optional[CustomerInfo] = None
        self.call_history: List[CallMetrics] = []
        
        # Agent state
        self.agent = None
        self.is_available = True
        self.current_queue_position = 0
        
        # Call center features
        self.escalation_keywords = [
            "supervisor", "manager", "escalate", "complaint", "angry",
            "unsatisfied", "legal", "lawsuit", "cancel account", "terrible service"
        ]
        
        self.authentication_required = True
        self.max_call_duration = timedelta(minutes=30)
        self.auto_wrap_up = True
        
        # Integration callbacks
        self.crm_integration: Optional[Callable] = None
        self.ticket_system_integration: Optional[Callable] = None
        self.queue_system_integration: Optional[Callable] = None
    
    async def initialize_agent(self) -> None:
        """Initialize the voice agent with call center configuration."""
        try:
            # Initialize SDK
            sdk = await initialize_sdk({
                "project_name": f"{self.company_name}_CallCenter",
                "environment": "production",
                "enable_monitoring": True
            })
            
            # Build agent with call center optimizations
            builder = sdk.create_builder()
            
            self.agent = (builder
                .with_name(self.agent_name)
                .with_stt("openai", language="en", model="whisper-1")
                .with_llm("openai", 
                         model="gpt-4-turbo",
                         temperature=0.3,  # Lower temperature for consistency
                         max_tokens=500)
                .with_tts("openai", voice="nova", speed=1.0)
                .with_vad("silero", sensitivity=0.7)
                .with_capability(AgentCapability.TURN_DETECTION)
                .with_capability(AgentCapability.INTERRUPTION_HANDLING)
                .with_capability(AgentCapability.CONTEXT_MANAGEMENT)
                .with_system_prompt(self._build_call_center_prompt())
                .with_callback("on_start", self._on_call_start)
                .with_callback("on_stop", self._on_call_end)
                .with_callback("on_user_speech", self._on_customer_speech)
                .build())
            
            print(f"✅ Call center agent '{self.agent_name}' initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize call center agent: {e}")
            raise
    
    def _build_call_center_prompt(self) -> str:
        """Build the system prompt for call center operations."""
        return f"""You are {self.agent_name}, a professional call center representative for {self.company_name} in the {self.department} department.

CALL CENTER GUIDELINES:
1. Always be professional, empathetic, and solution-focused
2. Listen actively and acknowledge customer concerns
3. Use positive language and avoid negative phrases
4. Follow the company's policies and procedures
5. Escalate to supervisor when appropriate
6. Maintain customer confidentiality at all times

CONVERSATION STRUCTURE:
1. **Opening**: Greet customer professionally with company name and your name
2. **Authentication**: Verify customer identity if required
3. **Issue Discovery**: Listen carefully to understand the customer's needs
4. **Resolution**: Provide solutions or next steps
5. **Verification**: Confirm customer satisfaction with the resolution
6. **Closing**: Thank customer and offer additional assistance

ESCALATION TRIGGERS:
- Customer requests supervisor/manager
- Complex technical issues beyond your knowledge
- Billing disputes over $500
- Legal threats or compliance issues
- Customer expressing extreme dissatisfaction
- Issues requiring policy exceptions

COMPANY INFORMATION:
- Company: {self.company_name}
- Department: {self.department}
- Agent ID: {self.agent_id}

IMPORTANT REMINDERS:
- Never promise what you cannot deliver
- Always get customer confirmation before making account changes
- Document all interactions accurately
- Follow up on promises made to customers
- Maintain a helpful and professional tone throughout the call

If you need to put the customer on hold, explain why and give an estimated time.
If you need to transfer the call, explain the reason and what the next agent will help with.
"""
    
    async def start_call(self, customer_phone: Optional[str] = None) -> str:
        """Start a new call session."""
        if not self.agent:
            await self.initialize_agent()
        
        # Create new call metrics
        call_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.current_call = CallMetrics(
            call_id=call_id,
            start_time=datetime.now()
        )
        
        # Initialize customer info
        self.customer_info = CustomerInfo()
        if customer_phone:
            self.customer_info.phone = customer_phone
            # In real implementation, lookup customer by phone
            await self._lookup_customer_by_phone(customer_phone)
        
        # Start the agent
        await self.agent.start()
        
        print(f"📞 Call started - ID: {call_id}")
        return call_id
    
    async def end_call(self, resolution_notes: str = "") -> CallMetrics:
        """End the current call and generate metrics."""
        if not self.current_call:
            raise ValueError("No active call to end")
        
        # Update call metrics
        self.current_call.end_time = datetime.now()
        self.current_call.resolution_time = (
            self.current_call.end_time - self.current_call.start_time
        )
        
        # Stop the agent
        if self.agent:
            await self.agent.stop()
        
        # Generate call summary
        await self._generate_call_summary(resolution_notes)
        
        # Store in history
        self.call_history.append(self.current_call)
        completed_call = self.current_call
        self.current_call = None
        
        print(f"📞 Call ended - Duration: {completed_call.resolution_time}")
        return completed_call
    
    async def _on_call_start(self) -> None:
        """Callback when call starts."""
        if self.current_call:
            print(f"📞 Call {self.current_call.call_id} started")
    
    async def _on_call_end(self) -> None:
        """Callback when call ends."""
        if self.current_call:
            print(f"📞 Call {self.current_call.call_id} ending")
    
    async def _on_customer_speech(self, text: str) -> None:
        """Analyze customer speech for call center insights."""
        if not self.current_call or not text:
            return
        
        text_lower = text.lower()
        
        # Check for escalation triggers
        if any(keyword in text_lower for keyword in self.escalation_keywords):
            await self._flag_for_escalation(text)
        
        # Categorize call content
        await self._categorize_call_content(text)
        
        # Sentiment analysis (basic)
        sentiment = await self._analyze_sentiment(text)
        if sentiment:
            self.current_call.sentiment_score = sentiment
    
    async def _flag_for_escalation(self, trigger_text: str) -> None:
        """Flag call for escalation based on customer speech."""
        if self.current_call:
            self.current_call.escalated = True
            self.current_call.priority = CallPriority.HIGH
            print(f"🚨 Call flagged for escalation: {trigger_text[:50]}...")
    
    async def _categorize_call_content(self, text: str) -> None:
        """Categorize call based on content."""
        text_lower = text.lower()
        
        category_keywords = {
            CallCategory.BILLING: ["bill", "charge", "payment", "invoice", "refund", "cost"],
            CallCategory.TECHNICAL_SUPPORT: ["not working", "error", "bug", "broken", "issue", "problem"],
            CallCategory.SALES: ["buy", "purchase", "upgrade", "pricing", "quote", "demo"],
            CallCategory.COMPLAINTS: ["complain", "disappointed", "terrible", "worst", "awful"],
            CallCategory.CANCELLATION: ["cancel", "close account", "terminate", "stop service"],
            CallCategory.ACCOUNT_MANAGEMENT: ["password", "login", "account", "profile", "settings"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                if self.current_call:
                    self.current_call.category = category
                break
    
    async def _analyze_sentiment(self, text: str) -> Optional[float]:
        """Basic sentiment analysis of customer speech."""
        # Simple keyword-based sentiment (in production, use proper NLP)
        positive_words = ["thank", "great", "excellent", "good", "helpful", "satisfied"]
        negative_words = ["angry", "frustrated", "terrible", "awful", "bad", "disappointed"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > 0 or negative_count > 0:
            total_sentiment_words = positive_count + negative_count
            return (positive_count - negative_count) / total_sentiment_words
        
        return None
    
    async def _lookup_customer_by_phone(self, phone: str) -> None:
        """Lookup customer information by phone number."""
        # Simulate CRM lookup
        if self.customer_info and self.crm_integration:
            try:
                customer_data = await self.crm_integration(phone)
                if customer_data:
                    self.customer_info = CustomerInfo(**customer_data)
            except Exception as e:
                print(f"⚠️ CRM lookup failed: {e}")
    
    async def _generate_call_summary(self, resolution_notes: str) -> None:
        """Generate a comprehensive call summary."""
        if not self.current_call:
            return
        
        print("\n" + "="*60)
        print("📊 CALL SUMMARY")
        print("="*60)
        print(f"Call ID: {self.current_call.call_id}")
        print(f"Agent: {self.agent_name} ({self.agent_id})")
        print(f"Department: {self.department}")
        print(f"Start Time: {self.current_call.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {self.current_call.resolution_time}")
        print(f"Category: {self.current_call.category.value if self.current_call.category else 'Not categorized'}")
        print(f"Priority: {self.current_call.priority.value}")
        print(f"Escalated: {'Yes' if self.current_call.escalated else 'No'}")
        print(f"Resolved: {'Yes' if self.current_call.resolved else 'No'}")
        
        if self.current_call.sentiment_score is not None:
            sentiment_label = "Positive" if self.current_call.sentiment_score > 0 else "Negative" if self.current_call.sentiment_score < 0 else "Neutral"
            print(f"Customer Sentiment: {sentiment_label} ({self.current_call.sentiment_score:.2f})")
        
        if self.customer_info and self.customer_info.customer_id:
            print(f"Customer ID: {self.customer_info.customer_id}")
            print(f"Customer Tier: {self.customer_info.tier}")
        
        if resolution_notes:
            print(f"Resolution Notes: {resolution_notes}")
        
        print("="*60)
    
    def get_call_metrics(self) -> Dict[str, Any]:
        """Get current call metrics."""
        if not self.current_call:
            return {}
        
        return {
            "call_id": self.current_call.call_id,
            "duration": str(self.current_call.resolution_time) if self.current_call.end_time else "Ongoing",
            "category": self.current_call.category.value if self.current_call.category else None,
            "priority": self.current_call.priority.value,
            "escalated": self.current_call.escalated,
            "sentiment_score": self.current_call.sentiment_score
        }


class CustomerServiceAgent(CallCenterAgent):
    """
    Specialized agent for general customer service.
    
    Handles billing inquiries, account management, general questions,
    and customer complaints with emphasis on resolution and satisfaction.
    """
    
    def __init__(self, company_name: str = "Your Company", agent_id: Optional[str] = None):
        super().__init__(
            agent_name="Customer Service Representative",
            company_name=company_name,
            department="Customer Service",
            agent_id=agent_id
        )
        
        # Customer service specific settings
        self.can_process_refunds = True
        self.refund_limit = 500.0  # USD
        self.can_modify_accounts = True
        self.escalation_threshold = 2  # Number of unresolved issues before escalation
    
    def _build_call_center_prompt(self) -> str:
        """Build customer service specific prompt."""
        base_prompt = super()._build_call_center_prompt()
        
        return base_prompt + f"""

CUSTOMER SERVICE SPECIFIC GUIDELINES:
- Focus on resolution and customer satisfaction
- Be empathetic and understanding
- Offer alternatives when direct resolution isn't possible
- Process refunds up to ${self.refund_limit} when appropriate
- Update customer accounts as needed
- Follow up on previous issues
- Always end with "Is there anything else I can help you with today?"

COMMON ISSUES & SOLUTIONS:
- Billing questions: Review account, explain charges, process adjustments
- Account access: Reset passwords, update contact information
- Service complaints: Acknowledge concern, investigate, provide resolution
- General inquiries: Provide accurate information, direct to resources
- Refund requests: Review policy, process if eligible, explain decisions

Remember: Your goal is to turn every customer interaction into a positive experience.
"""

    async def process_refund(self, amount: float, reason: str) -> bool:
        """Process a customer refund if within limits."""
        if not self.can_process_refunds:
            print("❌ This agent cannot process refunds")
            return False
        
        if amount > self.refund_limit:
            print(f"❌ Refund amount ${amount} exceeds limit ${self.refund_limit}")
            await self._flag_for_escalation(f"Refund request: ${amount}")
            return False
        
        # Simulate refund processing
        print(f"✅ Processing refund: ${amount} for reason: {reason}")
        
        if self.current_call:
            self.current_call.resolved = True
        
        return True


class TechnicalSupportAgent(CallCenterAgent):
    """
    Specialized agent for technical support.
    
    Handles technical issues, troubleshooting, product support,
    and escalates complex technical problems to specialists.
    """
    
    def __init__(
        self,
        company_name: str = "Your Company",
        support_level: str = "Level 1",
        knowledge_base_url: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        super().__init__(
            agent_name=f"Technical Support - {support_level}",
            company_name=company_name,
            department="Technical Support",
            agent_id=agent_id
        )
        
        self.support_level = support_level
        self.knowledge_base_url = knowledge_base_url
        
        # Technical support specific settings
        self.can_create_tickets = True
        self.can_remote_access = support_level in ["Level 2", "Level 3"]
        self.escalation_triggers.extend([
            "data loss", "security breach", "server down", "critical error"
        ])
    
    def _build_call_center_prompt(self) -> str:
        """Build technical support specific prompt."""
        base_prompt = super()._build_call_center_prompt()
        
        return base_prompt + f"""

TECHNICAL SUPPORT SPECIFIC GUIDELINES:
- Start with basic troubleshooting steps
- Ask clarifying questions to understand the technical issue
- Use simple, non-technical language when explaining solutions
- Document technical steps taken
- Create support tickets for unresolved issues
- Escalate complex problems to {self.support_level} specialists
- Provide ticket numbers for follow-up

TROUBLESHOOTING PROCESS:
1. Gather system information (OS, browser, version, etc.)
2. Reproduce the issue if possible
3. Try basic solutions first (restart, refresh, clear cache)
4. Check for known issues in knowledge base
5. Apply appropriate technical solutions
6. Verify resolution with customer
7. Document solution for future reference

ESCALATION CRITERIA:
- Issue requires {self.support_level} expertise
- Hardware replacement needed
- Software bugs requiring development team
- Security-related incidents
- Data recovery situations

Remember: If you don't know the answer, it's better to escalate than to guess.
"""

    async def create_support_ticket(self, issue_description: str, priority: CallPriority = CallPriority.NORMAL) -> str:
        """Create a support ticket for the current issue."""
        if not self.can_create_tickets:
            print("❌ This agent cannot create support tickets")
            return ""
        
        ticket_id = f"TECH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        ticket_data = {
            "ticket_id": ticket_id,
            "customer_id": self.customer_info.customer_id if self.customer_info else "Unknown",
            "issue_description": issue_description,
            "priority": priority.value,
            "created_by": self.agent_id,
            "call_id": self.current_call.call_id if self.current_call else None,
            "created_at": datetime.now().isoformat()
        }
        
        # Integrate with ticket system
        if self.ticket_system_integration:
            try:
                await self.ticket_system_integration(ticket_data)
            except Exception as e:
                print(f"⚠️ Ticket system integration failed: {e}")
        
        print(f"🎫 Support ticket created: {ticket_id}")
        return ticket_id


class SalesAgent(CallCenterAgent):
    """
    Specialized agent for sales and lead qualification.
    
    Handles sales inquiries, product demos, quote generation,
    and lead qualification with focus on conversion.
    """
    
    def __init__(
        self,
        company_name: str = "Your Company",
        agent_id: Optional[str] = None,
        sales_territory: Optional[str] = None
    ):
        super().__init__(
            agent_name="Sales Representative",
            company_name=company_name,
            department="Sales",
            agent_id=agent_id
        )
        
        self.sales_territory = sales_territory
        
        # Sales specific settings
        self.can_generate_quotes = True
        self.max_discount_percent = 15.0
        self.lead_qualification_required = True
        
        # Sales metrics
        self.leads_qualified = 0
        self.demos_scheduled = 0
        self.quotes_generated = 0
    
    def _build_call_center_prompt(self) -> str:
        """Build sales specific prompt."""
        base_prompt = super()._build_call_center_prompt()
        
        return base_prompt + f"""

SALES SPECIFIC GUIDELINES:
- Qualify leads using BANT criteria (Budget, Authority, Need, Timeline)
- Focus on understanding customer needs and pain points
- Present solutions that match customer requirements
- Build value before discussing price
- Handle objections professionally
- Always ask for the next step (demo, meeting, quote)
- Follow up promptly on all commitments

SALES PROCESS:
1. Build rapport and establish trust
2. Discover customer needs through questions
3. Present relevant solutions and benefits
4. Handle any objections or concerns
5. Create urgency and value
6. Ask for commitment (demo, trial, purchase)
7. Schedule follow-up actions

LEAD QUALIFICATION (BANT):
- Budget: Do they have budget allocated?
- Authority: Are they the decision maker?
- Need: Do they have a clear business need?
- Timeline: When do they plan to implement?

Remember: Focus on how our solution solves their specific business problems.
"""

    async def qualify_lead(self, customer_responses: Dict[str, str]) -> Dict[str, Any]:
        """Qualify a lead based on BANT criteria."""
        qualification_score = 0
        qualification_details = {}
        
        # Budget qualification
        budget_indicators = ["budget", "cost", "price", "investment", "money"]
        budget_response = customer_responses.get("budget", "").lower()
        if any(indicator in budget_response for indicator in budget_indicators):
            qualification_score += 25
            qualification_details["budget"] = "Qualified"
        else:
            qualification_details["budget"] = "Needs follow-up"
        
        # Authority qualification
        authority_indicators = ["decision", "approve", "authorize", "manager", "director"]
        authority_response = customer_responses.get("authority", "").lower()
        if any(indicator in authority_response for indicator in authority_indicators):
            qualification_score += 25
            qualification_details["authority"] = "Qualified"
        else:
            qualification_details["authority"] = "Needs follow-up"
        
        # Need qualification
        need_indicators = ["problem", "need", "challenge", "issue", "solution"]
        need_response = customer_responses.get("need", "").lower()
        if any(indicator in need_response for indicator in need_indicators):
            qualification_score += 25
            qualification_details["need"] = "Qualified"
        else:
            qualification_details["need"] = "Needs follow-up"
        
        # Timeline qualification
        timeline_indicators = ["soon", "month", "quarter", "year", "immediately"]
        timeline_response = customer_responses.get("timeline", "").lower()
        if any(indicator in timeline_response for indicator in timeline_indicators):
            qualification_score += 25
            qualification_details["timeline"] = "Qualified"
        else:
            qualification_details["timeline"] = "Needs follow-up"
        
        self.leads_qualified += 1
        
        return {
            "qualification_score": qualification_score,
            "details": qualification_details,
            "recommendation": "Hot lead" if qualification_score >= 75 else 
                            "Warm lead" if qualification_score >= 50 else "Cold lead"
        }


# Example usage and testing
async def demo_call_center_agents():
    """Demonstrate call center agent capabilities."""
    print("🏢 Call Center Voice Agents Demo")
    print("="*50)
    
    # Customer Service Demo
    print("\n📞 Customer Service Agent Demo")
    cs_agent = CustomerServiceAgent("TechCorp")
    await cs_agent.initialize_agent()
    
    call_id = await cs_agent.start_call("+1234567890")
    
    # Simulate customer interactions
    await cs_agent._on_customer_speech("I'm having trouble with my bill")
    await cs_agent._on_customer_speech("The charges seem wrong")
    
    # Process a refund
    await cs_agent.process_refund(50.0, "Billing error")
    
    metrics = await cs_agent.end_call("Processed $50 refund for billing error")
    
    # Technical Support Demo
    print("\n🔧 Technical Support Agent Demo")
    tech_agent = TechnicalSupportAgent("TechCorp", "Level 1")
    await tech_agent.initialize_agent()
    
    call_id = await tech_agent.start_call()
    
    await tech_agent._on_customer_speech("My software keeps crashing")
    ticket_id = await tech_agent.create_support_ticket("Software crash issue - requires investigation")
    
    await tech_agent.end_call(f"Created support ticket {ticket_id}")
    
    # Sales Agent Demo
    print("\n💼 Sales Agent Demo")
    sales_agent = SalesAgent("TechCorp")
    await sales_agent.initialize_agent()
    
    call_id = await sales_agent.start_call()
    
    # Simulate lead qualification
    customer_responses = {
        "budget": "We have budget allocated for this project",
        "authority": "I'm the IT director and can make this decision",
        "need": "We need a solution for our customer service problems",
        "timeline": "We want to implement within 3 months"
    }
    
    qualification = await sales_agent.qualify_lead(customer_responses)
    print(f"Lead qualification: {qualification}")
    
    await sales_agent.end_call("Qualified lead - scheduling demo next week")
    
    print("\n✅ Call center demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_call_center_agents())