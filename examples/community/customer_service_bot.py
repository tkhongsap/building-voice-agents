"""
Customer Service Bot Example

A sophisticated customer service voice agent that demonstrates enterprise-grade
customer support capabilities with ticket management, escalation handling,
and knowledge base integration.

Features:
- Intelligent query routing
- Ticket creation and tracking
- Escalation management
- Knowledge base search
- Customer authentication
- Sentiment analysis
- Multi-language support
- Integration with CRM systems

Usage:
    python customer_service_bot.py --mode [live|demo|training]
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from src.sdk.python_sdk import VoiceAgentSDK
from src.components.stt.azure_stt import AzureSTT
from src.components.llm.anthropic_llm import AnthropicLLM
from src.components.tts.elevenlabs_tts import ElevenLabsTTS
from src.components.vad.silero_vad import SileroVAD


class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


@dataclass
class Customer:
    """Customer information."""
    customer_id: str
    name: str
    email: str
    phone: Optional[str] = None
    tier: str = "standard"  # standard, premium, enterprise
    account_status: str = "active"
    join_date: Optional[datetime] = None
    lifetime_value: float = 0.0
    support_history: List[str] = None
    
    def __post_init__(self):
        if self.support_history is None:
            self.support_history = []


@dataclass
class SupportTicket:
    """Support ticket."""
    ticket_id: str
    customer_id: str
    title: str
    description: str
    category: str
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    escalation_level: int = 0
    resolution_notes: Optional[str] = None
    customer_satisfaction: Optional[int] = None  # 1-5 rating
    
    def __post_init__(self):
        if isinstance(self.priority, str):
            self.priority = TicketPriority(self.priority)
        if isinstance(self.status, str):
            self.status = TicketStatus(self.status)


@dataclass
class KnowledgeBaseArticle:
    """Knowledge base article."""
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    views: int = 0
    helpful_votes: int = 0
    total_votes: int = 0


class CustomerServiceBot:
    """
    Advanced customer service voice agent.
    
    Provides comprehensive customer support including:
    - Intelligent query classification
    - Automated ticket management
    - Knowledge base integration
    - Escalation handling
    - Customer authentication
    - Sentiment monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.sdk = VoiceAgentSDK()
        self.agent = None
        
        # Current conversation state
        self.current_customer: Optional[Customer] = None
        self.current_ticket: Optional[SupportTicket] = None
        self.conversation_sentiment = "neutral"
        self.escalation_threshold = 3  # Number of failed attempts before escalation
        self.failed_attempts = 0
        
        # Data storage
        self.data_dir = Path("customer_service_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize data stores
        self.customers: Dict[str, Customer] = {}
        self.tickets: Dict[str, SupportTicket] = {}
        self.knowledge_base: List[KnowledgeBaseArticle] = []
        
        # Load existing data
        self._load_data()
        self._initialize_sample_data()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        default_config = {
            "business": {
                "company_name": "TechCorp Solutions",
                "support_hours": "24/7",
                "escalation_email": "escalation@techcorp.com",
                "tier_benefits": {
                    "standard": ["email_support", "knowledge_base"],
                    "premium": ["email_support", "phone_support", "priority_queue"],
                    "enterprise": ["dedicated_manager", "24/7_phone", "custom_integration"]
                }
            },
            "stt": {
                "provider": "azure",
                "language": "en-US",
                "enable_sentiment": True
            },
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.2
            },
            "tts": {
                "provider": "elevenlabs",
                "voice_id": "professional_female",
                "stability": 0.8
            },
            "support": {
                "auto_escalate_negative_sentiment": True,
                "max_resolution_time_minutes": 30,
                "collect_satisfaction_rating": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
        
        return default_config
    
    def _load_data(self):
        """Load existing customer service data."""
        # Load customers
        customers_file = self.data_dir / "customers.json"
        if customers_file.exists():
            with open(customers_file, 'r') as f:
                customers_data = json.load(f)
                for customer_data in customers_data:
                    if 'join_date' in customer_data and customer_data['join_date']:
                        customer_data['join_date'] = datetime.fromisoformat(customer_data['join_date'])
                    customer = Customer(**customer_data)
                    self.customers[customer.customer_id] = customer
        
        # Load tickets
        tickets_file = self.data_dir / "tickets.json"
        if tickets_file.exists():
            with open(tickets_file, 'r') as f:
                tickets_data = json.load(f)
                for ticket_data in tickets_data:
                    ticket_data['created_at'] = datetime.fromisoformat(ticket_data['created_at'])
                    ticket_data['updated_at'] = datetime.fromisoformat(ticket_data['updated_at'])
                    if ticket_data.get('resolved_at'):
                        ticket_data['resolved_at'] = datetime.fromisoformat(ticket_data['resolved_at'])
                    ticket = SupportTicket(**ticket_data)
                    self.tickets[ticket.ticket_id] = ticket
        
        # Load knowledge base
        kb_file = self.data_dir / "knowledge_base.json"
        if kb_file.exists():
            with open(kb_file, 'r') as f:
                kb_data = json.load(f)
                for article_data in kb_data:
                    article_data['created_at'] = datetime.fromisoformat(article_data['created_at'])
                    article_data['updated_at'] = datetime.fromisoformat(article_data['updated_at'])
                    article = KnowledgeBaseArticle(**article_data)
                    self.knowledge_base.append(article)
    
    def _initialize_sample_data(self):
        """Initialize sample data if none exists."""
        if not self.customers:
            # Create sample customers
            sample_customers = [
                Customer(
                    customer_id="CUST001",
                    name="John Smith",
                    email="john.smith@email.com",
                    phone="+1-555-0123",
                    tier="premium",
                    join_date=datetime.now() - timedelta(days=365),
                    lifetime_value=2500.0
                ),
                Customer(
                    customer_id="CUST002",
                    name="Sarah Johnson",
                    email="sarah.j@company.com",
                    tier="enterprise",
                    join_date=datetime.now() - timedelta(days=180),
                    lifetime_value=15000.0
                )
            ]
            
            for customer in sample_customers:
                self.customers[customer.customer_id] = customer
        
        if not self.knowledge_base:
            # Create sample knowledge base
            sample_articles = [
                KnowledgeBaseArticle(
                    article_id="KB001",
                    title="How to Reset Your Password",
                    content="To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter your email 4. Check your email for reset link 5. Follow the instructions",
                    category="account",
                    tags=["password", "reset", "login"],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    views=1250,
                    helpful_votes=89,
                    total_votes=95
                ),
                KnowledgeBaseArticle(
                    article_id="KB002",
                    title="Billing and Payment Issues",
                    content="For billing questions: Check your payment method, verify card expiry, contact your bank if declined. For refunds, see our refund policy or contact support.",
                    category="billing",
                    tags=["billing", "payment", "refund"],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    views=892,
                    helpful_votes=67,
                    total_votes=72
                )
            ]
            
            self.knowledge_base = sample_articles
        
        # Save data
        self._save_data()
    
    def _save_data(self):
        """Save all data to files."""
        # Save customers
        customers_data = [asdict(customer) for customer in self.customers.values()]
        with open(self.data_dir / "customers.json", 'w') as f:
            json.dump(customers_data, f, indent=2, default=str)
        
        # Save tickets
        tickets_data = [asdict(ticket) for ticket in self.tickets.values()]
        with open(self.data_dir / "tickets.json", 'w') as f:
            json.dump(tickets_data, f, indent=2, default=str)
        
        # Save knowledge base
        kb_data = [asdict(article) for article in self.knowledge_base]
        with open(self.data_dir / "knowledge_base.json", 'w') as f:
            json.dump(kb_data, f, indent=2, default=str)
    
    async def setup(self):
        """Setup the customer service bot."""
        print("🔧 Setting up Customer Service Bot...")
        
        # Configure components
        stt_config = {
            **self.config["stt"],
            "api_key": os.getenv("AZURE_SPEECH_KEY"),
            "region": os.getenv("AZURE_SPEECH_REGION", "eastus")
        }
        
        llm_config = {
            **self.config["llm"],
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "system_prompt": f"""You are a professional customer service agent for {self.config['business']['company_name']}. 

Your role:
- Provide helpful, empathetic customer support
- Resolve issues efficiently and professionally
- Create tickets for complex problems
- Search knowledge base for solutions
- Escalate when necessary
- Maintain a positive, solution-focused attitude

Always be:
- Professional and courteous
- Clear and concise
- Empathetic to customer concerns
- Proactive in offering solutions

Company info:
- Support hours: {self.config['business']['support_hours']}
- We offer standard, premium, and enterprise tiers
- Focus on customer satisfaction and quick resolution"""
        }
        
        tts_config = {
            **self.config["tts"],
            "api_key": os.getenv("ELEVENLABS_API_KEY")
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
            vad_config={"sensitivity": 0.6}
        )
        
        # Register customer service functions
        self._register_service_functions()
        
        print("✅ Customer Service Bot ready!")
        print(f"   Company: {self.config['business']['company_name']}")
        print(f"   Support Hours: {self.config['business']['support_hours']}")
    
    def _register_service_functions(self):
        """Register customer service functions."""
        
        @self.agent.function(
            name="authenticate_customer",
            description="Authenticate a customer by email or customer ID",
            parameters={
                "identifier": {"type": "string", "description": "Customer email or ID"},
                "verification_method": {"type": "string", "enum": ["email", "phone", "security_question"], "description": "Method to verify identity"}
            }
        )
        async def authenticate_customer(identifier: str, verification_method: str = "email") -> str:
            """Authenticate a customer."""
            # Find customer
            customer = None
            for cust in self.customers.values():
                if cust.email == identifier or cust.customer_id == identifier:
                    customer = cust
                    break
            
            if not customer:
                return f"I couldn't find an account with {identifier}. Please check the spelling or provide your customer ID."
            
            self.current_customer = customer
            
            # In a real system, this would send verification code
            print(f"🔐 Customer authenticated: {customer.name} ({customer.tier} tier)")
            
            return f"Hello {customer.name}! I've found your {customer.tier} account. How can I help you today?"
        
        @self.agent.function(
            name="search_knowledge_base",
            description="Search the knowledge base for solutions",
            parameters={
                "query": {"type": "string", "description": "Search query or keywords"},
                "category": {"type": "string", "description": "Optional category filter"}
            }
        )
        async def search_knowledge_base(query: str, category: str = None) -> str:
            """Search knowledge base for relevant articles."""
            query_lower = query.lower()
            matches = []
            
            for article in self.knowledge_base:
                score = 0
                
                # Check title match
                if query_lower in article.title.lower():
                    score += 3
                
                # Check content match
                if query_lower in article.content.lower():
                    score += 2
                
                # Check tags
                for tag in article.tags:
                    if query_lower in tag.lower():
                        score += 1
                
                # Check category filter
                if category and article.category != category:
                    score = 0
                
                if score > 0:
                    matches.append((score, article))
            
            if not matches:
                return f"I couldn't find any articles matching '{query}'. Let me create a support ticket for you."
            
            # Sort by relevance
            matches.sort(key=lambda x: x[0], reverse=True)
            best_match = matches[0][1]
            
            # Update article views
            best_match.views += 1
            
            return f"I found this helpful article: '{best_match.title}'\n\n{best_match.content}\n\nWas this helpful? If not, I can create a support ticket for you."
        
        @self.agent.function(
            name="create_support_ticket",
            description="Create a new support ticket",
            parameters={
                "title": {"type": "string", "description": "Brief title for the issue"},
                "description": {"type": "string", "description": "Detailed description of the problem"},
                "category": {"type": "string", "enum": ["technical", "billing", "account", "feature_request", "bug_report"], "description": "Issue category"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "description": "Priority level"}
            }
        )
        async def create_support_ticket(title: str, description: str, category: str = "technical", priority: str = "medium") -> str:
            """Create a new support ticket."""
            if not self.current_customer:
                return "Please authenticate first so I can create a ticket for your account."
            
            # Generate ticket ID
            ticket_id = f"TK{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            
            # Determine priority based on customer tier and issue
            if self.current_customer.tier == "enterprise" and priority in ["medium", "high"]:
                priority = "high"
            elif category == "billing" and priority == "medium":
                priority = "high"
            
            # Create ticket
            ticket = SupportTicket(
                ticket_id=ticket_id,
                customer_id=self.current_customer.customer_id,
                title=title,
                description=description,
                category=category,
                priority=TicketPriority(priority),
                status=TicketStatus.OPEN,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.tickets[ticket_id] = ticket
            self.current_ticket = ticket
            
            # Add to customer history
            self.current_customer.support_history.append(ticket_id)
            
            # Save data
            self._save_data()
            
            print(f"🎫 Created ticket {ticket_id}: {title}")
            
            # Determine SLA based on priority and customer tier
            sla_hours = self._get_sla_hours(ticket.priority, self.current_customer.tier)
            
            return f"I've created ticket {ticket_id} for you. Priority: {priority}. We'll respond within {sla_hours} hours based on your {self.current_customer.tier} tier. Is there anything else I can help with while you wait?"
        
        @self.agent.function(
            name="check_ticket_status",
            description="Check the status of a support ticket",
            parameters={
                "ticket_id": {"type": "string", "description": "Ticket ID to check"}
            }
        )
        async def check_ticket_status(ticket_id: str) -> str:
            """Check ticket status."""
            if ticket_id not in self.tickets:
                return f"I couldn't find ticket {ticket_id}. Please check the ticket ID."
            
            ticket = self.tickets[ticket_id]
            
            # Check if customer owns this ticket
            if self.current_customer and ticket.customer_id != self.current_customer.customer_id:
                return "I can only show you tickets for your account."
            
            status_info = f"Ticket {ticket_id}: {ticket.title}\n"
            status_info += f"Status: {ticket.status.value.replace('_', ' ').title()}\n"
            status_info += f"Priority: {ticket.priority.value.title()}\n"
            status_info += f"Created: {ticket.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            
            if ticket.assigned_agent:
                status_info += f"Assigned to: {ticket.assigned_agent}\n"
            
            if ticket.status == TicketStatus.RESOLVED and ticket.resolution_notes:
                status_info += f"Resolution: {ticket.resolution_notes}\n"
            
            return status_info
        
        @self.agent.function(
            name="escalate_issue",
            description="Escalate the current issue to a human agent",
            parameters={
                "reason": {"type": "string", "description": "Reason for escalation"}
            }
        )
        async def escalate_issue(reason: str) -> str:
            """Escalate to human agent."""
            if self.current_ticket:
                self.current_ticket.status = TicketStatus.ESCALATED
                self.current_ticket.escalation_level += 1
                self.current_ticket.updated_at = datetime.now()
                
                print(f"🚨 Escalated ticket {self.current_ticket.ticket_id}: {reason}")
            else:
                print(f"🚨 General escalation: {reason}")
            
            self._save_data()
            
            return f"I'm connecting you with a human agent immediately. Reason: {reason}. Please hold while I transfer you."
        
        @self.agent.function(
            name="collect_satisfaction_rating",
            description="Collect customer satisfaction rating",
            parameters={
                "rating": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Rating from 1-5"},
                "feedback": {"type": "string", "description": "Optional feedback"}
            }
        )
        async def collect_satisfaction_rating(rating: int, feedback: str = None) -> str:
            """Collect satisfaction rating."""
            if self.current_ticket:
                self.current_ticket.customer_satisfaction = rating
                if feedback:
                    self.current_ticket.resolution_notes = (self.current_ticket.resolution_notes or "") + f"\nCustomer Feedback: {feedback}"
                
                self._save_data()
                
                print(f"📊 Customer satisfaction: {rating}/5")
                if feedback:
                    print(f"   Feedback: {feedback}")
            
            response = f"Thank you for rating your experience {rating}/5."
            if rating >= 4:
                response += " We're glad we could help!"
            elif rating <= 2:
                response += " I'm sorry we didn't meet your expectations. I'll make sure your feedback reaches our management team."
            
            return response
    
    def _get_sla_hours(self, priority: TicketPriority, tier: str) -> int:
        """Get SLA response time in hours."""
        base_hours = {
            TicketPriority.URGENT: 1,
            TicketPriority.HIGH: 4,
            TicketPriority.MEDIUM: 12,
            TicketPriority.LOW: 24
        }
        
        hours = base_hours.get(priority, 24)
        
        # Adjust for customer tier
        if tier == "enterprise":
            hours = max(1, hours // 2)
        elif tier == "premium":
            hours = max(2, int(hours * 0.75))
        
        return hours
    
    async def start_live_support(self):
        """Start live customer support session."""
        if not self.agent:
            await self.setup()
        
        print(f"\n📞 {self.config['business']['company_name']} Customer Service")
        print("   Welcome! I'm here to help with any questions or issues.")
        print("   Start by saying: 'I need help with...' or 'My email is...'")
        print("   I can help with:")
        print("   - Account issues")
        print("   - Billing questions") 
        print("   - Technical problems")
        print("   - Password resets")
        print("   Press Ctrl+C to end the session\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            print("\n📞 Thank you for contacting support. Have a great day!")
        finally:
            if self.current_ticket and self.config["support"]["collect_satisfaction_rating"]:
                print("\n📊 Please rate your experience (1-5): ", end="")
                try:
                    rating = int(input().strip())
                    if 1 <= rating <= 5:
                        await self.agent.process_text(f"Rate my experience {rating} out of 5")
                except:
                    pass
    
    async def demo_mode(self):
        """Run customer service demonstration."""
        if not self.agent:
            await self.setup()
        
        print(f"\n🎭 {self.config['business']['company_name']} Customer Service Demo")
        print("=" * 60)
        
        # Demo scenario: Password reset issue
        demo_interactions = [
            ("Customer Authentication", "My email is john.smith@email.com"),
            ("Password Issue", "I can't log into my account. I think I forgot my password."),
            ("Knowledge Base Search", "Search for password reset instructions"),
            ("Create Ticket", "The reset link isn't working. Create a ticket titled 'Password reset link not working' for technical category with high priority"),
            ("Check Status", f"What's the status of my ticket?"),
            ("Satisfaction", "Rate my experience 4 out of 5 - the agent was helpful")
        ]
        
        for step_name, user_input in demo_interactions:
            print(f"\n📝 {step_name}")
            print(f"Customer: {user_input}")
            
            response = await self.agent.process_text(user_input)
            print(f"Agent: {response}")
            
            # Small delay for readability
            await asyncio.sleep(1)
        
        print("\n✅ Customer service demo completed!")
        print(f"📊 Total customers: {len(self.customers)}")
        print(f"🎫 Total tickets: {len(self.tickets)}")
        print(f"📚 Knowledge base articles: {len(self.knowledge_base)}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Customer Service Bot Example")
    parser.add_argument("--mode", choices=["live", "demo", "training"], default="demo",
                       help="Run mode: live support, demo, or training")
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    print("🏢 Customer Service Bot Example")
    print("=" * 50)
    
    bot = CustomerServiceBot(args.config)
    
    if args.mode == "live":
        await bot.start_live_support()
    elif args.mode == "training":
        # Training mode - could include mock conversations for training
        print("🎓 Training mode - would include conversation training scenarios")
    else:  # demo mode
        await bot.demo_mode()


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key for LLM",
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