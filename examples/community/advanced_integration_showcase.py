"""
Advanced Integration Showcase

A comprehensive example demonstrating advanced integration patterns and
capabilities of the LiveKit Voice Agents Platform. This showcase combines
multiple features and demonstrates enterprise-ready patterns.

Features:
- Multi-modal conversation (voice + text + data)
- Real-time analytics and monitoring
- Dynamic provider switching and fallback
- Advanced conversation state management
- Integration with external APIs and databases
- Custom pipeline components
- Enterprise security patterns
- Scalable architecture patterns

Usage:
    python advanced_integration_showcase.py --scenario [all|analytics|fallback|security]
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import sqlite3
from contextlib import asynccontextmanager

from src.sdk.python_sdk import VoiceAgentSDK
from src.monitoring.performance_profiler import PerformanceProfiler
from src.monitoring.conversation_inspector import ConversationInspector
from src.components.stt.openai_stt import OpenAISTT
from src.components.stt.azure_stt import AzureSTT
from src.components.llm.openai_llm import OpenAILLM
from src.components.llm.anthropic_llm import AnthropicLLM
from src.components.tts.elevenlabs_tts import ElevenLabsTTS
from src.components.tts.openai_tts import OpenAITTS


@dataclass
class ConversationContext:
    """Rich conversation context with multi-modal data."""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    external_data: Dict[str, Any]
    security_context: Dict[str, Any]
    analytics_data: Dict[str, Any]


@dataclass
class ProviderConfig:
    """Configuration for AI provider with fallback options."""
    primary_provider: str
    fallback_providers: List[str]
    health_check_interval: float
    failure_threshold: int
    recovery_threshold: int


class ProviderHealthMonitor:
    """Monitor provider health and manage failover."""
    
    def __init__(self):
        self.provider_status: Dict[str, Dict[str, Any]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_health_check: Dict[str, datetime] = {}
    
    async def check_provider_health(self, provider_name: str) -> bool:
        """Check if a provider is healthy."""
        # Simulate health check
        # In production, this would make actual API calls
        
        import random
        is_healthy = random.random() > 0.1  # 90% success rate
        
        self.provider_status[provider_name] = {
            "healthy": is_healthy,
            "last_check": datetime.now(),
            "response_time": random.uniform(0.1, 2.0)
        }
        
        if not is_healthy:
            self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
        else:
            self.failure_counts[provider_name] = 0
        
        return is_healthy
    
    def get_provider_status(self, provider_name: str) -> Dict[str, Any]:
        """Get provider status information."""
        return self.provider_status.get(provider_name, {"healthy": False})


class SecurityManager:
    """Manage security policies and data protection."""
    
    def __init__(self):
        self.security_policies = {
            "pii_detection": True,
            "data_encryption": True,
            "access_logging": True,
            "rate_limiting": True,
            "content_filtering": True
        }
        self.blocked_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
    
    async def scan_content(self, content: str) -> Dict[str, Any]:
        """Scan content for security issues."""
        import re
        
        security_report = {
            "pii_detected": False,
            "sensitive_patterns": [],
            "risk_level": "low",
            "action_required": False
        }
        
        # Check for PII patterns
        for pattern in self.blocked_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                security_report["pii_detected"] = True
                security_report["sensitive_patterns"].extend(matches)
                security_report["risk_level"] = "high"
                security_report["action_required"] = True
        
        return security_report
    
    async def sanitize_content(self, content: str) -> str:
        """Remove or mask sensitive content."""
        import re
        
        sanitized = content
        
        # Mask SSN
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', sanitized)
        
        # Mask credit cards
        sanitized = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', sanitized)
        
        # Mask emails
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', sanitized)
        
        return sanitized


class AnalyticsEngine:
    """Real-time analytics and insights engine."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.events: List[Dict[str, Any]] = []
        self.insights: List[Dict[str, Any]] = []
        
        # Initialize database
        self.db_path = "analytics.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
                interaction_type TEXT,
                timestamp DATETIME,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversation_metrics (session_id, timestamp, metric_name, metric_value, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            metadata.get('session_id', 'unknown') if metadata else 'unknown',
            datetime.now(),
            metric_name,
            value,
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
    
    async def record_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record an event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": event_data
        }
        
        self.events.append(event)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_interactions (user_id, session_id, interaction_type, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            event_data.get('user_id', 'unknown'),
            event_data.get('session_id', 'unknown'),
            event_type,
            datetime.now(),
            json.dumps(event_data)
        ))
        
        conn.commit()
        conn.close()
    
    async def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from collected data."""
        insights = []
        
        # Average response time insight
        if 'response_time' in self.metrics:
            avg_response_time = sum(self.metrics['response_time']) / len(self.metrics['response_time'])
            
            if avg_response_time > 3.0:
                insights.append({
                    "type": "performance_warning",
                    "message": f"Average response time ({avg_response_time:.2f}s) exceeds recommended threshold",
                    "recommendation": "Consider optimizing models or adding caching",
                    "severity": "medium"
                })
        
        # User engagement insights
        user_sessions = {}
        for event in self.events:
            user_id = event['data'].get('user_id', 'unknown')
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            user_sessions[user_id].append(event)
        
        if len(user_sessions) > 5:
            insights.append({
                "type": "engagement_insight",
                "message": f"High user engagement: {len(user_sessions)} active users",
                "recommendation": "Monitor for capacity planning",
                "severity": "info"
            })
        
        return insights


class ExternalAPIIntegrator:
    """Integration with external APIs and services."""
    
    def __init__(self):
        self.api_cache = {}
        self.rate_limits = {}
    
    async def call_weather_api(self, location: str) -> Dict[str, Any]:
        """Mock weather API call."""
        # Simulate API call
        await asyncio.sleep(0.1)
        
        return {
            "location": location,
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45,
            "forecast": "Clear skies expected"
        }
    
    async def call_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Mock knowledge base query."""
        await asyncio.sleep(0.2)
        
        # Simulate knowledge base results
        knowledge_results = [
            {"title": "How to reset password", "content": "Click forgot password link...", "confidence": 0.9},
            {"title": "Account billing", "content": "Billing information can be found...", "confidence": 0.7},
            {"title": "Technical support", "content": "For technical issues, contact...", "confidence": 0.6}
        ]
        
        # Simple relevance matching
        relevant_results = [r for r in knowledge_results if query.lower() in r["title"].lower()]
        
        return {
            "query": query,
            "results": relevant_results[:3],
            "total_found": len(relevant_results)
        }
    
    async def call_crm_system(self, user_id: str) -> Dict[str, Any]:
        """Mock CRM system integration."""
        await asyncio.sleep(0.15)
        
        return {
            "user_id": user_id,
            "customer_tier": "premium",
            "account_status": "active",
            "last_contact": "2024-01-15",
            "open_tickets": 1,
            "satisfaction_score": 4.2
        }


class AdvancedIntegrationShowcase:
    """
    Showcase advanced integration patterns and enterprise features.
    
    Demonstrates:
    - Multi-provider fallback and health monitoring
    - Real-time analytics and insights
    - Security scanning and content protection
    - External API integration patterns
    - Advanced conversation state management
    - Performance monitoring and optimization
    """
    
    def __init__(self):
        self.sdk = VoiceAgentSDK()
        self.agent = None
        
        # Core components
        self.provider_monitor = ProviderHealthMonitor()
        self.security_manager = SecurityManager()
        self.analytics_engine = AnalyticsEngine()
        self.api_integrator = ExternalAPIIntegrator()
        
        # Monitoring tools
        self.profiler = PerformanceProfiler()
        self.inspector = ConversationInspector()
        
        # State management
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.current_session_id = None
        
        # Provider configurations
        self.provider_configs = {
            "stt": ProviderConfig(
                primary_provider="azure",
                fallback_providers=["openai"],
                health_check_interval=30.0,
                failure_threshold=3,
                recovery_threshold=5
            ),
            "llm": ProviderConfig(
                primary_provider="openai",
                fallback_providers=["anthropic"],
                health_check_interval=30.0,
                failure_threshold=2,
                recovery_threshold=3
            ),
            "tts": ProviderConfig(
                primary_provider="elevenlabs",
                fallback_providers=["openai"],
                health_check_interval=30.0,
                failure_threshold=3,
                recovery_threshold=5
            )
        }
    
    async def setup(self):
        """Setup the advanced integration showcase."""
        print("🔧 Setting up Advanced Integration Showcase...")
        
        # Start monitoring systems
        await self.profiler.start_profiling()
        await self.inspector.start_monitoring()
        
        # Configure agent with primary providers
        await self._configure_agent_with_fallback()
        
        # Register advanced functions
        self._register_advanced_functions()
        
        print("✅ Advanced Integration Showcase ready!")
        print("   Features enabled:")
        print("   • Multi-provider fallback")
        print("   • Real-time analytics")
        print("   • Security scanning")
        print("   • External API integration")
        print("   • Performance monitoring")
    
    async def _configure_agent_with_fallback(self):
        """Configure agent with fallback provider support."""
        # Start with primary providers
        stt_config = {
            "api_key": os.getenv("AZURE_SPEECH_KEY"),
            "region": os.getenv("AZURE_SPEECH_REGION", "eastus"),
            "language": "en-US"
        }
        
        llm_config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "system_prompt": """You are an advanced AI assistant showcasing enterprise integration capabilities.
            
            You have access to:
            - Real-time analytics and monitoring
            - External API integrations (weather, knowledge base, CRM)
            - Security and compliance features
            - Multi-provider fallback systems
            
            Be professional, helpful, and demonstrate these capabilities when appropriate."""
        }
        
        tts_config = {
            "api_key": os.getenv("ELEVENLABS_API_KEY"),
            "voice_id": "professional_female",
            "stability": 0.8
        }
        
        # Create agent
        self.agent = await self.sdk.create_agent(
            stt_provider="azure",
            stt_config=stt_config,
            llm_provider="openai",
            llm_config=llm_config,
            tts_provider="elevenlabs",
            tts_config=tts_config,
            vad_provider="silero",
            vad_config={"sensitivity": 0.6}
        )
        
        # Add profiler to agent
        self.agent.add_profiler(self.profiler)
    
    def _register_advanced_functions(self):
        """Register advanced integration functions."""
        
        @self.agent.function(
            name="get_weather_info",
            description="Get weather information for a location",
            parameters={
                "location": {"type": "string", "description": "Location to get weather for"}
            }
        )
        async def get_weather_info(location: str) -> str:
            """Get weather information with analytics."""
            start_time = datetime.now()
            
            try:
                # Call external API
                weather_data = await self.api_integrator.call_weather_api(location)
                
                # Record analytics
                response_time = (datetime.now() - start_time).total_seconds()
                await self.analytics_engine.record_metric("api_response_time", response_time * 1000, {
                    "api": "weather",
                    "session_id": self.current_session_id
                })
                
                await self.analytics_engine.record_event("weather_query", {
                    "location": location,
                    "session_id": self.current_session_id,
                    "response_time_ms": response_time * 1000
                })
                
                return f"Weather in {weather_data['location']}: {weather_data['temperature']}°F, {weather_data['condition']}. {weather_data['forecast']}"
                
            except Exception as e:
                await self.analytics_engine.record_event("api_error", {
                    "api": "weather",
                    "error": str(e),
                    "session_id": self.current_session_id
                })
                
                return f"Sorry, I couldn't get weather information for {location} right now."
        
        @self.agent.function(
            name="search_knowledge_base",
            description="Search the company knowledge base",
            parameters={
                "query": {"type": "string", "description": "Search query"}
            }
        )
        async def search_knowledge_base(query: str) -> str:
            """Search knowledge base with security scanning."""
            # Security scan
            security_report = await self.security_manager.scan_content(query)
            
            if security_report["pii_detected"]:
                await self.analytics_engine.record_event("security_alert", {
                    "type": "pii_detected",
                    "query": await self.security_manager.sanitize_content(query),
                    "session_id": self.current_session_id
                })
                
                return "I've detected sensitive information in your query. For your privacy, I cannot process requests containing personal data."
            
            try:
                # Search knowledge base
                kb_results = await self.api_integrator.call_knowledge_base(query)
                
                # Record analytics
                await self.analytics_engine.record_event("knowledge_base_search", {
                    "query": query,
                    "results_found": kb_results["total_found"],
                    "session_id": self.current_session_id
                })
                
                if kb_results["results"]:
                    response = f"I found {kb_results['total_found']} results for '{query}':\n\n"
                    for result in kb_results["results"][:2]:
                        response += f"• **{result['title']}**: {result['content'][:100]}...\n"
                    return response
                else:
                    return f"No results found for '{query}'. Would you like me to search for something else?"
                    
            except Exception as e:
                return "Sorry, the knowledge base is temporarily unavailable."
        
        @self.agent.function(
            name="get_user_profile",
            description="Get user profile and account information",
            parameters={
                "user_id": {"type": "string", "description": "User ID to look up"}
            }
        )
        async def get_user_profile(user_id: str) -> str:
            """Get user profile from CRM system."""
            try:
                # Get CRM data
                crm_data = await self.api_integrator.call_crm_system(user_id)
                
                # Record analytics
                await self.analytics_engine.record_event("profile_lookup", {
                    "user_id": user_id,
                    "customer_tier": crm_data["customer_tier"],
                    "session_id": self.current_session_id
                })
                
                response = f"User Profile for {user_id}:\n"
                response += f"• Customer Tier: {crm_data['customer_tier'].title()}\n"
                response += f"• Account Status: {crm_data['account_status'].title()}\n"
                response += f"• Satisfaction Score: {crm_data['satisfaction_score']}/5.0\n"
                
                if crm_data["open_tickets"] > 0:
                    response += f"• Open Support Tickets: {crm_data['open_tickets']}\n"
                
                return response
                
            except Exception as e:
                return f"Unable to retrieve profile for user {user_id}"
        
        @self.agent.function(
            name="get_system_analytics",
            description="Get real-time system analytics and insights"
        )
        async def get_system_analytics() -> str:
            """Get system analytics and performance insights."""
            # Generate insights
            insights = await self.analytics_engine.generate_insights()
            
            # Get performance data
            performance_report = await self.profiler.generate_performance_report()
            
            response = "📊 **System Analytics Dashboard**\n\n"
            
            # Performance metrics
            if "component_performance" in performance_report:
                response += "**Performance Overview:**\n"
                for component, stats in performance_report["component_performance"].items():
                    response += f"• {component}: {stats['avg_time_ms']:.1f}ms avg\n"
                response += "\n"
            
            # Analytics insights
            if insights:
                response += "**System Insights:**\n"
                for insight in insights:
                    response += f"• {insight['message']}\n"
                    if insight.get('recommendation'):
                        response += f"  Recommendation: {insight['recommendation']}\n"
                response += "\n"
            
            # Provider health
            response += "**Provider Status:**\n"
            for provider_type, config in self.provider_configs.items():
                status = self.provider_monitor.get_provider_status(config.primary_provider)
                health_indicator = "✅" if status.get("healthy", False) else "❌"
                response += f"• {provider_type.upper()} ({config.primary_provider}): {health_indicator}\n"
            
            return response
        
        @self.agent.function(
            name="switch_provider",
            description="Switch to a different AI provider for testing",
            parameters={
                "provider_type": {"type": "string", "enum": ["stt", "llm", "tts"], "description": "Type of provider to switch"},
                "provider_name": {"type": "string", "description": "Name of provider to switch to"}
            }
        )
        async def switch_provider(provider_type: str, provider_name: str) -> str:
            """Demonstrate provider switching capability."""
            # This is a simplified demonstration
            # In production, this would involve more complex configuration management
            
            config = self.provider_configs.get(provider_type)
            if not config:
                return f"Unknown provider type: {provider_type}"
            
            if provider_name not in [config.primary_provider] + config.fallback_providers:
                return f"Provider {provider_name} not configured for {provider_type}"
            
            # Record the switch
            await self.analytics_engine.record_event("provider_switch", {
                "provider_type": provider_type,
                "from_provider": config.primary_provider,
                "to_provider": provider_name,
                "session_id": self.current_session_id
            })
            
            return f"Switched {provider_type.upper()} provider from {config.primary_provider} to {provider_name}. This demonstrates our failover capabilities."
    
    async def start_session(self, user_id: str = "demo_user"):
        """Start an advanced integration session."""
        self.current_session_id = str(uuid.uuid4())
        
        # Create conversation context
        self.conversation_contexts[self.current_session_id] = ConversationContext(
            user_id=user_id,
            session_id=self.current_session_id,
            conversation_history=[],
            user_preferences={},
            external_data={},
            security_context={},
            analytics_data={}
        )
        
        # Record session start
        await self.analytics_engine.record_event("session_start", {
            "user_id": user_id,
            "session_id": self.current_session_id
        })
        
        print(f"\n🚀 Advanced Integration Session Started")
        print(f"   Session ID: {self.current_session_id}")
        print(f"   User ID: {user_id}")
        print(f"\n   Try these advanced capabilities:")
        print(f"   • 'Get weather for San Francisco'")
        print(f"   • 'Search knowledge base for password reset'")
        print(f"   • 'Show my user profile for demo_user'")
        print(f"   • 'Get system analytics'")
        print(f"   • 'Switch LLM provider to anthropic'")
        print(f"   Press Ctrl+C to end session\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            await self._end_session()
    
    async def _end_session(self):
        """End the current session."""
        if self.current_session_id:
            await self.analytics_engine.record_event("session_end", {
                "session_id": self.current_session_id,
                "duration_minutes": 5  # Simplified
            })
            
            print(f"\n📊 Session Analytics:")
            insights = await self.analytics_engine.generate_insights()
            for insight in insights:
                print(f"   • {insight['message']}")
            
            print(f"\n👋 Session ended. Analytics and performance data saved.")
    
    async def demo_scenario_analytics(self):
        """Demonstrate real-time analytics capabilities."""
        print("\n📊 Analytics Scenario Demo")
        print("=" * 50)
        
        # Simulate various interactions
        scenarios = [
            ("Weather Query", "Get weather for New York"),
            ("Knowledge Search", "Search knowledge base for account settings"),
            ("Profile Lookup", "Get user profile for demo_user"),
            ("System Status", "Get system analytics")
        ]
        
        for scenario_name, user_input in scenarios:
            print(f"\n📝 {scenario_name}")
            print(f"User: {user_input}")
            
            response = await self.agent.process_text(user_input)
            print(f"Assistant: {response[:200]}...")
            
            await asyncio.sleep(1)
        
        # Show analytics summary
        insights = await self.analytics_engine.generate_insights()
        print(f"\n📊 Generated Insights:")
        for insight in insights:
            print(f"   • {insight['message']}")
    
    async def demo_scenario_security(self):
        """Demonstrate security scanning capabilities."""
        print("\n🔒 Security Scenario Demo")
        print("=" * 50)
        
        # Test with sensitive content
        sensitive_queries = [
            "My social security number is 123-45-6789",
            "My credit card is 4532 1234 5678 9012",
            "Contact me at john.doe@email.com",
            "What's the weather like today?"  # Safe query
        ]
        
        for query in sensitive_queries:
            print(f"\n🔍 Testing: {query}")
            
            security_report = await self.security_manager.scan_content(query)
            print(f"   PII Detected: {security_report['pii_detected']}")
            print(f"   Risk Level: {security_report['risk_level']}")
            
            if security_report['pii_detected']:
                sanitized = await self.security_manager.sanitize_content(query)
                print(f"   Sanitized: {sanitized}")
    
    async def demo_scenario_failover(self):
        """Demonstrate provider failover capabilities."""
        print("\n🔄 Provider Failover Demo")
        print("=" * 50)
        
        # Check provider health
        providers_to_check = ["openai", "azure", "anthropic", "elevenlabs"]
        
        for provider in providers_to_check:
            health = await self.provider_monitor.check_provider_health(provider)
            status = self.provider_monitor.get_provider_status(provider)
            
            print(f"   {provider}: {'✅ Healthy' if health else '❌ Unhealthy'} "
                  f"(Response: {status.get('response_time', 0):.2f}s)")
        
        print(f"\n🔄 Simulating provider switch...")
        response = await self.agent.process_text("Switch LLM provider to anthropic")
        print(f"   Result: {response}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all features."""
        if not self.agent:
            await self.setup()
        
        print("\n🎯 Comprehensive Advanced Integration Demo")
        print("=" * 60)
        
        self.current_session_id = str(uuid.uuid4())
        
        # Run all demo scenarios
        await self.demo_scenario_analytics()
        await self.demo_scenario_security()
        await self.demo_scenario_failover()
        
        print("\n✅ Comprehensive demo completed!")
        print(f"   Session ID: {self.current_session_id}")
        
        # Cleanup
        await self.profiler.stop_profiling()
        await self.inspector.stop_monitoring()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Integration Showcase")
    parser.add_argument("--scenario", choices=["all", "analytics", "security", "failover", "live"],
                       default="all", help="Which scenario to demonstrate")
    
    args = parser.parse_args()
    
    print("🚀 Advanced Integration Showcase")
    print("=" * 50)
    
    showcase = AdvancedIntegrationShowcase()
    
    if args.scenario == "live":
        await showcase.setup()
        await showcase.start_session()
    elif args.scenario == "analytics":
        await showcase.setup()
        await showcase.demo_scenario_analytics()
    elif args.scenario == "security":
        await showcase.demo_scenario_security()
    elif args.scenario == "failover":
        await showcase.demo_scenario_failover()
    else:  # all scenarios
        await showcase.run_comprehensive_demo()


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key",
        "ANTHROPIC_API_KEY": "Anthropic API key (optional for fallback)",
        "AZURE_SPEECH_KEY": "Azure Speech Service key (optional for fallback)",
        "ELEVENLABS_API_KEY": "ElevenLabs API key (optional for fallback)"
    }
    
    # Only require OpenAI for basic demo
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing required environment variable:")
        print("  OPENAI_API_KEY: OpenAI API key")
        print("\nSet it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Warn about optional keys
    missing_optional = []
    for var, desc in list(required_vars.items())[1:]:
        if not os.getenv(var):
            missing_optional.append(f"  {var}: {desc}")
    
    if missing_optional:
        print("⚠️  Optional environment variables (for full demo):")
        for var in missing_optional:
            print(var)
        print("\nDemo will run with reduced functionality.")
    
    asyncio.run(main())