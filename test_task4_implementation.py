#!/usr/bin/env python3
"""
Task 4.0 Implementation Test Suite
Tests SDK functionality without external dependencies
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_sdk_imports():
    """Test that all SDK modules can be imported."""
    try:
        from sdk.exceptions import VoiceAgentError, ConfigurationError
        from sdk.utils import validate_type, ComponentRegistry
        from sdk.config_manager import ConfigManager, SDKConfig
        from sdk.agent_builder import VoiceAgentBuilder, AgentCapability
        from sdk.python_sdk import VoiceAgentSDK
        print("✅ All SDK imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_exception_hierarchy():
    """Test exception hierarchy and functionality."""
    try:
        from sdk.exceptions import VoiceAgentError, ConfigurationError, ValidationError
        
        # Test base exception
        error = VoiceAgentError("test", {"key": "value"})
        assert error.message == "test"
        assert error.details["key"] == "value"
        
        # Test inheritance
        config_error = ConfigurationError("config error")
        assert isinstance(config_error, VoiceAgentError)
        
        # Test validation error
        val_error = ValidationError("field", "value", "reason")
        assert "field" in str(val_error)
        
        print("✅ Exception hierarchy tests passed")
        return True
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration management."""
    try:
        from sdk.config_manager import ConfigManager, SDKConfig, LogLevel, Environment
        
        # Test default config
        config = SDKConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        
        # Test config manager
        manager = ConfigManager({"project_name": "test"})
        assert manager.config.project_name == "test"
        
        # Test config conversion
        config_dict = manager.to_dict()
        assert config_dict["project_name"] == "test"
        
        print("✅ Configuration system tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_agent_builder():
    """Test agent builder pattern."""
    try:
        from sdk.agent_builder import VoiceAgentBuilder, AgentCapability
        from sdk.config_manager import SDKConfig
        from sdk.utils import ComponentRegistry
        
        # Test builder creation
        builder = VoiceAgentBuilder(
            config=SDKConfig(),
            registry=ComponentRegistry()
        )
        
        # Test fluent API
        builder = (builder
            .with_name("Test Agent")
            .with_stt("openai", language="en")
            .with_llm("openai", model="gpt-4")
            .with_tts("elevenlabs")
            .with_capability(AgentCapability.TURN_DETECTION)
        )
        
        # Test configuration export
        config_dict = builder.to_dict()
        assert config_dict["metadata"]["name"] == "Test Agent"
        assert config_dict["providers"]["stt"]["provider"] == "openai"
        
        print("✅ Agent builder tests passed")
        return True
    except Exception as e:
        print(f"❌ Agent builder test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    try:
        from sdk.utils import validate_type, validate_range, ComponentRegistry
        
        # Test type validation
        result = validate_type("test", str, "field")
        assert result == "test"
        
        # Test range validation
        result = validate_range(5, 0, 10, "value")
        assert result == 5
        
        # Test component registry
        registry = ComponentRegistry()
        assert "stt" in registry.list()
        
        print("✅ Utility function tests passed")
        return True
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def test_quickstart_files():
    """Test that quickstart files exist and are valid."""
    try:
        # Check guide exists
        guide_path = Path("docs/quickstart/10_minute_guide.md")
        assert guide_path.exists(), "Quickstart guide missing"
        
        # Check hello world exists
        hello_path = Path("docs/quickstart/hello_world.py")
        assert hello_path.exists(), "Hello world example missing"
        
        # Check config templates
        config_path = Path("docs/quickstart/config_templates/basic_config.yaml")
        assert config_path.exists(), "Basic config template missing"
        
        print("✅ Quickstart files exist")
        return True
    except Exception as e:
        print(f"❌ Quickstart test failed: {e}")
        return False

def test_telehealth_recipe():
    """Test telehealth recipe functionality."""
    try:
        # Import telehealth recipe
        sys.path.insert(0, str(Path("docs/recipes")))
        from telehealth_recipe import TelehealthAgent, PrimaryCareTelehealthAgent
        
        # Test agent creation
        agent = TelehealthAgent("Test Provider", "General")
        assert agent.provider_name == "Test Provider"
        assert agent.specialization == "General"
        
        # Test specialized agent
        primary_agent = PrimaryCareTelehealthAgent()
        assert "Primary Care" in primary_agent.provider_name
        
        # Test prompt generation
        prompt = agent._build_medical_system_prompt()
        assert "NOT a doctor" in prompt
        assert "emergency" in prompt.lower()
        
        print("✅ Telehealth recipe tests passed")
        return True
    except Exception as e:
        print(f"❌ Telehealth test failed: {e}")
        return False

async def test_async_functionality():
    """Test async components."""
    try:
        from sdk.utils import ensure_async
        
        # Test sync function wrapping
        def sync_func(x):
            return x * 2
        
        async_func = ensure_async(sync_func)
        result = await async_func(5)
        assert result == 10
        
        print("✅ Async functionality tests passed")
        return True
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("🧪 Testing Task 4.0 Implementation")
    print("=" * 50)
    
    tests = [
        ("SDK Imports", test_sdk_imports),
        ("Exception Hierarchy", test_exception_hierarchy),
        ("Configuration System", test_configuration_system),
        ("Agent Builder", test_agent_builder),
        ("Utility Functions", test_utility_functions),
        ("Quickstart Files", test_quickstart_files),
        ("Telehealth Recipe", test_telehealth_recipe),
    ]
    
    async_tests = [
        ("Async Functionality", test_async_functionality),
    ]
    
    passed = 0
    total = len(tests) + len(async_tests)
    
    # Run sync tests
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {name} FAILED")
    
    # Run async tests
    async def run_async_tests():
        nonlocal passed
        for name, test_func in async_tests:
            print(f"\nTesting {name}...")
            if await test_func():
                passed += 1
            else:
                print(f"❌ {name} FAILED")
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Task 4.0 Implementation Successful!")
        return True
    else:
        print(f"⚠️  {total-passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)