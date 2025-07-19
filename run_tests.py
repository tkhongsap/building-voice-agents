#!/usr/bin/env python3
"""
Main Test Entry Point for Task 1.0 Core Voice Processing Pipeline

This script provides a convenient entry point to run all tests from the root directory.
It automatically changes to the tests directory and runs the appropriate test suite.

Usage:
    python3 run_tests.py                # Run simplified test suite (no dependencies)
    python3 run_tests.py --comprehensive # Run comprehensive test suite (requires dependencies)
    python3 run_tests.py --help         # Show help
"""

import sys
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 1.0 Voice Processing Pipeline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites Available:

1. Simplified Tests (default):
   - Tests core structure and interfaces
   - No external dependencies required
   - Fast execution (< 5 seconds)
   - Validates all 18 subtasks of Task 1.0

2. Comprehensive Tests (--comprehensive):
   - Full pytest-based test suite
   - Requires external dependencies (openai, httpx, etc.)
   - Detailed mocking and edge case testing
   - Use after installing: pip install -r tests/requirements.txt

3. Integration Tests (--integration):
   - Tests with real API connections
   - Requires API keys in .env file
   - Tests actual OpenAI connectivity

Examples:
    python3 run_tests.py                    # Quick validation
    python3 run_tests.py --comprehensive    # Full test suite
    python3 run_tests.py --integration      # API testing
        """
    )
    
    parser.add_argument(
        '--comprehensive', 
        action='store_true',
        help='Run comprehensive pytest-based test suite (requires dependencies)'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true', 
        help='Run integration tests with real API connections (requires API keys)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available test files'
    )
    
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests')
    
    if not os.path.exists(tests_dir):
        print("❌ Tests directory not found!")
        print(f"Expected: {tests_dir}")
        return 1
    
    if args.list:
        list_test_files(tests_dir)
        return 0
    
    print("🚀 Task 1.0 Core Voice Processing Pipeline - Test Runner")
    print("=" * 70)
    
    if args.comprehensive:
        return run_comprehensive_tests(tests_dir)
    elif args.integration:
        return run_integration_tests(tests_dir)
    else:
        return run_simplified_tests(tests_dir)


def run_simplified_tests(tests_dir):
    """Run the simplified test suite (default)."""
    print("Running: Simplified Test Suite (No Dependencies Required)")
    print("Testing: All 18 subtasks of Task 1.0 implementation")
    print()
    
    test_script = os.path.join(tests_dir, 'test_runner_simplified.py')
    
    if not os.path.exists(test_script):
        print(f"❌ Test script not found: {test_script}")
        return 1
    
    try:
        # Change to tests directory and run the test
        result = subprocess.run(
            [sys.executable, 'test_runner_simplified.py'],
            cwd=tests_dir,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_comprehensive_tests(tests_dir):
    """Run comprehensive pytest-based tests."""
    print("Running: Comprehensive Test Suite (pytest + dependencies)")
    print("Note: Requires dependencies from tests/requirements.txt")
    print()
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, '-c', 'import pytest'], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pytest not available!")
        print("Install with: pip install -r tests/requirements.txt")
        return 1
    
    # Run pytest on comprehensive test
    comprehensive_test = os.path.join(tests_dir, 'unit', 'test_task1_comprehensive.py')
    
    if not os.path.exists(comprehensive_test):
        print(f"❌ Comprehensive test not found: {comprehensive_test}")
        return 1
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'unit/test_task1_comprehensive.py',
            '-v', '--tb=short'
        ], cwd=tests_dir)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running comprehensive tests: {e}")
        return 1


def run_integration_tests(tests_dir):
    """Run integration tests with real API connections."""
    print("Running: Integration Test Suite (Real API Connections)")
    print("Note: Requires API keys in .env file")
    print()
    
    # Check for API keys
    env_file = os.path.join(os.path.dirname(tests_dir), '.env')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not openai_key and os.path.exists(env_file):
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('OPENAI_API_KEY'):
                        openai_key = line.split('=')[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    
    if not openai_key:
        print("⚠️  No OpenAI API key found!")
        print("Add OPENAI_API_KEY to .env file for integration testing")
        print("Continuing with mock testing...")
    
    # Run integration test
    integration_test = os.path.join(tests_dir, 'integration', 'test_openai_integration.py')
    
    if not os.path.exists(integration_test):
        print(f"❌ Integration test not found: {integration_test}")
        return 1
    
    try:
        result = subprocess.run([
            sys.executable, 'integration/test_openai_integration.py'
        ], cwd=tests_dir)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running integration tests: {e}")
        return 1


def list_test_files(tests_dir):
    """List all available test files."""
    print("📁 Available Test Files:")
    print("=" * 50)
    
    for root, dirs, files in os.walk(tests_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        level = root.replace(tests_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        rel_root = os.path.relpath(root, tests_dir)
        if rel_root == '.':
            rel_root = 'tests/'
        else:
            rel_root = f'tests/{rel_root}/'
            
        print(f"{indent}{rel_root}")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                print(f"{subindent}📄 {file}")


if __name__ == "__main__":
    sys.exit(main())