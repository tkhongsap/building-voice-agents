#!/usr/bin/env python3
"""Simple test runner for SDK unit tests without pytest"""

import sys
import importlib.util
from pathlib import Path

def run_test_file(test_file):
    """Run a test file and return results."""
    print(f"\n📋 Running {test_file.name}...")
    
    # Load the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    
    # Add src to path for imports
    sys.path.insert(0, str(test_file.parent.parent))
    
    try:
        spec.loader.exec_module(test_module)
        
        # Find and run test classes
        test_count = 0
        passed_count = 0
        
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if (isinstance(obj, type) and 
                name.startswith('Test') and 
                hasattr(obj, '__dict__')):
                
                print(f"  Running {name}...")
                instance = obj()
                
                # Run test methods
                for method_name in dir(instance):
                    if method_name.startswith('test_'):
                        test_count += 1
                        try:
                            method = getattr(instance, method_name)
                            method()
                            passed_count += 1
                            print(f"    ✅ {method_name}")
                        except Exception as e:
                            print(f"    ❌ {method_name}: {e}")
        
        print(f"  Results: {passed_count}/{test_count} tests passed")
        return passed_count, test_count
        
    except Exception as e:
        print(f"  ❌ Error loading test file: {e}")
        return 0, 0

def main():
    """Run all SDK unit tests."""
    print("🧪 Running SDK Unit Tests")
    print("=" * 50)
    
    sdk_dir = Path("src/sdk")
    test_files = list(sdk_dir.glob("*.test.py"))
    
    total_passed = 0
    total_tests = 0
    
    for test_file in test_files:
        passed, count = run_test_file(test_file)
        total_passed += passed
        total_tests += count
    
    print("\n" + "=" * 50)
    print("📊 OVERALL RESULTS")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
    
    if total_passed == total_tests:
        print("🎉 ALL UNIT TESTS PASSED!")
    else:
        print(f"⚠️  {total_tests - total_passed} tests failed")

if __name__ == "__main__":
    main()