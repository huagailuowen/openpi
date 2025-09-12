#!/usr/bin/env python3
"""Test script to verify RoboCasa OpenPI integration setup."""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import einops
        print("✅ einops imported successfully")
    except ImportError as e:
        print(f"❌ einops import failed: {e}")
        return False
        
    try:
        import imageio
        print("✅ imageio imported successfully")
    except ImportError as e:
        print(f"❌ imageio import failed: {e}")
        return False
        
    try:
        import tyro
        print("✅ tyro imported successfully")
    except ImportError as e:
        print(f"❌ tyro import failed: {e}")
        return False
    
    try:
        import robocasa
        from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
        print("✅ RoboCasa imported successfully")
        print(f"   Found {len(ALL_KITCHEN_ENVIRONMENTS)} available tasks")
    except ImportError as e:
        print(f"❌ RoboCasa import failed: {e}")
        return False
    
    try:
        from openpi_client import websocket_client_policy
        from openpi_client.runtime import runtime
        print("✅ OpenPI client imported successfully")
    except ImportError as e:
        print(f"❌ OpenPI client import failed: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test if RoboCasa environment can be created."""
    print("\n🏗️  Testing environment creation...")
    
    try:
        # Setup paths
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from env import RoboCasaEnvironment
        
        # Try to create environment
        env = RoboCasaEnvironment(
            task_name="PnPCounterToCab",
            robots="PandaOmron",
            horizon=10,  # Short horizon for testing
            seed=42
        )
        
        print("✅ RoboCasa environment created successfully")
        print(f"   Environment info: {env.get_env_info()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

def test_server_connection():
    """Test connection to OpenPI policy server."""
    print("\n🔗 Testing policy server connection...")
    
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8000/healthz"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✅ OpenPI policy server is running and accessible")
            return True
        else:
            print("⚠️  OpenPI policy server not accessible")
            print("   Make sure it's running on localhost:8000")
            return False
            
    except Exception as e:
        print(f"⚠️  Could not test server connection: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RoboCasa OpenPI Integration Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test environment creation  
    if test_environment_creation():
        tests_passed += 1
    
    # Test server connection
    if test_server_connection():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The integration is ready to use.")
        print("\nNext steps:")
        print("1. Start OpenPI policy server if not already running")
        print("2. Run: ./quick_start.sh")
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()
