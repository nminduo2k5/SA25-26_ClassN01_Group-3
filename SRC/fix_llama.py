#!/usr/bin/env python3
"""
Fix Llama setup automatically
"""

import subprocess
import requests
import sys
import time

def check_ollama_models():
    """Check what models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            print(f"Current models: {model_names}")
            return model_names
        return []
    except Exception as e:
        print(f"Cannot check models: {e}")
        return []

def pull_llama_model():
    """Pull llama model"""
    print("Pulling llama3.1:8b model...")
    print("This may take several minutes...")
    
    try:
        # Run ollama pull command
        result = subprocess.run(
            ["ollama", "pull", "llama3.1:8b"], 
            capture_output=True, 
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        if result.returncode == 0:
            print("Model pulled successfully!")
            return True
        else:
            print(f"Pull failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Pull timeout - model might be large, try manually")
        return False
    except Exception as e:
        print(f"Pull error: {e}")
        return False

def test_model():
    """Test the model works"""
    print("\nTesting model...")
    
    try:
        from litellm import completion
        
        response = completion(
            model="ollama/llama3.1:8b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            timeout=10
        )
        
        if response and response.choices:
            print("Model working!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("No response from model")
            return False
            
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def main():
    print("Llama Auto-Fix")
    print("=" * 30)
    
    # Check current models
    models = check_ollama_models()
    
    # Check if llama3.1:8b exists
    if "llama3.1:8b" in models:
        print("llama3.1:8b already exists")
        test_success = test_model()
        if test_success:
            print("\nLlama is working!")
            return
    else:
        print("llama3.1:8b not found")
    
    # Try to pull the model
    print("\nAttempting to pull model...")
    pull_success = pull_llama_model()
    
    if pull_success:
        # Test after pull
        time.sleep(2)  # Wait a bit
        test_success = test_model()
        
        if test_success:
            print("\nLlama fixed and working!")
        else:
            print("\nModel pulled but not working")
    else:
        print("\nCould not pull model")
        print("\n💡 Try manually:")
        print("   ollama pull llama3.1:8b")

if __name__ == "__main__":
    main()