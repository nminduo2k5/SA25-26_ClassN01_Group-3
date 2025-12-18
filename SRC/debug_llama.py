#!/usr/bin/env python3
"""
Debug Llama connection issues
"""

import requests
import subprocess

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Ollama server running with {len(models)} models")
            for model in models:
                print(f"  - {model.get('name')}")
            return True
        else:
            print(f"Ollama server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return False

def test_litellm_direct():
    """Test LiteLLM directly"""
    try:
        from litellm import completion
        print("LiteLLM imported successfully")
        
        # Test with very simple request
        response = completion(
            model="ollama/llama3.1:8b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            timeout=60  # Increase timeout significantly
        )
        
        if response and response.choices:
            print("LiteLLM test successful")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("LiteLLM no response")
            return False
            
    except Exception as e:
        print(f"LiteLLM test failed: {e}")
        return False

def restart_ollama():
    """Try to restart Ollama"""
    try:
        print("Attempting to restart Ollama...")
        
        # Kill existing ollama processes
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                      capture_output=True, check=False)
        
        # Wait a bit
        import time
        time.sleep(2)
        
        # Start ollama serve in background
        subprocess.Popen(["ollama", "serve"], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        # Wait for startup
        time.sleep(5)
        
        return check_ollama_server()
        
    except Exception as e:
        print(f"Restart failed: {e}")
        return False

def main():
    print("Debugging Llama Connection")
    print("=" * 35)
    
    # Step 1: Check Ollama server
    print("\n1. Checking Ollama server...")
    server_ok = check_ollama_server()
    
    if not server_ok:
        print("\n2. Attempting to restart Ollama...")
        server_ok = restart_ollama()
    
    if server_ok:
        print("\n3. Testing LiteLLM connection...")
        litellm_ok = test_litellm_direct()
        
        if litellm_ok:
            print("\nLlama is working!")
        else:
            print("\nLiteLLM connection failed")
            print("💡 Try: pip install litellm --upgrade")
    else:
        print("\nOllama server not accessible")
        print("💡 Manual steps:")
        print("   1. Open new terminal")
        print("   2. Run: ollama serve")
        print("   3. Test: ollama run llama3.1:8b")

if __name__ == "__main__":
    main()