#!/usr/bin/env python3
"""
Check Ollama status and fix common issues
"""

import requests
import subprocess
import sys

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running!")
            print(f"📦 Available models: {len(models)}")
            for model in models:
                print(f"  - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def check_required_models():
    """Check if required models are installed"""
    required_models = ["llama3.1:8b", "llama3:8b", "llama2:7b"]
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            installed_models = [m.get('name', '') for m in response.json().get('models', [])]
            
            print("\n🔍 Checking required models:")
            for model in required_models:
                if model in installed_models:
                    print(f"✅ {model} - installed")
                else:
                    print(f"❌ {model} - not installed")
                    print(f"   Install with: ollama pull {model}")
            
            return any(model in installed_models for model in required_models)
        else:
            return False
    except:
        return False

def test_llama_generation():
    """Test Llama generation"""
    print("\n🧪 Testing Llama generation...")
    
    try:
        from llm.llama_llm import LlamaLLM
        
        llama = LlamaLLM()
        status = llama.get_status()
        
        print(f"Configured: {status['configured']}")
        print(f"Available: {status['available']}")
        
        if status['available']:
            result = llama.generate("Hello", 20)
            if result['success']:
                print(f"✅ Generation successful!")
                print(f"Response: {result['response']}")
                return True
            else:
                print(f"❌ Generation failed: {result['error']}")
        else:
            print(f"❌ Llama not available: {status['setup_command']}")
        
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_fix_instructions():
    """Show how to fix common issues"""
    print("\n🔧 Fix Instructions:")
    print("=" * 40)
    
    print("\n1. If 'port already in use' error:")
    print("   - Ollama is already running (good!)")
    print("   - Just pull models if needed")
    
    print("\n2. Pull required model:")
    print("   ollama pull llama3.1:8b")
    
    print("\n3. Test model directly:")
    print("   ollama run llama3.1:8b")
    
    print("\n4. If still not working:")
    print("   - Restart Ollama: taskkill /f /im ollama.exe")
    print("   - Then: ollama serve")

def main():
    print("🚀 Ollama Status Check")
    print("=" * 30)
    
    # Check if Ollama is running
    ollama_running = check_ollama_running()
    
    if ollama_running:
        # Check models
        models_ok = check_required_models()
        
        # Test generation
        generation_ok = test_llama_generation()
        
        print("\n📊 Summary:")
        print(f"Ollama running: ✅")
        print(f"Models available: {'✅' if models_ok else '❌'}")
        print(f"Generation working: {'✅' if generation_ok else '❌'}")
        
        if not models_ok:
            print("\n💡 Run: ollama pull llama3.1:8b")
        elif not generation_ok:
            print("\n💡 Check Python dependencies: pip install litellm")
        else:
            print("\n🎉 Llama is fully working!")
    else:
        show_fix_instructions()

if __name__ == "__main__":
    main()