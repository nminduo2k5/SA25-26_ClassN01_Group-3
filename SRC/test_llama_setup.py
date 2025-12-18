#!/usr/bin/env python3
"""
Test Llama setup and configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_litellm_installation():
    """Test if litellm is installed"""
    print("🧪 Testing LiteLLM Installation...")
    
    try:
        from litellm import completion
        print("✅ LiteLLM installed successfully")
        return True
    except ImportError as e:
        print(f"❌ LiteLLM not installed: {e}")
        print("💡 Install with: pip install litellm")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\n🧪 Testing Ollama Connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama running with {len(models)} models")
            for model in models:
                print(f"  - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Start with: ollama serve")
        return False

def test_llama_llm():
    """Test LlamaLLM class"""
    print("\n🧪 Testing LlamaLLM Class...")
    
    try:
        from llm.llama_llm import LlamaLLM
        
        llama = LlamaLLM()
        status = llama.get_status()
        
        print(f"Configured: {status['configured']}")
        print(f"Available: {status['available']}")
        print(f"Current model: {status['current_model']}")
        print(f"Setup command: {status['setup_command']}")
        
        if status['available']:
            print("\n🧪 Testing generation...")
            result = llama.generate("Hello", 10)
            if result['success']:
                print(f"✅ Generation successful: {result['response']}")
            else:
                print(f"❌ Generation failed: {result['error']}")
        
        return status['available']
        
    except Exception as e:
        print(f"❌ LlamaLLM test failed: {e}")
        return False

def test_unified_llm():
    """Test UnifiedLLM with Llama"""
    print("\n🧪 Testing UnifiedLLM with Llama...")
    
    try:
        from llm.unified_llm import UnifiedLLM
        
        llm = UnifiedLLM(preferred="llama")
        all_status = llm.get_all_status()
        
        print("All LLM Status:")
        for name, status in all_status.items():
            print(f"  {name}: {status}")
        
        if 'llama' in llm.available_models:
            print("\n🧪 Testing Llama generation via UnifiedLLM...")
            result = llm.generate("Test prompt", force_model="llama")
            if result['success']:
                print(f"✅ Success: {result['response'][:100]}...")
            else:
                print(f"❌ Failed: {result['error']}")
        
    except Exception as e:
        print(f"❌ UnifiedLLM test failed: {e}")

def show_setup_instructions():
    """Show setup instructions"""
    print("\n📋 Llama Setup Instructions:")
    print("=" * 50)
    
    print("\n1. Install LiteLLM:")
    print("   pip install litellm")
    
    print("\n2. Install Ollama:")
    print("   Windows: Download from https://ollama.ai")
    print("   macOS: brew install ollama")
    print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
    
    print("\n3. Pull Llama model:")
    print("   ollama pull llama3.1:8b")
    
    print("\n4. Start Ollama server:")
    print("   ollama serve")
    
    print("\n5. Test in Python:")
    print("   python test_llama_setup.py")

def main():
    print("🚀 Llama Setup Test")
    print("=" * 50)
    
    # Test 1: LiteLLM installation
    litellm_ok = test_litellm_installation()
    
    # Test 2: Ollama connection
    ollama_ok = test_ollama_connection()
    
    # Test 3: LlamaLLM class
    llama_ok = test_llama_llm()
    
    # Test 4: UnifiedLLM integration
    test_unified_llm()
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"LiteLLM: {'✅' if litellm_ok else '❌'}")
    print(f"Ollama: {'✅' if ollama_ok else '❌'}")
    print(f"Llama LLM: {'✅' if llama_ok else '❌'}")
    
    if not (litellm_ok and ollama_ok and llama_ok):
        show_setup_instructions()
    else:
        print("\n🎉 Llama is fully configured and working!")

if __name__ == "__main__":
    main()