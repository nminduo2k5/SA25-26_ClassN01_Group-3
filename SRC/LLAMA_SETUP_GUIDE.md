# Hướng dẫn cài đặt Llama Local

## 🎯 Tổng quan
Llama chạy local qua Ollama - không cần API key, hoàn toàn miễn phí.

## 📥 Cài đặt Ollama

### Windows:
1. Download từ: https://ollama.ai
2. Chạy file installer
3. Mở Command Prompt/PowerShell

### macOS:
```bash
brew install ollama
```

### Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## 🚀 Cài đặt Models

```bash
# Model chính (khuyến nghị)
ollama pull llama3.1:8b

# Models khác (tùy chọn)
ollama pull llama3:8b
ollama pull llama2:7b
ollama pull codellama:7b
```

## ▶️ Chạy Ollama

```bash
# Khởi động server (cần chạy trước khi dùng)
ollama serve

# Hoặc test trực tiếp
ollama run llama3.1:8b
```

## 🔧 Kiểm tra

```bash
# Xem models đã cài
ollama list

# Test API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello"
}'
```

## 🐍 Test trong Python

```python
from llm.llama_llm import LlamaLLM

# Tạo instance
llama = LlamaLLM()
print(f"Available: {llama.is_available}")

# Test generation
if llama.is_available:
    result = llama.generate("Xin chào")
    print(result['response'])
```

## ⚠️ Troubleshooting

### Lỗi "connection refused":
- Chạy `ollama serve` trước
- Kiểm tra port 11434 có bị block không

### Lỗi "model not found":
- Chạy `ollama pull llama3.1:8b`
- Kiểm tra `ollama list`

### Performance chậm:
- Dùng model nhỏ hơn: `llama2:7b`
- Tăng RAM/CPU
- Giảm max_tokens

## 💡 Tips

- **RAM cần**: Tối thiểu 8GB cho model 7B
- **Tốc độ**: Local nên có thể chậm hơn cloud API
- **Offline**: Hoạt động hoàn toàn offline
- **Miễn phí**: Không giới hạn usage