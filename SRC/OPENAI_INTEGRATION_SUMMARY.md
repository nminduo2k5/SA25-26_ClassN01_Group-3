# OpenAI Integration Summary

## Tổng quan
Đã tích hợp thành công OpenAI GPT-4o vào hệ thống Multi-Agent Vietnam Stock với khả năng lựa chọn AI model theo preference của người dùng.

## Các thay đổi chính

### 1. `gemini_agent.py` - UnifiedAIAgent
**Thay đổi:**
- Cập nhật import từ `import openai` sang `from openai import OpenAI`
- Thêm parameter `preferred_model` vào constructor
- Thêm `self.openai_client` để sử dụng OpenAI client mới
- Cập nhật `select_best_model()` để tôn trọng user preference
- Sửa `generate_with_model()` để sử dụng OpenAI client mới
- Cập nhật `generate_with_fallback()` để sử dụng preferred model trước

**Tính năng mới:**
- Hỗ trợ 3 chế độ: `"gemini"`, `"openai"`, `"auto"`
- Auto mode ưu tiên Gemini cho tiếng Việt, fallback sang OpenAI
- Fallback thông minh giữa các models

### 2. `main_agent.py` - MainAgent
**Thay đổi:**
- Cập nhật `set_gemini_api_key()` để nhận `openai_api_key` và `preferred_model`
- Cập nhật `set_crewai_keys()` để truyền model preference
- Sửa constructor để truyền `preferred_model="auto"`

**Tính năng mới:**
- Hỗ trợ cấu hình cả Gemini và OpenAI keys
- Model preference được lưu và sử dụng xuyên suốt hệ thống

### 3. `app.py` - Streamlit Interface
**Thay đổi:**
- Thêm dropdown "🎯 Chọn AI Model" trong sidebar
- Lưu model preference vào database
- Hiển thị trạng thái AI models với preference
- Cập nhật các button setup để truyền model preference

**Tính năng mới:**
- User có thể chọn AI model preference
- Preference được lưu và restore giữa các sessions
- Hiển thị model đang được sử dụng

### 4. `crewai_collector.py` - CrewAI Integration
**Thay đổi:**
- Thêm `openai_api_key` và `preferred_model` parameters
- Cập nhật `_setup_agents()` để tôn trọng user preference
- Sửa `get_crewai_collector()` để handle preference changes

**Tính năng mới:**
- CrewAI có thể sử dụng OpenAI GPT-4o hoặc Gemini
- Model selection dựa trên user preference
- Fallback thông minh giữa các models

### 5. `vn_stock_api.py` - VN Stock API
**Thay đổi:**
- Cập nhật `set_crewai_keys()` để nhận `openai_api_key` và `preferred_model`
- Thêm error handling cho CrewAI setup

## Cách sử dụng

### 1. Trong Streamlit App
```python
# Trong sidebar
selected_ai_model = st.selectbox(
    "🎯 Chọn AI Model",
    options=["gemini", "openai", "auto"],
    format_func=lambda x: ai_model_options[x]
)

# Nhập API keys
gemini_key = st.text_input("Khóa API Gemini", type="password")
openai_key = st.text_input("Khóa API OpenAI", type="password")

# Setup với preference
main_agent.set_gemini_api_key(gemini_key, openai_key, selected_ai_model)
```

### 2. Programmatically
```python
from gemini_agent import UnifiedAIAgent

# Chỉ OpenAI
agent = UnifiedAIAgent(
    openai_api_key="sk-...",
    preferred_model="openai"
)

# Chỉ Gemini
agent = UnifiedAIAgent(
    gemini_api_key="AIza...",
    preferred_model="gemini"
)

# Cả hai với auto selection
agent = UnifiedAIAgent(
    gemini_api_key="AIza...",
    openai_api_key="sk-...",
    preferred_model="auto"  # Ưu tiên Gemini, fallback OpenAI
)
```

## Model Selection Logic

### Auto Mode (Mặc định)
1. **Ưu tiên Gemini** - Miễn phí, tốt cho tiếng Việt
2. **Fallback OpenAI** - Nếu Gemini không khả dụng
3. **Offline Mode** - Nếu cả hai đều không khả dụng

### User Preference
- **"gemini"**: Chỉ sử dụng Gemini, fallback offline nếu không có
- **"openai"**: Chỉ sử dụng OpenAI, fallback offline nếu không có
- **"auto"**: Thông minh selection như trên

## Supported Models

### OpenAI Models (Theo thứ tự ưu tiên)
1. `gpt-4o` - Latest GPT-4 Omni
2. `gpt-4-turbo` - GPT-4 Turbo
3. `gpt-4` - Standard GPT-4
4. `gpt-3.5-turbo` - Fallback

### Gemini Models (Theo thứ tự ưu tiên)
1. `gemini-3-pro-preview` - Flagship mới nhất
2. `gemini-3-flash-preview` - Tốc độ cao thế hệ 3
3. `gemini-2.5-pro` - Bản chuẩn tốt nhất
4. `gemini-2.5-flash` - Bản chuẩn tốc độ cao
5. `gemini-2.0-flash` - Fallback tin cậy

## Error Handling

### API Key Issues
- Hệ thống sẽ thử model khác nếu một model fail
- Offline mode nếu tất cả models fail
- Clear error messages cho user

### Quota Exceeded
- Automatic fallback sang model khác
- Offline responses với thông báo rõ ràng
- Hướng dẫn user cách khắc phục

## Testing

Chạy test để kiểm tra integration:
```bash
cd SRC
python test_openai_integration.py
```

## Lưu ý quan trọng

### API Keys
- **Gemini**: Miễn phí tại https://aistudio.google.com/apikey
- **OpenAI**: Trả phí tại https://platform.openai.com/api-keys
- **Serper**: Tùy chọn tại https://serper.dev/api-key

### Cost Considerations
- **Gemini**: Miễn phí với quota hàng ngày
- **OpenAI**: Trả phí theo usage (~$0.01-0.03 per 1K tokens)
- **Recommendation**: Sử dụng "auto" mode để tối ưu cost

### Performance
- **Gemini**: Nhanh, tốt cho tiếng Việt
- **OpenAI**: Chất lượng cao, tốt cho tiếng Anh
- **Auto mode**: Cân bằng giữa cost và performance

## Troubleshooting

### OpenAI không hoạt động
1. Kiểm tra API key có đúng format `sk-...`
2. Kiểm tra credit balance trong OpenAI account
3. Thử model khác (gpt-3.5-turbo thay vì gpt-4o)

### Gemini không hoạt động
1. Kiểm tra API key có đúng format `AIza...`
2. Kiểm tra quota daily limit
3. Thử sau 24h nếu hết quota

### Cả hai đều không hoạt động
1. Hệ thống sẽ chuyển sang offline mode
2. Vẫn có thể sử dụng các tính năng cơ bản
3. Kiểm tra kết nối internet

## Future Enhancements

### Planned Features
- [ ] Claude AI integration
- [ ] Custom model endpoints
- [ ] Model performance monitoring
- [ ] Cost tracking per model
- [ ] A/B testing between models

### Optimization
- [ ] Smart caching based on model
- [ ] Load balancing between models
- [ ] Model-specific prompt optimization
- [ ] Response quality scoring