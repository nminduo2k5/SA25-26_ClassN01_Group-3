# Hướng dẫn Setup OpenAI

## Bước 1: Lấy OpenAI API Key

1. Truy cập https://platform.openai.com/api-keys
2. Đăng nhập hoặc tạo tài khoản
3. Click "Create new secret key"
4. Copy API key (bắt đầu bằng `sk-`)

## Bước 2: Setup trong Streamlit App

1. Chạy app: `streamlit run app.py`
2. Trong sidebar, chọn "🎯 Chọn AI Model"
3. Chọn "🚀 OpenAI GPT (Trả phí)" hoặc "⚡ Tự động"
4. Nhập OpenAI API key vào ô "Khóa API OpenAI"
5. Click "🔧 Cài đặt AI Models"

## Bước 3: Kiểm tra

Sau khi setup, bạn sẽ thấy:
- ✅ AI Models: openai (hoặc gemini, openai)
- 🎯 Đang sử dụng: OPENAI (nếu chọn OpenAI)

## Test nhanh

```python
# Chạy test
python quick_test.py
```

## Troubleshooting

### Lỗi "OpenAI chưa được cấu hình"
- Kiểm tra API key có đúng format `sk-...`
- Kiểm tra credit balance trong OpenAI account

### Lỗi "No AI models available"
- Nhập ít nhất 1 API key (Gemini hoặc OpenAI)
- Click button setup sau khi nhập key

### Lỗi "API quota exceeded"
- Kiểm tra usage limits trong OpenAI dashboard
- Thêm payment method nếu cần

## Cost Estimate

- GPT-4o: ~$0.015 per 1K input tokens, ~$0.06 per 1K output tokens
- GPT-3.5-turbo: ~$0.001 per 1K tokens (rẻ hơn)
- Gemini: Miễn phí với quota hàng ngày

## Recommendation

Sử dụng chế độ "⚡ Tự động" để:
- Ưu tiên Gemini (miễn phí) cho tiếng Việt
- Fallback sang OpenAI khi cần
- Tối ưu cost và performance