from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
import os
import google.generativeai as genai
from typing import Optional, Dict, Any
import asyncio

app = FastAPI(title="LLM Hub Service", version="1.0.0")

# Redis connection
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class LLMRequest(BaseModel):
    prompt: str
    model: str = "gemini"
    temperature: float = 0.7
    max_tokens: int = 1000
    cache_key: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    model_used: str
    cached: bool = False
    tokens_used: Optional[int] = None

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "llm-hub",
        "models_available": ["gemini", "openai", "offline"],
        "redis_connected": redis_client.ping()
    }

@app.get("/models")
async def get_available_models():
    models = {
        "gemini": {
            "available": bool(GEMINI_API_KEY),
            "models": ["gemini-2.0-flash-exp", "gemini-1.5-pro"]
        },
        "openai": {
            "available": bool(os.getenv('OPENAI_API_KEY')),
            "models": ["gpt-4", "gpt-3.5-turbo"]
        },
        "offline": {
            "available": True,
            "models": ["fallback"]
        }
    }
    return models

@app.post("/generate", response_model=LLMResponse)
async def generate_response(request: LLMRequest):
    # Check cache first
    if request.cache_key:
        cached_response = redis_client.get(f"llm_cache:{request.cache_key}")
        if cached_response:
            cached_data = json.loads(cached_response)
            return LLMResponse(**cached_data, cached=True)
    
    try:
        if request.model == "gemini" and GEMINI_API_KEY:
            response = await generate_gemini_response(request)
        elif request.model == "openai" and os.getenv('OPENAI_API_KEY'):
            response = await generate_openai_response(request)
        else:
            response = generate_offline_response(request)
        
        # Cache the response
        if request.cache_key:
            cache_data = response.dict()
            cache_data['cached'] = False
            redis_client.setex(
                f"llm_cache:{request.cache_key}", 
                3600,  # 1 hour
                json.dumps(cache_data)
            )
        
        return response
        
    except Exception as e:
        # Fallback to offline mode
        return generate_offline_response(request)

async def generate_gemini_response(request: LLMRequest) -> LLMResponse:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            request.prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )
        )
        
        return LLMResponse(
            response=response.text,
            model_used="gemini-2.0-flash-exp",
            tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

async def generate_openai_response(request: LLMRequest) -> LLMResponse:
    # OpenAI implementation would go here
    raise HTTPException(status_code=501, detail="OpenAI integration not implemented yet")

def generate_offline_response(request: LLMRequest) -> LLMResponse:
    """Fallback offline responses for common stock analysis queries"""
    
    offline_responses = {
        "analysis": """📈 OFFLINE ANALYSIS:
        
Hệ thống đang hoạt động ở chế độ offline. Dưới đây là phân tích cơ bản:

💡 Nguyên tắc đầu tư cơ bản:
- P/E < 15 thường được coi là hấp dẫn
- Đa dạng hóa danh mục để giảm rủi ro  
- Chỉ đầu tư số tiền có thể chấp nhận mất

⚠️ Lưu ý: Đây là phân tích cơ bản. Vui lòng kiểm tra API key hoặc thử lại sau.""",
        
        "prediction": """🔮 DỰ ĐOÁN GIÁ (OFFLINE):
        
Không thể thực hiện dự đoán chính xác trong chế độ offline.

📊 Khuyến nghị chung:
- Theo dõi xu hướng thị trường tổng thể
- Phân tích kỹ thuật cơ bản (MA, RSI, MACD)
- Xem xét các yếu tố vĩ mô

⏰ API quota thường reset sau 24 giờ.""",
        
        "risk": """⚠️ ĐÁNH GIÁ RỦI RO (OFFLINE):
        
Nguyên tắc quản lý rủi ro cơ bản:
- Không đầu tư quá 5% tổng tài sản vào 1 cổ phiếu
- Đặt stop-loss ở mức 10-15%
- Đa dạng hóa theo ngành nghề

📈 Chỉ số rủi ro cần theo dõi:
- Beta (độ biến động so với thị trường)
- VaR (Value at Risk)
- Sharpe Ratio"""
    }
    
    # Simple keyword matching for offline responses
    prompt_lower = request.prompt.lower()
    
    if any(word in prompt_lower for word in ['dự đoán', 'predict', 'giá', 'price']):
        response_text = offline_responses['prediction']
    elif any(word in prompt_lower for word in ['rủi ro', 'risk', 'an toàn', 'safety']):
        response_text = offline_responses['risk']
    else:
        response_text = offline_responses['analysis']
    
    return LLMResponse(
        response=response_text,
        model_used="offline-fallback"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)