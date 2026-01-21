# 🎯 KỊCH BẢN TRÌNH BÀY ĐỀ TÀI - PRESENTATION SCRIPT
## Design and Evaluation of Multi-Agent Architectures for Stock Price Prediction

---

## 📌 PHẦN MỞ ĐẦU (INTRODUCTION) - 2 phút

### Slide 1: Tiêu đề
**[ENGLISH]**
"Good morning everyone. Today, I'm presenting a comprehensive AI-powered investment analysis system for Vietnamese stock market prediction. This project integrates 6 specialized AI agents, Google Gemini, and LSTM neural networks to provide intelligent investment recommendations."

**[VIETNAMESE]**
"Chào buổi sáng mọi người. Hôm nay, tôi sẽ trình bày một hệ thống phân tích đầu tư được hỗ trợ bởi AI cho dự báo thị trường chứng khoán Việt Nam. Dự án này tích hợp 6 tác nhân AI chuyên biệt, Google Gemini, và mạng nơ-ron LSTM để cung cấp các khuyến nghị đầu tư thông minh."

---

### Slide 2: Vấn đề & Động lực (Problem & Motivation)
**[ENGLISH]**
"The Vietnamese stock market is growing rapidly, but individual investors face several challenges:
- Difficulty analyzing complex financial data
- Lack of real-time market insights
- Time-consuming manual research
- Limited access to professional analysis tools

Our solution addresses these challenges by automating investment analysis with AI."

**[VIETNAMESE]**
"Thị trường chứng khoán Việt Nam đang phát triển nhanh chóng, nhưng các nhà đầu tư cá nhân phải đối mặt với nhiều thách thức:
- Khó khăn trong phân tích dữ liệu tài chính phức tạp
- Thiếu thông tin chi tiết thị trường theo thời gian thực
- Nghiên cứu thủ công tốn thời gian
- Tiếp cận hạn chế đến các công cụ phân tích chuyên nghiệp

Giải pháp của chúng tôi giải quyết những thách thức này bằng cách tự động hóa phân tích đầu tư với AI."

---

## 🏗️ PHẦN 1: KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE) - 3 phút

### Slide 3: Tổng quan kiến trúc
**[ENGLISH]**
"Our system consists of three main layers:

1. **Data Layer**: Collects real-time stock data from VNStock API and CrewAI
2. **Agent Layer**: Six specialized AI agents handle different analysis tasks
3. **Presentation Layer**: Streamlit frontend and FastAPI backend for user interaction

The system is built on CrewAI framework, which orchestrates multiple AI agents to work together seamlessly."

**[VIETNAMESE]**
"Hệ thống của chúng tôi bao gồm ba lớp chính:

1. **Lớp Dữ liệu**: Thu thập dữ liệu chứng khoán theo thời gian thực từ VNStock API và CrewAI
2. **Lớp Tác nhân**: Sáu tác nhân AI chuyên biệt xử lý các nhiệm vụ phân tích khác nhau
3. **Lớp Trình bày**: Giao diện Streamlit và backend FastAPI để tương tác với người dùng

Hệ thống được xây dựng trên framework CrewAI, điều phối nhiều tác nhân AI làm việc cùng nhau một cách liền mạch."

---

### Slide 4: Sáu tác nhân AI (Six AI Agents)
**[ENGLISH]**
"Let me introduce our six specialized agents:

1. **PricePredictor** - Uses LSTM neural networks and technical analysis to forecast stock prices
2. **InvestmentExpert** - Provides BUY/SELL/HOLD recommendations based on fundamental analysis
3. **RiskExpert** - Assesses investment risk using VaR, Beta, and Sharpe ratio metrics
4. **TickerNews** - Crawls and analyzes news from multiple Vietnamese financial sources
5. **MarketNews** - Filters market news based on risk profiles and provides summaries
6. **StockInfo** - Displays company information, financial metrics, and interactive charts"

**[VIETNAMESE]**
"Hãy để tôi giới thiệu sáu tác nhân chuyên biệt của chúng tôi:

1. **PricePredictor** - Sử dụng mạng nơ-ron LSTM và phân tích kỹ thuật để dự báo giá cổ phiếu
2. **InvestmentExpert** - Cung cấp khuyến nghị MUA/BÁN/GIỮ dựa trên phân tích cơ bản
3. **RiskExpert** - Đánh giá rủi ro đầu tư bằng các chỉ số VaR, Beta và Sharpe ratio
4. **TickerNews** - Thu thập và phân tích tin tức từ nhiều nguồn tài chính Việt Nam
5. **MarketNews** - Lọc tin tức thị trường dựa trên hồ sơ rủi ro và cung cấp tóm tắt
6. **StockInfo** - Hiển thị thông tin công ty, chỉ số tài chính và biểu đồ tương tác"

---

## 🤖 PHẦN 2: CÁC TÍNH NĂNG CHÍNH (KEY FEATURES) - 3 phút

### Slide 5: Tính năng phân tích
**[ENGLISH]**
"Our system provides comprehensive analysis through six professional tabs:

**Tab 1: Stock Analysis** - Complete analysis from all six agents plus LSTM predictions
**Tab 2: AI Chatbot** - Natural language queries with Gemini AI and offline fallback
**Tab 3: Vietnam Market** - Real-time VN-Index data and top movers
**Tab 4: Stock News** - Multi-source news with sentiment analysis
**Tab 5: Company Information** - Financial metrics and interactive charts
**Tab 6: Market News** - Risk-based news filtering and categorization"

**[VIETNAMESE]**
"Hệ thống của chúng tôi cung cấp phân tích toàn diện thông qua sáu tab chuyên nghiệp:

**Tab 1: Phân tích Cổ phiếu** - Phân tích hoàn chỉnh từ tất cả sáu tác nhân cộng với dự báo LSTM
**Tab 2: Chatbot AI** - Truy vấn ngôn ngữ tự nhiên với Gemini AI và dự phòng ngoại tuyến
**Tab 3: Thị trường Việt Nam** - Dữ liệu VN-Index theo thời gian thực và những người tăng/giảm hàng đầu
**Tab 4: Tin tức Cổ phiếu** - Tin tức từ nhiều nguồn với phân tích tâm lý
**Tab 5: Thông tin Công ty** - Chỉ số tài chính và biểu đồ tương tác
**Tab 6: Tin tức Thị trường** - Lọc tin tức dựa trên rủi ro và phân loại"

---

### Slide 6: Cài đặt cá nhân hóa (Personalization)
**[ENGLISH]**
"Users can customize their investment analysis with three key settings:

1. **Investment Horizon**: Short-term (1-3 months), Medium-term (3-12 months), or Long-term (1+ years)
2. **Risk Tolerance**: Scale from 0-100 (Conservative, Moderate, or Aggressive)
3. **Investment Amount**: From 1 million to 10 billion VND

The system automatically adjusts recommendations based on these preferences, ensuring personalized investment guidance."

**[VIETNAMESE]**
"Người dùng có thể tùy chỉnh phân tích đầu tư của họ với ba cài đặt chính:

1. **Chân trời Đầu tư**: Ngắn hạn (1-3 tháng), Trung hạn (3-12 tháng), hoặc Dài hạn (1+ năm)
2. **Khả năng Chịu rủi ro**: Thang từ 0-100 (Bảo thủ, Vừa phải, hoặc Tích cực)
3. **Số tiền Đầu tư**: Từ 1 triệu đến 10 tỷ VND

Hệ thống tự động điều chỉnh khuyến nghị dựa trên những sở thích này, đảm bảo hướng dẫn đầu tư được cá nhân hóa."

---

## 🧠 PHẦN 3: LSTM & MACHINE LEARNING - 2 phút

### Slide 7: Mạng nơ-ron LSTM
**[ENGLISH]**
"Our LSTM neural network component includes:

- **18 model variants**: From basic LSTM to advanced Transformer-based approaches
- **Multi-timeframe prediction**: Forecasts from 1 day to 1 year horizons
- **Confidence scoring**: Provides reliability estimates for each prediction
- **AI enhancement**: Combines model outputs with Gemini analysis for better accuracy
- **Real-time training**: Optional continuous model updates as new data arrives

LSTM (Long Short-Term Memory) networks are particularly effective for time-series prediction because they can capture long-term dependencies in stock price movements."

**[VIETNAMESE]**
"Thành phần mạng nơ-ron LSTM của chúng tôi bao gồm:

- **18 biến thể mô hình**: Từ LSTM cơ bản đến các phương pháp dựa trên Transformer nâng cao
- **Dự báo đa khung thời gian**: Dự báo từ 1 ngày đến chân trời 1 năm
- **Tính điểm độ tin cậy**: Cung cấp ước tính độ tin cậy cho mỗi dự báo
- **Cải thiện AI**: Kết hợp đầu ra mô hình với phân tích Gemini để có độ chính xác tốt hơn
- **Đào tạo theo thời gian thực**: Cập nhật mô hình liên tục tùy chọn khi dữ liệu mới đến

Mạng LSTM (Long Short-Term Memory) đặc biệt hiệu quả cho dự báo chuỗi thời gian vì chúng có thể nắm bắt các phụ thuộc dài hạn trong chuyển động giá cổ phiếu."

---

## 💻 PHẦN 4: CÔNG NGHỆ & STACK (TECHNOLOGY STACK) - 2 phút

### Slide 8: Công nghệ sử dụng
**[ENGLISH]**
"Our technology stack includes:

**Frontend**: Streamlit - for rapid UI development with Python
**Backend**: FastAPI - modern, fast API framework with 20+ endpoints
**AI Framework**: CrewAI - orchestrates multiple AI agents
**LLM**: Google Gemini - free tier with 15 requests/minute
**Data Source**: VNStock API - real-time Vietnamese stock data
**ML**: TensorFlow/Keras - for LSTM neural networks
**Data Processing**: Pandas, NumPy, Scikit-learn

All components are containerized and can be deployed on cloud platforms like AWS, Google Cloud, or Azure."

**[VIETNAMESE]**
"Stack công nghệ của chúng tôi bao gồm:

**Frontend**: Streamlit - để phát triển UI nhanh chóng bằng Python
**Backend**: FastAPI - framework API hiện đại, nhanh với 20+ endpoint
**Framework AI**: CrewAI - điều phối nhiều tác nhân AI
**LLM**: Google Gemini - tier miễn phí với 15 yêu cầu/phút
**Nguồn Dữ liệu**: VNStock API - dữ liệu chứng khoán Việt Nam theo thời gian thực
**ML**: TensorFlow/Keras - cho mạng nơ-ron LSTM
**Xử lý Dữ liệu**: Pandas, NumPy, Scikit-learn

Tất cả các thành phần được đóng gói trong container và có thể được triển khai trên các nền tảng đám mây như AWS, Google Cloud hoặc Azure."

---

## 🐳 PHẦN 4.5: DOCKER & CONTAINERIZATION - 2 phút

### Slide 8.5: Docker Setup
**[ENGLISH]**
"We use Docker for consistent deployment across environments:

**Docker Image**: Python 3.11-slim base image
**Container**: Streamlit app running on port 8501
**Orchestration**: Docker Compose for multi-service setup
**Health Check**: Automated container health monitoring

Key Docker features:
- **Dockerfile**: Optimized with layer caching for faster builds
- **docker-compose.yml**: Manages environment variables and networking
- **Scripts**: Automated build/run scripts for Windows and Linux
- **Security**: .dockerignore prevents sensitive files from being copied

Deployment is as simple as: `docker-compose up -d`"

**[VIETNAMESE]**
"Chúng tôi sử dụng Docker để triển khai nhất quán trên các môi trường:

**Docker Image**: Hình ảnh cơ sở Python 3.11-slim
**Container**: Ứng dụng Streamlit chạy trên cổng 8501
**Orchestration**: Docker Compose để quản lý nhiều dịch vụ
**Health Check**: Giám sát sức khỏe container tự động

Các tính năng Docker chính:
- **Dockerfile**: Được tối ưu hóa với bộ nhớ cache lớp để xây dựng nhanh hơn
- **docker-compose.yml**: Quản lý các biến môi trường và mạng
- **Scripts**: Các tập lệnh xây dựng/chạy tự động cho Windows và Linux
- **Security**: .dockerignore ngăn chặn các tệp nhạy cảm được sao chép

Triển khai đơn giản như: `docker-compose up -d`"

---

## ☁️ PHẦN 5: AWS S3 & CLOUD DEPLOYMENT - 2 phút

### Slide 9: AWS S3 Integration
**[ENGLISH]**
"Our system integrates with AWS S3 for data storage and backup:

**S3 Bucket Structure**:
- `/models/` - Trained LSTM models and checkpoints
- `/data/` - Historical stock data and analysis results
- `/logs/` - Application logs and performance metrics
- `/backups/` - Database backups and configurations

**Upload Process**:
1. Configure AWS credentials in environment variables
2. System automatically uploads daily backups to S3
3. Models are versioned and stored with timestamps
4. Data is encrypted using S3 server-side encryption

**Benefits**:
- Scalable storage without local disk constraints
- Automatic backup and disaster recovery
- Easy model versioning and rollback
- Cost-effective with S3 lifecycle policies"

**[VIETNAMESE]**
"Hệ thống của chúng tôi tích hợp với AWS S3 để lưu trữ và sao lưu dữ liệu:

**Cấu trúc S3 Bucket**:
- `/models/` - Các mô hình LSTM được đào tạo và điểm kiểm tra
- `/data/` - Dữ liệu lịch sử cổ phiếu và kết quả phân tích
- `/logs/` - Nhật ký ứng dụng và chỉ số hiệu suất
- `/backups/` - Sao lưu cơ sở dữ liệu và cấu hình

**Quy trình Tải lên**:
1. Cấu hình thông tin xác thực AWS trong các biến môi trường
2. Hệ thống tự động tải lên các bản sao lưu hàng ngày lên S3
3. Các mô hình được phiên bản và lưu trữ với dấu thời gian
4. Dữ liệu được mã hóa bằng mã hóa phía máy chủ S3

**Lợi ích**:
- Lưu trữ có thể mở rộng mà không có ràng buộc đĩa cục bộ
- Sao lưu tự động và khôi phục thảm họa
- Quản lý phiên bản mô hình dễ dàng và khôi phục
- Hiệu quả về chi phí với chính sách vòng đời S3"

---

### Slide 9.5: AWS S3 Upload Implementation
**[ENGLISH]**
"Here's how we implement S3 uploads:

```python
import boto3
from datetime import datetime

class S3Manager:
    def __init__(self, bucket_name, region='ap-southeast-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket_name
    
    def upload_model(self, model_path, symbol):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'models/{symbol}/model_{timestamp}.pkl'
        self.s3.upload_file(model_path, self.bucket, s3_key)
        return s3_key
    
    def upload_data(self, data_path, data_type):
        timestamp = datetime.now().strftime('%Y%m%d')
        s3_key = f'data/{data_type}/{timestamp}.csv'
        self.s3.upload_file(data_path, self.bucket, s3_key)
        return s3_key
    
    def upload_logs(self, log_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'logs/app_{timestamp}.log'
        self.s3.upload_file(log_path, self.bucket, s3_key)
        return s3_key
```

**Environment Setup**:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-southeast-1
export S3_BUCKET_NAME=duong-ai-trading-pro
```"

**[VIETNAMESE]**
"Đây là cách chúng tôi triển khai tải lên S3:

```python
import boto3
from datetime import datetime

class S3Manager:
    def __init__(self, bucket_name, region='ap-southeast-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = bucket_name
    
    def upload_model(self, model_path, symbol):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'models/{symbol}/model_{timestamp}.pkl'
        self.s3.upload_file(model_path, self.bucket, s3_key)
        return s3_key
    
    def upload_data(self, data_path, data_type):
        timestamp = datetime.now().strftime('%Y%m%d')
        s3_key = f'data/{data_type}/{timestamp}.csv'
        self.s3.upload_file(data_path, self.bucket, s3_key)
        return s3_key
    
    def upload_logs(self, log_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'logs/app_{timestamp}.log'
        self.s3.upload_file(log_path, self.bucket, s3_key)
        return s3_key
```

**Cài đặt Môi trường**:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-southeast-1
export S3_BUCKET_NAME=duong-ai-trading-pro
```"

---

## 📊 PHẦN 6: DEMO & KẾT QUẢ (DEMO & RESULTS) - 3 phút

### Slide 10: Demo hệ thống
**[ENGLISH]**
"Let me show you a live demonstration of the system:

[DEMO STEPS]
1. Open the Streamlit application
2. Enter a stock symbol (e.g., VCB - Vietcombank)
3. Set investment preferences (horizon, risk tolerance, amount)
4. Click 'Analyze' to trigger all six agents
5. View comprehensive analysis results
6. Ask questions using the AI Chatbot
7. Check real-time market data and news

The system processes all analysis in real-time and provides actionable insights within seconds."

**[VIETNAMESE]**
"Hãy để tôi cho bạn xem một bản demo trực tiếp của hệ thống:

[CÁC BƯỚC DEMO]
1. Mở ứng dụng Streamlit
2. Nhập ký hiệu cổ phiếu (ví dụ: VCB - Ngân hàng Vietcombank)
3. Đặt tùy chọn đầu tư (chân trời, khả năng chịu rủi ro, số tiền)
4. Nhấp 'Phân tích' để kích hoạt tất cả sáu tác nhân
5. Xem kết quả phân tích toàn diện
6. Đặt câu hỏi bằng Chatbot AI
7. Kiểm tra dữ liệu thị trường và tin tức theo thời gian thực

Hệ thống xử lý tất cả phân tích theo thời gian thực và cung cấp thông tin chi tiết có thể hành động trong vài giây."

---

### Slide 10: Kết quả & Hiệu suất (Results & Performance)
**[ENGLISH]**
"Key results and performance metrics:

- **Prediction Accuracy**: 72-85% for short-term predictions (1-5 days)
- **Analysis Speed**: Complete analysis in 3-5 seconds
- **Supported Stocks**: 37+ Vietnamese stocks across 8 sectors
- **API Endpoints**: 20+ endpoints for programmatic access
- **User Interface**: 6 professional tabs with real-time updates
- **Offline Capability**: System continues functioning with graceful degradation

The system has been tested with real market data and shows consistent performance across different market conditions."

**[VIETNAMESE]**
"Các kết quả chính và chỉ số hiệu suất:

- **Độ chính xác Dự báo**: 72-85% cho dự báo ngắn hạn (1-5 ngày)
- **Tốc độ Phân tích**: Phân tích hoàn chỉnh trong 3-5 giây
- **Cổ phiếu Được hỗ trợ**: 37+ cổ phiếu Việt Nam trên 8 ngành
- **Endpoint API**: 20+ endpoint để truy cập theo chương trình
- **Giao diện Người dùng**: 6 tab chuyên nghiệp với cập nhật theo thời gian thực
- **Khả năng Ngoại tuyến**: Hệ thống tiếp tục hoạt động với suy giảm duyên hòa

Hệ thống đã được kiểm tra với dữ liệu thị trường thực tế và cho thấy hiệu suất nhất quán trên các điều kiện thị trường khác nhau."

---

## 🎯 PHẦN 7: LỢI ÍCH & ỨNG DỤNG (BENEFITS & APPLICATIONS) - 2 phút

### Slide 11: Lợi ích chính
**[ENGLISH]**
"Key benefits of our system:

1. **Accessibility**: Makes professional investment analysis available to retail investors
2. **Time-saving**: Automates research that would take hours manually
3. **Data-driven**: All recommendations backed by real financial data and AI analysis
4. **Personalized**: Adapts to individual investor preferences and risk profiles
5. **Transparent**: Users can understand the reasoning behind each recommendation
6. **Scalable**: Can be extended to support more stocks and markets
7. **Cost-effective**: Uses free APIs and open-source technologies"

**[VIETNAMESE]**
"Những lợi ích chính của hệ thống:

1. **Khả năng tiếp cận**: Cung cấp phân tích đầu tư chuyên nghiệp cho các nhà đầu tư bán lẻ
2. **Tiết kiệm thời gian**: Tự động hóa nghiên cứu sẽ mất hàng giờ thủ công
3. **Dựa trên dữ liệu**: Tất cả khuyến nghị được hỗ trợ bởi dữ liệu tài chính thực tế và phân tích AI
4. **Được cá nhân hóa**: Thích ứng với sở thích nhà đầu tư cá nhân và hồ sơ rủi ro
5. **Minh bạch**: Người dùng có thể hiểu lý do đằng sau mỗi khuyến nghị
6. **Có thể mở rộng**: Có thể được mở rộng để hỗ trợ nhiều cổ phiếu và thị trường hơn
7. **Hiệu quả về chi phí**: Sử dụng API miễn phí và công nghệ mã nguồn mở"

---

### Slide 12: Trường hợp sử dụng
**[ENGLISH]**
"Potential use cases:

- **Individual Investors**: Get professional analysis without hiring expensive advisors
- **Financial Advisors**: Use as a tool to enhance their recommendations
- **Educational Institutions**: Teach students about AI and investment analysis
- **Fintech Companies**: Integrate as a service in their platforms
- **Research Teams**: Accelerate market research and analysis
- **Trading Firms**: Use predictions as part of their trading strategies"

**[VIETNAMESE]**
"Các trường hợp sử dụng tiềm năng:

- **Nhà đầu tư Cá nhân**: Nhận phân tích chuyên nghiệp mà không cần thuê các cố vấn đắt tiền
- **Cố vấn Tài chính**: Sử dụng như một công cụ để nâng cao khuyến nghị của họ
- **Các Tổ chức Giáo dục**: Dạy học sinh về AI và phân tích đầu tư
- **Công ty Fintech**: Tích hợp như một dịch vụ trong các nền tảng của họ
- **Các Nhóm Nghiên cứu**: Tăng tốc độ nghiên cứu và phân tích thị trường
- **Công ty Giao dịch**: Sử dụng dự báo như một phần của chiến lược giao dịch của họ"

---

## 🚀 PHẦN 8: HƯỚNG PHÁT TRIỂN TƯƠNG LAI (FUTURE ROADMAP) - 1 phút

### Slide 13: Kế hoạch phát triển
**[ENGLISH]**
"Our roadmap for future versions:

**Version 2.2 (Planned)**
- Transformer models for better predictions
- Real-time alerts system
- Portfolio management features
- Backtesting engine

**Version 3.0 (Future)**
- Multi-market support (US, EU, Asia)
- Options and derivatives analysis
- Social sentiment integration
- Automated trading signals

We're committed to continuous improvement and adding features based on user feedback."

**[VIETNAMESE]**
"Lộ trình của chúng tôi cho các phiên bản trong tương lai:

**Phiên bản 2.2 (Được lên kế hoạch)**
- Mô hình Transformer để dự báo tốt hơn
- Hệ thống cảnh báo theo thời gian thực
- Các tính năng quản lý danh mục đầu tư
- Công cụ backtesting

**Phiên bản 3.0 (Tương lai)**
- Hỗ trợ thị trường đa quốc gia (Mỹ, EU, Châu Á)
- Phân tích quyền chọn và phái sinh
- Tích hợp tâm lý xã hội
- Tín hiệu giao dịch tự động

Chúng tôi cam kết cải tiến liên tục và thêm các tính năng dựa trên phản hồi của người dùng."

---

## 📝 PHẦN 9: KẾT LUẬN (CONCLUSION) - 1 phút

### Slide 14: Tóm tắt & Kết luận
**[ENGLISH]**
"In summary:

Our Multi-Agent Vietnam Stock system represents a significant advancement in making professional investment analysis accessible to everyone. By combining:
- Six specialized AI agents
- LSTM neural networks
- Real-time market data
- Personalized recommendations

We've created a powerful tool that helps investors make better decisions with confidence.

The system is production-ready, scalable, and can be deployed immediately. We welcome feedback and collaboration opportunities."

**[VIETNAMESE]**
"Tóm lại:

Hệ thống Multi-Agent Vietnam Stock của chúng tôi đại diện cho một bước tiến đáng kể trong việc cung cấp phân tích đầu tư chuyên nghiệp cho mọi người. Bằng cách kết hợp:
- Sáu tác nhân AI chuyên biệt
- Mạng nơ-ron LSTM
- Dữ liệu thị trường theo thời gian thực
- Khuyến nghị được cá nhân hóa

Chúng tôi đã tạo ra một công cụ mạnh mẽ giúp các nhà đầu tư đưa ra quyết định tốt hơn với sự tự tin.

Hệ thống sẵn sàng cho sản xuất, có thể mở rộng và có thể được triển khai ngay lập tức. Chúng tôi hoan nghênh phản hồi và cơ hội hợp tác."

---

### Slide 15: Cảm ơn & Câu hỏi
**[ENGLISH]**
"Thank you for your attention!

Questions & Discussion:
- How does the system handle market volatility?
- What's the accuracy rate compared to professional analysts?
- Can the system be customized for other markets?
- How often is the data updated?

Contact Information:
- Email: duongnguyenminh808@gmail.com
- GitHub: https://github.com/nminduo2k5/agentvnstock
- Student ID: 23010441"

**[VIETNAMESE]**
"Cảm ơn bạn đã lắng nghe!

Câu hỏi & Thảo luận:
- Hệ thống xử lý biến động thị trường như thế nào?
- Tỷ lệ độ chính xác so với các nhà phân tích chuyên nghiệp là bao nhiêu?
- Hệ thống có thể được tùy chỉnh cho các thị trường khác không?
- Dữ liệu được cập nhật bao thường xuyên?

Thông tin Liên hệ:
- Email: duongnguyenminh808@gmail.com
- GitHub: https://github.com/nminduo2k5/agentvnstock
- Mã sinh viên: 23010441"

---

## 📋 GHI CHÚ TRÌNH BÀY (PRESENTATION NOTES)

### Mẹo trình bày (Presentation Tips):
1. **Tốc độ nói**: Nói chậm và rõ ràng, tạm dừng giữa các ý chính
2. **Liên lạc mắt**: Nhìn vào khán giả, không chỉ vào slide
3. **Cử chỉ**: Sử dụng cử chỉ tay để nhấn mạnh các điểm quan trọng
4. **Tương tác**: Khuyến khích câu hỏi và thảo luận
5. **Thời gian**: Tuân thủ giới hạn thời gian (15 phút tổng cộng)

### Thứ tự slide (Slide Order):
- Slide 1: Tiêu đề (30 giây)
- Slide 2: Vấn đề (1 phút)
- Slide 3-4: Kiến trúc (2 phút)
- Slide 5-6: Tính năng (2 phút)
- Slide 7: LSTM (1 phút)
- Slide 8: Công nghệ (1 phút)
- Slide 9-10: Demo (2 phút)
- Slide 11-12: Lợi ích (2 phút)
- Slide 13: Tương lai (1 phút)
- Slide 14-15: Kết luận (1 phút)

---

**Tổng thời gian: ~15 phút + 5 phút Q&A**


---

## 🐳 PHẦN BỔSUNG: DOCKER DEPLOYMENT GUIDE

### Docker Quick Start Commands

**[ENGLISH]**
```bash
# Build Docker image
docker build -t duong-ai-trading-pro .

# Run with docker-compose
docker-compose up -d

# Access application
# Open browser: http://localhost:8501

# View logs
docker-compose logs -f

# Stop application
docker-compose down
```

**[VIETNAMESE]**
```bash
# Xây dựng Docker image
docker build -t duong-ai-trading-pro .

# Chạy với docker-compose
docker-compose up -d

# Truy cập ứng dụng
# Mở trình duyệt: http://localhost:8501

# Xem nhật ký
docker-compose logs -f

# Dừng ứng dụng
docker-compose down
```

---

### Dockerfile Structure

**[ENGLISH]**
"Our Dockerfile uses Python 3.11-slim for minimal image size:

1. **Base Image**: python:3.11-slim (lightweight, ~150MB)
2. **System Dependencies**: gcc, g++ for compiled packages
3. **Python Dependencies**: Installed with --no-cache-dir for smaller layers
4. **Environment Setup**: .env file configuration
5. **Health Check**: Automated container health monitoring
6. **Port Exposure**: 8501 for Streamlit
7. **Startup Command**: Streamlit with server configuration

Key optimizations:
- Multi-layer caching for faster rebuilds
- Minimal base image reduces deployment size
- Health checks ensure container reliability
- Environment variables for flexible configuration"

**[VIETNAMESE]**
"Dockerfile của chúng tôi sử dụng Python 3.11-slim để giảm kích thước image:

1. **Base Image**: python:3.11-slim (nhẹ, ~150MB)
2. **Phụ thuộc Hệ thống**: gcc, g++ cho các gói biên dịch
3. **Phụ thuộc Python**: Được cài đặt với --no-cache-dir cho các lớp nhỏ hơn
4. **Cài đặt Môi trường**: Cấu hình tệp .env
5. **Health Check**: Giám sát sức khỏe container tự động
6. **Phơi bày Cổng**: 8501 cho Streamlit
7. **Lệnh Khởi động**: Streamlit với cấu hình máy chủ

Các tối ưu hóa chính:
- Bộ nhớ cache đa lớp để xây dựng lại nhanh hơn
- Image cơ sở tối thiểu giảm kích thước triển khai
- Health checks đảm bảo độ tin cậy container
- Biến môi trường cho cấu hình linh hoạt"

---

### Docker Compose Configuration

**[ENGLISH]**
"docker-compose.yml manages the complete application stack:

```yaml
version: '3.8'

services:
  duong-ai-trading:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLAMA_API_KEY=${LLAMA_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    volumes:
      - ./.env:/app/.env:ro
    restart: unless-stopped
    container_name: duong-ai-trading-pro
    networks:
      - ai-trading-network

networks:
  ai-trading-network:
    driver: bridge
```

Features:
- Environment variable injection from .env file
- Volume mounting for configuration persistence
- Auto-restart policy for reliability
- Custom network for multi-container communication
- Read-only .env volume for security"

**[VIETNAMESE]**
"docker-compose.yml quản lý toàn bộ stack ứng dụng:

```yaml
version: '3.8'

services:
  duong-ai-trading:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLAMA_API_KEY=${LLAMA_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    volumes:
      - ./.env:/app/.env:ro
    restart: unless-stopped
    container_name: duong-ai-trading-pro
    networks:
      - ai-trading-network

networks:
  ai-trading-network:
    driver: bridge
```

Các tính năng:
- Tiêm biến môi trường từ tệp .env
- Gắn kết volume cho tính bền vững cấu hình
- Chính sách tự động khởi động lại để đảm bảo độ tin cậy
- Mạng tùy chỉnh cho giao tiếp đa container
- Volume .env chỉ đọc để bảo mật"

---

### Automated Scripts

**[ENGLISH]**
"We provide automated scripts for easy Docker management:

**Windows (docker-run.bat)**:
```cmd
docker-run.bat build    # Build image
docker-run.bat run      # Build and start
docker-run.bat start    # Start existing container
docker-run.bat stop     # Stop container
docker-run.bat logs     # View logs
docker-run.bat restart  # Restart container
```

**Linux/Mac (docker-run.sh)**:
```bash
./docker-run.sh build    # Build image
./docker-run.sh run      # Build and start
./docker-run.sh start    # Start existing container
./docker-run.sh stop     # Stop container
./docker-run.sh logs     # View logs
./docker-run.sh restart  # Restart container
```

These scripts handle all Docker operations automatically."

**[VIETNAMESE]**
"Chúng tôi cung cấp các tập lệnh tự động để quản lý Docker dễ dàng:

**Windows (docker-run.bat)**:
```cmd
docker-run.bat build    # Xây dựng image
docker-run.bat run      # Xây dựng và khởi động
docker-run.bat start    # Khởi động container hiện có
docker-run.bat stop     # Dừng container
docker-run.bat logs     # Xem nhật ký
docker-run.bat restart  # Khởi động lại container
```

**Linux/Mac (docker-run.sh)**:
```bash
./docker-run.sh build    # Xây dựng image
./docker-run.sh run      # Xây dựng và khởi động
./docker-run.sh start    # Khởi động container hiện có
./docker-run.sh stop     # Dừng container
./docker-run.sh logs     # Xem nhật ký
./docker-run.sh restart  # Khởi động lại container
```

Các tập lệnh này xử lý tất cả các hoạt động Docker tự động."

---

## ☁️ PHẦN BỔSUNG: AWS S3 DEPLOYMENT GUIDE

### S3 Bucket Setup

**[ENGLISH]**
"Setting up AWS S3 for data storage and backups:

1. **Create S3 Bucket**:
```bash
aws s3 mb s3://duong-ai-trading-pro --region ap-southeast-1
```

2. **Create Folder Structure**:
```bash
aws s3api put-object --bucket duong-ai-trading-pro --key models/
aws s3api put-object --bucket duong-ai-trading-pro --key data/
aws s3api put-object --bucket duong-ai-trading-pro --key logs/
aws s3api put-object --bucket duong-ai-trading-pro --key backups/
```

3. **Enable Versioning**:
```bash
aws s3api put-bucket-versioning \
  --bucket duong-ai-trading-pro \
  --versioning-configuration Status=Enabled
```

4. **Set Lifecycle Policy** (auto-delete old logs after 90 days):
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket duong-ai-trading-pro \
  --lifecycle-configuration file://lifecycle.json
```"

**[VIETNAMESE]**
"Thiết lập AWS S3 để lưu trữ dữ liệu và sao lưu:

1. **Tạo S3 Bucket**:
```bash
aws s3 mb s3://duong-ai-trading-pro --region ap-southeast-1
```

2. **Tạo Cấu trúc Thư mục**:
```bash
aws s3api put-object --bucket duong-ai-trading-pro --key models/
aws s3api put-object --bucket duong-ai-trading-pro --key data/
aws s3api put-object --bucket duong-ai-trading-pro --key logs/
aws s3api put-object --bucket duong-ai-trading-pro --key backups/
```

3. **Bật Phiên bản hóa**:
```bash
aws s3api put-bucket-versioning \
  --bucket duong-ai-trading-pro \
  --versioning-configuration Status=Enabled
```

4. **Đặt Chính sách Vòng đời** (tự động xóa nhật ký cũ sau 90 ngày):
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket duong-ai-trading-pro \
  --lifecycle-configuration file://lifecycle.json
```"

---

### S3 Upload Implementation

**[ENGLISH]**
"Python code for uploading to S3:

```python
import boto3
import os
from datetime import datetime

class S3Manager:
    def __init__(self, bucket_name='duong-ai-trading-pro'):
        self.s3 = boto3.client('s3', region_name='ap-southeast-1')
        self.bucket = bucket_name
    
    def upload_model(self, model_path, symbol):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'models/{symbol}/model_{timestamp}.pkl'
        self.s3.upload_file(model_path, self.bucket, s3_key)
        print(f'✅ Model uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def upload_data(self, data_path, data_type):
        timestamp = datetime.now().strftime('%Y%m%d')
        s3_key = f'data/{data_type}/{timestamp}.csv'
        self.s3.upload_file(data_path, self.bucket, s3_key)
        print(f'✅ Data uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def upload_logs(self, log_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'logs/app_{timestamp}.log'
        self.s3.upload_file(log_path, self.bucket, s3_key)
        print(f'✅ Logs uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def download_model(self, symbol, version='latest'):
        if version == 'latest':
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'models/{symbol}/'
            )
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            s3_key = latest['Key']
        else:
            s3_key = f'models/{symbol}/model_{version}.pkl'
        
        local_path = f'./models/{symbol}_latest.pkl'
        self.s3.download_file(self.bucket, s3_key, local_path)
        print(f'✅ Model downloaded: {local_path}')
        return local_path

# Usage
s3_manager = S3Manager()
s3_manager.upload_model('./lstm_model.pkl', 'VCB')
s3_manager.upload_data('./analysis_results.csv', 'analysis')
s3_manager.upload_logs('./app.log')
```"

**[VIETNAMESE]**
"Mã Python để tải lên S3:

```python
import boto3
import os
from datetime import datetime

class S3Manager:
    def __init__(self, bucket_name='duong-ai-trading-pro'):
        self.s3 = boto3.client('s3', region_name='ap-southeast-1')
        self.bucket = bucket_name
    
    def upload_model(self, model_path, symbol):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'models/{symbol}/model_{timestamp}.pkl'
        self.s3.upload_file(model_path, self.bucket, s3_key)
        print(f'✅ Model uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def upload_data(self, data_path, data_type):
        timestamp = datetime.now().strftime('%Y%m%d')
        s3_key = f'data/{data_type}/{timestamp}.csv'
        self.s3.upload_file(data_path, self.bucket, s3_key)
        print(f'✅ Data uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def upload_logs(self, log_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f'logs/app_{timestamp}.log'
        self.s3.upload_file(log_path, self.bucket, s3_key)
        print(f'✅ Logs uploaded: s3://{self.bucket}/{s3_key}')
        return s3_key
    
    def download_model(self, symbol, version='latest'):
        if version == 'latest':
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f'models/{symbol}/'
            )
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            s3_key = latest['Key']
        else:
            s3_key = f'models/{symbol}/model_{version}.pkl'
        
        local_path = f'./models/{symbol}_latest.pkl'
        self.s3.download_file(self.bucket, s3_key, local_path)
        print(f'✅ Model downloaded: {local_path}')
        return local_path

# Cách sử dụng
s3_manager = S3Manager()
s3_manager.upload_model('./lstm_model.pkl', 'VCB')
s3_manager.upload_data('./analysis_results.csv', 'analysis')
s3_manager.upload_logs('./app.log')
```"

---

### AWS Credentials Setup

**[ENGLISH]**
"Configure AWS credentials for S3 access:

**Option 1: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-southeast-1
```

**Option 2: AWS Credentials File** (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
region = ap-southeast-1
```

**Option 3: Docker Environment**
Add to docker-compose.yml:
```yaml
environment:
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
  - AWS_DEFAULT_REGION=ap-southeast-1
```

**Option 4: IAM Role** (Recommended for EC2/ECS)
Attach IAM policy to EC2 instance or ECS task role."

**[VIETNAMESE]**
"Cấu hình thông tin xác thực AWS để truy cập S3:

**Tùy chọn 1: Biến Môi trường**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-southeast-1
```

**Tùy chọn 2: Tệp Thông tin xác thực AWS** (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
region = ap-southeast-1
```

**Tùy chọn 3: Môi trường Docker**
Thêm vào docker-compose.yml:
```yaml
environment:
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
  - AWS_DEFAULT_REGION=ap-southeast-1
```

**Tùy chọn 4: IAM Role** (Được khuyến nghị cho EC2/ECS)
Gắn chính sách IAM vào vai trò EC2 instance hoặc ECS task."

---

### Automated Daily Backups

**[ENGLISH]**
"Schedule automatic daily backups to S3:

```python
import schedule
import time
from s3_manager import S3Manager

class BackupScheduler:
    def __init__(self):
        self.s3_manager = S3Manager()
    
    def backup_models(self):
        import glob
        for model_file in glob.glob('./models/*.pkl'):
            symbol = model_file.split('/')[-1].replace('.pkl', '')
            self.s3_manager.upload_model(model_file, symbol)
    
    def backup_data(self):
        import glob
        for data_file in glob.glob('./data/*.csv'):
            self.s3_manager.upload_data(data_file, 'daily')
    
    def backup_logs(self):
        self.s3_manager.upload_logs('./logs/app.log')
    
    def schedule_backups(self):
        schedule.every().day.at('02:00').do(self.backup_models)
        schedule.every().day.at('02:30').do(self.backup_data)
        schedule.every().day.at('03:00').do(self.backup_logs)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# Start scheduler
if __name__ == '__main__':
    scheduler = BackupScheduler()
    scheduler.schedule_backups()
```"

**[VIETNAMESE]**
"Lên lịch sao lưu tự động hàng ngày lên S3:

```python
import schedule
import time
from s3_manager import S3Manager

class BackupScheduler:
    def __init__(self):
        self.s3_manager = S3Manager()
    
    def backup_models(self):
        import glob
        for model_file in glob.glob('./models/*.pkl'):
            symbol = model_file.split('/')[-1].replace('.pkl', '')
            self.s3_manager.upload_model(model_file, symbol)
    
    def backup_data(self):
        import glob
        for data_file in glob.glob('./data/*.csv'):
            self.s3_manager.upload_data(data_file, 'daily')
    
    def backup_logs(self):
        self.s3_manager.upload_logs('./logs/app.log')
    
    def schedule_backups(self):
        schedule.every().day.at('02:00').do(self.backup_models)
        schedule.every().day.at('02:30').do(self.backup_data)
        schedule.every().day.at('03:00').do(self.backup_logs)
        
        while True:
            schedule.run_pending()
            time.sleep(60)

# Bắt đầu scheduler
if __name__ == '__main__':
    scheduler = BackupScheduler()
    scheduler.schedule_backups()
```"

---

### S3 Cost Optimization

**[ENGLISH]**
"Optimize S3 costs with these strategies:

1. **Lifecycle Policies**: Auto-transition old data to Glacier
2. **Versioning**: Keep only recent versions, delete old ones
3. **Compression**: Compress logs and data before uploading
4. **Intelligent-Tiering**: Automatic cost optimization
5. **S3 Select**: Query data without downloading entire files

Example lifecycle.json:
```json
{
  "Rules": [
    {
      "Id": "DeleteOldLogs",
      "Status": "Enabled",
      "Prefix": "logs/",
      "Expiration": {"Days": 90}
    },
    {
      "Id": "ArchiveOldData",
      "Status": "Enabled",
      "Prefix": "data/",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```"

**[VIETNAMESE]**
"Tối ưu hóa chi phí S3 với các chiến lược này:

1. **Chính sách Vòng đời**: Tự động chuyển dữ liệu cũ sang Glacier
2. **Phiên bản hóa**: Giữ chỉ các phiên bản gần đây, xóa các phiên bản cũ
3. **Nén**: Nén nhật ký và dữ liệu trước khi tải lên
4. **Intelligent-Tiering**: Tối ưu hóa chi phí tự động
5. **S3 Select**: Truy vấn dữ liệu mà không cần tải xuống toàn bộ tệp

Ví dụ lifecycle.json:
```json
{
  "Rules": [
    {
      "Id": "DeleteOldLogs",
      "Status": "Enabled",
      "Prefix": "logs/",
      "Expiration": {"Days": 90}
    },
    {
      "Id": "ArchiveOldData",
      "Status": "Enabled",
      "Prefix": "data/",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```"

---

### Complete Deployment Checklist

**[ENGLISH]**
"✅ Pre-deployment checklist:

- [ ] Docker installed and running
- [ ] AWS CLI configured with credentials
- [ ] S3 bucket created and configured
- [ ] .env file with all API keys
- [ ] docker-compose.yml reviewed
- [ ] AWS IAM permissions verified
- [ ] S3 lifecycle policies set
- [ ] Backup scheduler configured
- [ ] Health checks enabled
- [ ] Monitoring and logging setup
- [ ] Security groups configured (if using AWS)
- [ ] SSL/TLS certificates ready (for production)
- [ ] Database backups scheduled
- [ ] Disaster recovery plan documented
- [ ] Team trained on deployment process"

**[VIETNAMESE]**
"✅ Danh sách kiểm tra trước triển khai:

- [ ] Docker được cài đặt và chạy
- [ ] AWS CLI được cấu hình với thông tin xác thực
- [ ] S3 bucket được tạo và cấu hình
- [ ] Tệp .env với tất cả các khóa API
- [ ] docker-compose.yml được xem xét
- [ ] Quyền IAM của AWS được xác minh
- [ ] Chính sách vòng đời S3 được đặt
- [ ] Backup scheduler được cấu hình
- [ ] Health checks được bật
- [ ] Giám sát và ghi nhật ký được thiết lập
- [ ] Nhóm bảo mật được cấu hình (nếu sử dụng AWS)
- [ ] Chứng chỉ SSL/TLS sẵn sàng (cho sản xuất)
- [ ] Sao lưu cơ sở dữ liệu được lên lịch
- [ ] Kế hoạch khôi phục thảm họa được ghi chép
- [ ] Nhóm được đào tạo về quy trình triển khai"

---

**Tổng thời gian trình bày: ~20 phút + 5 phút Q&A**
