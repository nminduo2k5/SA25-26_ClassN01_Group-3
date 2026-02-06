# Event-Driven Multi-Agent Architecture for Stock Price Prediction
## Design and Evaluation of Multi-Agent Architectures: A Vietnam Case Study

**Author:** Nguyen Minh Duong  
**Student ID:** 23010441  
**Group:** 3

---

## Table of Contents

1. [System Overview](#system-overview)
2. [C4 Model - Level 1: System Context](#c4-level-1-system-context)
3. [C4 Model - Level 2: Container Diagram](#c4-level-2-container-diagram)
4. [C4 Model - Level 3: Component Diagram](#c4-level-3-component-diagram)
5. [C4 Model - Level 4: Code Diagram](#c4-level-4-code-diagram)
6. [Event-Driven Architecture](#event-driven-architecture)
7. [Multi-Agent Architectures](#multi-agent-architectures)
8. [Agent Details](#agent-details)
9. [Data Flow](#data-flow)
10. [Deployment Architecture](#deployment-architecture)

---

## System Overview

### Purpose
An intelligent stock price prediction system using event-driven multi-agent architecture with 6 specialized AI agents, supporting both Vietnamese and international markets.

### Key Components
- **6 Specialized Agents**: PricePredictor, InvestmentExpert, RiskExpert, TickerNews, MarketNews, StockInfo
- **3 Multi-Agent Architectures**: Hierarchical, Round-Robin, Ensemble Voting
- **3 LLM Engines**: Gemini 2.0 Flash, OpenAI GPT-4o, Llama 3.1
- **Real-time Data Sources**: VNStock API, CrewAI, Yahoo Finance
- **Neural Networks**: LSTM with 18 model variants

### Technology Stack
```
Frontend:  Streamlit 1.28+
Backend:   FastAPI 0.104+
AI/ML:     Google Gemini, OpenAI, Llama, TensorFlow
Data:      VNStock 3.2+, yfinance, CrewAI
Analysis:  Pandas, NumPy, Scikit-learn
```

---

## C4 Level 1: System Context

### System Context Diagram

```mermaid
C4Context
    title System Context diagram for Multi-Agent Stock Prediction System
    
    Enterprise_Boundary(b0, "Vietnam Stock Market Ecosystem") {
        Person(investor, "Retail Investor", "Individual investor seeking stock analysis and predictions")
        Person(trader, "Professional Trader", "Experienced trader requiring risk assessment and portfolio analysis")
        Person_Ext(analyst, "Financial Analyst", "Market researcher needing comprehensive investment reports")

        System(stockSystem, "Multi-Agent Stock Prediction System", "Intelligent investment analysis platform with 6 AI Agents, 3 Architectures, LSTM Neural Network, and Gemini AI for Vietnamese and international stock markets")

        Enterprise_Boundary(b1, "External Data & AI Services") {
            SystemDb_Ext(vnstock, "VNStock API", "Vietnamese market data provider - real-time prices, financial ratios, historical data for 37+ VN stocks")
            
            System_Boundary(b2, "AI & ML Services") {
                System(gemini, "Google Gemini 2.0 Flash", "Primary AI engine for natural language analysis and investment recommendations")
                System(openai, "OpenAI GPT-4o", "Secondary AI engine for enhanced analysis and fallback support")
            }

            System_Ext(crewai, "CrewAI + Serper API", "Real-time news aggregation and search engine for market sentiment analysis")
            SystemDb(yahoo, "Yahoo Finance API", "International market data provider for global stock information")

            Boundary(b3, "News Sources", "boundary") {
                SystemQueue(cafef, "CafeF News", "Vietnamese financial news source")
                SystemQueue_Ext(vietstock, "VietStock News", "Vietnamese stock market news and analysis")
            }
        }
    }

    BiRel(investor, stockSystem, "Analyzes stocks, gets predictions")
    BiRel(trader, stockSystem, "Performs risk assessment")
    Rel(analyst, stockSystem, "Generates market reports")
    
    Rel(stockSystem, vnstock, "Fetches VN market data", "HTTPS/REST")
    Rel(stockSystem, yahoo, "Fetches international data", "HTTPS/REST")
    Rel(stockSystem, crewai, "Gets real-time news", "HTTPS/API")
    BiRel(stockSystem, gemini, "AI analysis & recommendations", "HTTPS/API")
    Rel(stockSystem, openai, "Fallback AI analysis", "HTTPS/API")
    
    Rel(crewai, cafef, "Crawls news")
    Rel(crewai, vietstock, "Crawls news")
    Rel(vnstock, stockSystem, "Returns price, ratios, history")
    Rel(gemini, stockSystem, "Returns AI insights")

    UpdateElementStyle(investor, $fontColor="white", $bgColor="#1f77b4", $borderColor="#1f77b4")
    UpdateElementStyle(stockSystem, $fontColor="white", $bgColor="#2ca02c", $borderColor="#2ca02c")
    UpdateRelStyle(investor, stockSystem, $textColor="blue", $lineColor="blue", $offsetX="5")
    UpdateRelStyle(stockSystem, vnstock, $textColor="green", $lineColor="green", $offsetY="-10")
    UpdateRelStyle(stockSystem, gemini, $textColor="orange", $lineColor="orange", $offsetY="-20")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

### Key Relationships

**Users:**
- Retail Investors: Basic stock analysis, price predictions, AI chatbot queries
- Professional Traders: Advanced risk assessment (VaR, Beta, Sharpe), multi-timeframe LSTM predictions
- Financial Analysts: Comprehensive reports, market insights, fundamental analysis

**External Systems:**
- VNStock API: Vietnamese market data (37+ stocks: VCB, BID, CTG, FPT, etc.)
- Yahoo Finance: International market data and historical prices
- CrewAI + Serper: Real-time news aggregation from CafeF, VietStock, VCI
- Google Gemini: Primary AI engine with offline fallback capability
- OpenAI GPT-4o: Secondary AI for enhanced multi-model analysis

---

## C4 Level 2: Container Diagram

### Container Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Streamlit Web App<br/>Python/Streamlit<br/>6 Professional Tabs]
        API[FastAPI REST API<br/>Python/FastAPI<br/>20+ Endpoints]
    end
    
    subgraph "Application Layer"
        MAIN[Main Agent Orchestrator<br/>Python<br/>Coordinates all agents]
        ARCH[Architecture Manager<br/>Python<br/>3 prediction architectures]
        LLM_MGR[Unified LLM Agent<br/>Python<br/>Multi-model support]
    end
    
    subgraph "Agent Layer"
        A1[PricePredictor Agent<br/>LSTM + Technical Analysis]
        A2[InvestmentExpert Agent<br/>Fundamental Analysis]
        A3[RiskExpert Agent<br/>Risk Assessment]
        A4[TickerNews Agent<br/>Stock News]
        A5[MarketNews Agent<br/>Market News]
        A6[StockInfo Agent<br/>Company Data]
    end
    
    subgraph "Data Layer"
        VN_API[VNStock API Client<br/>Vietnamese Market]
        CREW_API[CrewAI Collector<br/>News Aggregation]
        CACHE[Data Cache<br/>In-Memory]
    end
    
    subgraph "External Services"
        DB1[(VNStock<br/>VCI Source)]
        DB2[(Yahoo Finance<br/>International)]
        DB3[(Serper API<br/>Search Engine)]
        AI1[(Gemini 2.0<br/>Google AI)]
        AI2[(GPT-4o<br/>OpenAI)]
        AI3[(Llama 3.1<br/>Local/Cloud)]
    end
    
    WEB -->|HTTP/WebSocket| API
    WEB -->|Direct calls| MAIN
    API -->|Async calls| MAIN
    
    MAIN -->|Orchestrate| ARCH
    MAIN -->|AI enhancement| LLM_MGR
    MAIN -->|Delegate tasks| A1
    MAIN -->|Delegate tasks| A2
    MAIN -->|Delegate tasks| A3
    MAIN -->|Delegate tasks| A4
    MAIN -->|Delegate tasks| A5
    MAIN -->|Delegate tasks| A6
    
    ARCH -->|Execute architecture| A1
    ARCH -->|Execute architecture| A2
    ARCH -->|Execute architecture| A3
    
    A1 -->|Fetch data| VN_API
    A2 -->|Fetch data| VN_API
    A3 -->|Fetch data| VN_API
    A4 -->|Fetch news| CREW_API
    A5 -->|Fetch news| CREW_API
    A6 -->|Fetch data| VN_API
    
    VN_API -->|Query| DB1
    VN_API -->|Query| DB2
    VN_API -->|Cache| CACHE
    CREW_API -->|Search| DB3
    
    LLM_MGR -->|Generate| AI1
    LLM_MGR -->|Generate| AI2
    LLM_MGR -->|Generate| AI3
```

### Container Responsibilities

**User Interface Layer:**
- Streamlit Web App: Interactive UI with 6 tabs (Stock Analysis, Chatbot, VN Market, Stock News, Company Info, Market News)
- FastAPI REST API: RESTful endpoints for programmatic access

**Application Layer:**
- Main Agent: Orchestrates all agents, manages workflows
- Architecture Manager: Implements 3 prediction architectures
- Unified LLM Agent: Manages multiple AI models with fallback

**Agent Layer:**
- 6 specialized agents for different analysis tasks
- Each agent is autonomous and event-driven

**Data Layer:**
- VNStock API Client: Vietnamese market data
- CrewAI Collector: Real-time news aggregation
- Cache: Performance optimization

---

## C4 Level 3: Component Diagram

### Main Agent Components

```mermaid
graph TB
    subgraph "MainAgent [main_agent.py]"
        MA_INIT[Initialization<br/>- Initialize 6 agents<br/>- Setup LLM agent<br/>- Configure APIs]
        MA_ANALYZE[analyze_stock<br/>- Validate symbol<br/>- Parallel execution<br/>- Result aggregation]
        MA_QUERY[process_query<br/>- Parse user query<br/>- Get comprehensive data<br/>- Generate AI response]
        MA_MARKET[get_market_overview<br/>- VN market data<br/>- International news<br/>- Aggregate results]
        MA_HELPER[Helper Methods<br/>- Error handling<br/>- Data validation<br/>- Profile adjustment]
    end
    
    subgraph "Agent Instances"
        PP[PricePredictor]
        IE[InvestmentExpert]
        RE[RiskExpert]
        TN[TickerNews]
        MN[MarketNews]
        SI[StockInfo]
    end
    
    subgraph "LLM Integration"
        ULLM[UnifiedLLMAgent<br/>- Gemini<br/>- OpenAI<br/>- Llama]
    end
    
    subgraph "Architecture System"
        ARCH_MGR[ArchitectureManager<br/>- Hierarchical<br/>- Round-Robin<br/>- Ensemble Voting]
    end
    
    MA_INIT -->|Create| PP
    MA_INIT -->|Create| IE
    MA_INIT -->|Create| RE
    MA_INIT -->|Create| TN
    MA_INIT -->|Create| MN
    MA_INIT -->|Create| SI
    MA_INIT -->|Initialize| ULLM
    MA_INIT -->|Initialize| ARCH_MGR
    
    MA_ANALYZE -->|Call| PP
    MA_ANALYZE -->|Call| IE
    MA_ANALYZE -->|Call| RE
    MA_ANALYZE -->|Call| TN
    MA_ANALYZE -->|Call| MN
    MA_ANALYZE -->|Call| SI
    
    MA_QUERY -->|Generate| ULLM
    MA_HELPER -->|Validate| MA_ANALYZE
```


### PricePredictor Agent Components

```mermaid
graph TB
    subgraph "PricePredictor [price_predictor.py]"
        PP_INIT[Initialization<br/>- LSTM predictor<br/>- Technical indicators<br/>- Prediction periods]
        PP_PRED[predict_price_enhanced<br/>- LSTM priority<br/>- Traditional fallback<br/>- Multi-timeframe]
        PP_LSTM[LSTM Integration<br/>- 18 model variants<br/>- Confidence scoring<br/>- AI enhancement]
        PP_TECH[Technical Analysis<br/>- 20+ indicators<br/>- Trend analysis<br/>- ML predictions]
        PP_CALC[Calculations<br/>- Base change<br/>- Trend multiplier<br/>- Risk adjustments]
    end
    
    subgraph "LSTM Predictor"
        LSTM[LSTMPricePredictor<br/>- Neural networks<br/>- Time series<br/>- Confidence intervals]
    end
    
    subgraph "Data Sources"
        VN[VNStock API]
        YF[Yahoo Finance]
    end
    
    PP_INIT -->|Initialize| LSTM
    PP_PRED -->|Try first| PP_LSTM
    PP_PRED -->|Fallback| PP_TECH
    PP_LSTM -->|Use| LSTM
    PP_TECH -->|Calculate| PP_CALC
    PP_PRED -->|Fetch| VN
    PP_PRED -->|Fetch| YF
```

### InvestmentExpert Agent Components

```mermaid
graph TB
    subgraph "InvestmentExpert [investment_expert.py]"
        IE_INIT[Initialization<br/>- VN API client<br/>- AI agent<br/>- Risk profiles]
        IE_ANALYZE[analyze_stock<br/>- Fetch metrics<br/>- Calculate scores<br/>- Generate recommendation]
        IE_FINANCIAL[Financial Analysis<br/>- P/E, P/B, EPS<br/>- Dividend yield<br/>- Market cap]
        IE_TECHNICAL[Technical Analysis<br/>- 52-week range<br/>- Volume analysis<br/>- Beta calculation]
        IE_VALUATION[Valuation Analysis<br/>- Forward P/E<br/>- P/B valuation<br/>- Dividend sustainability]
        IE_SCORE[Score Calculation<br/>- Weighted scoring<br/>- Recommendation logic<br/>- Confidence level]
        IE_PROFILE[Profile Adjustment<br/>- Risk tolerance<br/>- Time horizon<br/>- Position sizing]
    end
    
    subgraph "AI Enhancement"
        AI[AI Agent<br/>- Gemini/OpenAI/Llama<br/>- Contextual advice<br/>- Reasoning]
    end
    
    IE_INIT -->|Setup| AI
    IE_ANALYZE -->|Execute| IE_FINANCIAL
    IE_ANALYZE -->|Execute| IE_TECHNICAL
    IE_ANALYZE -->|Execute| IE_VALUATION
    IE_FINANCIAL -->|Feed| IE_SCORE
    IE_TECHNICAL -->|Feed| IE_SCORE
    IE_VALUATION -->|Feed| IE_SCORE
    IE_SCORE -->|Adjust| IE_PROFILE
    IE_PROFILE -->|Enhance| AI
```

### RiskExpert Agent Components

```mermaid
graph TB
    subgraph "RiskExpert [risk_expert.py]"
        RE_INIT[Initialization<br/>- VN API client<br/>- Risk metrics<br/>- AI agent]
        RE_ASSESS[assess_risk<br/>- Calculate metrics<br/>- Determine level<br/>- Generate advice]
        RE_VAR[VaR Calculation<br/>- 95% confidence<br/>- Historical simulation<br/>- Monte Carlo]
        RE_METRICS[Risk Metrics<br/>- Volatility<br/>- Beta<br/>- Sharpe ratio<br/>- Max drawdown]
        RE_PROFILE[Profile Analysis<br/>- Risk tolerance<br/>- Time horizon<br/>- Position limits]
        RE_AI[AI Enhancement<br/>- Risk advice<br/>- Scenario analysis<br/>- Recommendations]
    end
    
    subgraph "Data Processing"
        HIST[Historical Data<br/>- Price history<br/>- Returns calculation<br/>- Statistical analysis]
    end
    
    RE_INIT -->|Setup| HIST
    RE_ASSESS -->|Calculate| RE_VAR
    RE_ASSESS -->|Calculate| RE_METRICS
    RE_METRICS -->|Analyze| RE_PROFILE
    RE_PROFILE -->|Enhance| RE_AI
```

---

## C4 Level 4: Code Diagram

### PricePredictor Class Structure

```python
class PricePredictor:
    # Attributes
    - name: str
    - vn_api: VNStockAPI
    - stock_info: StockInfoDisplay
    - ai_agent: UnifiedLLMAgent
    - lstm_predictor: LSTMPricePredictor
    - prediction_periods: dict
    
    # Core Methods
    + predict_price_enhanced(symbol, days, risk_tolerance, time_horizon, investment_amount)
    + predict_comprehensive(symbol, vn_api, stock_info)
    
    # Private Methods
    - _predict_vn_stock(symbol, vn_api)
    - _predict_international_stock(symbol)
    - _calculate_advanced_indicators(data)
    - _analyze_market_trend(data, predictions)
    - _generate_multi_timeframe_predictions(data, indicators, ml_predictions)
    - _apply_ml_predictions(data, indicators)
    - _calculate_confidence_scores(data, indicators, ml_predictions)
    - _analyze_risk_metrics(data)
    - _generate_recommendations(predictions, confidence_scores, risk_analysis)
    
    # Helper Methods
    - _calculate_trend_multiplier(indicators, rsi, bb_position)
    - _calculate_base_change(days, volatility, trend_multiplier)
    - _calculate_rsi_adjustment(rsi, days)
    - _calculate_bb_adjustment(bb_position, days)
    - _combine_lstm_with_traditional(lstm_result, symbol)
    - _get_risk_adjusted_analysis(result, risk_tolerance, time_horizon, investment_amount)
    - _get_ai_price_analysis(symbol, technical_data, days, risk_tolerance, time_horizon)
```

### InvestmentExpert Class Structure

```python
class InvestmentExpert:
    # Attributes
    - name: str
    - _vn_api: VNStockAPI
    - ai_agent: UnifiedLLMAgent
    
    # Core Methods
    + analyze_stock(symbol, risk_tolerance, time_horizon, investment_amount)
    + analyze_investment_decision(symbol)
    
    # Analysis Methods
    - _analyze_financial_metrics(metrics)
    - _analyze_technical_indicators(metrics)
    - _analyze_valuation(metrics)
    - _calculate_investment_score(financial, technical, valuation)
    - _make_investment_recommendation(score)
    
    # Data Methods
    - _fetch_real_detailed_metrics(symbol)
    - _generate_detailed_metrics(stock_data, symbol)
    - validate_metrics(metrics)
    
    # Profile Methods
    - _adjust_analysis_for_profile(base_analysis, risk_tolerance, time_horizon, investment_amount)
    - _get_risk_profile_name(risk_tolerance)
    - _calculate_max_position(risk_tolerance)
    - _get_time_horizon_days(time_horizon)
    
    # AI Methods
    + get_ai_enhancement(symbol, base_analysis)
    - _create_diverse_investment_context(symbol, base_analysis, risk_tolerance, time_horizon, investment_amount, risk_profile)
    - _create_diverse_investment_advice(symbol, base_analysis, risk_tolerance, time_horizon, investment_amount)
    - _parse_ai_advice(ai_response)
```

### MainAgent Class Structure

```python
class MainAgent:
    # Attributes
    - vn_api: VNStockAPI
    - stock_info: StockInfoDisplay
    - price_predictor: PricePredictor
    - ticker_news: TickerNews
    - market_news: MarketNews
    - investment_expert: InvestmentExpert
    - risk_expert: RiskExpert
    - international_news: InternationalMarketNews
    - llm_agent: UnifiedLLMAgent
    - architecture_manager: ArchitectureManager
    
    # Core Methods
    + analyze_stock(symbol, risk_tolerance, time_horizon, investment_amount)
    + get_market_overview()
    + process_query(query, symbol)
    
    # Configuration Methods
    + set_llm_keys(gemini_api_key, openai_api_key, llama_api_key, llama_base_url)
    + set_crewai_keys(gemini_api_key, openai_api_key, llama_api_key, llama_base_url, serper_api_key)
    
    # Architecture Methods
    + predict_price_with_architecture(symbol, architecture, timeframe)
    + compare_architectures(symbol, timeframe)
    + get_architecture_info()
    + get_architecture_performance()
    
    # Helper Methods
    - _integrate_llm_with_agents()
    - _safe_get_price_prediction(symbol)
    - _safe_get_ticker_news(symbol, limit)
    - _safe_get_investment_analysis(symbol, risk_tolerance, time_horizon, investment_amount)
    - _safe_get_risk_assessment(symbol, risk_tolerance, time_horizon, investment_amount)
    - _get_error_fallback(task_name, symbol, error)
    - _is_valid_international_symbol(symbol)
    - _get_risk_profile_name(risk_tolerance)
```

---

## Event-Driven Architecture

### Event Flow Overview

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant MainAgent
    participant Agents
    participant DataSources
    participant LLM
    
    User->>Streamlit: Request stock analysis
    Streamlit->>MainAgent: analyze_stock(symbol, profile)
    
    MainAgent->>MainAgent: Validate symbol
    MainAgent->>MainAgent: Create async tasks
    
    par Parallel Agent Execution
        MainAgent->>Agents: PricePredictor.predict()
        MainAgent->>Agents: InvestmentExpert.analyze()
        MainAgent->>Agents: RiskExpert.assess()
        MainAgent->>Agents: TickerNews.get_news()
        MainAgent->>Agents: MarketNews.get_news()
        MainAgent->>Agents: StockInfo.get_info()
    end
    
    par Parallel Data Fetching
        Agents->>DataSources: VNStock API
        Agents->>DataSources: Yahoo Finance
        Agents->>DataSources: CrewAI
    end
    
    DataSources-->>Agents: Return data
    Agents-->>MainAgent: Return results
    
    MainAgent->>LLM: Enhance with AI
    LLM-->>MainAgent: AI analysis
    
    MainAgent->>MainAgent: Aggregate results
    MainAgent-->>Streamlit: Complete analysis
    Streamlit-->>User: Display results
```

### Event Types

**User Events:**
- `ANALYZE_STOCK`: Trigger comprehensive stock analysis
- `PREDICT_PRICE`: Request price prediction
- `ASSESS_RISK`: Request risk assessment
- `GET_NEWS`: Fetch stock/market news
- `QUERY_AI`: Ask AI chatbot

**System Events:**
- `DATA_FETCH_START`: Begin data retrieval
- `DATA_FETCH_COMPLETE`: Data retrieval finished
- `AGENT_EXECUTION_START`: Agent begins processing
- `AGENT_EXECUTION_COMPLETE`: Agent finishes processing
- `AI_ENHANCEMENT_START`: AI analysis begins
- `AI_ENHANCEMENT_COMPLETE`: AI analysis finished
- `ERROR_OCCURRED`: Error handling triggered

**Data Events:**
- `PRICE_UPDATE`: New price data available
- `NEWS_UPDATE`: New news articles available
- `METRIC_UPDATE`: Financial metrics updated
- `CACHE_HIT`: Data retrieved from cache
- `CACHE_MISS`: Data fetched from source


---

## Multi-Agent Architectures

### Overview: Three Architecture Patterns

```mermaid
graph TB
    subgraph "Multi-Agent Architecture Comparison"
        subgraph "Input Layer"
            USER[User Request<br/>Symbol: VCB<br/>Timeframe: 30 days<br/>Risk Profile: Moderate]
        end
        
        subgraph "Architecture Selection"
            SELECTOR{Architecture<br/>Manager}
        end
        
        subgraph "Architecture 1: Hierarchical"
            H_COORD[Big Agent Coordinator]
            H_AGENTS[6 Agents Execute in Parallel]
            H_AGG[Weighted Aggregation]
            H_DEC[Centralized Decision]
            
            H_COORD --> H_AGENTS
            H_AGENTS --> H_AGG
            H_AGG --> H_DEC
        end
        
        subgraph "Architecture 2: Round-Robin"
            R_START[Initial Prediction]
            R_A1[Agent 1 Refines]
            R_A2[Agent 2 Refines]
            R_A3[Agent 3 Refines]
            R_A4[Agent 4 Refines]
            R_A5[Agent 5 Refines]
            R_A6[Agent 6 Refines]
            R_FINAL[Final Refined Result]
            
            R_START --> R_A1 --> R_A2 --> R_A3 --> R_A4 --> R_A5 --> R_A6 --> R_FINAL
        end
        
        subgraph "Architecture 3: Ensemble Voting"
            E_PARALLEL[6 Agents Vote in Parallel]
            E_BAYES[Bayesian Inference Engine]
            E_STATS[Statistical Analysis]
            E_FINAL[Weighted Consensus + Uncertainty]
            
            E_PARALLEL --> E_BAYES
            E_BAYES --> E_STATS
            E_STATS --> E_FINAL
        end
        
        subgraph "Output Layer"
            RESULT[Final Prediction<br/>Price: 112,000 VND<br/>Confidence: 85%<br/>Recommendation: BUY]
        end
        
        subgraph "Performance Metrics"
            METRICS[Execution Time<br/>Accuracy Score<br/>Confidence Level<br/>Uncertainty Range]
        end
    end
    
    USER --> SELECTOR
    
    SELECTOR -->|Pattern 1| H_COORD
    SELECTOR -->|Pattern 2| R_START
    SELECTOR -->|Pattern 3| E_PARALLEL
    
    H_DEC --> RESULT
    R_FINAL --> RESULT
    E_FINAL --> RESULT
    
    RESULT --> METRICS
```

### Architecture Characteristics Summary

| Characteristic | Hierarchical | Round-Robin | Ensemble Voting |
|---------------|--------------|-------------|------------------|
| **Pattern** | Coordinator-Worker | Sequential Pipeline | Parallel Voting |
| **Execution** | Parallel → Aggregate | Sequential Chain | Parallel → Bayesian |
| **Communication** | Hub-and-Spoke | Linear Chain | Broadcast-Collect |
| **Speed** | ⚡⚡ Medium (2-3s) | ⚡ Slow (4-6s) | ⚡⚡⚡ Fast (1-2s) |
| **Accuracy** | ⭐⭐⭐ Good (75-80%) | ⭐⭐⭐⭐ Very Good (80-85%) | ⭐⭐⭐⭐⭐ Excellent (85-90%) |
| **Robustness** | ❌ Low (SPOF) | ⚠️ Medium | ✅ High |
| **Scalability** | ⚠️ Limited | ❌ Poor | ✅ Excellent |
| **Complexity** | 🟢 Low | 🟡 Medium | 🔴 High |
| **Uncertainty** | ❌ Not quantified | ❌ Not quantified | ✅ Quantified (±range) |
| **Best Use Case** | Simple analysis | Iterative refinement | Complex predictions |
| **Failure Mode** | Coordinator fails → All fail | One agent fails → Chain breaks | One agent fails → Others compensate |

### Agent Roles Across Architectures

```mermaid
graph LR
    subgraph "6 Specialized Agents"
        A1[🔮 PricePredictor<br/>LSTM + Technical<br/>Weight: 25%]
        A2[💼 InvestmentExpert<br/>Fundamental Analysis<br/>Weight: 20%]
        A3[⚠️ RiskExpert<br/>Risk Assessment<br/>Weight: 20%]
        A4[📰 TickerNews<br/>Stock News<br/>Weight: 15%]
        A5[🌍 MarketNews<br/>Market Trends<br/>Weight: 10%]
        A6[🏢 StockInfo<br/>Company Data<br/>Weight: 10%]
    end
    
    subgraph "Data Sources"
        VN[VNStock API]
        YF[Yahoo Finance]
        CREW[CrewAI]
        AI[Gemini/OpenAI]
    end
    
    A1 --> VN
    A1 --> YF
    A2 --> VN
    A3 --> VN
    A4 --> CREW
    A5 --> CREW
    A6 --> VN
    
    A1 --> AI
    A2 --> AI
    A3 --> AI
```

### Architecture 1: Hierarchical (Big Agent)

```mermaid
graph TB
    subgraph "Hierarchical Architecture"
        BA[Big Agent<br/>Coordinator]
        
        subgraph "Layer 1: Data Collection"
            A1[PricePredictor]
            A2[InvestmentExpert]
            A3[RiskExpert]
            A4[TickerNews]
            A5[MarketNews]
            A6[StockInfo]
        end
        
        subgraph "Layer 2: Aggregation"
            AGG[Result Aggregator<br/>- Weighted combination<br/>- Conflict resolution<br/>- Confidence scoring]
        end
        
        subgraph "Layer 3: Decision"
            DEC[Decision Maker<br/>- Final recommendation<br/>- Risk adjustment<br/>- Profile matching]
        end
    end
    
    BA -->|Delegate| A1
    BA -->|Delegate| A2
    BA -->|Delegate| A3
    BA -->|Delegate| A4
    BA -->|Delegate| A5
    BA -->|Delegate| A6
    
    A1 -->|Results| AGG
    A2 -->|Results| AGG
    A3 -->|Results| AGG
    A4 -->|Results| AGG
    A5 -->|Results| AGG
    A6 -->|Results| AGG
    
    AGG -->|Aggregated data| DEC
    DEC -->|Final decision| BA
```

**Characteristics:**
- **Coordination**: Big Agent coordinates all sub-agents
- **Aggregation**: Results combined with weighted scoring
- **Decision**: Single point of decision-making
- **Pros**: Clear hierarchy, easy to understand, centralized control
- **Cons**: Single point of failure, bottleneck at coordinator

**Implementation:**
```python
# design_1_hierarchical.py
class HierarchicalPricePredictionSystem:
    async def predict_price(self, symbol, timeframe):
        # Step 1: Collect from all agents
        results = await self._collect_agent_predictions(symbol)
        
        # Step 2: Aggregate with weights
        aggregated = self._aggregate_results(results)
        
        # Step 3: Make final decision
        final = self._make_final_decision(aggregated)
        
        return final
```

### Architecture 2: Round-Robin (Sequential Improvement)

```mermaid
graph TB
    subgraph "Round-Robin Architecture"
        START[Initial Prediction<br/>Base price from market]
        
        subgraph "Sequential Refinement Pipeline"
            A1[Agent 1: PricePredictor<br/>- LSTM prediction<br/>- Technical indicators<br/>- Trend analysis]
            A2[Agent 2: InvestmentExpert<br/>- Fundamental analysis<br/>- Financial ratios<br/>- Valuation metrics]
            A3[Agent 3: RiskExpert<br/>- Risk assessment<br/>- Volatility analysis<br/>- Risk-adjusted price]
            A4[Agent 4: TickerNews<br/>- News sentiment<br/>- Market perception<br/>- Sentiment adjustment]
            A5[Agent 5: MarketNews<br/>- Market trends<br/>- Sector analysis<br/>- Macro adjustment]
            A6[Agent 6: StockInfo<br/>- Company metrics<br/>- Industry comparison<br/>- Final validation]
        end
        
        FINAL[Final Prediction<br/>Refined by all 6 agents<br/>Cumulative improvements]
    end
    
    START -->|Initial price| A1
    A1 -->|Improved v1<br/>+Technical| A2
    A2 -->|Improved v2<br/>+Fundamental| A3
    A3 -->|Improved v3<br/>+Risk| A4
    A4 -->|Improved v4<br/>+News| A5
    A5 -->|Improved v5<br/>+Market| A6
    A6 -->|Final refined<br/>+Validation| FINAL
```

**Characteristics:**
- **Sequential**: Each agent refines previous agent's output
- **Iterative**: Continuous improvement through pipeline
- **Cumulative**: Knowledge builds up through chain
- **Pros**: Progressive refinement, each agent adds value
- **Cons**: Slower execution, error propagation

**Implementation:**
```python
# design_2_round_robin.py
class RoundRobinPricePredictionSystem:
    async def predict_price(self, symbol, timeframe):
        prediction = self._initial_prediction(symbol)
        
        # Sequential refinement
        for agent in self.agents:
            prediction = await agent.refine(prediction, symbol)
        
        return prediction
```

### Architecture 3: Ensemble Voting (Bayesian Inference)

```mermaid
graph TB
    subgraph "Ensemble Voting Architecture"
        subgraph "Layer 1: Parallel Agent Execution"
            A1[Agent 1: PricePredictor<br/>Vote: 1250 VND<br/>Confidence: 0.85<br/>Weight: High]
            A2[Agent 2: InvestmentExpert<br/>Vote: 1280 VND<br/>Confidence: 0.75<br/>Weight: Medium]
            A3[Agent 3: RiskExpert<br/>Vote: 1230 VND<br/>Confidence: 0.90<br/>Weight: High]
            A4[Agent 4: TickerNews<br/>Vote: 1260 VND<br/>Confidence: 0.70<br/>Weight: Medium]
            A5[Agent 5: MarketNews<br/>Vote: 1270 VND<br/>Confidence: 0.65<br/>Weight: Low]
            A6[Agent 6: StockInfo<br/>Vote: 1240 VND<br/>Confidence: 0.80<br/>Weight: Medium]
        end
        
        subgraph "Layer 2: Bayesian Inference Engine"
            BAYES[Bayesian Aggregator<br/>- Prior: Market price distribution<br/>- Likelihood: Agent predictions<br/>- Posterior: Weighted combination<br/>- Confidence: Uncertainty quantification]
        end
        
        subgraph "Layer 3: Statistical Analysis"
            STATS[Statistical Processor<br/>- Mean calculation<br/>- Variance estimation<br/>- Confidence intervals<br/>- Outlier detection]
        end
        
        subgraph "Layer 4: Final Output"
            FINAL[Final Prediction<br/>Price: 1255 VND<br/>Confidence: 0.88<br/>Uncertainty: ±25 VND<br/>Range: 1230-1280]
        end
    end
    
    A1 -->|Vote + Confidence| BAYES
    A2 -->|Vote + Confidence| BAYES
    A3 -->|Vote + Confidence| BAYES
    A4 -->|Vote + Confidence| BAYES
    A5 -->|Vote + Confidence| BAYES
    A6 -->|Vote + Confidence| BAYES
    
    BAYES -->|Posterior distribution| STATS
    STATS -->|Statistical summary| FINAL
```

**Characteristics:**
- **Parallel**: All agents execute simultaneously
- **Voting**: Each agent provides prediction + confidence
- **Bayesian**: Statistical inference for final decision
- **Weighted**: Higher confidence = more influence
- **Pros**: Robust, handles disagreement, quantifies uncertainty
- **Cons**: Complex implementation, requires calibration

**Implementation:**
```python
# design_3_ensemble_voting.py
class EnsembleVotingPricePredictionSystem:
    async def predict_price(self, symbol, timeframe):
        # Parallel execution
        predictions = await asyncio.gather(*[
            agent.predict(symbol) for agent in self.agents
        ])
        
        # Bayesian inference
        final = self._bayesian_aggregation(predictions)
        
        return final
    
    def _bayesian_aggregation(self, predictions):
        # Weight by confidence
        weights = [p['confidence'] for p in predictions]
        prices = [p['price'] for p in predictions]
        
        # Weighted average
        final_price = np.average(prices, weights=weights)
        
        # Uncertainty quantification
        uncertainty = np.std(prices)
        
        return {
            'price': final_price,
            'confidence': np.mean(weights),
            'uncertainty': uncertainty
        }
```

### Architecture Comparison

| Feature | Hierarchical | Round-Robin | Ensemble Voting |
|---------|-------------|-------------|-----------------|
| **Execution** | Parallel → Sequential | Sequential | Parallel |
| **Speed** | Medium | Slow | Fast |
| **Accuracy** | Good | Very Good | Excellent |
| **Robustness** | Low | Medium | High |
| **Complexity** | Low | Medium | High |
| **Uncertainty** | Not quantified | Not quantified | Quantified |
| **Best for** | Simple tasks | Refinement tasks | Complex predictions |

---

## Agent Details

### Agent 1: PricePredictor

**Purpose:** Predict future stock prices using LSTM neural networks and technical analysis

**Key Features:**
- 18 LSTM model variants
- Multi-timeframe predictions (1d, 7d, 30d, 90d, 180d, 365d)
- 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Machine learning enhancements
- Confidence scoring
- AI-enhanced analysis

**Data Flow:**
```mermaid
graph LR
    INPUT[Symbol + Days] -->|Fetch| DATA[Historical Data]
    DATA -->|Calculate| TECH[Technical Indicators]
    DATA -->|Train| LSTM[LSTM Models]
    TECH -->|Analyze| TREND[Trend Analysis]
    LSTM -->|Predict| PRED[Price Predictions]
    TREND -->|Enhance| PRED
    PRED -->|Validate| CONF[Confidence Scores]
    CONF -->|AI Enhance| OUTPUT[Final Prediction]
```

**Output Structure:**
```json
{
  "symbol": "VCB",
  "current_price": 108000,
  "predicted_price": 112000,
  "change_percent": 3.7,
  "confidence": 85.5,
  "method_used": "LSTM Primary",
  "predictions": {
    "short_term": {"1_days": {...}, "7_days": {...}},
    "medium_term": {"30_days": {...}, "60_days": {...}},
    "long_term": {"90_days": {...}, "180_days": {...}}
  },
  "technical_indicators": {...},
  "trend_analysis": {...},
  "ai_advice": "...",
  "ai_reasoning": "..."
}
```

### Agent 2: InvestmentExpert

**Purpose:** Provide BUY/SELL/HOLD recommendations based on fundamental and technical analysis

**Key Features:**
- Financial metrics analysis (P/E, P/B, EPS, Dividend)
- Technical analysis (52-week range, volume, beta)
- Valuation analysis (Forward P/E, payout ratio)
- Risk-adjusted recommendations
- Profile-based position sizing
- AI-enhanced advice

**Analysis Components:**
```mermaid
graph TB
    INPUT[Symbol + Profile] -->|Fetch| METRICS[Financial Metrics]
    
    METRICS -->|Analyze| FIN[Financial Analysis<br/>P/E, P/B, EPS, Dividend<br/>Score: 0-100]
    METRICS -->|Analyze| TECH[Technical Analysis<br/>Price position, Volume, Beta<br/>Score: 0-100]
    METRICS -->|Analyze| VAL[Valuation Analysis<br/>Forward P/E, Sustainability<br/>Score: 0-100]
    
    FIN -->|40% weight| SCORE[Total Score]
    TECH -->|30% weight| SCORE
    VAL -->|30% weight| SCORE
    
    SCORE -->|Determine| REC[Recommendation<br/>STRONG BUY/BUY/WEAK BUY<br/>HOLD/WEAK SELL/SELL]
    
    REC -->|Adjust| PROFILE[Profile Adjustment<br/>Risk tolerance<br/>Time horizon<br/>Position sizing]
    
    PROFILE -->|Enhance| AI[AI Enhancement<br/>Contextual advice<br/>Reasoning]
    
    AI -->|Output| FINAL[Final Recommendation]
```

**Scoring Logic:**
```
Total Score = Financial(40%) + Technical(30%) + Valuation(30%)

Recommendation:
- 80-100: STRONG BUY (Confidence: 90%)
- 70-79:  BUY (Confidence: 80%)
- 60-69:  WEAK BUY (Confidence: 60%)
- 50-59:  HOLD (Confidence: 50%)
- 40-49:  WEAK SELL (Confidence: 60%)
- 0-39:   SELL (Confidence: 80%)
```

### Agent 3: RiskExpert

**Purpose:** Assess investment risk using advanced metrics

**Key Features:**
- Value at Risk (VaR) 95%
- Volatility calculation
- Beta (market correlation)
- Sharpe ratio
- Maximum drawdown
- Risk-adjusted position sizing
- AI-enhanced risk advice

**Risk Metrics:**
```mermaid
graph TB
    INPUT[Symbol + Profile] -->|Fetch| HIST[Historical Data]
    
    HIST -->|Calculate| VAR[VaR 95%<br/>Potential loss]
    HIST -->|Calculate| VOL[Volatility<br/>Price variation]
    HIST -->|Calculate| BETA[Beta<br/>Market correlation]
    HIST -->|Calculate| SHARPE[Sharpe Ratio<br/>Risk-adjusted return]
    HIST -->|Calculate| DD[Max Drawdown<br/>Peak-to-trough]
    
    VAR -->|Aggregate| RISK[Risk Level<br/>LOW/MEDIUM/HIGH]
    VOL -->|Aggregate| RISK
    BETA -->|Aggregate| RISK
    SHARPE -->|Aggregate| RISK
    DD -->|Aggregate| RISK
    
    RISK -->|Adjust| PROFILE[Profile Matching<br/>Risk tolerance<br/>Position limits<br/>Stop-loss levels]
    
    PROFILE -->|Enhance| AI[AI Risk Advice<br/>Scenario analysis<br/>Recommendations]
    
    AI -->|Output| FINAL[Risk Assessment]
```

**Risk Classification:**
```
Volatility-based:
- < 15%: Low Risk
- 15-25%: Medium Risk
- 25-40%: High Risk
- > 40%: Very High Risk

Position Sizing:
- Conservative (risk ≤ 30): Max 5% position
- Balanced (risk 31-70): Max 10% position
- Aggressive (risk > 70): Max 20% position
```

### Agent 4: TickerNews

**Purpose:** Crawl and analyze stock-specific news

**Key Features:**
- Multi-source crawling (CafeF, VietStock, VCI)
- CrewAI integration for real-time news
- Sentiment analysis
- Priority detection
- AI-enhanced summaries

**News Pipeline:**
```mermaid
graph LR
    INPUT[Symbol] -->|Search| CREW[CrewAI Search]
    INPUT -->|Crawl| CAFEF[CafeF]
    INPUT -->|Crawl| VIET[VietStock]
    INPUT -->|Crawl| VCI[VCI]
    
    CREW -->|Aggregate| NEWS[News Articles]
    CAFEF -->|Aggregate| NEWS
    VIET -->|Aggregate| NEWS
    VCI -->|Aggregate| NEWS
    
    NEWS -->|Analyze| SENT[Sentiment Analysis<br/>Positive/Negative/Neutral]
    NEWS -->|Detect| PRIOR[Priority Detection<br/>Symbol in title]
    
    SENT -->|Enhance| AI[AI Summary<br/>Impact analysis]
    PRIOR -->|Enhance| AI
    
    AI -->|Output| FINAL[Ticker News]
```

### Agent 5: MarketNews

**Purpose:** Provide market-wide news with risk-based filtering

**Key Features:**
- Risk-based news filtering
- Underground news (F319, F247, Facebook groups)
- Official news (CafeF, VnEconomy, DanTri)
- Sentiment analysis
- AI-enhanced market insights

**News Categories:**
```mermaid
graph TB
    INPUT[Risk Profile] -->|Route| FILTER{Risk Filter}
    
    FILTER -->|Conservative<br/>Risk ≤ 30| OFF[Official News Only<br/>CafeF, VnEconomy, DanTri]
    FILTER -->|Balanced<br/>Risk 31-70| MIX[Mixed News<br/>Official + Selected Underground]
    FILTER -->|Aggressive<br/>Risk > 70| ALL[All News<br/>Official + Underground]
    
    OFF -->|Fetch| NEWS1[News Articles]
    MIX -->|Fetch| NEWS2[News Articles]
    ALL -->|Fetch| NEWS3[News Articles]
    
    NEWS1 -->|Analyze| SENT[Sentiment Analysis]
    NEWS2 -->|Analyze| SENT
    NEWS3 -->|Analyze| SENT
    
    SENT -->|Enhance| AI[AI Market Insights<br/>Trend analysis<br/>Impact assessment]
    
    AI -->|Output| FINAL[Market News]
```

### Agent 6: StockInfo

**Purpose:** Display comprehensive company information

**Key Features:**
- Real-time stock data
- Financial ratios (P/E, P/B, EPS, ROE, ROA)
- Company overview
- Interactive charts
- Professional presentation

**Data Structure:**
```mermaid
graph TB
    INPUT[Symbol] -->|Fetch| VN[VNStock API]
    
    VN -->|Get| BASIC[Basic Data<br/>Price, Volume, Change]
    VN -->|Get| RATIOS[Financial Ratios<br/>P/E, P/B, EPS, Dividend]
    VN -->|Get| HIST[Price History<br/>30-365 days]
    VN -->|Get| COMPANY[Company Info<br/>Sector, Exchange, Market Cap]
    
    BASIC -->|Display| HEADER[Stock Header<br/>Current price, Change]
    RATIOS -->|Display| METRICS[Detailed Metrics<br/>Financial ratios]
    HIST -->|Display| CHART[Price Chart<br/>Interactive Plotly]
    COMPANY -->|Display| INFO[Company Overview]
    
    HEADER -->|Combine| OUTPUT[Stock Info Display]
    METRICS -->|Combine| OUTPUT
    CHART -->|Combine| OUTPUT
    INFO -->|Combine| OUTPUT
```

