# Lab 3: Multi-Agent Stock Prediction System - Layered Architecture

This project implements a **Layered Architecture** for the Multi-Agent Stock Price Prediction System, focusing on the "Generate Unified Price Prediction" feature (FR-03).

## Architecture Overview

The system follows a strict 3-layer architecture:

```
┌─────────────────────────────────────┐
│        Presentation Layer           │  ← HTTP Requests/Responses
│     (controllers.py, app.py)        │
├─────────────────────────────────────┤
│        Business Logic Layer         │  ← Core Business Rules
│  (coordination_service.py, models.py) │
├─────────────────────────────────────┤
│        Persistence Layer            │  ← Data Access
│   (market_data_repository.py)       │
└─────────────────────────────────────┘
```

**Dependency Flow**: Controller → Service → Repository (Unidirectional)

## Features

- ✅ **4 Coordination Architectures**: Ensemble, Hierarchical, Round Robin, Agent-Driven
- ✅ **Multi-Agent System**: Technical, Fundamental, Sentiment, LSTM agents
- ✅ **RESTful API**: Professional API with proper error handling
- ✅ **In-Memory Data**: Simulated market data for VNM, VCB, HPG stocks
- ✅ **Strict Layer Separation**: Clean architectural boundaries

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Server starts at: `http://127.0.0.1:5000`

### 3. Test the API

#### Get API Documentation
```bash
curl http://127.0.0.1:5000/
```

#### Health Check
```bash
curl http://127.0.0.1:5000/api/health
```

#### Price Prediction (Ensemble Architecture)
```bash
curl "http://127.0.0.1:5000/api/predictions/VNM?horizon=medium-term&arch=ensemble"
```

#### Price Prediction (Hierarchical Architecture)
```bash
curl "http://127.0.0.1:5000/api/predictions/VCB?horizon=short-term&arch=hierarchical"
```

#### Market Overview
```bash
curl http://127.0.0.1:5000/api/market/overview
```

#### Available Architectures
```bash
curl http://127.0.0.1:5000/api/architectures
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/api/health` | GET | Health check |
| `/api/predictions/<symbol>` | GET | Get price prediction |
| `/api/market/overview` | GET | Market overview |
| `/api/symbols/<symbol>/info` | GET | Symbol information |
| `/api/architectures` | GET | Available architectures |

## Supported Parameters

- **Symbols**: `VNM`, `VCB`, `HPG`
- **Horizons**: `short-term`, `medium-term`, `long-term`
- **Architectures**: `ensemble`, `hierarchical`, `round_robin`, `agent_driven`

## Example Response

```json
{
  "status": "success",
  "data": {
    "symbol": "VNM",
    "horizon": "medium-term",
    "predictions": {
      "14d": 108.5,
      "30d": 112.3,
      "60d": 118.7
    }
  },
  "timestamp": "2025-01-27T10:30:00",
  "architecture": "ensemble"
}
```

## Architecture Validation

### ASR-1: Flexibility via Configuration ✅
- Multiple coordination architectures selectable at runtime
- Easy to add new architectures without code changes

### ASR-2: Real-time Orchestration ✅
- Agents coordinate in real-time based on selected architecture
- Agent-driven architecture allows dynamic participation

### ASR-3: Modifiability ✅
- Strict layer separation enables independent modifications
- Clear dependency flow: Controller → Service → Repository

## Project Structure

```
multi_agent_stock_prediction/
├── app.py                              # Main Flask application
├── business_logic/
│   ├── __init__.py
│   ├── models.py                       # Data models
│   └── coordination_service.py         # Business logic service
├── persistence/
│   ├── __init__.py
│   └── market_data_repository.py       # Data access layer
├── presentation/
│   ├── __init__.py
│   └── controllers.py                  # API controllers
├── requirements.txt                    # Dependencies
└── README.md                          # This file
```

## Coordination Architectures

### 1. Ensemble Architecture (Default)
- **Pattern**: Parallel voting
- **Description**: All agents contribute equally
- **Use Case**: Balanced predictions with multiple perspectives

### 2. Hierarchical Architecture
- **Pattern**: Master-slave
- **Description**: LSTM acts as master agent
- **Use Case**: When AI/ML should have final say

### 3. Round Robin Architecture
- **Pattern**: Sequential processing
- **Description**: Each agent refines previous results
- **Use Case**: Iterative refinement of predictions

### 4. Agent-Driven Architecture
- **Pattern**: Autonomous participation
- **Description**: Agents decide participation based on market conditions
- **Use Case**: Adaptive system responding to volatility

## Development Notes

### Adding New Agents
1. Modify `_dummy_agent_predict()` in `coordination_service.py`
2. Add agent to the `agents` list in `_orchestrate_agents()`
3. Implement agent-specific logic

### Adding New Architectures
1. Add new case in `_orchestrate_agents()` method
2. Update `/api/architectures` endpoint
3. Add architecture documentation

### Production Considerations
- Replace in-memory storage with real database
- Add authentication and rate limiting
- Integrate real market data APIs
- Add comprehensive logging and monitoring

## License

MIT License - Educational purposes for Software Architecture Lab 3.