from flask import Blueprint, request, jsonify
from business_logic.coordination_service import PricePredictor
from datetime import datetime

# Create Blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize the service
predictor = PricePredictor()

@api_bp.route('/predictions/<symbol>', methods=['GET'])
def get_prediction(symbol):
    """Get price prediction for a symbol"""
    horizon = request.args.get('horizon', 'medium-term')
    arch = request.args.get('arch', 'ensemble')
    
    try:
        result = predictor.get_unified_prediction(symbol, horizon, arch)
        return jsonify({
            "status": "success",
            "data": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "architecture": arch
        }), 200
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/market/overview', methods=['GET'])
def get_market_overview():
    """Get market overview"""
    try:
        overview = predictor.get_market_overview()
        return jsonify({
            "status": "success",
            "data": overview,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/symbols/<symbol>/info', methods=['GET'])
def get_symbol_info(symbol):
    """Get detailed information for a symbol"""
    try:
        info = predictor.get_symbol_info(symbol.upper())
        return jsonify({
            "status": "success",
            "data": info,
            "timestamp": datetime.now().isoformat()
        }), 200
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/architectures', methods=['GET'])
def get_available_architectures():
    """Get list of available coordination architectures"""
    architectures = {
        "ensemble": {
            "name": "Ensemble Architecture",
            "description": "Parallel voting system where all agents contribute equally",
            "use_case": "Balanced predictions with multiple perspectives"
        },
        "hierarchical": {
            "name": "Hierarchical Architecture", 
            "description": "Master-slave system with LSTM as master agent",
            "use_case": "When you want AI/ML to have final say"
        },
        "round_robin": {
            "name": "Round Robin Architecture",
            "description": "Sequential processing where each agent refines the previous result",
            "use_case": "Iterative refinement of predictions"
        },
        "agent_driven": {
            "name": "Agent-Driven Architecture",
            "description": "Agents autonomously decide their participation based on market conditions",
            "use_case": "Adaptive system that responds to market volatility"
        }
    }
    
    return jsonify({
        "status": "success",
        "data": {
            "architectures": architectures,
            "default": "ensemble",
            "total_count": len(architectures)
        },
        "timestamp": datetime.now().isoformat()
    }), 200

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Multi-Agent Stock Prediction System",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "layers": {
            "presentation": "active",
            "business_logic": "active", 
            "persistence": "active"
        }
    }), 200