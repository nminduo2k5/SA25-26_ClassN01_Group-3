from flask import Flask, request, jsonify
from presentation.controllers import api_bp
from business_logic.coordination_service import PricePredictor
from datetime import datetime

app = Flask(__name__)

# Register API Blueprint
app.register_blueprint(api_bp)

# Initialize predictor for direct routes
predictor = PricePredictor()

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Multi-Agent Stock Prediction System - Lab 3 Implementation",
        "description": "Layered Architecture with Presentation, Business Logic, and Persistence layers",
        "version": "1.0.0",
        "architecture": "3-Layer Architecture (Controller → Service → Repository)",
        "endpoints": {
            "predictions": "/api/predictions/<symbol>?horizon=<horizon>&arch=<architecture>",
            "market_overview": "/api/market/overview",
            "symbol_info": "/api/symbols/<symbol>/info",
            "architectures": "/api/architectures",
            "health": "/api/health"
        },
        "supported_symbols": ["VNM", "VCB", "HPG"],
        "supported_horizons": ["short-term", "medium-term", "long-term"],
        "supported_architectures": ["ensemble", "hierarchical", "round_robin", "agent_driven"],
        "timestamp": datetime.now().isoformat()
    })

# Legacy endpoint for backward compatibility
@app.route('/api/predictions/<symbol>', methods=['GET'])
def get_prediction(symbol):
    horizon = request.args.get('horizon', 'medium-term')
    arch = request.args.get('arch', 'ensemble')
    try:
        result = predictor.get_unified_prediction(symbol, horizon, arch)
        return jsonify(result.to_dict()), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint not found",
        "message": "The requested resource does not exist",
        "available_endpoints": [
            "/",
            "/api/predictions/<symbol>",
            "/api/market/overview", 
            "/api/symbols/<symbol>/info",
            "/api/architectures",
            "/api/health"
        ],
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Multi-Agent Stock Prediction System - Lab 3")
    print("📚 API Documentation: http://127.0.0.1:5000/")
    print("🔍 Health Check: http://127.0.0.1:5000/api/health")
    print("📈 Example: http://127.0.0.1:5000/api/predictions/VNM?horizon=medium-term&arch=ensemble")
    print("⚡ System ready!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)