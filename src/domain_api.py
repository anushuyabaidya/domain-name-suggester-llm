"""
Created By: Anushuya Baidya
Date: 8/1/25
"""
from flask import Flask, request, jsonify

from src.domain_trainer import DomainModel

class SimpleDomainAPI:
    def __init__(self):
        self.app = Flask(__name__)
        print("Loading your trained model...")
        model_options = ["../models/domain_model_v2", "../models/domain_model", "gpt2"]
        for model_path in model_options:
            try:
                self.model = DomainModel(model_path)
                print(f"Loaded: {model_path}")
                break
            except:
                print(f"Failed: {model_path}")
                continue
        self.setup_routes()

    def convert_to_api_format(self, model_result):
        """Convert your model output to API format"""
        if model_result["status"] != "success":
            return {
                "suggestions": [],
                "status": model_result["status"],
                "message": model_result["message"]
            }

        suggestions = []
        for i, domain in enumerate(model_result["domains"]):
            confidence = 0.90 - (i * 0.05)
            suggestions.append({
                "domain": domain,
                "confidence": round(confidence, 2)
            })

        return {
            "suggestions": suggestions,
            "status": "success"
        }

    def setup_routes(self):
        """Setup API endpoints"""
        @self.app.route('/', methods=['GET'])
        def home():
            return jsonify({
                "message": "Domain Generator API",
                "endpoints": {
                    "POST /generate": "Generate domain suggestions",
                    "GET /health": "Health check"
                },
                "example": {
                    "request": {"business_description": "organic coffee shop"},
                    "response": {
                        "suggestions": [{"domain": "organiccafe.com", "confidence": 0.90}],
                        "status": "success"
                    }
                }
            })

        @self.app.route('/generate', methods=['POST'])
        def generate():
            try:
                data = request.get_json()

                if not data or 'business_description' not in data:
                    return jsonify({
                        "suggestions": [],
                        "status": "error",
                        "message": "Missing business_description field"
                    }), 400

                business_description = data['business_description']
                model_result = self.model.generate_domains(business_description)
                api_result = self.convert_to_api_format(model_result)

                if api_result['status'] == 'success':
                    return jsonify(api_result), 200
                elif api_result['status'] == 'blocked':
                    return jsonify(api_result), 403
                else:
                    return jsonify(api_result), 400

            except Exception as e:
                return jsonify({
                    "suggestions": [],
                    "status": "error",
                    "message": f"Server error: {str(e)}"
                }), 500

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "model_loaded": True,
                "device": str(self.model.device)
            })

    def run(self, port=5002):
        """Start the API server"""
        print(f"Starting API on http://localhost:{port}")
        print("Test with:")
        print(
            f'curl -X POST http://localhost:{port}/generate -H "Content-Type: application/json" -d \'{{"business_description":"coffee shop"}}\'')

        self.app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    api = SimpleDomainAPI()
    api.run(port=5002)