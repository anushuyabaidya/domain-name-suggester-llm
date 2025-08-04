"""
Created By: Anushuya Baidya
Date: 8/1/25
"""
from flask import Flask, request, jsonify
from src.domain_trainer import DomainModel

class DomainAPI:
    def __init__(self):
        self.app = Flask(__name__)

        models = ["../models/domain_model_v2", "../models/domain_model", "gpt2"]
        for model_path in models:
            try:
                self.model = DomainModel(model_path)
                print(f"Loaded: {model_path}")
                break
            except:
                continue

        self.setup_routes()

    def convert_format(self, result):
        """Convert model result to API format"""
        if result["status"] != "success":
            return {"suggestions": [], "status": result["status"], "message": result["message"]}

        suggestions = [{"domain": domain, "confidence": round(0.90 - i*0.05, 2)}
                      for i, domain in enumerate(result["domains"])]

        return {"suggestions": suggestions, "status": "success"}

    def setup_routes(self):
        @self.app.route('/generate', methods=['POST'])
        def generate():
            data = request.get_json()
            if not data or 'business_description' not in data:
                return jsonify({"suggestions": [], "status": "error", "message": "Missing business_description"}), 400

            result = self.model.generate_domains(data['business_description'])
            api_result = self.convert_format(result)

            status_code = 200 if api_result['status'] == 'success' else 403 if api_result['status'] == 'blocked' else 400
            return jsonify(api_result), status_code

        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy"})

    def run(self, port=5002):
        print(f"API running on http://localhost:{port}")
        self.app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == '__main__':
    DomainAPI().run()