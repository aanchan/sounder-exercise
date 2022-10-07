from flask import request, jsonify
import uuid


def configure_routes(app):

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    @app.post('/analyze')
    def analyze_endpoint():
        key = uuid.uuid4()
        print(f'Key:{key}')
        return {"id": key}

