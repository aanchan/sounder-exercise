from flask import request
import uuid

fake_db = {}


def configure_routes(app):

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    @app.post('/analyze')
    def analyze_endpoint():
        key = uuid.uuid4()
        data = request.get_json()
        fake_db[key] = data
        print(f'Key:{key}')
        print(f'fake_db_entry:{fake_db[key]}')
        return {"id": key}

