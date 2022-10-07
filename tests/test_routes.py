from flask import Flask
from handlers.routes import configure_routes
import pytest
import json
import uuid

@pytest.fixture
def test_app():
    app = Flask(__name__)
    configure_routes(app)
    yield app


@pytest.fixture
def test_client(test_app):
    return test_app.test_client()


@pytest.fixture
def task_id():
    yield str(uuid.uuid4())

def test_base_route(test_client):
    url = '/'
    response = test_client.get(url)
    assert response.get_data() == b'Hello, World!'
    assert response.status_code == 200


@pytest.fixture
def input_data():
    file_name='input_text_data.json'
    with open(file_name) as f:
        data = json.load(f)
        yield data


def test_analyze_route(test_client, input_data):
    url = '/analyze'
    response = test_client.post(url, json=input_data)
    assert isinstance(response.get_json()["id"], str)
    assert response.status_code == 200


def test_highlight_route(test_client, task_id):
    url = f'/highlights/{task_id}'
    response = test_client.get(url)
    response_data = response.get_json()
    assert response_data["id"] == task_id
    assert isinstance(response_data["highlight"], str)


