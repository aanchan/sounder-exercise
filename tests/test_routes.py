from flask import Flask
from handlers.routes import configure_routes
import pytest


@pytest.fixture
def test_app():
    app = Flask(__name__)
    configure_routes(app)
    yield app


@pytest.fixture
def test_client(test_app):
    return test_app.test_client()


def test_base_route(test_client):
    url = '/'
    response = test_client.get(url)
    assert response.get_data() == b'Hello, World!'
    assert response.status_code == 200


def test_analyze_route(test_client):
    url = '/analyze'
    response = test_client.post(url)
    assert isinstance(response.get_json()["id"], str)

