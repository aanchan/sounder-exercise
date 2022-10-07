from flask import request


def configure_routes(app):

    @app.route('/')
    def hello_world():
        return 'Hello, World!'
