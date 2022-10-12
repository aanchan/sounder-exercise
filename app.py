from flask import Flask
from .handlers.routes import configure_routes


app = Flask(__name__)
db = {}
configure_routes(app, db)

if __name__ == '__main__':
    app.run()
