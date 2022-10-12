from flask import request
import uuid
from ..tasks import background_task
from ..tasks.task import Task


def configure_routes(app, db):

    @app.route('/')
    def hello_world():
        return 'Hello, World!'

    @app.post('/analyze')
    def analyze_endpoint():
        key = uuid.uuid4()
        data = request.get_json()
        task = Task(task_id=key, data=data)
        background_task(task, db)
        return {"id": key}

    @app.get('/highlights/<task_id>')
    def get_highlights(task_id):
        highlight_dict = {
            "id": task_id,
            "highlight": db[task_id]
        }
        return highlight_dict
