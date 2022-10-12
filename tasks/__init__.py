from ..analysis.processing import process
from .task import Task


def background_task(active_task: Task, db):
    # TODO: Make this truly a background task
    process(active_task, db)
