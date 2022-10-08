import multiprocessing
from analysis.processing import process
from .task import Task


def background_task(active_task: Task):
    procname = multiprocessing.Process(target=process,
                                       args=(active_task,),
                                       name=active_task.task_id)
    procname.daemon = True
    procname.start()
    procname.join()
