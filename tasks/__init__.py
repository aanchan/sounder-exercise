import multiprocessing
from time import sleep
from .task import Task


fake_db = {}


def func(active_task: Task):
    name = multiprocessing.current_process().name
    print("starting of process named: ", name)
    fake_db[active_task.task_id] = active_task.data
    print(f'Key:{active_task.task_id}')
    print(f'fake_db_entry:{fake_db[active_task.task_id]}')
    sleep(2)
    print("exiting process")


def background_task(active_task: Task):
    procname = multiprocessing.Process(target=func,
                                       args=(active_task,),
                                       name=active_task.task_id)
    procname.daemon = True
    procname.start()
    procname.join()
