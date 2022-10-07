import multiprocessing
from time import sleep


def func():
    name = multiprocessing.current_process().name
    print("starting of process named: ", name)
    sleep(2)
    print("exiting process")


def background_task(task_id):
    procname = multiprocessing.Process(target=func, name=task_id)
    procname.daemon = True
    procname.start()
    procname.join()
