from tasks.task import Task
from time import sleep
from transcription import Transcription
fake_db = {}


def process(active_task: Task):
    fake_db[active_task.task_id] = active_task.data
    transcription = Transcription(active_task.data)
    topic_change_list = transcription.segment()
    print(f'Key:{active_task.task_id}')
    print(f'fake_db_entry:{fake_db[active_task.task_id]}')
    sleep(2)
    print("exiting process")