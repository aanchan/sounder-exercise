from ..tasks.task import Task
from .transcription import Transcription
from .emotion import pick_interesting_segment


def process(active_task: Task, db):
    db[str(active_task.task_id)] = active_task.result
    transcription = Transcription(active_task.data)
    topic_change_list = transcription.segment()
    collected_segments = transcription.collect_segments(topic_change_list)
    most_interesting_segment = pick_interesting_segment(collected_segments)
    db[str(active_task.task_id)] = most_interesting_segment

