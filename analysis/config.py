from .constants import TopicSegmentationAlgorithm


class TopicSegmentationConfig:
    def __init__(self):
        self.SENTENCE_COMPARISON_WINDOW = 15
        self.SMOOTHING_PASSES = 2
        self.SMOOTHING_WINDOW = 1
        self.TOPIC_CHANGE_THRESHOLD = 0.1
