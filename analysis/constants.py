from enum import Enum

MIN_SENT_THRESHOLD = 4  # used for even segmentation


class TopicSegmentationAlgorithm(Enum):
    EVEN = 0
    EMBED = 1
