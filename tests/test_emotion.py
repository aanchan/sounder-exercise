import pytest
from analysis.transcription import Transcription
from analysis.emotion import pick_interesting_segment
import json


@pytest.fixture
def input_data():
    file_name = 'this-american-life_getting-out.json'
    with open(file_name) as f:
        data = json.load(f)
        yield data


@pytest.fixture
def trans(input_data):
    transcription = Transcription(input_data)
    yield transcription


@pytest.fixture
def segments(trans):
    seg_list = trans.segment()
    collected_segments = trans.collect_segments(seg_list)
    yield collected_segments


def test_detect_sentiment(segments):
    max_emotional_segment = pick_interesting_segment(segments)
    assert max_emotional_segment in segments
