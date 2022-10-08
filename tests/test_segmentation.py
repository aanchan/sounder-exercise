import pytest
import json
from analysis.transcription import Transcription


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


def test_segments(trans):
    seg_list = trans.segment()
    assert len(seg_list) == len(trans.tokenized_sent)


def test_collect_segments(trans):
    seg_list = trans.segment()
    collected_segments = trans.collect_segments(seg_list)
    n_ones = seg_list.count(1)
    n_segments = len(collected_segments)
    assert n_ones + 1 == n_segments
