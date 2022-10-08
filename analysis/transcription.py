from nltk.tokenize import sent_tokenize
from .segmentation import even_segmentation


def get_transcription_from_json(transcription_data):
    return transcription_data['results']['transcripts'][0]['transcript']


class Transcription:
    def __init__(self, transcription_data):
        self.transcription_text = get_transcription_from_json(transcription_data)
        self.tokenized_sent = sent_tokenize(self.transcription_text)

    def segment(self, segmentation_method=even_segmentation):
        segment_list = segmentation_method(self.tokenized_sent)
        return segment_list

    def collect_segments(self, segment_list):
        collected_segments = []
        curr_seg = []
        for sentence, seg_ind in zip(self.tokenized_sent, segment_list):
            if seg_ind == 0:
                curr_seg.append(sentence)
            elif seg_ind == 1:
                seg = " ".join(curr_seg)
                collected_segments.append(seg)
                curr_seg = [sentence]
        seg = "".join(curr_seg)
        collected_segments.append(seg)
        return collected_segments
