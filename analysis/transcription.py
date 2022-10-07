class Transcription:
    def __init__(self, transcription_text):
        self.transcription_text = transcription_text

    def segment(self, segmentation_method):
        segments = segmentation_method(self.transcription_text)
        return segments
