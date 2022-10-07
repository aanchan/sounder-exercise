from nltk.tokenize import sent_tokenize
import itertools
from constants import MIN_SENT_THRESHOLD


def even_segmentation(transcription_text: str):
    tokenized_sent = sent_tokenize(transcription_text)
    n_sent = len(tokenized_sent)
    min_sent_thrshold = 3
    if n_sent == 0:
        raise ValueError("Empty list of sentences")
    elif n_sent < MIN_SENT_THRESHOLD:
        return list(itertools.repeat(0, n_sent))
    else:
        [1 if i % MIN_SENT_THRESHOLD == 1 else 0 for i in range(0, n_sent)]



