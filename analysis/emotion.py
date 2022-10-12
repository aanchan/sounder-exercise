from textblob import TextBlob


def pick_interesting_segment(segments):
    segment_blobs = []
    for segment in segments:
        segment_blob = TextBlob(segment)
        segment_blobs.append(segment_blob)
    max_blob = max(segment_blobs, key=lambda k: abs(k.sentiment.subjectivity))
    max_index = segment_blobs.index(max_blob)
    return segments[max_index]
