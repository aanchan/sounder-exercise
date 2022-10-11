import torch
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def depth_score(timeseries):
    """
    The depth score corresponds to how strongly the cues for a subtopic changed on both sides of a
    given token-sequence gap and is based on the distance from the peaks on both sides of the valleyto that valley.
    returns depth_scores
    """
    depth_scores = []
    for i in range(1, len(timeseries) - 1):
        left, right = i - 1, i + 1
        while left > 0 and timeseries[left - 1] > timeseries[left]:
            left -= 1
        while (
                right < (len(timeseries) - 1) and timeseries[right + 1] > timeseries[right]
        ):
            right += 1
        depth_scores.append(
            (timeseries[right] - timeseries[i]) + (timeseries[left] - timeseries[i])
        )
    return depth_scores


def smooth(timeseries, n, s):
    smoothed_timeseries = timeseries[:]
    for _ in range(n):
        for index in range(len(smoothed_timeseries)):
            neighbours = smoothed_timeseries[
                         max(0, index - s): min(len(timeseries) - 1, index + s)
                         ]
            smoothed_timeseries[index] = sum(neighbours) / len(neighbours)
    return smoothed_timeseries


def sentences_similarity(first_sentence_features, second_sentence_features) -> float:
    """
    Given two senteneces embedding features compute cosine similarity
    """
    similarity_metric = torch.nn.CosineSimilarity()
    return float(similarity_metric(first_sentence_features, second_sentence_features))


def compute_window(timeseries, start_index, end_index):
    """given start and end index of embedding, compute pooled window value
    [window_size, 768] -> [1, 768]
    """
    stack = torch.stack([features[0] for features in timeseries[start_index:end_index]])
    stack = stack.unsqueeze(
        0
    )  # https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
    stack_size = end_index - start_index
    pooling = torch.nn.MaxPool2d((stack_size - 1, 1))
    return pooling(stack)


def block_comparison_score(timeseries, k):
    """
    comparison score for a gap (i)
    cfr. docstring of block_comparison_score
    """
    res = []
    for i in range(k, len(timeseries) - k):
        first_window_features = compute_window(timeseries, i - k, i + 1)
        second_window_features = compute_window(timeseries, i + 1, i + k + 2)
        res.append(
            sentences_similarity(first_window_features[0], second_window_features[0])
        )

    return res


def arsort2(array1, array2):
    x = np.array(array1)
    y = np.array(array2)

    sorted_idx = x.argsort()[::-1]
    return x[sorted_idx], y[sorted_idx]


def get_local_maxima(array):
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def get_features_from_sentences(sentences):
    embeddings = model.encode(sentences, convert_to_numpy=False)
    # the unsqueeze below is needed because
    # the top level assumes torch vectors of dimensions n_feat, 1
    # here n_feat denotes the number of elements for a sentence
    # embedding for a chosen model
    embeddings = [torch.unsqueeze(vec, 0) for vec in embeddings]
    return embeddings


def depth_score_to_topic_change_indexes(
        depth_score_timeseries,
        topic_segmentation_config,
):
    threshold = topic_segmentation_config.TOPIC_CHANGE_THRESHOLD * max(depth_score_timeseries)

    if not depth_score_timeseries:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)

    if not local_maxima:
        return []

    local_maxima_np = np.array(local_maxima)
    local_maxima_indices_np = np.array(local_maxima_indices)
    filt_indices = np.nonzero(local_maxima_np > threshold)
    filt_indices_orig = local_maxima_indices_np[filt_indices]
    filt_vals = local_maxima_np[filt_indices]
    # adjust indices to reflect original sentence indices
    filt_indices = filt_indices_orig \
                   + topic_segmentation_config.SENTENCE_COMPARISON_WINDOW \
                   + topic_segmentation_config.SMOOTHING_WINDOW

    return filt_indices


def topic_segmentation(sentences, topic_segmentation_config):
    timeseries = get_features_from_sentences(sentences)

    block_comparison_score_timeseries = block_comparison_score(
        timeseries, k=topic_segmentation_config.SENTENCE_COMPARISON_WINDOW
    )

    block_comparison_score_timeseries = smooth(
        block_comparison_score_timeseries,
        n=topic_segmentation_config.SMOOTHING_PASSES,
        s=topic_segmentation_config.SMOOTHING_WINDOW,
    )

    depth_score_timeseries = depth_score(block_comparison_score_timeseries)

    segment_indices = depth_score_to_topic_change_indexes(
        depth_score_timeseries,
        topic_segmentation_config)
    segments_np = np.zeros(len(sentences))
    np.put(segments_np, segment_indices, 1)
    segments_np = segments_np.astype(int)
    return segments_np.tolist()
