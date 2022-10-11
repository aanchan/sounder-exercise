import torch
import numpy as np
from .constants import (
    TopicSegmentationAlgorithm,
    TopicSegmentationConfig,
    TextTilingHyperparameters,
)


# pretrained roberta model
roberta_model = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta_model.eval()

PARALLEL_INFERENCE_INSTANCES = 20
SENTENCE_COMPARISON_WINDOW = 15
SMOOTHING_PASSES = 2
SMOOTHING_WINDOW = 1
TOPIC_CHANGE_THRESHOLD = 0.6



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


def get_features_from_sentence(batch_sentences, layer=-2):
    """
    extracts the BERT semantic representation
    from a sentence, using an averaged value of
    the `layer`-th layer
    returns a 1-dimensional tensor of size 758
    """
    batch_features = []
    for sentence in batch_sentences:
        tokens = roberta_model.encode(sentence)
        all_layers = roberta_model.extract_features(tokens, return_all_hiddens=True)
        pooling = torch.nn.AvgPool2d((len(tokens), 1))
        sentence_features = pooling(all_layers[layer])
        batch_features.append(sentence_features[0])
    return batch_features


def depth_score_to_topic_change_indexes(
        depth_score_timeseries,
        topic_segmentation_configs=TopicSegmentationConfig,
):

    #TOPIC_CHANGE_THRESHOLD * max(
        #depth_score_timeseries
    #)

    if not depth_score_timeseries:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_score_timeseries)
    np.savetxt('local_maxima_indices.csv', local_maxima_indices, delimiter=',')
    np.savetxt('local_maxima.csv', local_maxima, delimiter=',')
    if local_maxima == []:
        return []



    threshold = np.mean(local_maxima)
    filtered_local_maxima_indices = []
    filtered_local_maxima = []

    for i, m in enumerate(local_maxima):
        if m > threshold:
            filtered_local_maxima.append(m)
            filtered_local_maxima_indices.append(i)

    local_maxima = filtered_local_maxima
    local_maxima_indices = filtered_local_maxima_indices

    return local_maxima_indices


def flatten_features(batches_features):
    res = []
    for batch_features in batches_features:
        res += batch_features
    return res


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (
        a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(min(len(a), n))
    )


def topic_segmentation(
        topic_segmentation_algorithm: TopicSegmentationAlgorithm,
        sentences,
        topic_segmentation_config):
    if topic_segmentation_algorithm == TopicSegmentationAlgorithm.BERT:
        return topic_segmentation_bert(
            sentences,
            topic_segmentation_config)
    elif topic_segmentation_algorithm == TopicSegmentationAlgorithm.EVEN:
        raise NotImplementedError
    else:
        raise NotImplementedError


def topic_segmentation_bert(sentences,
                            topic_segmentation_configs=TopicSegmentationConfig):

    textiling_hyperparameters = TextTilingHyperparameters

    # parallel inference
    batches_features = []
    for batch_sentences in split_list(
            sentences, PARALLEL_INFERENCE_INSTANCES
    ):
        batches_features.append(get_features_from_sentence(batch_sentences))
    timeseries = flatten_features(batches_features)

    block_comparison_score_timeseries = block_comparison_score(
        timeseries, k=SENTENCE_COMPARISON_WINDOW
    )

    block_comparison_score_timeseries = smooth(
        block_comparison_score_timeseries,
        n=SMOOTHING_PASSES,
        s=SMOOTHING_WINDOW,
    )

    depth_score_timeseries = depth_score(block_comparison_score_timeseries)
    depth_scores = np.array(depth_score_timeseries)
    np.savetxt("depth_scores.csv", depth_scores, delimiter=",")
    segments = depth_score_to_topic_change_indexes(
        depth_score_timeseries,
        topic_segmentation_configs=topic_segmentation_configs)

    return segments
