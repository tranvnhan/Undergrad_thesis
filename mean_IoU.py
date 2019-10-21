import numpy as np

def mean_iou(labels, predictions, n_classes):
    mean_iou = 0.0
    seen_classes = 0

    for c in range(n_classes):
        labels_c = (labels == c)
        pred_c = (predictions == c)

        labels_c_sum = (labels_c).sum()
        pred_c_sum = (pred_c).sum()

        if (labels_c_sum > 0) or (pred_c_sum > 0):
            seen_classes += 1

            intersect = np.logical_and(labels_c, pred_c).sum()
            union = labels_c_sum + pred_c_sum - intersect

            mean_iou += intersect / union

    return mean_iou / seen_classes if seen_classes else 0