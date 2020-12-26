import cv2
import numpy as np


def draw_bboxes(image, bboxes, labels, label_indices, probabilities):
    if len(bboxes) == 0 or len(labels) == 0 or len(label_indices) == 0 or len(probabilities) == 0:
        return
    np.random.seed(42)
    colors = np.random.randint(0, 256, (np.max(label_indices) + 1, 3))
    for bbox, label, label_index, probability in zip(bboxes, labels, label_indices, probabilities):
        bbox[2:] += bbox[:2]
        bbox *= image.shape[1::-1] * 2
        bbox = bbox.astype('int32')
        color = colors[label_index]
        cv2.rectangle(image, tuple(bbox[:2].tolist()), tuple(bbox[2:].tolist()), tuple(color.tolist()), 2)
        cv2.putText(image, "%s: %f%%" % (label, probability * 100), tuple((bbox[:2] + [0, 10]).tolist()), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(color.tolist()), 1)