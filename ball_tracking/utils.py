import cv2
import numpy as np


def draw_bboxes(img, bboxes, labels, label_indices, probabilities):
    if len(bboxes) == 0 or len(labels) == 0 or len(label_indices) == 0 or len(probabilities) == 0:
        return
    np.random.seed(42)
    colors = np.random.randint(0, 256, (np.max(label_indices) + 1, 3))
    for bbox, label, label_index, probability in zip(bboxes, labels, label_indices, probabilities):
        bbox[2:] += bbox[:2]
        bbox *= img.shape[1::-1] * 2
        bbox = bbox.astype('int32')
        color = colors[label_index]
        cv2.rectangle(img, tuple(bbox[:2].tolist()), tuple(bbox[2:].tolist()), tuple(color.tolist()), 2)
        cv2.putText(img, "%s: %f%%" % (label, probability * 100), tuple((bbox[:2] + [0, 10]).tolist()), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(color.tolist()), 1)


def draw_circles(img, centers, radiuses, probabilities=None, color=(0, 255, 0), thickness=1, draw_center_dot=True, center_dot_color=(0,0,255), center_dot_radius=5, probability_text_color=(255, 255, 255)):
    num_centers = len(centers)
    num_radiuses = len(radiuses)
    if num_centers == 0 or num_centers != num_radiuses:
        return
    centers *= img.shape[1::-1]
    radiuses *= np.mean(img.shape[1::-1])
    if draw_center_dot:
        def draw_circle(img, center, radius):
            cv2.circle(img, (int(center[0]), int(center[1])), int(radius), color, thickness)
            cv2.circle(img, (int(center[0]), int(center[1])), int(center_dot_radius), center_dot_color, -1)
    else:
        def draw_circle(img, center, radius):
            cv2.circle(img, (int(center[0]), int(center[1])), int(radius), color, thickness)
    if probabilities is not None:
        for i in range(num_centers):
            draw_circle(img, centers[i], radiuses[i])            
            cv2.putText(img, '%f%%' % np.round(probabilities[i] * 100, 2), tuple((centers[i].astype('int32') + [0, 10]).tolist()), cv2.FONT_HERSHEY_COMPLEX, 0.5, probability_text_color)
    else:
        for i in range(num_centers):
            draw_circle(img, centers[i], radiuses[i])            
