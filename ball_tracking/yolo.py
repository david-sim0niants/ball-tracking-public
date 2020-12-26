import tensorflow as tf
import numpy as np
import cv2
from .core import *
import os



class YoloModel:

    input_shape = (416, 416)
    anchors_matrix = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    num_boxes_per_cell = 3
    nms_thresh = 0.5
    class_thresh = 0.7
    confidence_thresh = 0.04
    labels = np.array(["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
    

    def __init__(self, **kwargs):    
        self.nms_thresh = kwargs.get('nms_thresh', self.nms_thresh)
        self.class_thresh = kwargs.get('class_thresh', self.class_thresh)
        self.confidence_thresh = kwargs.get('confidence_thresh', self.confidence_thresh)
        model_path = os.path.join(os.path.dirname(__file__), 'nn/yolov3-spp-model.h5')
        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, h5_path):
        self.model = tf.keras.models.load_model(h5_path)

    def prepare_image(self, img):
        img = cv2.resize(img, self.input_shape)
        img = img.astype('float32') / 255.0
        return img[None]

    def predict(self, img):
        predictions = self.model.predict(self.prepare_image(img))
        
        total_bboxes = []
        total_classes = []
        total_confidences = []

        for prediction, anchors in zip(predictions, self.anchors_matrix):
            bboxes, classes, confidences = decode_output(prediction[0], anchors, self.confidence_thresh, \
                self.input_shape[0], self.input_shape[1], self.num_boxes_per_cell)
            
            total_bboxes.append(bboxes)
            total_classes.append(classes)
            total_confidences.append(confidences)
        
        bboxes = np.concatenate(total_bboxes, axis=0)
        classes = np.concatenate(total_classes, axis=0)
        confidences = np.concatenate(total_confidences, axis=0)

        classes_mask = (1 >= classes) * (classes > self.class_thresh)
        bbox_indices, label_indices = np.where(classes_mask)
        bboxes = bboxes[bbox_indices]
        classes = classes[np.unique(bbox_indices)]

        non_maximum_suppression(bboxes, classes, self.nms_thresh)

        classes_mask = (1 >= classes) * (classes > self.class_thresh)
        bbox_indices, label_indices = np.where(classes_mask)
        bboxes = bboxes[bbox_indices]
        labels = self.labels[label_indices]
        probabilities = classes[classes_mask]

        return bboxes, (labels, label_indices), (probabilities, confidences)


def _sigmoid(X):
    return 1. / (1. + np.exp(-X))

def decode_output(output, anchors, confidence_thresh, input_width, input_height, num_boxes_per_cell):
    grid_height, grid_width = output.shape[:2]
    output = output.reshape((grid_height, grid_width, num_boxes_per_cell, -1))
    output[..., :2] = _sigmoid(output[..., :2])
    output[..., 4:] = _sigmoid(output[..., 4:])
    # output[..., 5:] = output[..., 4, None] * output[..., 5:]
    output[..., 5:] *= output[..., 5:] >= confidence_thresh

    anchors = np.array(anchors)
    X = (output[..., 0] + np.repeat(np.tile(np.arange(0, grid_width), (grid_height, 1))[..., None], num_boxes_per_cell, axis=-1)) / grid_width
    Y = (output[..., 1] + np.repeat(np.tile(np.arange(0, grid_height)[..., None], (grid_width, ))[..., None], num_boxes_per_cell, axis=-1)) / grid_height
    W = anchors[np.arange(0, num_boxes_per_cell * 2, 2)] * np.exp(output[..., 2]) / input_width
    H = anchors[np.arange(1, num_boxes_per_cell * 2, 2)] * np.exp(output[..., 3]) / input_height
    
    X = X.flatten()
    Y = Y.flatten()
    W = W.flatten()
    H = H.flatten()
    
    X -= W / 2
    Y -= H / 2
    
    confidences = output[..., 4].flatten()
    confident_mask = confidences >= confidence_thresh
    confidences = confidences[confident_mask]
    bboxes = np.concatenate([X[confident_mask, ..., None], Y[confident_mask, ..., None], W[confident_mask, ..., None], H[confident_mask, ..., None]], axis=-1)
    classes = output[..., 5:].reshape(-1, output.shape[-1] - 5)[confident_mask]
    
    return bboxes.astype('float64'), classes.astype('float64'), confidences.astype('float64')
