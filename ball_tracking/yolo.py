from numpy.lib.function_base import meshgrid
import tensorflow as tf
import numpy as np
import cv2
from .core import *
import os
import PIL as pil



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
    stretch_input_image = False
    max_input_image_resizing = (1.05, 1.05)


    def __init__(self, **kwargs):    
        self.nms_thresh = kwargs.get('nms_thresh', self.nms_thresh)
        self.class_thresh = kwargs.get('class_thresh', self.class_thresh)
        self.confidence_thresh = kwargs.get('confidence_thresh', self.confidence_thresh)
        model_path = os.path.join(os.path.dirname(__file__), 'nn/yolov3-spp-model.h5')
        if os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, h5_path):
        self.model = tf.keras.models.load_model(h5_path)

    def predict(self, img):

        if not self.stretch_input_image:
            processed_images, (x_axis, y_axis) = preprocess_image(img, self.input_shape, 'divide', self.max_input_image_resizing)
            processed_images = processed_images.reshape(-1, *processed_images.shape[2:])
            grid_height, grid_width = len(y_axis), len(x_axis)
        else:
            processed_images = preprocess_image(img, self.input_shape, 'stretch')
        predictions = self.model.predict(processed_images)
        
        total_bboxes = []
        total_classes = []
        total_confidences = []

        for prediction, anchors in zip(predictions, self.anchors_matrix):
            if self.stretch_input_image:
                bboxes, classes, confidences = decode_output(prediction[0], anchors, self.confidence_thresh, \
                    self.input_shape[0], self.input_shape[1], self.num_boxes_per_cell)
                bboxes = [bboxes]
                classes = [classes]
                confidences = [confidences]
            else:
                prediction = prediction.reshape(grid_height, grid_width, *prediction.shape[1:])
                print(prediction.shape)
                bboxes = [None] * (grid_height * grid_width)
                classes = [None] * (grid_height * grid_width)
                confidences = [None] * (grid_height * grid_width)
                flat_idx = 0
                print(grid_height, grid_width)
                for i in range(grid_height):
                    for j in range(grid_width):
                        print(x_axis[j], y_axis[i])
                        bboxes[flat_idx], classes[flat_idx], confidences[flat_idx] = decode_output(prediction[i, j],\
                             anchors, self.confidence_thresh, self.input_shape[0], self.input_shape[1], self.num_boxes_per_cell)
                        bboxes[flat_idx] = [x_axis[j], y_axis[i]][::-1] + [0, 0] + bboxes[flat_idx] * ([self.input_shape[1] / img.shape[1], self.input_shape[0] / img.shape[0]] * 2)
                        
                        flat_idx += 1
            
            total_bboxes += bboxes
            total_classes += classes
            total_confidences += confidences
        
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
        confidences = confidences[bbox_indices]

        return bboxes, (labels, label_indices), (probabilities, confidences)


def preprocess_image(img, model_input_shape, method='stretch', max_resizing=(1.05, 1.05)):

    if method == 'stretch' or img.shape[1] / model_input_shape[1] <= max_resizing[0] and img.shape[0] / model_input_shape[0] <= max_resizing[1]:
        img = img.astype('float32') / 255.0
        imgs = tf.image.resize([img], model_input_shape, method=tf.image.ResizeMethod.BILINEAR, antialias=True).numpy()
        return imgs
        
    elif method == 'divide':
        half_input_shape = (model_input_shape[0] // 2, model_input_shape[1] // 2)
        grid_shape = [img.shape[0] // (half_input_shape[0]), img.shape[1] // (half_input_shape[1])]

        imgs = np.zeros(tuple(grid_shape) + model_input_shape + img.shape[-1:], dtype=img.dtype)
        min_i, max_i, min_j, max_j = 0, 0, 0, 0

        x_axis = np.arange(grid_shape[1]) / grid_shape[1]
        x_axis[-1] = 1 - model_input_shape[1] / img.shape[1]
        y_axis = np.arange(grid_shape[0]) / grid_shape[0]
        y_axis[-1] = 1 - model_input_shape[0] / img.shape[0]

        for i in range(grid_shape[0] - 1):
            max_i = min_i + model_input_shape[0]
            for j in range(grid_shape[1] - 1):
                max_j = min_j + model_input_shape[1]
                imgs[i, j] = img[min_i:max_i, min_j:max_j]
                min_j = max_j - half_input_shape[1]
            imgs[i, -1] = img[min_i:max_i, -model_input_shape[1]:]
            min_j = 0
            min_i = max_i - half_input_shape[0]
        for j in range(grid_shape[1] - 1):
            max_j = min_j + model_input_shape[1]
            imgs[-1, j] = img[-model_input_shape[0]:, min_j:max_j]
            min_j = max_j - half_input_shape[1]
        imgs[-1, -1] = img[-model_input_shape[0]:, -model_input_shape[1]:]
        
        return imgs.astype('float32') / 255, (x_axis, y_axis)

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
