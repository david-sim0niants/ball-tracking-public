import tensorflow as tf
import numpy as np
import cv2
from .core import *
import os


INPUT_SHAPE = (416, 416)
ANCHORS_MATRIX = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
CONFIDENCE_THRESH = 0.002
NUM_BOXES_PER_CELL = 3
NMS_THRESH = 0.8


def _sigmoid(X):
    return 1. / (1. + np.exp(-X))

def decode_output(output, anchors, confidence_thresh, input_width, input_height):
    grid_height, grid_width = output.shape[:2]
    output = output.reshape((grid_height, grid_width, NUM_BOXES_PER_CELL, -1))
    output[..., :2] = _sigmoid(output[..., :2])
    output[..., 4:] = _sigmoid(output[..., 4:])
    output[..., 5:] = output[..., 4, None] * output[..., 5:]
    output[..., 5:] *= output[..., 5:] >= confidence_thresh
    anchors = np.array(anchors)
    
    X = (output[..., 0] + np.repeat(np.tile(np.arange(0, grid_width), (grid_height, 1))[..., None], NUM_BOXES_PER_CELL, axis=-1)) / grid_width
    Y = (output[..., 1] + np.repeat(np.tile(np.arange(0, grid_height)[..., None], (grid_width, ))[..., None], NUM_BOXES_PER_CELL, axis=-1)) / grid_height
    W = anchors[np.arange(0, NUM_BOXES_PER_CELL * 2, 2)] * np.exp(output[..., 2]) / input_width
    H = anchors[np.arange(1, NUM_BOXES_PER_CELL * 2, 2)] * np.exp(output[..., 3]) / input_height
    
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
    
    return bboxes, classes, confidences

def ready_image(img):
    img = cv2.resize(img, INPUT_SHAPE)
    img = img.astype('float32') / 255.0
    return img[None]

def predict(img):
    model_predictions = model.predict(ready_image(img))
    predictions = []
    for prediction, anchors in zip(model_predictions, ANCHORS_MATRIX):
        bboxes, classes, confidences = decode_output(prediction[0], anchors, CONFIDENCE_THRESH, INPUT_SHAPE[0], INPUT_SHAPE[1])
        print(classes)
        non_maximum_suppression(bboxes, classes, NMS_THRESH)
        print(classes)
        predictions.append((bboxes, classes, confidences))
    return predictions
    
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'nn/yolov3-tiny-model.h5'))