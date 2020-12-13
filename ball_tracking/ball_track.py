import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model('nn/yolov3-tiny-model.h5')
img = cv2.imread('temp.png')
img = cv2.resize(img, (416, 416))

img = img.astype('float32') / 255.0
img = img[None]

predictions = model.predict(img)
print([prediction.shape for prediction in predictions])