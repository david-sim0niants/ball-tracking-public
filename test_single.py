from ball_tracking.utils import draw_bboxes
from ball_tracking import *

np.set_printoptions(suppress=True)
img = cv2.imread('examples/ball-0.jpg')
predictions = predict(img)

for pred in predictions:
    (bboxes, classes, confidences), (bboxes_indices, label_indices), (labels, probabilities) = pred
    print(bboxes, labels, probabilities, classes)
    draw_bboxes(img, bboxes, labels, label_indices, probabilities)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()