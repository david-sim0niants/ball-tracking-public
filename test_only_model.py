from ball_tracking.utils import draw_bboxes
from ball_tracking import *

np.set_printoptions(suppress=True)
img = cv2.imread('examples/balls.jpg')

yolo_model = YoloModel()
bboxes, (labels, label_indices), (probabilities, confidences) = yolo_model.predict(img)
draw_bboxes(img, bboxes, labels, label_indices, probabilities)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()