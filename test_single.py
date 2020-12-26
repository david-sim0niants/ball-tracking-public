from ball_tracking.utils import draw_bboxes
from ball_tracking import *

np.set_printoptions(suppress=True)
img = cv2.imread('examples/ball-2.jpg')

bboxes, (labels, label_indices), (probabilities, confidences) = predict(img)
draw_bboxes(img, bboxes, labels, label_indices, probabilities)

cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()