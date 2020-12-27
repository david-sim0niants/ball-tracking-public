from ball_tracking import *
from ball_tracking.utils import *


np.set_printoptions(suppress=True)

img = cv2.imread('examples/ball-0.jpg')
ball_tracker = BallTracker(lowest_probability=0.7)
predictions = ball_tracker.detect_ball(img)
if predictions is not None:
    (centers, radiuses), (probabilities, confidences) = predictions
    draw_circles(img, centers, radiuses, probabilities)
    cv2.imshow('result', img)
    cv2.waitKey(0)