from ball_tracking import *
from ball_tracking.utils import *


np.set_printoptions(suppress=True)

img = cv2.imread('examples/balls.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

ball_tracker = BallTracker(lowest_probability=0.7)
ball_tracker.yolo_model.stretch_input_image = False
predictions = ball_tracker.detect_ball(img)
if predictions is not None:
    (centers, radiuses), (probabilities, confidences) = predictions
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    draw_circles(img, centers, radiuses, probabilities)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()