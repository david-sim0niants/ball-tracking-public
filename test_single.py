from ball_tracking import *

np.set_printoptions(suppress=True)
img = cv2.imread('examples/temp.png')
predictions = predict(img)
print(predictions)