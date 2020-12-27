from .yolo import *


class BallTracker:

    yolo_model = YoloModel()
    target_label = 'sports ball'
    target_label_index = 32

    def __init__(self, target='sports ball', lowest_probability=0):
        self.lowest_probability = lowest_probability
        if isinstance(target, str) and target in self.yolo_model.labels: 
            self.target_label = target
        if isinstance(target, int):
            self.target_label_index = target

    def set_target(self, target):
        if isinstance(target, str) and target in self.yolo_model.labels: 
            self.target_label = target
        if isinstance(target, int):
            self.target_label_index = target

    def detect_ball(self, img, take_all=True):
        bboxes, (labels, label_indices), (probabilities, confidences) = self.yolo_model.predict(img)
        if len(bboxes) == 0 or len(label_indices) == 0 or len(probabilities) == 0 or len(confidences) == 0:
            return

        targets_mask = (label_indices == self.target_label_index) * (probabilities >= self.lowest_probability)
        probabilities = probabilities[targets_mask]
        bboxes = bboxes[targets_mask]
        confidences = confidences[targets_mask]
        
        if len(bboxes) == 0 or len(probabilities) == 0 or len(confidences) == 0:
            return
        if not take_all:
            highest_prob_index = np.argmax(probabilities)
            probabilities = probabilities[highest_prob_index]
            bboxes = bboxes[highest_prob_index]
            confidences = confidences[highest_prob_index]

        ball_radiuses = bboxes[..., 2:] / 2
        ball_centers = bboxes[..., :2] + ball_radiuses
        ball_radiuses = np.mean(ball_radiuses * img.shape[1::-1], axis=-1) / np.mean(img.shape[1::-1])

        return (ball_centers, ball_radiuses), (probabilities, confidences)
        
        