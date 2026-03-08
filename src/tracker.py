import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.kalman_predictor import KalmanMotionPredictor
from src.trajectory_lstm import TrajectoryPredictor
from scipy.interpolate import splprep, splev


def smooth_trajectory(points, smoothing=3):

    if len(points) < 6:
        return points

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    try:
        tck, _ = splprep([x, y], s=smoothing)
        u_new = np.linspace(0, 1, len(points) * 3)
        out = splev(u_new, tck)

        smooth_points = list(zip(out[0], out[1]))
        smooth_points = [(int(px), int(py)) for px, py in smooth_points]

        return smooth_points

    except:
        return points


class ObjectTracker:

    def __init__(self):

        self.model = YOLO("models/yolov8n.pt")

        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7
        )

        self.trajectory_history = {}
        self.smoothed_positions = {}
        self.kalman_filters = {}

        self.max_history = 30
        self.alpha = 0.6

        self.lstm_predictor = TrajectoryPredictor()

    def smooth_position(self, track_id, cx, cy):

        if track_id not in self.smoothed_positions:
            self.smoothed_positions[track_id] = (cx, cy)
            return cx, cy

        prev_x, prev_y = self.smoothed_positions[track_id]

        smooth_x = int(self.alpha * cx + (1 - self.alpha) * prev_x)
        smooth_y = int(self.alpha * cy + (1 - self.alpha) * prev_y)

        self.smoothed_positions[track_id] = (smooth_x, smooth_y)

        return smooth_x, smooth_y

    def process_frame(self, frame):

        results = self.model(frame, conf=0.3, iou=0.4)[0]

        detections = []

        vehicle_classes = [2, 3, 5, 7]  

        for r in results.boxes.data.tolist():

            x1, y1, x2, y2, score, class_id = r

            if int(class_id) not in vehicle_classes:
                continue

            detections.append(([x1, y1, x2-x1, y2-y1], score, int(class_id)))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l, t, r, b = track.to_ltrb()

            cx = int((l + r) / 2)
            cy = int((t + b) / 2)

            cx, cy = self.smooth_position(track_id, cx, cy)

            if track_id not in self.trajectory_history:
                self.trajectory_history[track_id] = []

            self.trajectory_history[track_id].append((cx, cy))

            if len(self.trajectory_history[track_id]) > self.max_history:
                self.trajectory_history[track_id].pop(0)

            points = self.trajectory_history[track_id]

            # Smooth trajectory
            smooth_points = smooth_trajectory(points)

            for i in range(1, len(smooth_points)):

                pt1 = smooth_points[i - 1]
                pt2 = smooth_points[i]

                thickness = int(np.sqrt(self.max_history / float(i + 1)) * 2)

                cv2.line(frame, pt1, pt2, (255, 0, 0), thickness)

            # Kalman prediction
            if track_id not in self.kalman_filters:
                self.kalman_filters[track_id] = KalmanMotionPredictor()

            predictor = self.kalman_filters[track_id]

            if len(points) > 3:

                predictor.predict()

                px, py = predictor.update(cx, cy)

                cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), -1)

            # LSTM prediction
            prediction = self.lstm_predictor.predict(points)

            if prediction is not None:

                lx, ly = prediction

                cv2.circle(frame, (lx, ly), 6, (0, 255, 0), -1)

        return tracks
