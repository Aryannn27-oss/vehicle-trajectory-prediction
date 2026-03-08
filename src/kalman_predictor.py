import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanMotionPredictor:
    

    def __init__(self, dt=1.0, min_history=5):

        self.dt = dt
        self.min_history = min_history
        self.initialized = False

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # state transition matrix
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # measurement function
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # initial state
        self.kf.x = np.zeros((4, 1))

        # covariance
        self.kf.P *= 300.

        # process noise
        q = 0.05
        self.kf.Q = np.eye(4) * q

        # measurement noise
        r = 3.0
        self.kf.R = np.eye(2) * r

    def initialize_velocity(self, trajectory):

        if len(trajectory) < 2:
            return 0.0, 0.0

        x1, y1 = trajectory[-2]
        x2, y2 = trajectory[-1]

        vx = x2 - x1
        vy = y2 - y1

        return vx, vy

    def predict(self):
        """
        Predict next state
        """
        self.kf.predict()

        px = float(self.kf.x[0])
        py = float(self.kf.x[1])

        return px, py

    def update(self, x, y, trajectory=None):
        """
        Update filter with measurement
        """

        if not self.initialized and trajectory is not None:

            if len(trajectory) >= self.min_history:

                vx, vy = self.initialize_velocity(trajectory)

                self.kf.x = np.array([
                    [x],
                    [y],
                    [vx],
                    [vy]
                ])

                self.initialized = True

        measurement = np.array([[x], [y]])

        self.kf.update(measurement)

        px = float(self.kf.x[0])
        py = float(self.kf.x[1])

        return px, py
