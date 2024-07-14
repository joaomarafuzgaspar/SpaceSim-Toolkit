import numpy as np

from scipy.optimize import minimize
from dynamics import SatelliteDynamics


class WLSTSQ:
    """
    This class implements the (constrained) Weighted Least Squares (WLSTSQ).
    """

    def __init__(self, Q, R):
        """
        Initialize the WLSTSQ class.

        Parameters:
        Q (np.array): Process noise covariance.
        R (np.array): Measurement noise covariance.
        SatelliteDynamics (class): Satellite dynamics model.
        """
        # Define noise covariances
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

        self.dynamic_model = SatelliteDynamics()

    def h_function_chief(self, x_vec):
        """
        Computes the measurement vector based on the current state vector.
        The measurement vector includes position components.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).
        """
        return x_vec[0:3]

    def h_function_deputy(self, x_vec):
        """
        Computes the measurement vector based on the current state vector.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        y (np.array): The measurement vector of the satellite (range [km]).
        """
        r_chief = x_vec[:3]
        r_deputy1 = x_vec[6:9]
        r_deputy2 = x_vec[12:15]
        r_deputy3 = x_vec[18:21]

        range_deputy1_chief = np.linalg.norm(r_deputy1 - r_chief)
        range_deputy1_deputy2 = np.linalg.norm(r_deputy1 - r_deputy2)
        range_deputy1_deputy3 = np.linalg.norm(r_deputy1 - r_deputy3)
        range_deputy2_chief = np.linalg.norm(r_deputy2 - r_chief)
        range_deputy2_deputy3 = np.linalg.norm(r_deputy2 - r_deputy3)
        range_deputy3_chief = np.linalg.norm(r_deputy3 - r_chief)

        return np.array(
            [
                [range_deputy1_chief],
                [range_deputy1_deputy2],
                [range_deputy1_deputy3],
                [range_deputy2_chief],
                [range_deputy2_deputy3],
                [range_deputy3_chief],
            ]
        )

    def J_function(self, X, dt, T, Y):
        X = X.reshape((24, 1, T))
        obj_fun = 0
        for t in range(T):
            V = Y[:, :, t] - np.concatenate(
                (
                    self.h_function_chief(X[:, :, t]),
                    self.h_function_deputy(X[:, :, t]),
                )
            )
            obj_fun += np.dot(V.T, np.dot(np.linalg.inv(self.R), V))
        for t in range(T - 1):
            W = X[:, :, t + 1] - self.dynamic_model.x_new(dt, X[:, :, t])
            obj_fun += np.dot(W.T, np.dot(np.linalg.inv(self.Q), W))
        return obj_fun

    def apply(self, dt, x_initial, X_true, Y):
        # Initializations
        T = Y.shape[2]
        X_est = np.zeros((24, 1, T))
        X_initial = np.zeros((24, 1, T))
        X_initial[:, :, 0] = x_initial

        # Optimization
        solution = minimize(
            fun=self.J_function,
            x0=X_true.flatten(),
            args=(dt, T, Y),
            method="SLSQP",
            options={"maxiter": 1000, "disp": True},
        )

        # Extract the solution
        if solution.success:
            X_est = solution.x.reshape((24, 1, T))

            print("Optimization successful")
        else:
            print("Optimization failed")

        return X_est
