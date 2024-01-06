import numpy as np

from dynamics import SatelliteDynamics


class EKF:
    """
    This class implements the Extended Kalman Filter (EKF).
    """

    def __init__(self, Q, R):
        """
        Initialize the FCEKF.

        Parameters:
        Q (np.array): Process noise covariance.
        R (np.array): Measurement noise covariance.
        x_priori (np.array): State a priori estimate vector.
        P_priori (np.array): State covariance a priori estimate.
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

    def H_jacobian_chief(self):
        """
        Computes the Jacobian of the measurement function.
        """
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        return H

    def apply(self, dt, x_est_old, P_est_old, y):
        """
        Applies the Extended Kalman Filter to the current state vector and state covariance estimate.

        Parameters:
        dt (float): Time step.
        x_est_old (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).
        P_est_old (np.array): The current state covariance estimate.
        y (np.array): The measurement vector of the satellite (range [km]).

        Returns:
        x_est_new (np.array): The new state vector of the satellite (position [km] and velocity [km / s]).
        P_est_new (np.array): The new state covariance estimate.
        """
        # Prediction Step
        x_pred, F = self.dynamic_model.x_new_and_F(dt, x_est_old)
        P_pred = F @ P_est_old @ F.T + self.Q

        # Kalman Gain
        H = self.H_jacobian_chief()
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + self.R)

        # Update Step
        y_est = self.h_function_chief(x_pred)
        x_est_new = x_pred + K @ (y - y_est)
        P_est_new = P_pred - K @ H @ P_pred

        return x_est_new, P_est_new


class CCEKF:
    """
    This class implements the Consider Covariance Extended Kalman Filter (FCEKF).
    """

    def __init__(self, Q, R):
        """
        Initialize the FCEKF.

        Parameters:
        Q (np.array): Process noise covariance.
        R (np.array): Measurement noise covariance.
        x_priori (np.array): State a priori estimate vector.
        P_priori (np.array): State covariance a priori estimate.
        SatelliteDynamics (class): Satellite dynamics model.
        """
        # Define noise covariances
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

        self.dynamic_model = SatelliteDynamics()

    def h_function_deputy(self, x_vec, c_vec):
        """
        Computes the measurement vector based on the current state vector.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        y (np.array): The measurement vector of the satellite (range [km]).
        """
        r_chief = c_vec[:3]
        r_deputy1 = x_vec[:3]
        r_deputy2 = x_vec[6:9]
        r_deputy3 = x_vec[12:15]

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

    def H_x(self, x_vec, c_vec):
        r_chief = c_vec[:3]
        r_deputy1 = x_vec[:3]
        r_deputy2 = x_vec[6:9]
        r_deputy3 = x_vec[12:15]

        range_deputy1_chief = np.linalg.norm(r_deputy1 - r_chief)
        range_deputy1_deputy2 = np.linalg.norm(r_deputy1 - r_deputy2)
        range_deputy1_deputy3 = np.linalg.norm(r_deputy1 - r_deputy3)
        range_deputy2_chief = np.linalg.norm(r_deputy2 - r_chief)
        range_deputy2_deputy3 = np.linalg.norm(r_deputy2 - r_deputy3)
        range_deputy3_chief = np.linalg.norm(r_deputy3 - r_chief)

        H = np.zeros((6, 18))
        H[0, :3] = (r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[1, :3] = (r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[1, 6:9] = -(r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[2, :3] = (r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[2, 12:15] = -(r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[3, 6:9] = (r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[4, 6:9] = (r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[4, 12:15] = -(r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[5, 12:15] = (r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        return H

    def H_c(self, x_vec, c_vec):
        r_chief = c_vec[:3]
        r_deputy1 = x_vec[:3]
        r_deputy2 = x_vec[6:9]
        r_deputy3 = x_vec[12:15]

        range_deputy1_chief = np.linalg.norm(r_deputy1 - r_chief)
        range_deputy2_chief = np.linalg.norm(r_deputy2 - r_chief)
        range_deputy3_chief = np.linalg.norm(r_deputy3 - r_chief)

        H = np.zeros((6, 6))
        H[0, 0:3] = -(r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[3, 0:3] = -(r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[5, 0:3] = -(r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        return H

    def apply(
        self, dt, x_est_old, P_xx_est_old, P_xc_est_old, y, c_est_new, P_cc_est_new
    ):
        # Prediction Step
        x_pred, F = self.dynamic_model.x_new_and_F(dt, x_est_old)
        P_xx_pred = F @ P_xx_est_old @ F.T + self.Q
        P_xc_pred = F @ P_xc_est_old
        P_cx_pred = P_xc_pred.T

        # Kalman Gain
        H_x = self.H_x(x_pred, c_est_new)
        H_c = self.H_c(x_pred, c_est_new)
        K = (P_xx_pred @ H_x.T + P_xc_pred @ H_c.T) @ np.linalg.inv(
            H_x @ P_xx_pred @ H_x.T
            + H_x @ P_xc_pred @ H_c.T
            + H_c @ P_cx_pred @ H_x.T
            + H_c @ P_cc_est_new @ H_c.T
            + self.R
        )

        # Update Step
        y_est = self.h_function_deputy(x_pred, c_est_new)
        x_est_new = x_pred + K @ (y - y_est)
        P_xx_est_new = P_xx_pred - K @ H_x @ P_xx_pred - K @ H_c @ P_cx_pred
        P_xc_est_new = P_xc_pred - K @ H_x @ P_xc_pred - K @ H_c @ P_cc_est_new

        return x_est_new, P_xx_est_new, P_xc_est_new
