import numpy as np

from dynamics import SatelliteDynamics
from scipy.linalg import fractional_matrix_power


class WLSTSQ_LM:
    """
    This class implements the Levenberg-Marquardt algorithm for the Weighted Least Squares (WLSTSQ) estimator.
    """

    def __init__(self, Q, R):
        """
        Initialize the WLSTSQ class.

        Parameters:
        Q (np.array): Process noise covariance matrix.
        R (np.array): Measurement noise covariance matrix.
        SatelliteDynamics (class): Satellite dynamics model instance.
        """
        # Define noise covariances
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

        self.dynamic_model = SatelliteDynamics()

    def h_function_chief(self, x_vec):
        """
        Compute the measurement vector for the chief satellite's position.

        Parameters:
        x_vec (np.array): Current state vector of the satellite (position [km] and velocity [km/s]).

        Returns:
        np.array: Position components of the state vector.
        """
        return x_vec[0:3]

    def H_jacobian_chief(self):
        """
        Compute the Jacobian of the measurement function for the chief satellite.

        Returns:
        np.array: Jacobian matrix of the measurement function.
        """
        H = np.zeros((3, 24))
        H[0:3, 0:3] = np.eye(3)
        return H

    def h_function_deputy(self, x_vec):
        """
        Compute the measurement vector for relative distances between satellites.

        Parameters:
        x_vec (np.array): Current state vector of the satellites.

        Returns:
        np.array: Measurement vector containing distances between satellites.
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

    def H_jacobian_deputy(self, x_vec):
        """
        Compute the Jacobian of the measurement function for relative distances between satellites.

        Parameters:
        x_vec (np.array): Current state vector of the satellites.

        Returns:
        np.array: Jacobian matrix for the relative distances.
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

        H = np.zeros((6, 24))
        H[0, 0:3] = -(r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[0, 6:9] = (r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[1, 6:9] = (r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[1, 12:15] = -(r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[2, 6:9] = (r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[2, 18:21] = -(r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[3, 0:3] = -(r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[3, 12:15] = (r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[4, 12:15] = (r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[4, 18:21] = -(r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[5, 0:3] = -(r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        H[5, 18:21] = (r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        return H

    def h(self, x_vec):
        """
        Combine the measurement functions for both the chief and deputies.

        Parameters:
        x_vec (np.array): State vector of the satellites.

        Returns:
        np.array: Combined measurement vector.
        """
        return np.concatenate(
            [self.h_function_chief(x_vec), self.h_function_deputy(x_vec)]
        )

    def H(self, x_vec):
        """
        Combine the Jacobians for both the chief and deputies.

        Parameters:
        x_vec (np.array): State vector of the satellites.

        Returns:
        np.array: Combined Jacobian matrix.
        """
        return np.concatenate((self.H_jacobian_chief(), self.H_jacobian_deputy(x_vec)))

    def r_function(self, X, Y):
        """
        Compute the residual vector for the state and measurements.

        Parameters:
        X (np.array): State vector across all time steps.
        Y (np.array): Measurement data.

        Returns:
        np.array: Residual vector.
        """
        K = Y.shape[2]
        n_x = 6
        n_y_1 = 3
        n_y_2 = 2
        n_y_3 = 2
        n_y_4 = 2
        r = np.zeros(((4 * n_x + n_y_1 + n_y_2 + n_y_3 + n_y_4) * K, 1))
        # Process noise
        for k in range(K - 1):
            X_k = X[4 * n_x * k : 4 * n_x * (k + 1)]
            X_k_1 = X[4 * n_x * (k + 1) : 4 * n_x * (k + 2)]
            r[
                k * 4 * n_x
                + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4) : (k + 1) * 4 * n_x
                + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4),
                :,
            ] = fractional_matrix_power(self.Q, -1 / 2) @ (
                X_k_1 - self.dynamic_model.x_new(self.dt, X_k)
            )
        # Observation noise
        for k in range(K):
            X_k = X[4 * n_x * k : 4 * n_x * (k + 1)]
            Y_k = Y[:, :, k]
            r[
                (k + 1) * 4 * n_x
                + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4) : (k + 1) * 4 * n_x
                + (k + 1) * (n_y_1 + n_y_2 + n_y_3 + n_y_4),
                :,
            ] = fractional_matrix_power(self.R, -1 / 2) @ (Y_k - self.h(X_k))
        return r

    def J_jacobian(self, X, Y):
        """
        Compute the Jacobian of the residual function.

        Parameters:
        X (np.array): State vector across all time steps.
        Y (np.array): Measurement data.

        Returns:
        np.array: Jacobian of the residual function.
        """
        K = Y.shape[2]
        n_x = 6
        n_y_1 = 3
        n_y_2 = 2
        n_y_3 = 2
        n_y_4 = 2
        J = np.zeros(((4 * n_x + n_y_1 + n_y_2 + n_y_3 + n_y_4) * K, 4 * n_x * K))
        for k in range(K):
            X_k = X[4 * n_x * k : 4 * n_x * (k + 1)]
            _, F_k = self.dynamic_model.x_new_and_F(self.dt, X_k)
            if k < K - 1:
                J[
                    k * 4 * n_x
                    + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4) : (k + 1) * 4 * n_x
                    + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4),
                    k * 4 * n_x : (k + 1) * 4 * n_x,
                ] = (
                    -fractional_matrix_power(self.Q, -1 / 2) @ F_k
                )
                J[
                    k * 4 * n_x
                    + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4) : (k + 1) * 4 * n_x
                    + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4),
                    (k + 1) * 4 * n_x : (k + 2) * 4 * n_x,
                ] = fractional_matrix_power(self.Q, -1 / 2)
            J[
                (k + 1) * 4 * n_x
                + k * (n_y_1 + n_y_2 + n_y_3 + n_y_4) : (k + 1) * 4 * n_x
                + (k + 1) * (n_y_1 + n_y_2 + n_y_3 + n_y_4),
                k * 4 * n_x : (k + 1) * 4 * n_x,
            ] = -fractional_matrix_power(self.R, -1 / 2) @ self.H(X_k)
        return J

    def cost_function(self, X, Y):
        """
        Compute the cost function (residual norm squared).

        Parameters:
        X (np.array): State vector across all time steps.
        Y (np.array): Measurement data.

        Returns:
        float: Cost function value.
        """
        return np.linalg.norm(self.r_function(X, Y)) ** 2

    def apply(self, dt, X_initial, Y):
        """
        Run the Levenberg-Marquardt algorithm to estimate the state.

        Parameters:
        dt (float): Time step.
        X_initial (np.array): Initial guess of the state vector.
        Y (np.array): Measurement data.

        Returns:
        np.array: Estimated state vector.
        """
        self.dt = dt
        lambda_0 = 1.0
        X_est_i = X_initial.reshape(-1, 1, order="F")
        lambda_i = lambda_0
        epsilon = 1e-6
        max_iter = 100

        i = 0

        while i < max_iter:
            r = self.r_function(X_est_i, Y)
            J = self.J_jacobian(X_est_i, Y)

            # For plotting the results
            grad_i_norm = np.linalg.norm(J.T @ r)

            if grad_i_norm < epsilon:
                print(
                    f"STOOOOP\nIteration {i}\nGrad_i_norm = {grad_i_norm}\nx_pred_i cost = {self.cost_function(X_pred_i, Y)} | x_bold_i cost = {self.cost_function(X_est_i, Y)}\nlambda_i = {lambda_i}\n"
                )
                break

            # Solve for the update step
            delta_x = np.linalg.inv(J.T @ J + lambda_i * np.eye(J.shape[1])) @ J.T @ r
            X_pred_i = X_est_i - delta_x

            print(
                f"Iteration {i}\nGrad_i_norm = {grad_i_norm}\nx_pred_i cost = {self.cost_function(X_pred_i, Y)} | x_bold_i cost = {self.cost_function(X_est_i, Y)}\nlambda_i = {lambda_i}\n"
            )

            if self.cost_function(
                X_pred_i,
                Y,
            ) < self.cost_function(X_est_i, Y):
                X_est_i = X_pred_i
                lambda_i *= 0.7
            else:
                lambda_i *= 2.0
            i += 1

        return X_est_i.reshape(Y.shape[2], 1, 24).T
