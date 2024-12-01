import numpy as np

from tqdm import tqdm
from scipy.linalg import solve
from dynamics import SatelliteDynamics


class UNKKT:
    """
    This class implements the Levenberg-Marquardt algorithm for the Weighted Least Squares (WLSTSQ) estimator.
    """

    def __init__(self, W, R_chief, r_deputy_pos):
        """
        Initialize the WLSTSQ class.

        Parameters:
        Q (np.array): Process noise covariance.
        R (np.array): Measurement noise covariance.
        SatelliteDynamics (class): Satellite dynamics model.
        """
        # Define window size
        self.W = W

        # Define noise covariances
        self.R_chief = R_chief  # Process noise covariance
        self.r_deputy_pos = r_deputy_pos

        # Define state to position transformation matrix
        self.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

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
        H = np.zeros((3, 24))
        H[0:3, 0:3] = np.eye(3)
        return H

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

    def H_jacobian_deputy(self, x_vec):
        """
        Computes the Jacobian of the measurement function.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        H (np.array): The Jacobian of the measurement function.
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
        return np.concatenate(
            [self.h_function_chief(x_vec), self.h_function_deputy(x_vec)]
        )

    def H(self, x_vec):
        return np.concatenate((self.H_jacobian_chief(), self.H_jacobian_deputy(x_vec)))

    def obj_function(self, x_0, STM, y):
        n_x = 6
        n_y_1 = 3
        f_x_0 = 0

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]
        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for k in range(self.W):
            # Absolute residual term: observed data y for each state
            y_1_k = y[:n_y_1, :, k]
            y_rel_k = y[n_y_1:, :, k]

            # Update the cost function with the residuals for self-measurements
            residual = y_1_k - self.P @ x_1_k
            f_x_0 += 1 / 2 * residual.T @ np.linalg.inv(self.R_chief) @ residual

            # Pairwise relative measurements
            y_21_k = y_rel_k[0]
            y_23_k = y_rel_k[1]
            y_24_k = y_rel_k[2]
            y_31_k = y_rel_k[3]
            y_34_k = y_rel_k[4]
            y_41_k = y_rel_k[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_k, x_1_k, y_21_k),  # (2, 1)
                (2, 3, x_2_k, x_3_k, y_23_k),  # (2, 3)
                (2, 4, x_2_k, x_4_k, y_24_k),  # (2, 4)
                (3, 1, x_3_k, x_1_k, y_31_k),  # (3, 1)
                (3, 4, x_3_k, x_4_k, y_34_k),  # (3, 4)
                (4, 1, x_4_k, x_1_k, y_41_k),  # (4, 1)
            ]

            # Iterate over each pair and update the cost function with the residuals for pairwise measurements
            for i, j, x_i_k, x_j_k, y_ij_k in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)
                f_x_0 += (y_ij_k - d_ij_k) ** 2 / (2 * self.r_deputy_pos**2)

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                x_1_k = STM[:n_x, :n_x, k + 1] @ x_1_k
                x_2_k = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ x_2_k
                x_3_k = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ x_3_k
                x_4_k = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ x_4_k
                STM_t0_1 = STM[:n_x, :n_x, k + 1] @ STM_t0_1
                STM_t0_2 = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ STM_t0_2
                STM_t0_3 = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ STM_t0_3
                STM_t0_4 = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ STM_t0_4

        return f_x_0

    def grad_obj_function(self, x_0, STM, y):
        n_x = 6
        n_y_1 = 3
        grad_f_x_0 = np.zeros_like(x_0)

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]
        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for k in range(self.W):
            # Absolute residual term: observed data y for each state
            y_1_k = y[:n_y_1, :, k]
            y_rel_k = y[n_y_1:, :, k]

            # Compute gradients for the absolute residual terms
            grad_f_x_0[:n_x, :] -= (
                STM_t0_1.T
                @ self.P.T
                @ np.linalg.inv(self.R_chief)
                @ (y_1_k - self.P @ x_1_k)
            )

            # Pairwise relative measurements
            y_21_k = y_rel_k[0]
            y_23_k = y_rel_k[1]
            y_24_k = y_rel_k[2]
            y_31_k = y_rel_k[3]
            y_34_k = y_rel_k[4]
            y_41_k = y_rel_k[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_k, x_1_k, y_21_k, STM_t0_2, STM_t0_1),  # (2, 1)
                (2, 3, x_2_k, x_3_k, y_23_k, STM_t0_2, STM_t0_3),  # (2, 3)
                (2, 4, x_2_k, x_4_k, y_24_k, STM_t0_2, STM_t0_4),  # (2, 4)
                (3, 1, x_3_k, x_1_k, y_31_k, STM_t0_3, STM_t0_1),  # (3, 1)
                (3, 4, x_3_k, x_4_k, y_34_k, STM_t0_3, STM_t0_4),  # (3, 4)
                (4, 1, x_4_k, x_1_k, y_41_k, STM_t0_4, STM_t0_1),  # (4, 1)
            ]

            # Iterate over each pair and compute gradients
            for i, j, x_i_k, x_j_k, y_ij_k, STM_t0_i, STM_t0_j in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)

                # Update the gradient of f(x) with respect to x_i(k) and x_j(k)
                grad_f_x_0[(i - 1) * n_x : i * n_x, :] -= (
                    (y_ij_k - d_ij_k)
                    / self.r_deputy_pos**2
                    * (STM_t0_i.T @ self.P.T @ d_ij_k_vec)
                    / d_ij_k
                )
                grad_f_x_0[(j - 1) * n_x : j * n_x, :] += (
                    (y_ij_k - d_ij_k)
                    / self.r_deputy_pos**2
                    * (STM_t0_j.T @ self.P.T @ d_ij_k_vec)
                    / d_ij_k
                )

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                x_1_k = STM[:n_x, :n_x, k + 1] @ x_1_k
                x_2_k = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ x_2_k
                x_3_k = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ x_3_k
                x_4_k = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ x_4_k
                STM_t0_1 = STM[:n_x, :n_x, k + 1] @ STM_t0_1
                STM_t0_2 = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ STM_t0_2
                STM_t0_3 = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ STM_t0_3
                STM_t0_4 = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ STM_t0_4

        return grad_f_x_0

    def hessian_obj_function(self, x_0, STM, y):
        n_x = 6
        n_y_1 = 3
        hessian_f_x_0 = np.zeros(
            (x_0.shape[0], x_0.shape[0])
        )  # Initialize Hessian matrix

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]
        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for k in range(self.W):
            # Absolute residual term: observed data y for each state
            y_rel_k = y[n_y_1:, :, k]

            # Absolute measurement residuals' Hessian
            hessian_f_x_0[:n_x, :n_x] += (
                STM_t0_1.T @ self.P.T @ np.linalg.inv(self.R_chief) @ self.P @ STM_t0_1
            )

            # Pairwise relative measurements
            y_21_k = y_rel_k[0]
            y_23_k = y_rel_k[1]
            y_24_k = y_rel_k[2]
            y_31_k = y_rel_k[3]
            y_34_k = y_rel_k[4]
            y_41_k = y_rel_k[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_k, x_1_k, y_21_k, STM_t0_2, STM_t0_1),  # (2, 1)
                (2, 3, x_2_k, x_3_k, y_23_k, STM_t0_2, STM_t0_3),  # (2, 3)
                (2, 4, x_2_k, x_4_k, y_24_k, STM_t0_2, STM_t0_4),  # (2, 4)
                (3, 1, x_3_k, x_1_k, y_31_k, STM_t0_3, STM_t0_1),  # (3, 1)
                (3, 4, x_3_k, x_4_k, y_34_k, STM_t0_3, STM_t0_4),  # (3, 4)
                (4, 1, x_4_k, x_1_k, y_41_k, STM_t0_4, STM_t0_1),  # (4, 1)
            ]

            # Iterate over each pair and compute gradients
            for i, j, x_i_k, x_j_k, y_ij_k, STM_t0_i, STM_t0_j in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)

                # Update the Hessian for x_i(k) and x_j(k)
                idx_i = n_x * (i - 1)
                idx_j = n_x * (j - 1)

                hessian_f_x_0[idx_i : idx_i + n_x, idx_i : idx_i + n_x] -= (
                    1
                    / (self.r_deputy_pos**2)
                    * (
                        STM_t0_i.T
                        @ (
                            (y_ij_k - d_ij_k) / d_ij_k * self.P.T @ self.P
                            - y_ij_k
                            / d_ij_k**3
                            * self.P.T
                            @ d_ij_k_vec
                            @ d_ij_k_vec.T
                            @ self.P
                        )
                        @ STM_t0_i
                    )
                )
                hessian_f_x_0[idx_i : idx_i + n_x, idx_j : idx_j + n_x] += (
                    1
                    / (self.r_deputy_pos**2)
                    * (
                        STM_t0_i.T
                        @ (
                            (y_ij_k - d_ij_k) / d_ij_k * self.P.T @ self.P
                            - y_ij_k
                            / d_ij_k**3
                            * self.P.T
                            @ d_ij_k_vec
                            @ d_ij_k_vec.T
                            @ self.P
                        )
                        @ STM_t0_j
                    )
                )
                hessian_f_x_0[idx_j : idx_j + n_x, idx_i : idx_i + n_x] += (
                    1
                    / (self.r_deputy_pos**2)
                    * (
                        STM_t0_j.T
                        @ (
                            (y_ij_k - d_ij_k) / d_ij_k * self.P.T @ self.P
                            - y_ij_k
                            / d_ij_k**3
                            * self.P.T
                            @ d_ij_k_vec
                            @ d_ij_k_vec.T
                            @ self.P
                        )
                        @ STM_t0_i
                    )
                )
                hessian_f_x_0[idx_j : idx_j + n_x, idx_j : idx_j + n_x] -= (
                    1
                    / (self.r_deputy_pos**2)
                    * (
                        STM_t0_j.T
                        @ (
                            (y_ij_k - d_ij_k) / d_ij_k * self.P.T @ self.P
                            - y_ij_k
                            / d_ij_k**3
                            * self.P.T
                            @ d_ij_k_vec
                            @ d_ij_k_vec.T
                            @ self.P
                        )
                        @ STM_t0_j
                    )
                )

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                x_1_k = STM[:n_x, :n_x, k + 1] @ x_1_k
                x_2_k = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ x_2_k
                x_3_k = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ x_3_k
                x_4_k = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ x_4_k
                STM_t0_1 = STM[:n_x, :n_x, k + 1] @ STM_t0_1
                STM_t0_2 = STM[n_x : 2 * n_x, n_x : 2 * n_x, k + 1] @ STM_t0_2
                STM_t0_3 = STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, k + 1] @ STM_t0_3
                STM_t0_4 = STM[3 * n_x : 4 * n_x, 3 * n_x : 4 * n_x, k + 1] @ STM_t0_4

        return hessian_f_x_0

    def lagrangian(self, X, STM, Y):
        return self.obj_function(X, STM, Y)

    def grad_lagrangian(self, X, STM, Y):
        return self.grad_obj_function(X, STM, Y)

    def hessian_lagrangian(self, X, STM, Y):
        return self.hessian_obj_function(X, STM, Y)

    def solve_for_each_window(self, dt, x_init, lambda_init, Y):
        n_x = 6
        tolerance = 1e-6
        max_iter = 100
        x = x_init
        lmbda = lambda_init
        for iteration in range(max_iter):
            # Compute the cost function, gradient of the Lagrangian and Hessian of the Lagrangian
            L_x = self.lagrangian(dt, x, lmbda, Y)
            grad_L_x = self.grad_lagrangian(dt, x, lmbda, Y)
            hessian_L_x = self.hessian_lagrangian(x, Y)

            # Compute the constraints and its Jacobian
            h_x = self.eq_const_function(dt, x)
            grad_h_x = self.jacobian_eq_const_function(dt, x)

            # Calculate norms for convergence tracking
            L_norm = np.linalg.norm(L_x)
            grad_L_norm = np.linalg.norm(grad_L_x)
            h_norm = np.linalg.norm(h_x)
            # Check convergence and print metrics
            if (
                grad_L_norm < tolerance and h_norm < tolerance
            ) or iteration + 1 == max_iter:
                print(
                    f"STOP on Iteration {iteration}\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\nh_norm = {h_norm}\n"
                )
                break
            else:
                if iteration == 0:
                    print(
                        f"Before applying the algorithm\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\nh_norm = {h_norm}\n"
                    )
                else:
                    print(
                        f"Iteration {iteration}\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\nh_norm = {h_norm}\n"
                    )

            # Form the KKT matrix
            KKT_matrix = np.block(
                [
                    [hessian_L_x, grad_h_x.T],
                    [
                        grad_h_x,
                        np.zeros((4 * n_x * (self.W - 1), 4 * n_x * (self.W - 1))),
                    ],
                ]
            )

            # Form the right-hand side
            rhs = -np.block([[grad_L_x], [h_x]])

            # Solve for the Newton step - this is one iteration
            delta = solve(KKT_matrix, rhs)
            delta_x = delta[: x.size]
            delta_lmbda = delta[x.size :]

            # Update x and lambda
            x += delta_x
            lmbda += delta_lmbda

        return x

    def apply(self, X_initial, STM, Y):
        n_x = 6
        K = Y.shape[2]

        # Before applying the Newton algorithm for the first time, initialize the initial conditions
        # guess randomly (cold-start) and the next states depending on the initial condition guess and dynamics
        # so that the KKT primal feasibility condition h(x) = 0 is verified except for the first propagation iteration
        x_init = np.zeros((4 * n_x, 1))
        x_init[: 4 * n_x, :] = X_initial

        X_est = np.zeros((4 * n_x, 1, K - self.W + 1))
        for n in tqdm(range(K - self.W + 1), desc="Windows", leave=False):
            # For the lambdas try to solve the least squares problem that arises from the stationarity KKT condition
            x_est = self.solve_for_each_window(
                x_init, STM[:, :, n : n + self.W], Y[:, :, n : n + self.W]
            )
            X_est[:, :, n] = x_est[: 4 * n_x, :]

            # Get next new guess (warm-start)
            # The initial guess is the previous window estimate second timestamp value
            x_init[:n_x, :] = STM[:n_x, :n_x, n + 1] @ x_est[:n_x, :]
            x_init[n_x : 2 * n_x, :] = (
                STM[n_x : 2 * n_x, n_x : 2 * n_x, n + 1] @ x_est[n_x : 2 * n_x, :]
            )
            x_init[2 * n_x : 3 * n_x, :] = (
                STM[2 * n_x : 3 * n_x, 2 * n_x : 3 * n_x, n + 1]
                @ x_est[2 * n_x : 3 * n_x, :]
            )
            x_init[3 * n_x : 4 * n_x, :] = (
                STM[3 * n_x :, 3 * n_x :, n + 1] @ x_est[3 * n_x :, :]
            )

        return X_est
