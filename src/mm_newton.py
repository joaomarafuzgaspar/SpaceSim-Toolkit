# src/mm_newton.py
import numpy as np


from scipy.linalg import solve


from dynamics import SatelliteDynamics
from config import SimulationConfig as config


class MMNewton:
    def __init__(self):
        # Define window size
        self.H = config.H
        self.dt = config.dt

        # Define noise covariances
        self.R_chief = config.R_chief  # Process noise covariance
        self.r_deputy_pos = config.r_deputy_pos

        # Define state to position transformation matrix
        self.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        # Define the dynamic model
        self.dyn = SatelliteDynamics()

        # Define Newton parameters
        self.grad_norm_order_mag = config.grad_norm_order_mag
        self.grad_norm_tol = config.grad_norm_tol
        self.max_iterations = config.max_iterations

        # Define Majorization-Minimization parameters
        self.mm_tol = config.mm_tol
        self.mm_max_iter = config.mm_max_iter

        # Store the cost function and gradient norm values
        self.cost_function_values = []
        self.grad_norm_values = []
        self.surrogate_function_values = []
        self.surrogate_grad_norm_values = []

    def h_function_chief(self, x_vec):
        return x_vec[0:3]

    def h_function_deputy(self, x_vec):
        """
        Compute the measurement vector for relative distances between satellites.

        Parameters:
        x_vec (np.array): Current state vector of all satellites.

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

    def h(self, x_vec):
        return np.concatenate(
            [self.h_function_chief(x_vec), self.h_function_deputy(x_vec)]
        )

    def surrogate_function(self, x_i, x_j, z_i, z_j, d_ij, sigma):
        """
        Compute the surrogate function Φ_ij.

        Args:
            x_i, x_j: Current position vectors for agents i and j
            z_i, z_j: Previous iteration position vectors (denoted as bar_x in the equation)
            d_ij: True measurement between agents i and j
            sigma: Standard deviation parameter
        """
        # Compute the norm of the difference between previous positions
        z_ij_norm = np.linalg.norm(self.P @ z_i - self.P @ z_j)

        # Term 1: ||x_i - bar_x_i||^2 / sigma^2
        term1 = np.linalg.norm(self.P @ x_i - self.P @ z_i) ** 2 / sigma**2

        # Term 2: (||bar_x_i - bar_x_j|| - d_ij) * (x_i - bar_x_i)^T(bar_x_i - bar_x_j) / (sigma^2 * ||bar_x_i - bar_x_j||)
        term2 = (
            ((z_ij_norm - d_ij) / (sigma**2 * z_ij_norm))
            * (self.P @ x_i - self.P @ z_i).T
            @ (self.P @ z_i - self.P @ z_j)
        )

        # Term 3: ||x_j - bar_x_j||^2 / sigma^2
        term3 = np.linalg.norm(self.P @ x_j - self.P @ z_j) ** 2 / sigma**2

        # Term 4: -(||bar_x_i - bar_x_j|| - d_ij) * (x_j - bar_x_j)^T(bar_x_i - bar_x_j) / (sigma^2 * ||bar_x_i - bar_x_j||)
        term4 = (
            -((z_ij_norm - d_ij) / (sigma**2 * z_ij_norm))
            * (self.P @ x_j - self.P @ z_j).T
            @ (self.P @ z_i - self.P @ z_j)
        )

        # Term 5: ||bar_x_i - bar_x_j||^2 / (2*sigma^2)
        term5 = z_ij_norm**2 / (2 * sigma**2)

        # Term 6: -d_ij * ||bar_x_i - bar_x_j|| / sigma^2
        term6 = -(d_ij * z_ij_norm) / sigma**2

        # Term 7: d_ij^2 / (2*sigma^2)
        term7 = d_ij**2 / (2 * sigma**2)

        return term1 + term2 + term3 + term4 + term5 + term6 + term7

    def surrogate_gradient(self, x_i, x_j, z_i, z_j, d_ij, sigma):
        """
        Compute the gradient of the surrogate function Φ_ij.
        """
        z_ij_norm = np.linalg.norm(self.P @ z_i - self.P @ z_j)
        z_ij = self.P @ z_i - self.P @ z_j

        # Gradient with respect to x_i
        grad_i = (2 / sigma**2) * self.P.T @ (self.P @ x_i - self.P @ z_i) + (
            (z_ij_norm - d_ij) / (sigma**2 * z_ij_norm)
        ) * self.P.T @ z_ij

        # Gradient with respect to x_j
        grad_j = (2 / sigma**2) * self.P.T @ (self.P @ x_j - self.P @ z_j) - (
            (z_ij_norm - d_ij) / (sigma**2 * z_ij_norm)
        ) * self.P.T @ z_ij

        return grad_i, grad_j

    def surrogate_hessian(self, x_i, x_j, z_i, z_j, d_ij, sigma):
        """
        Compute the Hessian of the surrogate function Φ_ij.
        """
        n = 3

        # The Hessian is constant with respect to x_i and x_j
        H_ii = (2 / sigma**2) * self.P.T @ self.P
        H_ij = self.P.T @ np.zeros((n, n)) @ self.P
        H_ji = self.P.T @ np.zeros((n, n)) @ self.P
        H_jj = (2 / sigma**2) * self.P.T @ self.P

        return H_ii, H_ij, H_ji, H_jj

    def obj_function(self, x_0, y, z_0=None):
        n_x = 6
        n_y_1 = 3
        f_x_0 = 0
        if z_0 is None:
            z_0 = x_0.copy()

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]

        z_1_k = z_0[:n_x, :]
        z_2_k = z_0[n_x : 2 * n_x, :]
        z_3_k = z_0[2 * n_x : 3 * n_x, :]
        z_4_k = z_0[3 * n_x : 4 * n_x, :]

        # Iterate over all sliding window time steps
        for k in range(self.H):
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
                (3, 1, x_3_k, x_1_k, y_31_k),  # (3, 1)
                (4, 1, x_4_k, x_1_k, y_41_k),  # (4, 1)
            ]

            # Iterate over each pair and update the cost function with the residuals for pairwise measurements
            for i, j, x_i_k, x_j_k, y_ij_k in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)
                f_x_0 += (y_ij_k - d_ij_k) ** 2 / (2 * self.r_deputy_pos**2)

            pairs_surrogate = [
                (2, 3, x_2_k, x_3_k, z_2_k, z_3_k, y_23_k),  # (2, 3)
                (2, 4, x_2_k, x_4_k, z_2_k, z_4_k, y_24_k),  # (2, 4)
                (3, 4, x_3_k, x_4_k, z_3_k, z_4_k, y_34_k),  # (3, 4)
            ]

            for i, j, x_i_k, x_j_k, z_i_k, z_j_k, y_ij_k in pairs_surrogate:
                f_x_0 += self.surrogate_function(
                    x_i_k, x_j_k, z_i_k, z_j_k, y_ij_k, self.r_deputy_pos
                )

            if k < self.H - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                x_1_k = self.dyn.x_new(self.dt, x_1_k)
                x_2_k = self.dyn.x_new(self.dt, x_2_k)
                x_3_k = self.dyn.x_new(self.dt, x_3_k)
                x_4_k = self.dyn.x_new(self.dt, x_4_k)

                z_1_k = self.dyn.x_new(self.dt, z_1_k)
                z_2_k = self.dyn.x_new(self.dt, z_2_k)
                z_3_k = self.dyn.x_new(self.dt, z_3_k)
                z_4_k = self.dyn.x_new(self.dt, z_4_k)

        return f_x_0 / self.H

    def grad_obj_function(self, x_0, y, z_0=None):
        n_x = 6
        n_y_1 = 3
        grad_f_x_0 = np.zeros_like(x_0)
        if z_0 is None:
            z_0 = x_0.copy()

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]

        z_1_k = z_0[:n_x, :]
        z_2_k = z_0[n_x : 2 * n_x, :]
        z_3_k = z_0[2 * n_x : 3 * n_x, :]
        z_4_k = z_0[3 * n_x : 4 * n_x, :]

        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for k in range(self.H):
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
                (3, 1, x_3_k, x_1_k, y_31_k, STM_t0_3, STM_t0_1),  # (3, 1)
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

            pairs_surrogate = [
                (
                    2,
                    3,
                    x_2_k,
                    x_3_k,
                    z_2_k,
                    z_3_k,
                    y_23_k,
                    STM_t0_2,
                    STM_t0_3,
                ),  # (2, 3)
                (
                    2,
                    4,
                    x_2_k,
                    x_4_k,
                    z_2_k,
                    z_4_k,
                    y_24_k,
                    STM_t0_2,
                    STM_t0_4,
                ),  # (2, 4)
                (
                    3,
                    4,
                    x_3_k,
                    x_4_k,
                    z_3_k,
                    z_4_k,
                    y_34_k,
                    STM_t0_3,
                    STM_t0_4,
                ),  # (3, 4)
            ]

            for (
                i,
                j,
                x_i_k,
                x_j_k,
                z_i_k,
                z_j_k,
                y_ij_k,
                STM_t0_i,
                STM_t0_j,
            ) in pairs_surrogate:
                grad_i, grad_j = self.surrogate_gradient(
                    x_i_k, x_j_k, z_i_k, z_j_k, y_ij_k, self.r_deputy_pos
                )
                grad_f_x_0[(i - 1) * n_x : i * n_x, :] += STM_t0_i.T @ grad_i
                grad_f_x_0[(j - 1) * n_x : j * n_x, :] += STM_t0_j.T @ grad_j

            if k < self.H - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                STM_t0_1_old = STM_t0_1
                STM_t0_2_old = STM_t0_2
                STM_t0_3_old = STM_t0_3
                STM_t0_4_old = STM_t0_4

                x_1_k, STM_t0_1 = self.dyn.x_new_and_F(self.dt, x_1_k)
                x_2_k, STM_t0_2 = self.dyn.x_new_and_F(self.dt, x_2_k)
                x_3_k, STM_t0_3 = self.dyn.x_new_and_F(self.dt, x_3_k)
                x_4_k, STM_t0_4 = self.dyn.x_new_and_F(self.dt, x_4_k)

                STM_t0_1 = STM_t0_1 @ STM_t0_1_old
                STM_t0_2 = STM_t0_2 @ STM_t0_2_old
                STM_t0_3 = STM_t0_3 @ STM_t0_3_old
                STM_t0_4 = STM_t0_4 @ STM_t0_4_old

                z_1_k = self.dyn.x_new(self.dt, z_1_k)
                z_2_k = self.dyn.x_new(self.dt, z_2_k)
                z_3_k = self.dyn.x_new(self.dt, z_3_k)
                z_4_k = self.dyn.x_new(self.dt, z_4_k)

        return grad_f_x_0 / self.H

    def hessian_obj_function(self, x_0, y, z_0=None):
        n_x = 6
        n_y_1 = 3
        hessian_f_x_0 = np.zeros(
            (x_0.shape[0], x_0.shape[0])
        )  # Initialize Hessian matrix
        if z_0 is None:
            z_0 = x_0.copy()

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]

        z_1_k = z_0[:n_x, :]
        z_2_k = z_0[n_x : 2 * n_x, :]
        z_3_k = z_0[2 * n_x : 3 * n_x, :]
        z_4_k = z_0[3 * n_x : 4 * n_x, :]

        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for k in range(self.H):
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
                (3, 1, x_3_k, x_1_k, y_31_k, STM_t0_3, STM_t0_1),  # (3, 1)
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

            pairs_surrogate = [
                (
                    2,
                    3,
                    x_2_k,
                    x_3_k,
                    z_2_k,
                    z_3_k,
                    y_23_k,
                    STM_t0_2,
                    STM_t0_3,
                ),  # (2, 3)
                (
                    2,
                    4,
                    x_2_k,
                    x_4_k,
                    z_2_k,
                    z_4_k,
                    y_24_k,
                    STM_t0_2,
                    STM_t0_4,
                ),  # (2, 4)
                (
                    3,
                    4,
                    x_3_k,
                    x_4_k,
                    z_3_k,
                    z_4_k,
                    y_34_k,
                    STM_t0_3,
                    STM_t0_4,
                ),  # (3, 4)
            ]

            for (
                i,
                j,
                x_i_k,
                x_j_k,
                z_i_k,
                z_j_k,
                y_ij_k,
                STM_t0_i,
                STM_t0_j,
            ) in pairs_surrogate:
                H_ii, H_ij, H_ji, H_jj = self.surrogate_hessian(
                    x_i_k, x_j_k, z_i_k, z_j_k, y_ij_k, self.r_deputy_pos
                )

                idx_i = n_x * (i - 1)
                idx_j = n_x * (j - 1)

                hessian_f_x_0[idx_i : idx_i + n_x, idx_i : idx_i + n_x] += (
                    STM_t0_i.T @ H_ii @ STM_t0_i
                )
                hessian_f_x_0[idx_i : idx_i + n_x, idx_j : idx_j + n_x] += (
                    STM_t0_i.T @ H_ij @ STM_t0_j
                )
                hessian_f_x_0[idx_j : idx_j + n_x, idx_i : idx_i + n_x] += (
                    STM_t0_j.T @ H_ji @ STM_t0_i
                )
                hessian_f_x_0[idx_j : idx_j + n_x, idx_j : idx_j + n_x] += (
                    STM_t0_j.T @ H_jj @ STM_t0_j
                )

            if k < self.H - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                STM_t0_1_old = STM_t0_1
                STM_t0_2_old = STM_t0_2
                STM_t0_3_old = STM_t0_3
                STM_t0_4_old = STM_t0_4

                x_1_k, STM_t0_1 = self.dyn.x_new_and_F(self.dt, x_1_k)
                x_2_k, STM_t0_2 = self.dyn.x_new_and_F(self.dt, x_2_k)
                x_3_k, STM_t0_3 = self.dyn.x_new_and_F(self.dt, x_3_k)
                x_4_k, STM_t0_4 = self.dyn.x_new_and_F(self.dt, x_4_k)

                STM_t0_1 = STM_t0_1 @ STM_t0_1_old
                STM_t0_2 = STM_t0_2 @ STM_t0_2_old
                STM_t0_3 = STM_t0_3 @ STM_t0_3_old
                STM_t0_4 = STM_t0_4 @ STM_t0_4_old

                z_1_k = self.dyn.x_new(self.dt, z_1_k)
                z_2_k = self.dyn.x_new(self.dt, z_2_k)
                z_3_k = self.dyn.x_new(self.dt, z_3_k)
                z_4_k = self.dyn.x_new(self.dt, z_4_k)

        return hessian_f_x_0 / self.H

    def solve_MHE_problem(self, k, Y, x_init, x_true_initial, x_true_end):
        x = x_init
        z = x_init.copy()

        for mm_iter in range(self.mm_max_iter):
            # print(f"\nMajorization-Minimization Iteration {mm_iter + 1}")

            prev_cost_value = None
            prev_grad_norm_value = None
            prev_global_error = None
            grad_norm_order_history = []

            # Solve the surrogate problem
            for iteration in range(self.max_iterations):
                # Compute the cost function, gradient and approximated Hessian
                L_x = self.obj_function(x, Y, z)
                grad_L_x = self.grad_obj_function(x, Y, z)
                hessian_L_x = self.hessian_obj_function(x, Y, z)

                # Convergence tracking
                cost_value = L_x[0][0]
                gradient_norm_value = np.linalg.norm(grad_L_x)

                # Store the cost function and gradient norm values
                self.surrogate_function_values.append(cost_value)
                self.surrogate_grad_norm_values.append(gradient_norm_value)
                self.cost_function_values.append(self.obj_function(x, Y, None)[0][0])
                self.grad_norm_values.append(
                    np.linalg.norm(self.grad_obj_function(x, Y, None))
                )

                # Compute the changes in the cost function, gradient and global error
                if prev_cost_value is not None:
                    cost_value_change = (
                        (cost_value - prev_cost_value) / abs(prev_cost_value) * 100
                    )
                    gradient_norm_value_change = (
                        (gradient_norm_value - prev_grad_norm_value)
                        / abs(prev_grad_norm_value)
                        * 100
                    )
                    global_estimation_error_change = (
                        (np.linalg.norm(x - x_true_initial) - prev_global_error)
                        / abs(prev_global_error)
                        * 100
                    )
                prev_cost_value = cost_value
                prev_grad_norm_value = gradient_norm_value
                prev_global_error = np.linalg.norm(x - x_true_initial)

                # Track gradient norm order of magnitude
                current_order = int(
                    np.floor(np.log10(gradient_norm_value + 1e-12))
                )  # avoid log(0)
                grad_norm_order_history.append(current_order)

                if self.grad_norm_order_mag:
                    if len(grad_norm_order_history) >= 3:
                        if (
                            grad_norm_order_history[-1]
                            == grad_norm_order_history[-2]
                            == grad_norm_order_history[-3]
                        ):
                            stagnant_order = True
                            # if k == self.H - 1:
                            #     stagnant_order = False
                        else:
                            stagnant_order = False
                    else:
                        stagnant_order = False
                else:
                    stagnant_order = False

                # Propagate window initial conditions for metrics
                x_end = x.copy()
                for _ in range(self.H - 1):
                    x_end = self.dyn.x_new(self.dt, x_end)

                # Check convergence and print metrics
                if (
                    gradient_norm_value < self.grad_norm_tol
                    or iteration == self.max_iterations
                    or stagnant_order
                ):
                    reason = (
                        "tolerance reached"
                        if gradient_norm_value < self.grad_norm_tol
                        else (
                            "max iteration reached"
                            if iteration == self.max_iterations
                            else "gradient norm stagnated"
                        )
                    )
                    print(
                        f"[MMNewton] STOP on Iteration {iteration} | Majorization #{mm_iter + 1} ({reason})"
                    )
                    print(
                        f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)"
                    )
                    print(
                        f"Final initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m"
                    )
                    print(
                        f"Final position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n"
                    )
                    break
                else:
                    if iteration == 0:
                        print(
                            f"[MMNewton] Before applying the algorithm | Majorization #{mm_iter + 1}\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}"
                        )
                    else:
                        print(
                            f"[MMNewton] Iteration {iteration} | Majorization #{mm_iter + 1}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)"
                        )

                # Print estimation errors
                print(
                    f"Initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m"
                )
                print(
                    f"Position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n"
                )

                # Solve for the Newton step
                delta_x = solve(hessian_L_x, -grad_L_x)
                x += delta_x

            # Update z for next majorization step
            print(
                f"Majorization-Minimization Iteration {mm_iter + 1} | Norm: {np.linalg.norm(x - z)}"
            )
            if np.linalg.norm(x - z) < self.mm_tol:
                print(
                    f"Majorization-Minimization converged after {mm_iter + 1} iterations"
                )
                break

            # Update z for next majorization step
            z = x.copy()

        # Propagate window initial conditions getting estimate at timestamp k
        x_init = x
        for _ in range(self.H - 1):
            x = self.dyn.x_new(self.dt, x)

        return x_init, x
