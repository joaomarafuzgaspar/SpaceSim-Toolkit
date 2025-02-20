import numpy as np

from tqdm import tqdm
from scipy.linalg import solve
from dynamics import SatelliteDynamics


class Tree_Newton:
    """
    This class implements the Newton method with an approximated Hessian for optimization.
    """

    def __init__(self, W, R_chief, r_deputy_pos):
        # Define window size
        self.W = W

        # Define noise covariances
        self.R_chief = R_chief  # Process noise covariance
        self.r_deputy_pos = r_deputy_pos

        # Define state to position transformation matrix
        self.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        # Define the dynamic model
        self.dynamic_model = SatelliteDynamics()

        # Define Newton parameters
        self.grad_tol = 1e0
        self.max_iter = 20
        
        # Store the cost function and gradient norm values
        self.cost_function_values = []
        self.grad_norm_values = []

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

    def original_obj_function(self, dt, x_0, y):
        n_x = 6
        n_y_1 = 3
        f_x_0 = 0

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]

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
                x_1_k = self.dynamic_model.x_new(dt, x_1_k)
                x_2_k = self.dynamic_model.x_new(dt, x_2_k)
                x_3_k = self.dynamic_model.x_new(dt, x_3_k)
                x_4_k = self.dynamic_model.x_new(dt, x_4_k)

        return f_x_0 / self.W

    def obj_function(self, dt, x_0, y):
        n_x = 6
        n_y_1 = 3
        f_x_0 = 0

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_0
        x_1_k = x_0[:n_x, :]
        x_2_k = x_0[n_x : 2 * n_x, :]
        x_3_k = x_0[2 * n_x : 3 * n_x, :]
        x_4_k = x_0[3 * n_x : 4 * n_x, :]

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
                (3, 1, x_3_k, x_1_k, y_31_k),  # (3, 1)
                (4, 1, x_4_k, x_1_k, y_41_k),  # (4, 1)
            ]

            # Iterate over each pair and update the cost function with the residuals for pairwise measurements
            for i, j, x_i_k, x_j_k, y_ij_k in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)
                f_x_0 += (y_ij_k - d_ij_k) ** 2 / (2 * self.r_deputy_pos**2)

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                x_1_k = self.dynamic_model.x_new(dt, x_1_k)
                x_2_k = self.dynamic_model.x_new(dt, x_2_k)
                x_3_k = self.dynamic_model.x_new(dt, x_3_k)
                x_4_k = self.dynamic_model.x_new(dt, x_4_k)

        return f_x_0 / self.W

    def grad_obj_function(self, dt, x_0, y):
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

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                STM_t0_1_old = STM_t0_1
                STM_t0_2_old = STM_t0_2
                STM_t0_3_old = STM_t0_3
                STM_t0_4_old = STM_t0_4
                x_1_k, STM_t0_1 = self.dynamic_model.x_new_and_F(dt, x_1_k)
                x_2_k, STM_t0_2 = self.dynamic_model.x_new_and_F(dt, x_2_k)
                x_3_k, STM_t0_3 = self.dynamic_model.x_new_and_F(dt, x_3_k)
                x_4_k, STM_t0_4 = self.dynamic_model.x_new_and_F(dt, x_4_k)
                STM_t0_1 = STM_t0_1 @ STM_t0_1_old
                STM_t0_2 = STM_t0_2 @ STM_t0_2_old
                STM_t0_3 = STM_t0_3 @ STM_t0_3_old
                STM_t0_4 = STM_t0_4 @ STM_t0_4_old

        return grad_f_x_0 / self.W

    def hessian_obj_function(self, dt, x_0, y):
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

            if k < self.W - 1:
                # Get x_1(k), x_2(k), x_3(k), x_4(k) from the state vector x_0
                STM_t0_1_old = STM_t0_1
                STM_t0_2_old = STM_t0_2
                STM_t0_3_old = STM_t0_3
                STM_t0_4_old = STM_t0_4
                x_1_k, STM_t0_1 = self.dynamic_model.x_new_and_F(dt, x_1_k)
                x_2_k, STM_t0_2 = self.dynamic_model.x_new_and_F(dt, x_2_k)
                x_3_k, STM_t0_3 = self.dynamic_model.x_new_and_F(dt, x_3_k)
                x_4_k, STM_t0_4 = self.dynamic_model.x_new_and_F(dt, x_4_k)
                STM_t0_1 = STM_t0_1 @ STM_t0_1_old
                STM_t0_2 = STM_t0_2 @ STM_t0_2_old
                STM_t0_3 = STM_t0_3 @ STM_t0_3_old
                STM_t0_4 = STM_t0_4 @ STM_t0_4_old

        return hessian_f_x_0 / self.W

    def solve_for_each_window(self, dt, x_init, Y, x_true):
        x = x_init

        prev_cost_function_value = None
        prev_grad_norm_value = None
        prev_global_error = None

        for iteration in range(self.max_iter):
            # Compute the cost function, gradient and approximated Hessian
            L_x = self.original_obj_function(dt, x, Y)
            grad_L_x = self.grad_obj_function(dt, x, Y)
            hessian_L_x = self.hessian_obj_function(dt, x, Y)

            # Convergence tracking
            cost_function_value = L_x[0][0]
            grad_norm_value = np.linalg.norm(grad_L_x)
            
            # Store the cost function and gradient norm values
            self.cost_function_values.append(cost_function_value)
            self.grad_norm_values.append(grad_norm_value)

            # Compute the changes in the cost function, gradient and global error
            if prev_cost_function_value is not None:
                cost_function_change = (
                    (cost_function_value - prev_cost_function_value)
                    / abs(prev_cost_function_value)
                    * 100
                )
                grad_norm_change = (
                    (grad_norm_value - prev_grad_norm_value)
                    / abs(prev_grad_norm_value)
                    * 100
                )
                global_error_change = (
                    (np.linalg.norm(x - x_true) - prev_global_error)
                    / abs(prev_global_error)
                    * 100
                )
            prev_cost_function_value = cost_function_value
            prev_grad_norm_value = grad_norm_value
            prev_global_error = np.linalg.norm(x - x_true)

            # Check convergence and print metrics
            if grad_norm_value < self.grad_tol or iteration + 1 == self.max_iter:
                print(
                    f"STOP on Iteration {iteration}\nCost function = {cost_function_value} ({cost_function_change:.2f}%)\nGradient norm = {grad_norm_value} ({grad_norm_change:.2f}%)\nGlobal relative error = {np.linalg.norm(x - x_true)} ({global_error_change:.2f}%)"
                )
                print(
                    f"Final position relative errors: {np.linalg.norm(x[0:3, :] - x_true[0:3, :])} m, {np.linalg.norm(x[6:9, :] - x_true[6:9, :])} m, {np.linalg.norm(x[12:15, :] - x_true[12:15, :])} m, {np.linalg.norm(x[18:21, :] - x_true[18:21, :])} m\n"
                )
                break
            else:
                if iteration == 0:
                    print(
                        f"Before applying the algorithm\nCost function: {cost_function_value}\nGradient norm: {grad_norm_value}\nGlobal relative error: {np.linalg.norm(x - x_true)}"
                    )
                else:
                    print(
                        f"Iteration {iteration}\nCost function: {cost_function_value} ({cost_function_change:.2f}%)\nGradient norm: {grad_norm_value} ({grad_norm_change:.2f}%)\nGlobal relative error: {np.linalg.norm(x - x_true)} ({global_error_change:.2f}%)"
                    )

            # Print relative errors
            print(
                f"Position relative errors: {np.linalg.norm(x[0:3, :] - x_true[0:3, :])} m, {np.linalg.norm(x[6:9, :] - x_true[6:9, :])} m, {np.linalg.norm(x[12:15, :] - x_true[12:15, :])} m, {np.linalg.norm(x[18:21, :] - x_true[18:21, :])} m\n"
            )

            # Solve for the Newton step
            # delta_x = solve(hessian_L_x, -grad_L_x)
            delta_x = -np.linalg.pinv(hessian_L_x) @ grad_L_x
            x += delta_x

        return x

    def apply(self, dt, x_init, Y, X_true):
        K = Y.shape[2]
        
        # Initialize storage for results
        X_est = np.zeros_like(X_true)
        for n in tqdm(range(K - self.W + 1), desc="Windows", leave=False):
            x_est = self.solve_for_each_window(dt, x_init, Y[:, :, n : n + self.W], X_true[:, :, n])
            X_est[:, :, n] = x_est

            # Get next new guess (warm-start)
            # The initial guess is the previous window propagated forward
            x_init = self.dynamic_model.x_new(dt, x_est)

        return X_est
