import numpy as np

from scipy.linalg import solve
from dynamics import SatelliteDynamics


class CNKKT:
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

    def obj_function(self, X, Y):
        n_x = 6
        n_y_1 = 3

        # Iterate over all sliding window time steps
        obj_fun = 0
        for k in range(self.W):
            # Extract x_1(k), x_2(k), x_3(k), x_4(k) from flattened state vector x
            x_1_k = X[n_x * (4 * k) : n_x * (4 * k + 1), :]  # x_1(k)
            x_2_k = X[n_x * (4 * k + 1) : n_x * (4 * k + 2), :]  # x_2(k)
            x_3_k = X[n_x * (4 * k + 2) : n_x * (4 * k + 3), :]  # x_3(k)
            x_4_k = X[n_x * (4 * k + 3) : n_x * (4 * k + 4), :]  # x_4(k)

            # Absolute residual term: observed data y for each state
            y_1_k = Y[:n_y_1, :, k]
            y_rel_k = Y[n_y_1:, :, k]

            # Update the cost function with the residuals for self-measurements
            residual = y_1_k - self.P @ x_1_k
            obj_fun += 1 / 2 * residual.T @ np.linalg.inv(self.R_chief) @ residual

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
                obj_fun += (y_ij_k - d_ij_k) ** 2 / (2 * self.r_deputy_pos**2)

        return obj_fun

    def grad_obj_function(self, X, Y):
        n_x = 6
        n_y_1 = 3

        # Iterate over all sliding window time steps
        grad_obj_fun = np.zeros_like(X)
        for k in range(self.W):
            # Extract x_1(k), x_2(k), x_3(k), x_4(k) from flattened state vector x
            x_1_k = X[n_x * (4 * k) : n_x * (4 * k + 1), :]  # x_1(k)
            x_2_k = X[n_x * (4 * k + 1) : n_x * (4 * k + 2), :]  # x_2(k)
            x_3_k = X[n_x * (4 * k + 2) : n_x * (4 * k + 3), :]  # x_3(k)
            x_4_k = X[n_x * (4 * k + 3) : n_x * (4 * k + 4), :]  # x_4(k)

            # Absolute residual term: observed data y for each state
            y_1_k = Y[:n_y_1, :, k]
            y_rel_k = Y[n_y_1:, :, k]

            # Compute gradients for the absolute residual terms
            grad_obj_fun[n_x * (4 * k) : n_x * (4 * k + 1), :] -= (
                self.P.T @ np.linalg.inv(self.R_chief) @ (y_1_k - self.P @ x_1_k)
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
                (2, 1, x_2_k, x_1_k, y_21_k),  # (2, 1)
                (2, 3, x_2_k, x_3_k, y_23_k),  # (2, 3)
                (2, 4, x_2_k, x_4_k, y_24_k),  # (2, 4)
                (3, 1, x_3_k, x_1_k, y_31_k),  # (3, 1)
                (3, 4, x_3_k, x_4_k, y_34_k),  # (3, 4)
                (4, 1, x_4_k, x_1_k, y_41_k),  # (4, 1)
            ]

            # Iterate over each pair and compute gradients
            for i, j, x_i_k, x_j_k, y_ij_k in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)

                # Update the gradient of f(x) with respect to x_i(k) and x_j(k)
                grad_obj_fun[n_x * (4 * k + (i - 1)) : n_x * (4 * k + i), :] -= (
                    (y_ij_k - d_ij_k)
                    / self.r_deputy_pos**2
                    * self.P.T
                    @ d_ij_k_vec
                    / d_ij_k
                )
                grad_obj_fun[n_x * (4 * k + (j - 1)) : n_x * (4 * k + j), :] += (
                    (y_ij_k - d_ij_k)
                    / self.r_deputy_pos**2
                    * self.P.T
                    @ d_ij_k_vec
                    / d_ij_k
                )

        return grad_obj_fun

    def hessian_obj_function(self, X, Y):
        n_x = 6
        n_y_1 = 3

        # Iterate over all sliding window time steps
        hessian_obj_fun = np.zeros(
            (X.shape[0], X.shape[0])
        )  # Initialize Hessian matrix
        for k in range(self.W):
            # Extract x_1(k), x_2(k), x_3(k), x_4(k) from flattened state vector x
            x_1_k = X[n_x * (4 * k) : n_x * (4 * k + 1), :]
            x_2_k = X[n_x * (4 * k + 1) : n_x * (4 * k + 2), :]
            x_3_k = X[n_x * (4 * k + 2) : n_x * (4 * k + 3), :]
            x_4_k = X[n_x * (4 * k + 3) : n_x * (4 * k + 4), :]

            # Absolute residual term: observed data y for each state
            y_1_k = Y[:n_y_1, :, k]
            y_rel_k = Y[n_y_1:, :, k]

            # Absolute measurement residuals' Hessian
            hessian_obj_fun[
                n_x * (4 * k) : n_x * (4 * k + 1), n_x * (4 * k) : n_x * (4 * k + 1)
            ] += (self.P.T @ np.linalg.inv(self.R_chief) @ self.P)

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

            # Iterate over each pair and compute gradients
            for i, j, x_i_k, x_j_k, y_ij_k in pairs:
                d_ij_k_vec = self.P @ x_i_k - self.P @ x_j_k
                d_ij_k = np.linalg.norm(d_ij_k_vec)

                # Second derivative (Hessian) of relative measurement residual
                hessian_rel = (
                    -1
                    / (self.r_deputy_pos**2)
                    * (
                        (y_ij_k - d_ij_k) / d_ij_k * self.P.T @ self.P
                        - y_ij_k
                        / d_ij_k**3
                        * self.P.T
                        @ d_ij_k_vec
                        @ d_ij_k_vec.T
                        @ self.P
                    )
                )

                # Update the Hessian for x_i(k) and x_j(k)
                idx_i = n_x * (4 * k + (i - 1))
                idx_j = n_x * (4 * k + (j - 1))

                hessian_obj_fun[idx_i : idx_i + n_x, idx_i : idx_i + n_x] += hessian_rel
                hessian_obj_fun[idx_j : idx_j + n_x, idx_j : idx_j + n_x] += hessian_rel
                hessian_obj_fun[idx_i : idx_i + n_x, idx_j : idx_j + n_x] -= hessian_rel
                hessian_obj_fun[idx_j : idx_j + n_x, idx_i : idx_i + n_x] -= hessian_rel

        return hessian_obj_fun

    def eq_const_function(self, dt, X):
        n_x = 6

        # Iterate over each time step k from 0 to W-2
        eq_const_fun = []
        for k in range(self.W - 1):
            # Extract x(k) and x(k+1) from the flattened x vector
            x_k = X[
                4 * n_x * k : 4 * n_x * (k + 1), :
            ]  # Extract x_k (4 state variables at time step k)
            x_k_next = X[
                4 * n_x * (k + 1) : 4 * n_x * (k + 2), :
            ]  # Extract x_k+1 (4 state variables at time step k+1)

            # Iterate over each state i = 1, ..., 4
            for i in range(4):
                x_i_k = x_k[n_x * i : n_x * (i + 1)]  # State i at time step k
                x_i_k_next = x_k_next[
                    n_x * i : n_x * (i + 1)
                ]  # State i at time step k+1

                # Compute the constraint: x_i(k+1) - A_i * x_i(k) - b_i
                h_i_k = x_i_k_next - self.dynamic_model.x_new(dt, x_i_k)

                # Append the constraint for this state i at time step k
                eq_const_fun.append(h_i_k)

        # Convert the list of constraint values to a numpy array and flatten it
        return np.concatenate(eq_const_fun)

    def jacobian_eq_const_function(self, dt, X):
        n_x = 6

        # Iterate over each time step k from 0 to W-2
        jacobian_eq_const_fun = np.zeros(
            (4 * n_x * (self.W - 1), 4 * n_x * self.W)
        )  # Jacobian matrix
        for k in range(self.W - 1):
            X_k = X[4 * n_x * k : 4 * n_x * (k + 1)]
            _, F_k = self.dynamic_model.x_new_and_F(dt, X_k)

            # Set the block for the derivative of h(x) w.r.t. x_k
            jacobian_eq_const_fun[
                4 * n_x * k : 4 * n_x * (k + 1), 4 * n_x * k : 4 * n_x * (k + 1)
            ] = -F_k

            # Set the block for the derivative of h(x) w.r.t. x_(k+1)
            jacobian_eq_const_fun[
                4 * n_x * k : 4 * n_x * (k + 1), 4 * n_x * (k + 1) : 4 * n_x * (k + 2)
            ] = np.eye(4 * n_x)

        return jacobian_eq_const_fun

    def lagrangian(self, dt, X, Lambda, Y):
        return self.obj_function(X, Y) + Lambda.T @ self.eq_const_function(dt, X)

    def grad_lagrangian(self, dt, X, Lambda, Y):
        return (
            self.grad_obj_function(X, Y)
            + self.jacobian_eq_const_function(dt, X).T @ Lambda
        )

    def hessian_lagrangian(self, X, Y):
        return self.hessian_obj_function(X, Y)

    def solve_for_each_window(self, dt, x_init, lambda_init, Y):
        n_x = 6
        tolerance = 1e-6
        max_iter = 100
        x = x_init
        lmbda = lambda_init
        self.i = None
        self.L_norms = []
        self.grad_L_norms = []
        self.h_norms = []
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

            # Store the norms
            self.L_norms.append(L_norm)
            self.grad_L_norms.append(grad_L_norm)
            self.h_norms.append(h_norm)

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

            # Save the current iteration
            self.i = iteration + 1

        return x

    def apply(self, dt, X_initial, Y, X_true):
        n_x = 6
        n_p = 3
        K = Y.shape[2]
        rmse_1 = []
        rmse_2 = []
        rmse_3 = []
        rmse_4 = []
        NKKT_first_round_stop_iteration = []
        NKKT_K_minus_W_next_rounds_stop_iteration = []

        # Before applying the Newton algorithm for the first time, initialize the initial conditions
        # guess randomly (cold-start) and the next states depending on the initial condition guess and dynamics
        # so that the KKT primal feasibility condition h(x) = 0 is verified except for the first propagation iteration
        x_init = np.zeros((4 * n_x * self.W, 1))
        x_init[: 4 * n_x, :] = X_initial
        for k in range(self.W - 1):
            x_init[4 * n_x * (k + 1) : 4 * n_x * (k + 2), :] = self.dynamic_model.x_new(
                dt, x_init[4 * n_x * k : 4 * n_x * (k + 1), :]
            )

        p_1_est = np.zeros((n_p, 1, K - self.W + 1))
        p_2_est = np.zeros((n_p, 1, K - self.W + 1))
        p_3_est = np.zeros((n_p, 1, K - self.W + 1))
        p_4_est = np.zeros((n_p, 1, K - self.W + 1))
        x_1_est = np.zeros((n_x, 1, K - self.W + 1))
        x_2_est = np.zeros((n_x, 1, K - self.W + 1))
        x_3_est = np.zeros((n_x, 1, K - self.W + 1))
        x_4_est = np.zeros((n_x, 1, K - self.W + 1))
        for n in range(K - self.W + 1):
            # For the lambdas try to solve the least squares problem that arises from the stationarity KKT condition
            # nabla f + nabla h^T @ lambda = 0
            # this is also part of the warm-start
            lambda_init = np.linalg.lstsq(
                self.eq_const_function(dt, x_init).T,
                -self.obj_function(x_init, Y[:, :, n : n + self.W]).flatten(),
                rcond=None,
            )[0].reshape(-1, 1)
            x_est = self.solve_for_each_window(
                dt, x_init, lambda_init, Y[:, :, n : n + self.W]
            )
            p_1_est[:, :, n] = self.P @ x_est[:n_x, :]
            p_2_est[:, :, n] = self.P @ x_est[n_x : 2 * n_x, :]
            p_3_est[:, :, n] = self.P @ x_est[2 * n_x : 3 * n_x, :]
            p_4_est[:, :, n] = self.P @ x_est[3 * n_x : 4 * n_x, :]
            x_1_est[:, :, n] = x_est[:n_x, :]
            x_2_est[:, :, n] = x_est[n_x : 2 * n_x, :]
            x_3_est[:, :, n] = x_est[2 * n_x : 3 * n_x, :]
            x_4_est[:, :, n] = x_est[3 * n_x : 4 * n_x, :]

            # give_me_the_plots(nkkt.i, nkkt.L_norms, nkkt.grad_L_norms, nkkt.h_norms)

            if n == 0:  # Check divergence in the beginning
                abs_error_init_1 = np.zeros(self.W)
                abs_error_init_2 = np.zeros(self.W)
                abs_error_init_3 = np.zeros(self.W)
                abs_error_init_4 = np.zeros(self.W)
                for k in range(self.W):
                    abs_error_init_1[k] = np.linalg.norm(
                        self.P @ x_est[n_x * (4 * k) : n_x * (4 * k + 1), :]
                        - self.P @ X_true[:n_x, :, k]
                    )
                    abs_error_init_2[k] = np.linalg.norm(
                        self.P @ x_est[n_x * (4 * k + 1) : n_x * (4 * k + 2), :]
                        - self.P @ X_true[n_x : 2 * n_x, :, k]
                    )
                    abs_error_init_3[k] = np.linalg.norm(
                        self.P @ x_est[n_x * (4 * k + 2) : n_x * (4 * k + 3), :]
                        - self.P @ X_true[2 * n_x : 3 * n_x, :, k]
                    )
                    abs_error_init_4[k] = np.linalg.norm(
                        self.P @ x_est[n_x * (4 * k + 3) : n_x * (4 * k + 4), :]
                        - self.P @ X_true[3 * n_x : 4 * n_x, :, k]
                    )
                rmse_init_1 = np.sqrt(np.mean(abs_error_init_1**2))
                rmse_init_2 = np.sqrt(np.mean(abs_error_init_2**2))
                rmse_init_3 = np.sqrt(np.mean(abs_error_init_3**2))
                rmse_init_4 = np.sqrt(np.mean(abs_error_init_4**2))
                if (
                    rmse_init_1 > 1e2
                    or rmse_init_2 > 1e2
                    or rmse_init_3 > 1e2
                    or rmse_init_4 > 1e2
                ):
                    print(f"This Monte Carlo run diverged!")
                    # Mimic the propagation for rmse computation below
                    for k in range(K - self.W + 1):
                        p_1_est[:, :, k] = self.P @ self.dynamic_model.x_new(
                            dt, x_est[:n_x, :]
                        )
                        p_2_est[:, :, k] = self.P @ self.dynamic_model.x_new(
                            dt, x_est[n_x : 2 * n_x, :]
                        )
                        p_3_est[:, :, k] = self.P @ self.dynamic_model.x_new(
                            dt, x_est[2 * n_x : 3 * n_x, :]
                        )
                        p_4_est[:, :, k] = self.P @ self.dynamic_model.x_new(
                            dt, x_est[3 * n_x : 4 * n_x, :]
                        )
                    break
                else:
                    NKKT_first_round_stop_iteration.append(self.i)
            else:
                NKKT_K_minus_W_next_rounds_stop_iteration.append(self.i)

            # Get next new guess (warm-start)
            # The initial guess is the second timestamp of the previous window and then propagate the states
            # This approach is better to faster verify the KKT condition h(x) = 0
            x_init[: 4 * n_x, :] = x_est[4 * n_x : 8 * n_x, :]
            for k in range(self.W - 1):
                x_init[4 * n_x * (k + 1) : 4 * n_x * (k + 2), :] = (
                    self.dynamic_model.x_new(
                        dt, x_init[4 * n_x * k : 4 * n_x * (k + 1), :]
                    )
                )

        # After each Monte Carlo Run compute the RMSE_m for each first K - W + 1 iterations
        abs_error_1 = np.zeros(K - self.W + 1)
        abs_error_2 = np.zeros(K - self.W + 1)
        abs_error_3 = np.zeros(K - self.W + 1)
        abs_error_4 = np.zeros(K - self.W + 1)
        for k in range(K - self.W + 1):
            abs_error_1[k] = np.linalg.norm(
                p_1_est[:, :, k] - self.P @ X_true[:n_x, :, k]
            )
            abs_error_2[k] = np.linalg.norm(
                p_2_est[:, :, k] - self.P @ X_true[n_x : 2 * n_x, :, k]
            )
            abs_error_3[k] = np.linalg.norm(
                p_3_est[:, :, k] - self.P @ X_true[2 * n_x : 3 * n_x, :, k]
            )
            abs_error_4[k] = np.linalg.norm(
                p_4_est[:, :, k] - self.P @ X_true[3 * n_x : 4 * n_x, :, k]
            )
        rmse_m_1 = np.sqrt(np.mean(abs_error_1**2))
        rmse_m_2 = np.sqrt(np.mean(abs_error_2**2))
        rmse_m_3 = np.sqrt(np.mean(abs_error_3**2))
        rmse_m_4 = np.sqrt(np.mean(abs_error_4**2))
        print(f"This MC run RMSE for the first {K - self.W + 1} iterations:")
        print(f"RMSE_run?_1 = {rmse_m_1}")
        print(f"RMSE_run?_2 = {rmse_m_2}")
        print(f"RMSE_run?_3 = {rmse_m_3}")
        print(f"RMSE_run?_4 = {rmse_m_4}\n")
        if rmse_m_1 > 0.5 or rmse_m_2 > 0.5 or rmse_m_3 > 0.5 or rmse_m_4 > 0.5:
            print(
                "Discarding this Monte Carlo run and going to the next Monte Carlo run...\n"
            )
        else:
            rmse_1.append(rmse_m_1)
            rmse_2.append(rmse_m_2)
            rmse_3.append(rmse_m_3)
            rmse_4.append(rmse_m_4)

        print(
            f"Average RMSE for the first {K - self.W + 1} iterations for {len(rmse_1)} valid Monte Carlo runs:"
        )
        print(f"RMSE_1 = {np.mean(rmse_1)}")
        print(f"RMSE_2 = {np.mean(rmse_2)}")
        print(f"RMSE_3 = {np.mean(rmse_3)}")
        print(f"RMSE_4 = {np.mean(rmse_4)}")
        print(
            f"The first round of NKKT converged with {np.mean(NKKT_first_round_stop_iteration)} iterations on average."
        )
        print(
            f"The next {K - self.W} rounds of NKKT converged with {np.mean(NKKT_K_minus_W_next_rounds_stop_iteration)} iterations on average."
        )
        return np.concatenate(x_1_est, x_2_est, x_3_est, x_4_est, axis=1)
