import numpy as np

from tqdm import tqdm
from scipy.linalg import solve
from dynamics import SatelliteDynamics


class UNKKT:
    """
    This class implements the unconstrained version of the Karush-Kuhn-Tucker (KKT)
    conditions using the Newton method for optimization in satellite dynamics.
    """

    def __init__(self, W, R_chief, r_deputy_pos):
        """
        Initialize the UNKKT class.

        Parameters:
        W (int): Sliding window size for optimization.
        R_chief (np.array): Measurement noise covariance matrix for the chief satellite.
        r_deputy_pos (float): Measurement noise standard deviation for deputy positions.
        """
        # Define window size
        self.W = W

        # Define noise covariances
        self.R_chief = R_chief  # Process noise covariance
        self.r_deputy_pos = r_deputy_pos

        # Define state to position transformation matrix
        self.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

        self.dynamic_model = SatelliteDynamics()
        
        # Store the cost function and gradient norm values
        self.cost_function_values = []
        self.grad_norm_values = []

    def h_function_chief(self, x_vec):
        """
        Compute the measurement vector for the chief satellite's position.

        Parameters:
        x_vec (np.array): Current state vector of the satellite.

        Returns:
        np.array: Position components of the state vector.
        """
        return x_vec[0:3]

    def H_jacobian_chief(self):
        """
        Compute the Jacobian of the measurement function for the chief satellite.

        Returns:
        np.array: Jacobian matrix for the chief satellite measurements.
        """
        H = np.zeros((3, 24))
        H[0:3, 0:3] = np.eye(3)
        return H

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

    def H_jacobian_deputy(self, x_vec):
        """
        Compute the Jacobian of the measurement function for relative distances between satellites.

        Parameters:
        x_vec (np.array): Current state vector of all satellites.

        Returns:
        np.array: Jacobian matrix for relative distances.
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
        Combine measurement functions for the chief and deputies.

        Parameters:
        x_vec (np.array): State vector of all satellites.

        Returns:
        np.array: Combined measurement vector.
        """
        return np.concatenate(
            [self.h_function_chief(x_vec), self.h_function_deputy(x_vec)]
        )

    def H(self, x_vec):
        """
        Combine Jacobians for both the chief and deputies.

        Parameters:
        x_vec (np.array): State vector of all satellites.

        Returns:
        np.array: Combined Jacobian matrix.
        """
        return np.concatenate((self.H_jacobian_chief(), self.H_jacobian_deputy(x_vec)))

    def obj_function(self, dt, x_0, y):
        """
        Compute the objective function for the optimization problem.

        Parameters:
        dt (float): Time step.
        x_0 (np.array): Initial state vector.
        y (np.array): Measurement data over the time window.

        Returns:
        float: Value of the objective function.
        """
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

        return f_x_0

    def grad_obj_function(self, dt, x_0, y):
        """
        Compute the gradient of the objective function.

        Parameters:
        dt (float): Time step.
        x_0 (np.array): Initial state vector.
        y (np.array): Measurement data over the time window.

        Returns:
        np.array: Gradient of the objective function.
        """
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

        return grad_f_x_0

    def hessian_obj_function(self, dt, x_0, y):
        """
        Compute the Hessian of the objective function.

        Parameters:
        dt (float): Time step.
        x_0 (np.array): Initial state vector.
        y (np.array): Measurement data over the time window.

        Returns:
        np.array: Hessian matrix of the objective function.
        """
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

        return hessian_f_x_0

    def lagrangian(self, dt, X, Y):
        """
        Compute the Lagrangian (equivalent to the objective function for unconstrained optimization).

        Parameters:
        dt (float): Time step.
        X (np.array): State vector over the time window.
        Y (np.array): Measurement data.

        Returns:
        float: Value of the Lagrangian.
        """
        return self.obj_function(dt, X, Y)

    def grad_lagrangian(self, dt, X, Y):
        """
        Compute the gradient of the Lagrangian.

        Parameters:
        dt (float): Time step.
        X (np.array): State vector over the time window.
        Y (np.array): Measurement data.
        
        Returns:
        np.array: Gradient of the Lagrangian.
        """
        return self.grad_obj_function(dt, X, Y)

    def hessian_lagrangian(self, dt, X, Y):
        """
        Compute the Hessian of the Lagrangian.

        Parameters:
        dt (float): Time step.
        X (np.array): State vector over the time window.
        Y (np.array): Measurement data.
        
        Returns:
        np.array: Hessian of the Lagrangian.
        """
        return self.hessian_obj_function(dt, X, Y)

    def solve_for_each_window(self, dt, x_init, Y):
        """
        Solve the optimization problem for a single sliding window using Newton's method.

        Parameters:
        dt (float): Time step.
        x_init (np.array): Initial state guess.
        Y (np.array): Measurement data for the sliding window.
        
        Returns:
        np.array: Optimized state vector for the sliding window.
        """
        n_x = 6
        tolerance = 1e0
        max_iter = 20
        x = x_init
        for iteration in range(max_iter):
            # Compute the cost function, gradient of the Lagrangian and Hessian of the Lagrangian
            L_x = self.lagrangian(dt, x, Y)
            grad_L_x = self.grad_lagrangian(dt, x, Y)
            hessian_L_x = self.hessian_lagrangian(dt, x, Y)

            # Calculate norms for convergence tracking
            L_norm = np.linalg.norm(L_x)
            grad_L_norm = np.linalg.norm(grad_L_x)
            
            # Store the norms
            self.cost_function_values.append(L_x[0][0])
            self.grad_norm_values.append(grad_L_norm)

            # Check convergence and print metrics
            if grad_L_norm < tolerance or iteration + 1 == max_iter:
                print(
                    f"STOP on Iteration {iteration}\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\n"
                )
                break
            else:
                if iteration == 0:
                    print(
                        f"Before applying the algorithm\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\n"
                    )
                else:
                    print(
                        f"Iteration {iteration}\nL_norm = {L_norm}\nGrad_L_norm = {grad_L_norm}\n"
                    )

            # Solve for the Newton step - this is one iteration
            delta_x = solve(hessian_L_x, -grad_L_x)
            x += delta_x

        return x

    def apply(self, dt, X_initial, Y):
        """
        Apply the UNKKT method across all sliding windows.

        Parameters:
        dt (float): Time step.
        X_initial (np.array): Initial guess of the state vector.
        Y (np.array): Measurement data.
        
        Returns:
        np.array: Estimated state vector over all time steps.
        """
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
            x_est = self.solve_for_each_window(dt, x_init, Y[:, :, n : n + self.W])
            X_est[:, :, n] = x_est[: 4 * n_x, :]

            # Get next new guess (warm-start)
            # The initial guess is the previous window estimate second timestamp value
            x_init = self.dynamic_model.x_new(dt, x_est)

        return X_est
