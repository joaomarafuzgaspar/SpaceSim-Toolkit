import numpy as np

from scipy.linalg import solve
from dynamics import SatelliteDynamics
from config import SimulationConfig as config


class Hybrid:
    def __init__(self, H, K, o, R, grad_norm_order_mag, grad_norm_tol, max_iterations):   
        # Simulation parameters
        self.H = H
        self.K = K
        self.o = o
        self.R = R
        self.dyn = SatelliteDynamics()
        
        # Define state to position transformation matrix
        self.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
             
        # Stopping criteria
        self.grad_norm_order_mag = grad_norm_order_mag
        self.grad_norm_tol = grad_norm_tol
        self.max_iterations = max_iterations
        
        # Storage for results
        self.iterations = None
        self.cost_values = []
        self.gradient_norm_values = []
        
        # Retrieve observation noise covariance matrix (chief) and standard deviation (deputies)
        self.R_chief = self.R[:config.n_p, :config.n_p]
        self.r_deputy_pos = np.sqrt(self.R[-1, -1])

    def h_function_chief(self, x_vec):
        return x_vec[0:3]

    def h_function_deputy(self, x_vec):
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

    def J(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        n_x = 6
        n_y_1 = 3
        f_x_0 = 0

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_vec
        x_1_tau = x_vec[:n_x, :]
        x_2_tau = x_vec[n_x : 2 * n_x, :]
        x_3_tau = x_vec[2 * n_x : 3 * n_x, :]
        x_4_tau = x_vec[3 * n_x : 4 * n_x, :]

        # Iterate over all sliding window time steps
        for tau in range(k - self.H + 1, k + 1):
            # Absolute residual term: observed data y for each state
            y_1_tau = Y[:n_y_1, :, tau]
            y_rel_tau = Y[n_y_1:, :, tau]

            # Update the cost function with the residuals for self-measurements
            residual = y_1_tau - self.P @ x_1_tau
            f_x_0 += 1 / 2 * residual.T @ np.linalg.inv(self.R_chief) @ residual

            # Pairwise relative measurements
            y_21_tau = y_rel_tau[0]
            y_23_tau = y_rel_tau[1]
            y_24_tau = y_rel_tau[2]
            y_31_tau = y_rel_tau[3]
            y_34_tau = y_rel_tau[4]
            y_41_tau = y_rel_tau[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_tau, x_1_tau, y_21_tau),  # (2, 1)
                (2, 3, x_2_tau, x_3_tau, y_23_tau),  # (2, 3)
                (2, 4, x_2_tau, x_4_tau, y_24_tau),  # (2, 4)
                (3, 1, x_3_tau, x_1_tau, y_31_tau),  # (3, 1)
                (3, 4, x_3_tau, x_4_tau, y_34_tau),  # (3, 4)
                (4, 1, x_4_tau, x_1_tau, y_41_tau),  # (4, 1)
            ]

            # Iterate over each pair and update the cost function with the residuals for pairwise measurements
            for i, j, x_i_tau, x_j_tau, y_ij_tau in pairs:
                d_ij_tau_vec = self.P @ x_i_tau - self.P @ x_j_tau
                d_ij_tau = np.linalg.norm(d_ij_tau_vec)
                f_x_0 += (y_ij_tau - d_ij_tau) ** 2 / (2 * self.r_deputy_pos**2)

            # Get x_1(tau), x_2(tau), x_3(tau), x_4(tau) from the state vector x_vec
            x_1_tau = self.dyn.x_new(dt, x_1_tau)
            x_2_tau = self.dyn.x_new(dt, x_2_tau)
            x_3_tau = self.dyn.x_new(dt, x_3_tau)
            x_4_tau = self.dyn.x_new(dt, x_4_tau)

        return f_x_0

    def DJ(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        n_x = 6
        n_y_1 = 3
        grad_f_x_0 = np.zeros_like(x_vec)

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_vec
        x_1_tau = x_vec[:n_x, :]
        x_2_tau = x_vec[n_x : 2 * n_x, :]
        x_3_tau = x_vec[2 * n_x : 3 * n_x, :]
        x_4_tau = x_vec[3 * n_x : 4 * n_x, :]
        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for tau in range(k - self.H + 1, k + 1):
            # Absolute residual term: observed data y for each state
            y_1_tau = Y[:n_y_1, :, tau]
            y_rel_tau = Y[n_y_1:, :, tau]

            # Compute gradients for the absolute residual terms
            grad_f_x_0[:n_x, :] -= (
                STM_t0_1.T
                @ self.P.T
                @ np.linalg.inv(self.R_chief)
                @ (y_1_tau - self.P @ x_1_tau)
            )

            # Pairwise relative measurements
            y_21_tau = y_rel_tau[0]
            y_23_tau = y_rel_tau[1]
            y_24_tau = y_rel_tau[2]
            y_31_tau = y_rel_tau[3]
            y_34_tau = y_rel_tau[4]
            y_41_tau = y_rel_tau[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_tau, x_1_tau, y_21_tau, STM_t0_2, STM_t0_1),  # (2, 1)
                (2, 3, x_2_tau, x_3_tau, y_23_tau, STM_t0_2, STM_t0_3),  # (2, 3)
                (2, 4, x_2_tau, x_4_tau, y_24_tau, STM_t0_2, STM_t0_4),  # (2, 4)
                (3, 1, x_3_tau, x_1_tau, y_31_tau, STM_t0_3, STM_t0_1),  # (3, 1)
                (3, 4, x_3_tau, x_4_tau, y_34_tau, STM_t0_3, STM_t0_4),  # (3, 4)
                (4, 1, x_4_tau, x_1_tau, y_41_tau, STM_t0_4, STM_t0_1),  # (4, 1)
            ]

            # Iterate over each pair and compute gradients
            for i, j, x_i_tau, x_j_tau, y_ij_tau, STM_t0_i, STM_t0_j in pairs:
                d_ij_tau_vec = self.P @ x_i_tau - self.P @ x_j_tau
                d_ij_tau = np.linalg.norm(d_ij_tau_vec)

                # Update the gradient of f(x) with respect to x_i(tau) and x_j(tau)
                grad_f_x_0[(i - 1) * n_x : i * n_x, :] -= (
                    (y_ij_tau - d_ij_tau)
                    / self.r_deputy_pos**2
                    * (STM_t0_i.T @ self.P.T @ d_ij_tau_vec)
                    / d_ij_tau
                )
                grad_f_x_0[(j - 1) * n_x : j * n_x, :] += (
                    (y_ij_tau - d_ij_tau)
                    / self.r_deputy_pos**2
                    * (STM_t0_j.T @ self.P.T @ d_ij_tau_vec)
                    / d_ij_tau
                )

            # Get x_1(tau), x_2(tau), x_3(tau), x_4(tau) from the state vector x_vec
            STM_t0_1_old = STM_t0_1
            STM_t0_2_old = STM_t0_2
            STM_t0_3_old = STM_t0_3
            STM_t0_4_old = STM_t0_4
            x_1_tau, STM_t0_1 = self.dyn.x_new_and_F(dt, x_1_tau)
            x_2_tau, STM_t0_2 = self.dyn.x_new_and_F(dt, x_2_tau)
            x_3_tau, STM_t0_3 = self.dyn.x_new_and_F(dt, x_3_tau)
            x_4_tau, STM_t0_4 = self.dyn.x_new_and_F(dt, x_4_tau)
            STM_t0_1 = STM_t0_1 @ STM_t0_1_old
            STM_t0_2 = STM_t0_2 @ STM_t0_2_old
            STM_t0_3 = STM_t0_3 @ STM_t0_3_old
            STM_t0_4 = STM_t0_4 @ STM_t0_4_old

        return grad_f_x_0

    def HJ(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        n_x = 6
        n_y_1 = 3
        hessian_f_x_0 = np.zeros(
            (x_vec.shape[0], x_vec.shape[0])
        )  # Initialize Hessian matrix

        # Extract x_1(0), x_2(0), x_3(0), x_4(0) from flattened state vector x_vec
        x_1_tau = x_vec[:n_x, :]
        x_2_tau = x_vec[n_x : 2 * n_x, :]
        x_3_tau = x_vec[2 * n_x : 3 * n_x, :]
        x_4_tau = x_vec[3 * n_x : 4 * n_x, :]
        STM_t0_1 = np.eye(n_x)
        STM_t0_2 = np.eye(n_x)
        STM_t0_3 = np.eye(n_x)
        STM_t0_4 = np.eye(n_x)

        # Iterate over all sliding window time steps
        for tau in range(k - self.H + 1, k + 1):
            # Absolute residual term: observed data y for each state
            y_rel_tau = Y[n_y_1:, :, tau]

            # Absolute measurement residuals' Hessian
            hessian_f_x_0[:n_x, :n_x] += (
                STM_t0_1.T @ self.P.T @ np.linalg.inv(self.R_chief) @ self.P @ STM_t0_1
            )

            # Pairwise relative measurements
            y_21_tau = y_rel_tau[0]
            y_23_tau = y_rel_tau[1]
            y_24_tau = y_rel_tau[2]
            y_31_tau = y_rel_tau[3]
            y_34_tau = y_rel_tau[4]
            y_41_tau = y_rel_tau[5]

            # Process pairwise relative residuals for each pair
            pairs = [
                (2, 1, x_2_tau, x_1_tau, y_21_tau, STM_t0_2, STM_t0_1),  # (2, 1)
                (2, 3, x_2_tau, x_3_tau, y_23_tau, STM_t0_2, STM_t0_3),  # (2, 3)
                (2, 4, x_2_tau, x_4_tau, y_24_tau, STM_t0_2, STM_t0_4),  # (2, 4)
                (3, 1, x_3_tau, x_1_tau, y_31_tau, STM_t0_3, STM_t0_1),  # (3, 1)
                (3, 4, x_3_tau, x_4_tau, y_34_tau, STM_t0_3, STM_t0_4),  # (3, 4)
                (4, 1, x_4_tau, x_1_tau, y_41_tau, STM_t0_4, STM_t0_1),  # (4, 1)
            ]

            # Iterate over each pair and compute gradients
            for i, j, x_i_tau, x_j_tau, y_ij_tau, STM_t0_i, STM_t0_j in pairs:
                d_ij_tau_vec = self.P @ x_i_tau - self.P @ x_j_tau
                d_ij_tau = np.linalg.norm(d_ij_tau_vec)

                # Update the Hessian for x_i(tau) and x_j(tau)
                idx_i = n_x * (i - 1)
                idx_j = n_x * (j - 1)

                hessian_f_x_0[idx_i : idx_i + n_x, idx_i : idx_i + n_x] -= (
                    1
                    / (self.r_deputy_pos**2)
                    * (
                        STM_t0_i.T
                        @ (
                            (y_ij_tau - d_ij_tau) / d_ij_tau * self.P.T @ self.P
                            - y_ij_tau
                            / d_ij_tau**3
                            * self.P.T
                            @ d_ij_tau_vec
                            @ d_ij_tau_vec.T
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
                            (y_ij_tau - d_ij_tau) / d_ij_tau * self.P.T @ self.P
                            - y_ij_tau
                            / d_ij_tau**3
                            * self.P.T
                            @ d_ij_tau_vec
                            @ d_ij_tau_vec.T
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
                            (y_ij_tau - d_ij_tau) / d_ij_tau * self.P.T @ self.P
                            - y_ij_tau
                            / d_ij_tau**3
                            * self.P.T
                            @ d_ij_tau_vec
                            @ d_ij_tau_vec.T
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
                            (y_ij_tau - d_ij_tau) / d_ij_tau * self.P.T @ self.P
                            - y_ij_tau
                            / d_ij_tau**3
                            * self.P.T
                            @ d_ij_tau_vec
                            @ d_ij_tau_vec.T
                            @ self.P
                        )
                        @ STM_t0_j
                    )
                )

            # Get x_1(tau), x_2(tau), x_3(tau), x_4(tau) from the state vector x_vec
            STM_t0_1_old = STM_t0_1
            STM_t0_2_old = STM_t0_2
            STM_t0_3_old = STM_t0_3
            STM_t0_4_old = STM_t0_4
            x_1_tau, STM_t0_1 = self.dyn.x_new_and_F(dt, x_1_tau)
            x_2_tau, STM_t0_2 = self.dyn.x_new_and_F(dt, x_2_tau)
            x_3_tau, STM_t0_3 = self.dyn.x_new_and_F(dt, x_3_tau)
            x_4_tau, STM_t0_4 = self.dyn.x_new_and_F(dt, x_4_tau)
            STM_t0_1 = STM_t0_1 @ STM_t0_1_old
            STM_t0_2 = STM_t0_2 @ STM_t0_2_old
            STM_t0_3 = STM_t0_3 @ STM_t0_3_old
            STM_t0_4 = STM_t0_4 @ STM_t0_4_old

        return hessian_f_x_0

    def solve_MHE_problem(self, k, dt, Y, x_init, x_true_initial, x_true_end):
        x = x_init.copy()

        prev_cost_value = None
        prev_gradient_norm_value = None
        prev_global_estimation_error = None
        grad_norm_order_history = []

        for iteration in range(self.max_iterations + 1):
            # Compute the cost function, gradient of the Lagrangian and Hessian of the Lagrangian
            J_x = self.J(k, dt, Y, x)
            DJ_x = self.DJ(k, dt, Y, x)
            HJ_x = self.HJ(k, dt, Y, x)

            # Convergence tracking
            cost_value = J_x[0][0]
            gradient_norm_value = np.linalg.norm(DJ_x)

            # Store the values
            self.cost_values.append(cost_value)
            self.gradient_norm_values.append(gradient_norm_value)

            # Metrics
            if prev_cost_value is not None:
                cost_value_change = (cost_value - prev_cost_value) / abs(prev_cost_value) * 100
                gradient_norm_value_change = (gradient_norm_value - prev_gradient_norm_value) / abs(prev_gradient_norm_value) * 100
                global_estimation_error_change = (np.linalg.norm(x - x_true_initial) - prev_global_estimation_error) / abs(prev_global_estimation_error) * 100
            prev_cost_value = cost_value
            prev_gradient_norm_value = gradient_norm_value
            prev_global_estimation_error = np.linalg.norm(x - x_true_initial)
            
            # Track gradient norm order of magnitude
            current_order = int(np.floor(np.log10(gradient_norm_value + 1e-12)))  # avoid log(0)
            grad_norm_order_history.append(current_order)

            if self.grad_norm_order_mag:
                if len(grad_norm_order_history) >= 3:
                    if grad_norm_order_history[-1] == grad_norm_order_history[-2] == grad_norm_order_history[-3]:
                        stagnant_order = True
                        if k == self.H - 1:
                            stagnant_order = False
                    else:
                        stagnant_order = False
                else:
                    stagnant_order = False
            else:
                stagnant_order = False
                
            # Propagate window initial conditions for metrics 
            x_end = x.copy()
            for _ in range(self.H - 1):
                x_end = self.dyn.x_new(dt, x_end)

            # Check convergence and print metrics
            if gradient_norm_value < self.grad_norm_tol or iteration == self.max_iterations or stagnant_order:
                reason = "tolerance reached" if gradient_norm_value < self.grad_norm_tol else \
                        "max iteration reached" if iteration == self.max_iterations else \
                        "gradient norm stagnated"
                print(f"[Hybrid] STOP on Iteration {iteration} ({reason})")
                print(f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                print(f"Final initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m")
                print(f"Final position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n")
                break
            else:
                if iteration == 0:
                    print(f"[Hybrid] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}")
                else:
                    print(f"[Hybrid] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                    
            # Print estimation errors 
            print(f"Initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m")
            print(f"Position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n")
                
            # Solve for the Newton step - this is one iteration
            delta_x = solve(HJ_x, -DJ_x)
            x += delta_x
        
            # Save the current iteration
            self.iterations = iteration + 1
            
        # Propagate window initial conditions getting estimate at timestamp k
        x_init = x
        for _ in range(self.H - 1):
            x = self.dyn.x_new(dt, x)

        return x_init, x