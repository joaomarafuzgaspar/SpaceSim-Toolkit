# src/inewton.py
import numpy as np


from scipy.linalg import solve


from dynamics import Dynamics
from config import SimulationConfig as config


class iNewton:
    def __init__(self):
        # Simulation parameters
        self.H = config.H
        self.K = config.K
        self.dt = config.dt
        self.o = config.o
        self.R = config.R
        self.dyn = Dynamics()

        # Stopping criteria
        self.grad_norm_order_mag = config.grad_norm_order_mag
        self.grad_norm_tol = config.grad_norm_tol
        self.max_iterations = config.max_iterations
        self.iterative_loop_max_iterations = config.iterative_loop_max_iterations

        # Storage for results
        self.iterations = None
        self.cost_values = []
        self.gradient_norm_values = []
        self.grad_norm_order_history = []
        self.HJ_x_eigenvalues_history = []
        self.A_norm_history, self.B_norm_history, self.C_norm_history = [], [], []

        # Observation model
        self.h = config.h
        self.Dh = config.Dh
        self.Hh = config.Hh

    def J(self, k, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        J_x = 0
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            h_x = self.h(x_vec)
            J_x += 1 / 2 * (y - h_x).T @ R_inv @ (y - h_x)
            x_vec = self.dyn.f(self.dt, x_vec)
        return J_x

    def DJ(self, k, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(config.n)
        DJ_x = np.zeros((config.n, 1))
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            DJ_x += -STM.T @ self.Dh(x_vec).T @ R_inv @ (y - self.h(x_vec))
            STM = self.dyn.Df(self.dt, x_vec) @ STM
            x_vec = self.dyn.f(self.dt, x_vec)
        return DJ_x

    def HJ(self, k, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(config.n)
        DSTM = np.zeros((config.n * config.n, config.n))
        HJ_x = np.zeros((config.n, config.n))
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            h_x = self.h(x_vec)
            Dh_x = self.Dh(x_vec)
            Hh_x = self.Hh(x_vec)
            Df_x = STM
            Hf_x = DSTM
            HJ_x += (
                -(
                    np.kron(R_inv @ (y - h_x), Df_x).T @ Hh_x @ Df_x
                    + np.kron(Dh_x.T @ R_inv @ (y - h_x), np.eye(config.n)).T @ Hf_x
                )
                + Df_x.T @ Dh_x.T @ R_inv @ Dh_x @ Df_x
            )
            DSTM = (
                np.kron(np.eye(config.n), STM).T @ self.dyn.Hf(self.dt, x_vec) @ STM
                + np.kron(self.dyn.Df(self.dt, x_vec), np.eye(config.n)) @ DSTM
            )
            STM = self.dyn.Df(self.dt, x_vec) @ STM
            x_vec = self.dyn.f(self.dt, x_vec)
        return HJ_x

    def HJ_GN(self, k, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(config.n)
        HJ_x = np.zeros((config.n, config.n))
        for tau in range(k - self.H + 1, k + 1):
            Dh_x = self.Dh(x_vec)
            Df_x = STM
            HJ_x += Df_x.T @ Dh_x.T @ R_inv @ Dh_x @ Df_x
            STM = self.dyn.Df(self.dt, x_vec) @ STM
            x_vec = self.dyn.f(self.dt, x_vec)
        return HJ_x

    def solve_MHE_problem(self, k, Y, x_init, x_true_initial, x_true_end):
        x = x_init.copy()

        prev_cost_value = None
        prev_gradient_norm_value = None
        prev_global_estimation_error = None
        grad_norm_order_history = []

        for iteration in range(self.max_iterations + 1):
            # Compute the cost function, gradient of the Lagrangian and Hessian of the Lagrangian
            J_x = self.J(k, Y, x)
            DJ_x = self.DJ(k, Y, x)
            HJ_x = self.HJ_GN(k, Y, x)

            # Convergence tracking
            cost_value = J_x[0][0]
            gradient_norm_value = np.linalg.norm(DJ_x)

            # Store the values
            self.cost_values.append(cost_value)
            self.gradient_norm_values.append(gradient_norm_value)

            # Metrics
            if prev_cost_value is not None:
                cost_value_change = (
                    (cost_value - prev_cost_value) / abs(prev_cost_value) * 100
                )
                gradient_norm_value_change = (
                    (gradient_norm_value - prev_gradient_norm_value)
                    / abs(prev_gradient_norm_value)
                    * 100
                )
                global_estimation_error_change = (
                    (np.linalg.norm(x - x_true_initial) - prev_global_estimation_error)
                    / abs(prev_global_estimation_error)
                    * 100
                )
            prev_cost_value = cost_value
            prev_gradient_norm_value = gradient_norm_value
            prev_global_estimation_error = np.linalg.norm(x - x_true_initial)

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
                x_end = self.dyn.f(self.dt, x_end)

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
                print(f"[Newton] STOP on Iteration {iteration} ({reason})")
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
                        f"[Newton] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}"
                    )
                else:
                    print(
                        f"[Newton] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)"
                    )

            # Print estimation errors
            print(
                f"Initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m"
            )
            print(
                f"Position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n"
            )

            # Solve for the Newton step - this is one iteration
            A_tree_plus_residual_diag = np.zeros_like(HJ_x)
            for i in range(config.N):
                for j in range(config.N):
                    block = HJ_x[
                        i * config.n_x : (i + 1) * config.n_x,
                        j * config.n_x : (j + 1) * config.n_x,
                    ]
                    if i == j or i == 0 or j == 0:
                        A_tree_plus_residual_diag[
                            i * config.n_x : (i + 1) * config.n_x,
                            j * config.n_x : (j + 1) * config.n_x,
                        ] = block
            A_residual_off_diag = HJ_x - A_tree_plus_residual_diag

            # Fixed point iteration
            print(
                f"Spectral radius before going into inner-loop: {np.max(np.abs(np.linalg.eigvals(solve(A_tree_plus_residual_diag, A_residual_off_diag))))}"
            )
            delta_sol = solve(HJ_x, -DJ_x)
            delta_x = np.zeros_like(x)
            for inner_loop_iteration in range(self.iterative_loop_max_iterations):
                delta_x = solve(
                    A_tree_plus_residual_diag, -(A_residual_off_diag @ delta_x + DJ_x)
                )
                if inner_loop_iteration % 10 == 0:
                    print(
                        f"[Inner-loop] Iteration {inner_loop_iteration + 1} | Relative error: {np.linalg.norm(delta_x - delta_sol) / np.linalg.norm(delta_sol) * 100:.2f}%"
                    )
            x += delta_x

            # Save the current iteration
            self.iterations = iteration + 1

        # Propagate window initial conditions getting estimate at timestamp k
        x_init = x
        for _ in range(self.H - 1):
            x = self.dyn.f(self.dt, x)

        return x_init, x
