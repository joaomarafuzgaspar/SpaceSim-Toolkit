# src/lm.py
import numpy as np


from scipy.linalg import fractional_matrix_power


from dynamics import Dynamics
from config import SimulationConfig as config


class LM:
    def __init__(self, Q):
        # Dynamics model
        self.Q_invsqrt = fractional_matrix_power(Q, -1 / 2)
        self.dyn = Dynamics()

        # Observation model
        self.R_invsqrt = fractional_matrix_power(config.R, -1 / 2)
        self.h = config.h
        self.Dh = config.Dh

        # Storage for results
        self.iterations = None
        self.cost_values = []
        self.gradient_norm_values = []

    def r(self, X, Y):
        r_z = np.zeros(((config.n + config.o) * config.H, 1))
        # Process noise
        for tau in range(config.H - 1):
            X_tau = X[:, :, tau]
            X_tau_plus_1 = X[:, :, tau + 1]
            r_z[
                (config.n + config.o) * tau : (config.n + config.o) * tau + config.n, :
            ] = self.Q_invsqrt @ (X_tau_plus_1 - self.dyn.f(config.dt, X_tau))
        # Observation noise
        for tau in range(config.H):
            X_tau = X[:, :, tau]
            Y_k = Y[:, :, tau]
            r_z[
                (config.n + config.o) * tau
                + config.n : (config.n + config.o) * (tau + 1),
                :,
            ] = self.R_invsqrt @ (Y_k - self.h(X_tau))
        return r_z

    def J(self, X):
        J_z = np.zeros(((config.n + config.o) * config.H, config.n * config.H))
        for tau in range(config.H):
            X_tau = X[:, :, tau]
            Df_X_tau = self.dyn.Df(config.dt, X_tau)
            if tau < config.H - 1:
                J_z[
                    (config.n + config.o) * tau : (config.n + config.o) * tau
                    + config.n,
                    config.n * tau : config.n * (tau + 1),
                ] = (
                    -self.Q_invsqrt @ Df_X_tau
                )
                J_z[
                    (config.n + config.o) * tau : (config.n + config.o) * tau
                    + config.n,
                    config.n * (tau + 1) : config.n * (tau + 2),
                ] = self.Q_invsqrt
            J_z[
                (config.n + config.o) * tau
                + config.n : (config.n + config.o) * (tau + 1),
                config.n * tau : config.n * (tau + 1),
            ] = -self.R_invsqrt @ self.Dh(X_tau)
        return J_z

    def cost_function(self, X, Y):
        return 1 / 2 * np.linalg.norm(self.r(X, Y)) ** 2

    def solve_MHE_problem(self, k, Y, X_init, X_true):
        if k < config.H - 1 or k + 1 > config.K:
            raise ValueError("k is out of bounds")

        X_est_m = X_init.copy()

        prev_cost_value = None
        prev_gradient_norm_value = None
        prev_global_estimation_error = None

        lambda_m = config.lambda_0

        for iteration in range(config.max_iter + 1):
            # Compute the cost function and gradient
            r_z_m = self.r(X_est_m, Y)
            J_m = self.J(X_est_m)

            # Convergence tracking
            cost_value = self.cost_function(X_est_m, Y)
            gradient_norm_value = np.linalg.norm(J_m.T @ r_z_m)

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
                    (np.linalg.norm(X_est_m - X_true) - prev_global_estimation_error)
                    / abs(prev_global_estimation_error)
                    * 100
                )
            prev_cost_value = cost_value
            prev_gradient_norm_value = gradient_norm_value
            prev_global_estimation_error = np.linalg.norm(X_est_m - X_true)

            # Check convergence and print metrics
            if gradient_norm_value < config.epsilon or iteration == config.max_iter:
                reason = (
                    "tolerance reached"
                    if gradient_norm_value < config.epsilon
                    else (
                        "max iteration reached"
                        if iteration == config.max_iter
                        else "gradient norm stagnated"
                    )
                )
                print(f"[Levenberg-Marquardt] STOP on Iteration {iteration} ({reason})")
                print(
                    f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(X_est_m - X_true)} ({global_estimation_error_change:.2f}%)"
                )
                print(
                    f"Final initial conditions estimation errors: {np.linalg.norm(X_est_m[:config.n_p, :, 0] - X_true[:config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[config.n_x : config.n_x + config.n_p, :, 0] - X_true[config.n_x : config.n_x + config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[2 * config.n_x : 2 * config.n_x + config.n_p, :, 0] - X_true[2 * config.n_x : 2 * config.n_x + config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[3 * config.n_x : 3 * config.n_x + config.n_p, :, 0] - X_true[3 * config.n_x : 3 * config.n_x + config.n_p, :, 0])} m"
                )
                print(
                    f"Final position estimation errors: {np.linalg.norm(X_est_m[:config.n_p, :, -1] - X_true[:config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[config.n_x : config.n_x + config.n_p, :, -1] - X_true[config.n_x : config.n_x + config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[2 * config.n_x : 2 * config.n_x + config.n_p, :, -1] - X_true[2 * config.n_x : 2 * config.n_x + config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[3 * config.n_x : 3 * config.n_x + config.n_p, :, -1] - X_true[3 * config.n_x : 3 * config.n_x + config.n_p, :, -1])} m\n"
                )
                break
            else:
                if iteration == 0:
                    print(
                        f"[Levenberg-Marquardt] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(X_est_m - X_true)}"
                    )
                else:
                    print(
                        f"[Levenberg-Marquardt] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(X_est_m - X_true)} ({global_estimation_error_change:.2f}%)"
                    )

            # Print estimation errors
            print(
                f"Initial conditions estimation errors: {np.linalg.norm(X_est_m[:config.n_p, :, 0] - X_true[:config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[config.n_x : config.n_x + config.n_p, :, 0] - X_true[config.n_x : config.n_x + config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[2 * config.n_x : 2 * config.n_x + config.n_p, :, 0] - X_true[2 * config.n_x : 2 * config.n_x + config.n_p, :, 0])} m, {np.linalg.norm(X_est_m[3 * config.n_x : 3 * config.n_x + config.n_p, :, 0] - X_true[3 * config.n_x : 3 * config.n_x + config.n_p, :, 0])} m"
            )
            print(
                f"Position estimation errors: {np.linalg.norm(X_est_m[:config.n_p, :, -1] - X_true[:config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[config.n_x : config.n_x + config.n_p, :, -1] - X_true[config.n_x : config.n_x + config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[2 * config.n_x : 2 * config.n_x + config.n_p, :, -1] - X_true[2 * config.n_x : 2 * config.n_x + config.n_p, :, -1])} m, {np.linalg.norm(X_est_m[3 * config.n_x : 3 * config.n_x + config.n_p, :, -1] - X_true[3 * config.n_x : 3 * config.n_x + config.n_p, :, -1])} m\n"
            )

            # Solve for the update step
            delta_x = (
                -np.linalg.inv(J_m.T @ J_m + lambda_m * np.eye(config.n * config.H))
                @ J_m.T
                @ r_z_m
            )
            X_pred_m = X_est_m + delta_x.reshape(config.H, 1, config.n).T

            # Update the estimate
            if self.cost_function(X_pred_m, Y) < self.cost_function(X_est_m, Y):
                X_est_m = X_pred_m
                lambda_m *= 0.7
            else:
                lambda_m *= 2.0

            # Save the current iteration
            self.iterations = iteration + 1

        return X_est_m
