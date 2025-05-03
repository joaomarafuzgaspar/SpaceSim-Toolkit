import numpy as np

from tqdm import tqdm
from scipy.linalg import solve

from dynamics import Dynamics
from config import SimulationConfig as config


class CentralizedNewton:
    def __init__(self, H, K, o, R, grad_norm_order_mag, grad_norm_tol, max_iterations):   
        # Simulation parameters
        self.H = H
        self.K = K
        self.o = o
        self.R = R
        self.dyn = Dynamics()
             
        # Stopping criteria
        self.grad_norm_order_mag = grad_norm_order_mag
        self.grad_norm_tol = grad_norm_tol
        self.max_iterations = max_iterations
        
        # Storage for results
        self.iterations = None
        self.cost_values = []
        self.gradient_norm_values = []
        self.grad_norm_order_history = []
        self.HJ_x_eigenvalues_history = []
        self.A_norm_history, self.B_norm_history, self.C_norm_history = [], [], []
        
    def h(self, x_vec):
        p_vecs = [x_vec[i : i + config.n_p] for i in range(0, config.n, config.n_x)]
        distances = [np.linalg.norm(p_vecs[j] - p_vecs[i]) for (i, j) in [(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)]]
        return np.concatenate((p_vecs[0], np.array(distances).reshape(-1, 1)))

    def Dh(self, x_vec):
        first_order_der = np.zeros((self.o, config.n))
        p_vecs = [x_vec[i : i + config.n_p] for i in range(0, config.n, config.n_x)]
        
        first_order_der[:config.n_p, :config.n_p] = np.eye(config.n_p)
        
        for k, (i, j) in enumerate([(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)], start=config.n_p):
            d = p_vecs[j] - p_vecs[i]
            norm_d = np.linalg.norm(d)
            first_order_der[k, i * config.n_x : i * config.n_x + config.n_p] = -d.T / norm_d
            first_order_der[k, j * config.n_x : j * config.n_x + config.n_p] = d.T / norm_d
        
        return first_order_der

    def Hh(self, x_vec):
        second_order_der = np.zeros((self.o, config.n, config.n))
        p_vecs = [x_vec[i : i + config.n_p] for i in range(0, config.n, config.n_x)]

        def hessian_distance(d, norm_d):
            I = np.eye(config.n_p)
            return -(I / norm_d - np.outer(d, d) / norm_d**3)

        for k, (i, j) in enumerate([(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)], start=config.n_p):
            d = p_vecs[j] - p_vecs[i]
            norm_d = np.linalg.norm(d)
            hess_d = hessian_distance(d, norm_d)
            
            second_order_der[k, i * config.n_x : i * config.n_x + config.n_p, i * config.n_x : i * config.n_x + config.n_p] = -hess_d
            second_order_der[k, i * config.n_x : i * config.n_x + config.n_p, j * config.n_x : j * config.n_x + config.n_p] = hess_d
            second_order_der[k, j * config.n_x : j * config.n_x + config.n_p, i * config.n_x : i * config.n_x + config.n_p] = hess_d
            second_order_der[k, j * config.n_x : j * config.n_x + config.n_p, j * config.n_x : j * config.n_x + config.n_p] = -hess_d
        
        return second_order_der.reshape((self.o * config.n, config.n))
    
    def J(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        J_x = 0
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            h_x = self.h(x_vec)
            J_x += 1 / 2 * (y - h_x).T @ R_inv @ (y - h_x)
            x_vec = self.dyn.f(dt, x_vec)
        return J_x

    def DJ(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(config.n)
        DJ_x = np.zeros((config.n, 1))
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            DJ_x += -STM.T @ self.Dh(x_vec).T @ R_inv @ (y - self.h(x_vec))
            STM = self.dyn.Df(dt, x_vec) @ STM
            x_vec = self.dyn.f(dt, x_vec)
        return DJ_x

    def HJ(self, k, dt, Y, x_vec):
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
            HJ_x += - (np.kron(R_inv @ (y - h_x), Df_x).T @ Hh_x @ Df_x + np.kron(Dh_x.T @ R_inv @ (y - h_x), np.eye(config.n)).T @ Hf_x) + Df_x.T @ Dh_x.T @ R_inv @ Dh_x @ Df_x
            DSTM = np.kron(np.eye(config.n), STM).T @ self.dyn.Hf(dt, x_vec) @ STM + np.kron(self.dyn.Df(dt, x_vec), np.eye(config.n)) @ DSTM
            STM = self.dyn.Df(dt, x_vec) @ STM
            x_vec = self.dyn.f(dt, x_vec)
        return HJ_x
    
    def HJ_with_metrics(self, k, dt, Y, x_vec):
        if k < self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(config.n)
        DSTM = np.zeros((config.n * config.n, config.n))
        HJ_x = np.zeros((config.n, config.n))
        A = np.zeros((config.n, config.n))
        B = np.zeros((config.n, config.n))
        C = np.zeros((config.n, config.n))
        for tau in range(k - self.H + 1, k + 1):
            y = Y[:, :, tau]
            h_x = self.h(x_vec)
            Dh_x = self.Dh(x_vec)
            Hh_x = self.Hh(x_vec)
            Df_x = STM
            Hf_x = DSTM
            A_aux = -np.kron(R_inv @ (y - h_x), Df_x).T @ Hh_x @ Df_x
            B_aux = -np.kron(Dh_x.T @ R_inv @ (y - h_x), np.eye(config.n)).T @ Hf_x
            C_aux = Df_x.T @ Dh_x.T @ R_inv @ Dh_x @ Df_x
            A += A_aux
            B += B_aux
            C += C_aux
            HJ_x += A_aux + B_aux + C_aux
            DSTM = np.kron(np.eye(config.n), STM).T @ self.dyn.Hf(dt, x_vec) @ STM + np.kron(self.dyn.Df(dt, x_vec), np.eye(config.n)) @ DSTM
            STM = self.dyn.Df(dt, x_vec) @ STM
            x_vec = self.dyn.f(dt, x_vec)
        return HJ_x, np.linalg.eigvalsh(HJ_x), np.linalg.norm(A), np.linalg.norm(B), np.linalg.norm(C)

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
                x_end = self.dyn.f(dt, x_end)

            # Check convergence and print metrics
            if gradient_norm_value < self.grad_norm_tol or iteration == self.max_iterations or stagnant_order:
                reason = "tolerance reached" if gradient_norm_value < self.grad_norm_tol else \
                        "max iteration reached" if iteration == self.max_iterations else \
                        "gradient norm stagnated"
                print(f"[Centralized Newton] STOP on Iteration {iteration} ({reason})")
                print(f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                print(f"Final initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m")
                print(f"Final position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n")
                break
            else:
                if iteration == 0:
                    print(f"[Centralized Newton] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}")
                else:
                    print(f"[Centralized Newton] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                    
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
            x = self.dyn.f(dt, x)

        return x_init, x
    
    def solve_MHE_problem_with_metrics(self, k, dt, Y, x_init, x_true_initial, x_true_end):
        x = x_init.copy()

        prev_cost_value = None
        prev_gradient_norm_value = None
        prev_global_estimation_error = None
        grad_norm_order_history = []

        for iteration in range(self.max_iterations + 1):
            # Compute the cost function, gradient of the Lagrangian and Hessian of the Lagrangian
            J_x = self.J(k, dt, Y, x)
            DJ_x = self.DJ(k, dt, Y, x)
            HJ_x, HJ_x_eigenvalues, A_norm, B_norm, C_norm = self.HJ_with_metrics(k, dt, Y, x)

            # Convergence tracking
            cost_value = J_x[0][0]
            gradient_norm_value = np.linalg.norm(DJ_x)

            # Store the values
            self.cost_values.append(cost_value)
            self.gradient_norm_values.append(gradient_norm_value)
            self.HJ_x_eigenvalues_history.append(HJ_x_eigenvalues)
            self.A_norm_history.append(A_norm)
            self.B_norm_history.append(B_norm)
            self.C_norm_history.append(C_norm)

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
                
            # Distributed framework validation
            T_matrix_x = np.zeros_like(HJ_x)
            R_matrix_x = np.zeros_like(HJ_x)
            for i in range(config.N):
                for j in range(config.N):
                    block = HJ_x[i*config.n_x:(i+1)*config.n_x, j*config.n_x:(j+1)*config.n_x]
                    if i == j or i == 0 or j == 0:
                        T_matrix_x[i*config.n_x:(i+1)*config.n_x, j*config.n_x:(j+1)*config.n_x] = block
                    else:
                        R_matrix_x[i*config.n_x:(i+1)*config.n_x, j*config.n_x:(j+1)*config.n_x] = -block
            spectral_radius_T_inv_R = max(abs(np.linalg.eigvals(np.linalg.inv(T_matrix_x) @ R_matrix_x)))
                
            # Propagate window initial conditions for metrics 
            x_end = x.copy()
            for _ in range(self.H - 1):
                x_end = self.dyn.f(dt, x_end)

            # Check convergence and print metrics
            if gradient_norm_value < self.grad_norm_tol or iteration == self.max_iterations or stagnant_order:
                reason = "tolerance reached" if gradient_norm_value < self.grad_norm_tol else \
                        "max iteration reached" if iteration == self.max_iterations else \
                        "gradient norm stagnated"
                print(f"[Centralized Newton] STOP on Iteration {iteration} ({reason})")
                print(f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)\nHessian matrix minimum eigenvalue: {min(self.HJ_x_eigenvalues_history[-1])}\nRatios ||A|| / ||C||, ||B|| / ||C||: {self.A_norm_history[-1] / self.C_norm_history[-1]}, {self.B_norm_history[-1] / self.C_norm_history[-1]}\nSpectral radius inv(T) @ R: {spectral_radius_T_inv_R}\nT minimum eigenvalue: {min(np.linalg.eigvalsh(T_matrix_x))}")
                print(f"Final initial conditions estimation errors: {np.linalg.norm(x[:config.n_p, :] - x_true_initial[:config.n_p, :])} m, {np.linalg.norm(x[config.n_x : config.n_x + config.n_p, :] - x_true_initial[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_initial[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_initial[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m")
                print(f"Final position estimation errors: {np.linalg.norm(x_end[:config.n_p, :] - x_true_end[:config.n_p, :])} m, {np.linalg.norm(x_end[config.n_x : config.n_x + config.n_p, :] - x_true_end[config.n_x : config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[2 * config.n_x : 2 * config.n_x + config.n_p, :] - x_true_end[2 * config.n_x : 2 * config.n_x + config.n_p, :])} m, {np.linalg.norm(x_end[3 * config.n_x : 3 * config.n_x + config.n_p, :] - x_true_end[3 * config.n_x : 3 * config.n_x + config.n_p, :])} m\n")
                break
            else:
                if iteration == 0:
                    print(f"[Centralized Newton] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}\nHessian matrix minimum eigenvalue: {min(self.HJ_x_eigenvalues_history[-1])}\nRatios ||A|| / ||C||, ||B|| / ||C||: {self.A_norm_history[-1] / self.C_norm_history[-1]}, {self.B_norm_history[-1] / self.C_norm_history[-1]}\nSpectral radius inv(T) @ R: {spectral_radius_T_inv_R}\nT minimum eigenvalue: {min(np.linalg.eigvalsh(T_matrix_x))}")
                else:
                    print(f"[Centralized Newton] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)\nHessian matrix minimum eigenvalue: {min(self.HJ_x_eigenvalues_history[-1])}\nRatios ||A|| / ||C||, ||B|| / ||C||: {self.A_norm_history[-1] / self.C_norm_history[-1]}, {self.B_norm_history[-1] / self.C_norm_history[-1]}\nSpectral radius inv(T) @ R: {spectral_radius_T_inv_R}\nT minimum eigenvalue: {min(np.linalg.eigvalsh(T_matrix_x))}")
                    
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
            x = self.dyn.f(dt, x)

        return x_init, x