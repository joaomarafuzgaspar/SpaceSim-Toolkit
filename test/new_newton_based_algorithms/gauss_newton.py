import numpy as np

from tqdm import tqdm
from scipy.linalg import solve

from dynamics import Dynamics
from config import SimulationConfig as CFG


class GaussNewton:
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
        
    def h(self, x_vec):
        p_vecs = [x_vec[i : i + CFG.n_p] for i in range(0, CFG.n, CFG.n_x)]
        distances = [np.linalg.norm(p_vecs[j] - p_vecs[i]) for (i, j) in [(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)]]
        return np.concatenate((p_vecs[0], np.array(distances).reshape(-1, 1)))

    def Dh(self, x_vec):
        first_order_der = np.zeros((self.o, CFG.n))
        p_vecs = [x_vec[i : i + CFG.n_p] for i in range(0, CFG.n, CFG.n_x)]
        
        first_order_der[:CFG.n_p, :CFG.n_p] = np.eye(CFG.n_p)
        
        for k, (i, j) in enumerate([(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)], start=CFG.n_p):
            d = p_vecs[j] - p_vecs[i]
            norm_d = np.linalg.norm(d)
            first_order_der[k, i * CFG.n_x : i * CFG.n_x + CFG.n_p] = -d.T / norm_d
            first_order_der[k, j * CFG.n_x : j * CFG.n_x + CFG.n_p] = d.T / norm_d
        
        return first_order_der

    def Hh(self, x_vec):
        second_order_der = np.zeros((self.o, CFG.n, CFG.n))
        p_vecs = [x_vec[i : i + CFG.n_p] for i in range(0, CFG.n, CFG.n_x)]

        def hessian_distance(d, norm_d):
            I = np.eye(CFG.n_p)
            return -(I / norm_d - np.outer(d, d) / norm_d**3)

        for k, (i, j) in enumerate([(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)], start=CFG.n_p):
            d = p_vecs[j] - p_vecs[i]
            norm_d = np.linalg.norm(d)
            hess_d = hessian_distance(d, norm_d)
            
            second_order_der[k, i * CFG.n_x : i * CFG.n_x + CFG.n_p, i * CFG.n_x : i * CFG.n_x + CFG.n_p] = -hess_d
            second_order_der[k, i * CFG.n_x : i * CFG.n_x + CFG.n_p, j * CFG.n_x : j * CFG.n_x + CFG.n_p] = hess_d
            second_order_der[k, j * CFG.n_x : j * CFG.n_x + CFG.n_p, i * CFG.n_x : i * CFG.n_x + CFG.n_p] = hess_d
            second_order_der[k, j * CFG.n_x : j * CFG.n_x + CFG.n_p, j * CFG.n_x : j * CFG.n_x + CFG.n_p] = -hess_d
        
        return second_order_der.reshape((self.o * CFG.n, CFG.n))
    
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
        if k <self.H - 1 or k + 1 > self.K:
            raise ValueError("k is out of bounds")
        R_inv = np.linalg.inv(self.R)
        STM = np.eye(CFG.n)
        DJ_x = np.zeros((CFG.n, 1))
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
        STM = np.eye(CFG.n)
        HJ_x = np.zeros((CFG.n, CFG.n))
        for tau in range(k - self.H + 1, k + 1):
            Dh_x = self.Dh(x_vec)
            Df_x = STM
            HJ_x += Df_x.T @ Dh_x.T @ R_inv @ Dh_x @ Df_x
            STM = self.dyn.Df(dt, x_vec) @ STM
            x_vec = self.dyn.f(dt, x_vec)
        return HJ_x

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
                print(f"[Gauss-Newton] STOP on Iteration {iteration} ({reason})")
                print(f"Cost function = {cost_value} ({cost_value_change:.2f}%)\nGradient norm = {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error = {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                print(f"Final initial conditions estimation errors: {np.linalg.norm(x[:CFG.n_p, :] - x_true_initial[:CFG.n_p, :])} m, {np.linalg.norm(x[CFG.n_x : CFG.n_x + CFG.n_p, :] - x_true_initial[CFG.n_x : CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :] - x_true_initial[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :] - x_true_initial[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :])} m")
                print(f"Final position estimation errors: {np.linalg.norm(x_end[:CFG.n_p, :] - x_true_end[:CFG.n_p, :])} m, {np.linalg.norm(x_end[CFG.n_x : CFG.n_x + CFG.n_p, :] - x_true_end[CFG.n_x : CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x_end[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :] - x_true_end[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x_end[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :] - x_true_end[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :])} m\n")
                break
            else:
                if iteration == 0:
                    print(f"[Gauss-Newton] Before applying the algorithm\nCost function: {cost_value}\nGradient norm: {gradient_norm_value}\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)}")
                else:
                    print(f"[Gauss-Newton] Iteration {iteration}\nCost function: {cost_value} ({cost_value_change:.2f}%)\nGradient norm: {gradient_norm_value} ({gradient_norm_value_change:.2f}%)\nGlobal estimation error: {np.linalg.norm(x - x_true_initial)} ({global_estimation_error_change:.2f}%)")
                    
            # Print estimation errors 
            print(f"Initial conditions estimation errors: {np.linalg.norm(x[:CFG.n_p, :] - x_true_initial[:CFG.n_p, :])} m, {np.linalg.norm(x[CFG.n_x : CFG.n_x + CFG.n_p, :] - x_true_initial[CFG.n_x : CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :] - x_true_initial[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :] - x_true_initial[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :])} m")
            print(f"Position estimation errors: {np.linalg.norm(x_end[:CFG.n_p, :] - x_true_end[:CFG.n_p, :])} m, {np.linalg.norm(x_end[CFG.n_x : CFG.n_x + CFG.n_p, :] - x_true_end[CFG.n_x : CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x_end[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :] - x_true_end[2 * CFG.n_x : 2 * CFG.n_x + CFG.n_p, :])} m, {np.linalg.norm(x_end[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :] - x_true_end[3 * CFG.n_x : 3 * CFG.n_x + CFG.n_p, :])} m\n")
                
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