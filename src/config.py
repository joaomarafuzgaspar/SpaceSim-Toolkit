# src/config.py
import numpy as np
from scipy.linalg import block_diag


class SimulationConfig:
    # Simulation parameters
    dt = 10.0  # Time step [s] (1, 5, 10, 20, 30, 60)
    K = 570  # Simulation duration in timesteps (57000, 11400, 5700, 2850, 1900, 950)
    H = 5  # Window size [timesteps]
    seed = 42 # Random seed for reproducibility

    N = 4  # Number of spacecrafts
    C = 1  # Number of chiefs
    D = N - C  # Number of deputies

    n_p = 3
    n_v = 3
    n_x = n_p + n_v
    n = N * n_x

    o_chief = 3
    o_deputy = 3
    o = 3 + 3 + 2 + 1
    # FIXME: o = C * o_chief + D * o_deputy
    o_tree = 3 + 1 + 1 + 1

    # Spacecraft parameters
    C_drag = 2.2  # Drag coefficient
    A_drag = 0.01  # Drag area [m^2]
    m = 1.0  # Mass [kg]

    # Observation noise
    r_chief_pos = 1e-1  # [m]
    R_chief = np.diag(np.concatenate([r_chief_pos * np.ones(3)])) ** 2
    r_deputy_pos = 1e0  # [m]
    R_deputies = np.diag(np.concatenate([r_deputy_pos * np.ones(6)])) ** 2
    R = block_diag(R_chief, R_deputies)
    R_deputies_tree = np.diag(np.concatenate([r_deputy_pos * np.ones(3)])) ** 2
    R_tree = block_diag(R_chief, R_deputies_tree)

    # Initial deviation noise
    p_pos_initial = 1e0  # [m]
    p_vel_initial = 1e-2  # [m / s]
    # p_pos_initial = 1e2  # [m]
    # p_vel_initial = 1e0  # [m / s]
    P_0_spacecraft = (
        np.diag(
            np.concatenate([p_pos_initial * np.ones(n_p), p_vel_initial * np.ones(n_v)])
        )
        ** 2
    )
    # P_0 = block_diag(*[P_0_spacecraft for _ in range(N)])
    P_0 = block_diag(P_0_spacecraft, P_0_spacecraft, P_0_spacecraft, P_0_spacecraft)

    # Consensus parameters
    L = 1  # Number of consensus iterations
    N = 4  # Number of satellites
    gamma = N  # Consensus gain 1
    pi = 1 / N  # Consensus gain 2

    # Newton's method-based algorithms parameters
    grad_norm_order_mag = True
    grad_norm_tol = 1e-1
    max_iterations = 20

    # Post-processing parameters
    invalid_rmse = 1e2  # [m]
    K_RMSE = 570  # Index from which the RMSE is calculated


class SpacecraftConfig:
    mass = 1.0  # Mass [kg]
    C_drag = 2.2  # Drag coefficient
    A_drag = 0.01  # Drag area [m^2]
    C_SRP = 1.2  # SRP coefficient
    A_SRP = 0.01  # SRP area [m^2]
