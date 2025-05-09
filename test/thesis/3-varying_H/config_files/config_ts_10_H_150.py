import numpy as np


from dataclasses import dataclass
from scipy.linalg import block_diag


@dataclass
class SimulationConfig:
    name_of_file: str = "config_ts_10_H_150.py"

    # Simulation parameters
    dt: float = 10.0  # Time step [s] (1, 5, 10, 20, 30, 60)
    K: int = (
        570  # Simulation duration in timesteps (57000, 11400, 5700, 2850, 1900, 950)
    )
    H: int = 150  # Window size [timesteps]
    seed: int = 42  # Random seed for reproducibility

    # Network parameters
    N: int = 4  # Number of systems
    number_of_chiefs: int = 1  # Number of chiefs
    number_of_deputies: int = N - number_of_chiefs  # Number of deputies
    n_p: int = 3  # Position vector dimension
    n_v: int = 3  # Velocity vector dimension
    n_x: int = n_p + n_v  # State vector dimension
    n: int = N * n_x  # Global state vector dimension
    o_chief: int = 3  # Chief observation vector dimension
    o_deputies: int = 6  # Deputy observation vector dimension
    o: int = (
        number_of_chiefs * o_chief + o_deputies
    )  # Global observation vector dimension

    # Observation noise
    r_chief_pos: float = 1e-1  # [m]
    R_chief: np.ndarray = np.diag(np.concatenate([r_chief_pos * np.ones(o_chief)])) ** 2
    r_deputy_pos: float = 1e0  # [m]
    R_deputies: np.ndarray = (
        np.diag(np.concatenate([r_deputy_pos * np.ones(o_deputies)])) ** 2
    )
    R: np.ndarray = block_diag(R_chief, R_deputies)

    # Initial deviation noise
    # Warm-start parameters
    p_pos_initial: float = 1e2  # [m]
    p_vel_initial: float = 1e-2  # [m / s]
    # Cold-start parameters
    # p_pos_initial: float = 1e2  # [m]
    # p_vel_initial: float = 1e0  # [m / s]
    P_0_spacecraft: np.ndarray = (
        np.diag(
            np.concatenate([p_pos_initial * np.ones(n_p), p_vel_initial * np.ones(n_v)])
        )
        ** 2
    )
    P_0: np.ndarray = block_diag(
        P_0_spacecraft, P_0_spacecraft, P_0_spacecraft, P_0_spacecraft
    )

    # Levenberg-Marquardt parameters
    lambda_0: float = 1.0
    epsilon: float = 1e-6
    max_iter: int = 100

    # Consensus parameters
    L: int = 1  # Number of consensus iterations
    gamma: float = N  # Consensus gain 1
    pi: float = 1 / N  # Consensus gain 2

    # Newton's method-based algorithms parameters
    grad_norm_order_mag: bool = True
    grad_norm_tol: float = 1e-6
    max_iterations: int = 20

    # Post-processing parameters
    invalid_rmse: float = 1e2  # [m]
    K_RMSE: int = 300  # Index from which the RMSE is calculated

    # Spacecraft parameters
    mass: float = 1.0  # Mass [kg]
    C_drag: float = 2.2  # Drag coefficient
    A_drag: float = 0.01  # Drag area [m^2]
    C_SRP: float = 1.2  # SRP coefficient
    A_SRP: float = 0.01  # SRP area [m^2]

    # Observation model
    @classmethod
    def h(cls, x_vec):
        p_vecs = [x_vec[i : i + cls.n_p] for i in range(0, cls.n, cls.n_x)]
        distances = [
            np.linalg.norm(p_vecs[i] - p_vecs[j])
            for (i, j) in [
                (1, 0),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 0),
            ]
        ]
        return np.concatenate((p_vecs[0], np.array(distances).reshape(-1, 1)))

    @classmethod
    def Dh(cls, x_vec):
        first_order_der = np.zeros((cls.o, cls.n))
        p_vecs = [x_vec[i : i + cls.n_p] for i in range(0, cls.n, cls.n_x)]

        first_order_der[: cls.n_p, : cls.n_p] = np.eye(cls.n_p)

        for k, (i, j) in enumerate(
            [(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)],
            start=cls.n_p,
        ):
            d = p_vecs[i] - p_vecs[j]
            norm_d = np.linalg.norm(d)
            first_order_der[k, i * cls.n_x : i * cls.n_x + cls.n_p] = d.T / norm_d
            first_order_der[k, j * cls.n_x : j * cls.n_x + cls.n_p] = -d.T / norm_d

        return first_order_der

    @classmethod
    def Hh(cls, x_vec):
        second_order_der = np.zeros((cls.o, cls.n, cls.n))
        p_vecs = [x_vec[i : i + cls.n_p] for i in range(0, cls.n, cls.n_x)]

        for k, (i, j) in enumerate(
            [(1, 0), (1, 2), (1, 3), (2, 0), (2, 3), (3, 0)],
            start=cls.n_p,
        ):
            d = p_vecs[i] - p_vecs[j]
            norm_d = np.linalg.norm(d)
            hess_d = np.eye(cls.n_p) / norm_d - np.outer(d, d) / norm_d**3

            second_order_der[
                k,
                i * cls.n_x : i * cls.n_x + cls.n_p,
                i * cls.n_x : i * cls.n_x + cls.n_p,
            ] = hess_d
            second_order_der[
                k,
                i * cls.n_x : i * cls.n_x + cls.n_p,
                j * cls.n_x : j * cls.n_x + cls.n_p,
            ] = -hess_d
            second_order_der[
                k,
                j * cls.n_x : j * cls.n_x + cls.n_p,
                i * cls.n_x : i * cls.n_x + cls.n_p,
            ] = -hess_d
            second_order_der[
                k,
                j * cls.n_x : j * cls.n_x + cls.n_p,
                j * cls.n_x : j * cls.n_x + cls.n_p,
            ] = hess_d

        return second_order_der.reshape((cls.o * cls.n, cls.n))
