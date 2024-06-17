import os
import numpy as np
import pandas as pd

# from numba import jit  # FIXME: speeding up the code
from tqdm import tqdm
from utils import rmse
from fcekf import FCEKF
from hcmci import HCMCI
from ccekf import EKF, CCEKF
from scipy.linalg import block_diag
from dynamics import SatelliteDynamics


def run_propagation(args):
    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = 360  # Duration [min]

    # Initial state vector and state covariance
    if args.formation == 1:
        # Formation Setup for REx-Lab Mission
        x_initial = np.array(
            [
                6895.6,
                0,
                0,
                0,
                -0.99164,
                7.5424,
                6895.6,
                3e-05,
                1e-05,
                -0.0015,
                -0.99214,
                7.5426,
                6895.6,
                1e-05,
                3e-06,
                0.005,
                -0.98964,
                7.5422,
                6895.6,
                -2e-05,
                4e-06,
                0.00545,
                -0.99594,
                7.5423,
            ]
        ).reshape(24, 1)
    elif args.formation == 2:
        # Formation Setup for Higher-Orbit Difference
        # x_initial = np.array(
        #     [
        #         -1295.9,
        #         -929.67,
        #         6793.4,
        #         -7.4264,
        #         0.1903,
        #         -1.3906,
        #         171.77,
        #         6857.3,
        #         -1281.5,
        #         1.0068,
        #         -1.3998,
        #         -7.3585,
        #         -4300.5,
        #         1654.8,
        #         -5241,
        #         -4.6264,
        #         3.4437,
        #         4.8838,
        #         -2197.8,
        #         -3973.4,
        #         -5298.7,
        #         1.377,
        #         5.6601,
        #         -4.8155,
        #     ]
        # ).reshape(24, 1)

        x_initial = np.array(
            [
                -1295.584507660883,
                -929.374090790716,
                6793.411681382508,
                -7.426511598857044,
                0.1901977700449196,
                -1.390296386102339,
                -171.9053207863305,
                6857.222886707187,
                -1281.649066658106,
                1.00700996311517,
                -1.399928822038102,
                -7.358476293144188,
                -4251.965594108594,
                1619.96360478974,
                -5291.008175332099,
                -4.678160885930184,
                3.46354521932496,
                4.820200886960163,
                -2214.904606485801,
                -3967.018695794123,
                -5296.066519714539,
                1.361455163361585,
                5.66586154474086,
                -4.813337608337905,
            ]
        ).reshape(24, 1)

    # Get the true state vectors and Jacobians
    X_true = np.zeros((24, 1, T + 1))
    X_true[:, :, 0] = x_initial
    F = np.zeros((24, 24, T + 1))
    F[:, :, 0] = SatelliteDynamics().F_jacobian(x_initial)
    STM = np.zeros((24, 24, T + 1))
    STM[:, :, 0] = np.eye(24)
    for i in range(T):
        X_true[:, :, i + 1] = SatelliteDynamics().x_new(dt, X_true[:, :, i])
        F[:, :, i + 1] = SatelliteDynamics().F_jacobian(X_true[:, :, i + 1])
        STM[:, :, i + 1] = SatelliteDynamics().Phi(dt, X_true[:, :, i]) @ STM[:, :, i]

    header = "x_chief,y_chief,z_chief,x_deputy1,y_deputy1,z_deputy1,x_deputy2,y_deputy2,z_deputy2,x_deputy3,y_deputy3,z_deputy3"
    data_x_true = np.column_stack(
        (
            X_true[0, :, :].reshape(-1, 1),
            X_true[1, :, :].reshape(-1, 1),
            X_true[2, :, :].reshape(-1, 1),
            X_true[6, :, :].reshape(-1, 1),
            X_true[7, :, :].reshape(-1, 1),
            X_true[8, :, :].reshape(-1, 1),
            X_true[12, :, :].reshape(-1, 1),
            X_true[13, :, :].reshape(-1, 1),
            X_true[14, :, :].reshape(-1, 1),
            X_true[18, :, :].reshape(-1, 1),
            X_true[19, :, :].reshape(-1, 1),
            X_true[20, :, :].reshape(-1, 1),
        )
    )
    os.makedirs("data", exist_ok=True)
    np.savetxt(
        "data/data_x_true.csv", data_x_true, delimiter=",", header=header, comments=""
    )

    def extract_and_serialize_submatrices(F):
        submatrices = []
        for i in range(0, 24, 6):
            submatrix = F[i : i + 6, i : i + 6]
            submatrices.append(submatrix)
        return submatrices
    extracted_data = [
        extract_and_serialize_submatrices(F[:, :, t]) for t in range(T + 1)
    ]
    columns = ["F_chief", "F_deputy1", "F_deputy2", "F_deputy3"]
    matrix_evolution = {col: [] for col in columns}
    for matrices in extracted_data:
        for idx, matrix in enumerate(matrices):
            matrix_evolution[columns[idx]].append(matrix.flatten())
    df = pd.DataFrame(
        {
            col: [" ".join(map(str, row)) for row in matrix_evolution[col]]
            for col in columns
        }
    )
    df.to_csv("data/data_F.csv", index=False)
    
    extracted_data = [
        extract_and_serialize_submatrices(STM[:, :, t]) for t in range(T + 1)
    ]
    columns = ["STM_chief", "STM_deputy1", "STM_deputy2", "STM_deputy3"]
    matrix_evolution = {col: [] for col in columns}
    for matrices in extracted_data:
        for idx, matrix in enumerate(matrices):
            matrix_evolution[columns[idx]].append(matrix.flatten())
    df = pd.DataFrame(
        {
            col: [" ".join(map(str, row)) for row in matrix_evolution[col]]
            for col in columns
        }
    )
    df.to_csv("data/data_STM.csv", index=False)


def run_simulation(args):
    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = 395  # Duration [min]
    T_RMSE = 300  # Index from which the RMSE is calculated
    n_simulations = args.monte_carlo_sims  # Number of Monte-Carlo simulations
    L = 1  # Number of consensus iterations
    N = 4  # Number of satellites
    gamma = N  # Consensus gain 1
    pi = 1 / N  # Consensus gain 2

    # Initial state vector and state covariance
    if args.formation == 1:
        # Formation Setup for REx-Lab Mission
        x_initial = np.array(
            [
                6895.6,
                0,
                0,
                0,
                -0.99164,
                7.5424,
                6895.6,
                3e-05,
                1e-05,
                -0.0015,
                -0.99214,
                7.5426,
                6895.6,
                1e-05,
                3e-06,
                0.005,
                -0.98964,
                7.5422,
                6895.6,
                -2e-05,
                4e-06,
                0.00545,
                -0.99594,
                7.5423,
            ]
        ).reshape(24, 1)
    elif args.formation == 2:
        # Formation Setup for Higher-Orbit Difference
        x_initial = np.array(
            [
                -1295.9,
                -929.67,
                6793.4,
                -7.4264,
                0.1903,
                -1.3906,
                171.77,
                6857.3,
                -1281.5,
                1.0068,
                -1.3998,
                -7.3585,
                -4300.5,
                1654.8,
                -5241,
                -4.6264,
                3.4437,
                4.8838,
                -2197.8,
                -3973.4,
                -5298.7,
                1.377,
                5.6601,
                -4.8155,
            ]
        ).reshape(24, 1)
    p_pos_initial = 1e-1  # [km]
    p_vel_initial = 1e-5  # [km / s]

    # Process noise
    q_chief_pos = 1e-4  # [km]
    q_chief_vel = 1e-5  # [km / s]
    Q_chief = (
        np.diag(np.concatenate([q_chief_pos * np.ones(3), q_chief_vel * np.ones(3)]))
        ** 2
    )
    q_deputy_pos = 1e-3  # [km]
    q_deputy_vel = 1e-5  # [km / s]
    Q_deputy = (
        np.diag(np.concatenate([q_deputy_pos * np.ones(3), q_deputy_vel * np.ones(3)]))
        ** 2
    )
    Q_deputies = block_diag(Q_deputy, Q_deputy, Q_deputy)
    Q = block_diag(Q_chief, Q_deputies)

    # Observation noise
    r_chief_pos = 1e-4  # [km]
    R_chief = np.diag(np.concatenate([r_chief_pos * np.ones(3)])) ** 2
    r_deputy_pos = 1e-3  # [km]
    R_deputies = np.diag(np.concatenate([r_deputy_pos * np.ones(6)])) ** 2
    R = block_diag(R_chief, R_deputies)

    # Get the true state vectors
    X_true = np.zeros((24, 1, T))
    X_true[:, :, 0] = x_initial
    for i in range(T - 1):
        X_true[:, :, i + 1] = SatelliteDynamics().x_new(dt, X_true[:, :, i])

    # Simulation
    rmse_chief_values, rmse_deputy1_values, rmse_deputy2_values, rmse_deputy3_values = (
        [],
        [],
        [],
        [],
    )
    dev_chief_values, dev_deputy1_values, dev_deputy2_values, dev_deputy3_values = (
        [],
        [],
        [],
        [],
    )

    # Estimation technique
    if args.algorithm == "fcekf":
        fcekf = FCEKF(Q, R)
        for i in tqdm(range(n_simulations)):
            # Observations
            Y = np.zeros((9, 1, T))
            for t in range(T):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            initial_dev = np.concatenate(
                (
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est[:, :, 0] = x_initial + initial_dev
            P = np.diag(initial_dev.reshape(-1) ** 2)

            for t in range(1, T):
                X_est[:, :, t], P = fcekf.apply(dt, X_est[:, :, t - 1], P, Y[:, :, t])

            # Compute RMSE
            rmse_chief_values.append(
                rmse(X_est[:3, :, T_RMSE:], X_true[:3, :, T_RMSE:])
            )
            rmse_deputy1_values.append(
                rmse(X_est[6:9, :, T_RMSE:], X_true[6:9, :, T_RMSE:])
            )
            rmse_deputy2_values.append(
                rmse(X_est[12:15, :, T_RMSE:], X_true[12:15, :, T_RMSE:])
            )
            rmse_deputy3_values.append(
                rmse(X_est[18:21, :, T_RMSE:], X_true[18:21, :, T_RMSE:])
            )

            # Compute deviation
            dev_chief_values.append(
                np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0)
            )
            dev_deputy1_values.append(
                np.linalg.norm(X_est[6:9, :, :] - X_true[6:9, :, :], axis=0)
            )
            dev_deputy2_values.append(
                np.linalg.norm(X_est[12:15, :, :] - X_true[12:15, :, :], axis=0)
            )
            dev_deputy3_values.append(
                np.linalg.norm(X_est[18:21, :, :] - X_true[18:21, :, :], axis=0)
            )
    elif args.algorithm == "hcmci":
        fcekf = FCEKF(Q, R)
        chief_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy1_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy2_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy3_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        for i in tqdm(range(n_simulations)):
            # Observations
            Y = np.zeros((9, 1, T))
            for t in range(T):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            X_est_chief = np.zeros_like(X_true)
            X_est_deputy1 = np.zeros_like(X_true)
            X_est_deputy2 = np.zeros_like(X_true)
            X_est_deputy3 = np.zeros_like(X_true)
            initial_dev = np.concatenate(
                (
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est[:, :, 0] = x_initial + initial_dev
            X_est_chief[:, :, 0] = x_initial + initial_dev
            X_est_deputy1[:, :, 0] = x_initial + initial_dev
            X_est_deputy2[:, :, 0] = x_initial + initial_dev
            X_est_deputy3[:, :, 0] = x_initial + initial_dev
            Omega_chief = np.linalg.inv(np.diag(initial_dev.reshape(-1) ** 2))
            Omega_deputy1 = np.linalg.inv(np.diag(initial_dev.reshape(-1) ** 2))
            Omega_deputy2 = np.linalg.inv(np.diag(initial_dev.reshape(-1) ** 2))
            Omega_deputy3 = np.linalg.inv(np.diag(initial_dev.reshape(-1) ** 2))

            for t in range(1, T):
                (
                    q_chief,
                    Omega_chief,
                    delta_q_chief,
                    delta_Omega_chief,
                ) = chief_hcmci.prediction(
                    dt, X_est_chief[:, :, t - 1], Omega_chief, Y[:, :, t]
                )
                (
                    q_deputy1,
                    Omega_deputy1,
                    delta_q_deputy1,
                    delta_Omega_deputy1,
                ) = deputy1_hcmci.prediction(
                    dt, X_est_deputy1[:, :, t - 1], Omega_deputy1, Y[:, :, t]
                )
                (
                    q_deputy2,
                    Omega_deputy2,
                    delta_q_deputy2,
                    delta_Omega_deputy2,
                ) = deputy2_hcmci.prediction(
                    dt, X_est_deputy2[:, :, t - 1], Omega_deputy2, Y[:, :, t]
                )
                (
                    q_deputy3,
                    Omega_deputy3,
                    delta_q_deputy3,
                    delta_Omega_deputy3,
                ) = deputy3_hcmci.prediction(
                    dt, X_est_deputy3[:, :, t - 1], Omega_deputy3, Y[:, :, t]
                )

                # Consensus
                (
                    delta_q_vec_chief,
                    delta_Omega_vec_chief,
                    q_vec_chief,
                    Omega_vec_chief,
                ) = chief_hcmci.init_consensus(
                    delta_q_chief, delta_Omega_chief, q_chief, Omega_chief, L
                )
                (
                    delta_q_vec_deputy1,
                    delta_Omega_vec_deputy1,
                    q_vec_deputy1,
                    Omega_vec_deputy1,
                ) = deputy1_hcmci.init_consensus(
                    delta_q_deputy1, delta_Omega_deputy1, q_deputy1, Omega_deputy1, L
                )
                (
                    delta_q_vec_deputy2,
                    delta_Omega_vec_deputy2,
                    q_vec_deputy2,
                    Omega_vec_deputy2,
                ) = deputy2_hcmci.init_consensus(
                    delta_q_deputy2, delta_Omega_deputy2, q_deputy2, Omega_deputy2, L
                )
                (
                    delta_q_vec_deputy3,
                    delta_Omega_vec_deputy3,
                    q_vec_deputy3,
                    Omega_vec_deputy3,
                ) = deputy3_hcmci.init_consensus(
                    delta_q_deputy3, delta_Omega_deputy3, q_deputy3, Omega_deputy3, L
                )

                for l in range(1, L + 1):
                    delta_q_vec_chief[:, :, l] = pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy1[:, :, l] = pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy2[:, :, l] = pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy3[:, :, l] = pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )

                    delta_Omega_vec_chief[:, :, l] = pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy1[:, :, l] = pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy2[:, :, l] = pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy3[:, :, l] = pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )

                    q_vec_chief[:, :, l] = pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy1[:, :, l] = pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy2[:, :, l] = pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy3[:, :, l] = pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )

                    Omega_vec_chief[:, :, l] = pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy1[:, :, l] = pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy2[:, :, l] = pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy3[:, :, l] = pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )

                X_est_chief[:, :, t], Omega_chief = chief_hcmci.correction(
                    delta_q_vec_chief,
                    delta_Omega_vec_chief,
                    q_vec_chief,
                    Omega_vec_chief,
                    gamma,
                )
                X_est_deputy1[:, :, t], Omega_deputy1 = deputy1_hcmci.correction(
                    delta_q_vec_deputy1,
                    delta_Omega_vec_deputy1,
                    q_vec_deputy1,
                    Omega_vec_deputy1,
                    gamma,
                )
                X_est_deputy2[:, :, t], Omega_deputy2 = deputy2_hcmci.correction(
                    delta_q_vec_deputy2,
                    delta_Omega_vec_deputy2,
                    q_vec_deputy2,
                    Omega_vec_deputy2,
                    gamma,
                )
                X_est_deputy3[:, :, t], Omega_deputy3 = deputy3_hcmci.correction(
                    delta_q_vec_deputy3,
                    delta_Omega_vec_deputy3,
                    q_vec_deputy3,
                    Omega_vec_deputy3,
                    gamma,
                )
                X_est[:, :, t] = np.concatenate(
                    (
                        X_est_chief[:6, :, t],
                        X_est_deputy1[6:12, :, t],
                        X_est_deputy2[12:18, :, t],
                        X_est_deputy3[18:24, :, t],
                    ),
                    axis=0,
                )

            # Compute RMSE
            rmse_chief_values.append(
                rmse(X_est[:3, :, T_RMSE:], X_true[:3, :, T_RMSE:])
            )
            rmse_deputy1_values.append(
                rmse(X_est[6:9, :, T_RMSE:], X_true[6:9, :, T_RMSE:])
            )
            rmse_deputy2_values.append(
                rmse(X_est[12:15, :, T_RMSE:], X_true[12:15, :, T_RMSE:])
            )
            rmse_deputy3_values.append(
                rmse(X_est[18:21, :, T_RMSE:], X_true[18:21, :, T_RMSE:])
            )

            # Compute deviation
            dev_chief_values.append(
                np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0)
            )
            dev_deputy1_values.append(
                np.linalg.norm(X_est[6:9, :, :] - X_true[6:9, :, :], axis=0)
            )
            dev_deputy2_values.append(
                np.linalg.norm(X_est[12:15, :, :] - X_true[12:15, :, :], axis=0)
            )
            dev_deputy3_values.append(
                np.linalg.norm(X_est[18:21, :, :] - X_true[18:21, :, :], axis=0)
            )
    elif args.algorithm == "ccekf":
        fcekf = FCEKF(Q, R)
        chief_ekf = EKF(Q_chief, R_chief)
        deputy1_ccekf = CCEKF(Q_deputies, R_deputies)
        deputy2_ccekf = CCEKF(Q_deputies, R_deputies)
        deputy3_ccekf = CCEKF(Q_deputies, R_deputies)
        for i in tqdm(range(n_simulations)):
            # Observations
            Y = np.zeros((9, 1, T))
            for t in range(T):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            X_est_chief = np.zeros_like(X_true[:6])
            X_est_deputy1 = np.zeros_like(X_true[6:])
            X_est_deputy2 = np.zeros_like(X_true[6:])
            X_est_deputy3 = np.zeros_like(X_true[6:])
            initial_dev = np.concatenate(
                (
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                    p_pos_initial * np.random.randn(3, 1),
                    p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est[:, :, 0] = x_initial + initial_dev
            X_est_chief[:, :, 0] = x_initial[:6] + initial_dev[:6]
            X_est_deputy1[:, :, 0] = x_initial[6:] + initial_dev[6:]
            X_est_deputy2[:, :, 0] = x_initial[6:] + initial_dev[6:]
            X_est_deputy3[:, :, 0] = x_initial[6:] + initial_dev[6:]
            P_chief = np.diag(initial_dev[:6].reshape(-1) ** 2)
            P_deputy1 = np.diag(initial_dev[6:].reshape(-1) ** 2)
            P_deputy2 = np.diag(initial_dev[6:].reshape(-1) ** 2)
            P_deputy3 = np.diag(initial_dev[6:].reshape(-1) ** 2)
            P_deputy1_chief = np.zeros((18, 6))
            P_deputy2_chief = np.zeros((18, 6))
            P_deputy3_chief = np.zeros((18, 6))

            for t in range(1, T):
                X_est_chief[:, :, t], P_chief = chief_ekf.apply(
                    dt, X_est_chief[:, :, t - 1], P_chief, Y[:3, :, t]
                )
                (
                    X_est_deputy1[:, :, t],
                    P_deputy1,
                    P_deputy1_chief,
                ) = deputy1_ccekf.apply(
                    dt,
                    X_est_deputy1[:, :, t - 1],
                    P_deputy1,
                    P_deputy1_chief,
                    Y[3:, :, t],
                    X_est_chief[:, :, t],
                    P_chief,
                )
                (
                    X_est_deputy2[:, :, t],
                    P_deputy2,
                    P_deputy2_chief,
                ) = deputy2_ccekf.apply(
                    dt,
                    X_est_deputy2[:, :, t - 1],
                    P_deputy2,
                    P_deputy2_chief,
                    Y[3:, :, t],
                    X_est_chief[:, :, t],
                    P_chief,
                )
                (
                    X_est_deputy3[:, :, t],
                    P_deputy3,
                    P_deputy3_chief,
                ) = deputy3_ccekf.apply(
                    dt,
                    X_est_deputy3[:, :, t - 1],
                    P_deputy3,
                    P_deputy3_chief,
                    Y[3:, :, t],
                    X_est_chief[:, :, t],
                    P_chief,
                )
                X_est[:, :, t] = np.concatenate(
                    (
                        X_est_chief[:, :, t],
                        X_est_deputy1[:6, :, t],
                        X_est_deputy2[6:12, :, t],
                        X_est_deputy3[12:18, :, t],
                    ),
                    axis=0,
                )

            # Compute RMSE
            rmse_chief_values.append(
                rmse(X_est[:3, :, T_RMSE:], X_true[:3, :, T_RMSE:])
            )
            rmse_deputy1_values.append(
                rmse(X_est[6:9, :, T_RMSE:], X_true[6:9, :, T_RMSE:])
            )
            rmse_deputy2_values.append(
                rmse(X_est[12:15, :, T_RMSE:], X_true[12:15, :, T_RMSE:])
            )
            rmse_deputy3_values.append(
                rmse(X_est[18:21, :, T_RMSE:], X_true[18:21, :, T_RMSE:])
            )

            # Compute deviation
            dev_chief_values.append(
                np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0)
            )
            dev_deputy1_values.append(
                np.linalg.norm(X_est[6:9, :, :] - X_true[6:9, :, :], axis=0)
            )
            dev_deputy2_values.append(
                np.linalg.norm(X_est[12:15, :, :] - X_true[12:15, :, :], axis=0)
            )
            dev_deputy3_values.append(
                np.linalg.norm(X_est[18:21, :, :] - X_true[18:21, :, :], axis=0)
            )

    # Calculate average RMSE
    rmse_chief_average = np.mean(rmse_chief_values)
    rmse_deputy1_average = np.mean(rmse_deputy1_values)
    rmse_deputy2_average = np.mean(rmse_deputy2_values)
    rmse_deputy3_average = np.mean(rmse_deputy3_values)

    print(f"Average RMSE for chief: {rmse_chief_average * 1e3} m")
    print(f"Average RMSE for deputy 1: {rmse_deputy1_average * 1e3} m")
    print(f"Average RMSE for deputy 2: {rmse_deputy2_average * 1e3} m")
    print(f"Average RMSE for deputy 3: {rmse_deputy3_average * 1e3} m")

    # Calculate the average deviation
    dev_chief_average = np.mean(dev_chief_values, axis=0)
    dev_deputy1_average = np.mean(dev_deputy1_values, axis=0)
    dev_deputy2_average = np.mean(dev_deputy2_values, axis=0)
    dev_deputy3_values = np.mean(dev_deputy3_values, axis=0)

    data = np.column_stack(
        (
            dev_chief_average.reshape(-1, 1),
            dev_deputy1_average.reshape(-1, 1),
            dev_deputy2_average.reshape(-1, 1),
            dev_deputy3_values.reshape(-1, 1),
        )
    )
    header = "dev_chief,dev_deputy1,dev_deputy2,dev_deputy3"
    os.makedirs(os.path.dirname("data/data.csv"), exist_ok=True)
    np.savetxt("data/data.csv", data, delimiter=",", header=header, comments="")
