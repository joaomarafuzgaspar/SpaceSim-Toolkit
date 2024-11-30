import os
import numpy as np
import pandas as pd

# from numba import jit  # FIXME: speeding up the code
from tqdm import tqdm
from cnkkt import CNKKT
from fcekf import FCEKF
from hcmci import HCMCI
from ccekf import EKF, CCEKF
from wlstsq_lm import WLSTSQ_LM
from scipy.linalg import block_diag
from dynamics import SatelliteDynamics
from utils import (
    rmse,
    save_simulation_data,
    save_propagation_data,
    get_form_initial_conditions,
)


def run_tudat_propagation(args):
    return print("Tudat propagation is not implemented yet.")


def run_propagation(args):
    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = 360  # Duration [min]

    # Initial state vector
    X_initial = get_form_initial_conditions(args.formation)

    # Get the true state vectors and Jacobians
    X = np.zeros((24, 1, T))
    X[:, :, 0] = X_initial
    for t in range(T - 1):
        X[:, :, t + 1] = SatelliteDynamics().x_new(dt, X[:, :, t])

    # Save data to pickle file
    save_propagation_data(args, dt, T, X)


def run_simulation(args):
    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = 395  # Duration [min]
    T_RMSE = 300  # Index from which the RMSE is calculated
    M = args.monte_carlo_sims  # Number of Monte-Carlo simulations
    L = 1  # Number of consensus iterations
    N = 4  # Number of satellites
    gamma = N  # Consensus gain 1
    pi = 1 / N  # Consensus gain 2
    W = 100  # Window size

    # Initial state vector and get the true state vectors (propagation) FIXME: Add run_propagation() here
    X_initial = get_form_initial_conditions(args.formation)
    X_true = np.zeros((24, 1, T))
    X_true[:, :, 0] = X_initial
    for t in range(T - 1):
        X_true[:, :, t + 1] = SatelliteDynamics().x_new(dt, X_true[:, :, t])

    # Process noise
    q_chief_pos = 1e-1  # [m]
    q_chief_vel = 1e-2  # [m / s]
    Q_chief = (
        np.diag(np.concatenate([q_chief_pos * np.ones(3), q_chief_vel * np.ones(3)]))
        ** 2
    )
    q_deputy_pos = 1e0  # [m]
    q_deputy_vel = 1e-2  # [m / s]
    Q_deputy = (
        np.diag(np.concatenate([q_deputy_pos * np.ones(3), q_deputy_vel * np.ones(3)]))
        ** 2
    )
    Q_deputies = block_diag(Q_deputy, Q_deputy, Q_deputy)
    Q = block_diag(Q_chief, Q_deputies)

    # Observation noise
    r_chief_pos = 1e-1  # [m]
    R_chief = np.diag(np.concatenate([r_chief_pos * np.ones(3)])) ** 2
    r_deputy_pos = 1e0  # [m]
    R_deputies = np.diag(np.concatenate([r_deputy_pos * np.ones(6)])) ** 2
    R = block_diag(R_chief, R_deputies)

    # Initial deviation noise
    p_pos_initial = 1e2  # [m]
    p_vel_initial = 1e-2  # [m / s]

    # Simulation
    X_est_all = []
    if args.algorithm == "wlstsq-lm":
        fcekf = FCEKF(Q, R)
        wlstsq_lm = WLSTSQ_LM(Q, R)
        for m in tqdm(range(M)):
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
            X_est[:, :, 0] = X_initial + initial_dev
            for t in range(1, T):
                X_est[:, :, t] = SatelliteDynamics().x_new(dt, X_est[:, :, t - 1])
            X_est = wlstsq_lm.apply(dt, X_est, Y)
            X_est_all.append(X_est)
    if args.algorithm == "fcekf":
        fcekf = FCEKF(Q, R)
        for m in tqdm(range(M)):
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
            X_est[:, :, 0] = X_initial + initial_dev
            P = np.diag(initial_dev.reshape(-1) ** 2)
            for t in range(1, T):
                X_est[:, :, t], P = fcekf.apply(dt, X_est[:, :, t - 1], P, Y[:, :, t])
            X_est_all.append(X_est)
    elif args.algorithm == "hcmci":
        fcekf = FCEKF(Q, R)
        chief_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy1_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy2_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        deputy3_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(R))
        for m in tqdm(range(M)):
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
            X_est[:, :, 0] = X_initial + initial_dev
            X_est_chief[:, :, 0] = X_initial + initial_dev
            X_est_deputy1[:, :, 0] = X_initial + initial_dev
            X_est_deputy2[:, :, 0] = X_initial + initial_dev
            X_est_deputy3[:, :, 0] = X_initial + initial_dev
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
            X_est_all.append(X_est)
    elif args.algorithm == "ccekf":
        fcekf = FCEKF(Q, R)
        chief_ekf = EKF(Q_chief, R_chief)
        deputy1_ccekf = CCEKF(Q_deputies, R_deputies)
        deputy2_ccekf = CCEKF(Q_deputies, R_deputies)
        deputy3_ccekf = CCEKF(Q_deputies, R_deputies)
        for m in tqdm(range(M)):
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
            X_est[:, :, 0] = X_initial + initial_dev
            X_est_chief[:, :, 0] = X_initial[:6] + initial_dev[:6]
            X_est_deputy1[:, :, 0] = X_initial[6:] + initial_dev[6:]
            X_est_deputy2[:, :, 0] = X_initial[6:] + initial_dev[6:]
            X_est_deputy3[:, :, 0] = X_initial[6:] + initial_dev[6:]
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
            X_est_all.append(X_est)
    elif args.algorithm == "cnkkt":
        fcekf = FCEKF(Q, R)
        cnkkt = CNKKT(W, R_chief, r_deputy_pos)
        for m in tqdm(range(M)):
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
            X_est[:, :, 0] = X_initial + initial_dev
            X_est = cnkkt.apply(dt, X_est[:, :, 0], Y, X_true)
            X_est_all.append(X_est)

    # Compute average RMSE
    rmse_chief_values = []
    rmse_deputy1_values = []
    rmse_deputy2_values = []
    rmse_deputy3_values = []
    for m in range(M):
        rmse_chief = rmse(X_est_all[m][:6, :, T_RMSE:], X_true[:6, :, T_RMSE:])
        rmse_deputy1 = rmse(X_est_all[m][6:12, :, T_RMSE:], X_true[6:12, :, T_RMSE:])
        rmse_deputy2 = rmse(X_est_all[m][12:18, :, T_RMSE:], X_true[12:18, :, T_RMSE:])
        rmse_deputy3 = rmse(X_est_all[m][18:24, :, T_RMSE:], X_true[18:24, :, T_RMSE:])
        invalid_rmse = 1e2
        if (
            rmse_chief < invalid_rmse
            and rmse_deputy1 < invalid_rmse
            and rmse_deputy2 < invalid_rmse
            and rmse_deputy3 < invalid_rmse
        ):
            rmse_chief_values.append(rmse_chief)
            rmse_deputy1_values.append(rmse_deputy1)
            rmse_deputy2_values.append(rmse_deputy2)
            rmse_deputy3_values.append(rmse_deputy3)
        else:
            print(
                f"(!!) For Monte Carlo Run #{m + 1} the algorithm diverged with RMSEs:"
            )
            print(f"    - Chief: {rmse_chief} m")
            print(f"    - Deputy 1: {rmse_deputy1} m")
            print(f"    - Deputy 2: {rmse_deputy2} m")
            print(f"    - Deputy 3: {rmse_deputy3} m")
    print(f"Average RMSEs:")
    print(f"    - Chief: {np.mean(rmse_chief_values)} m")
    print(f"    - Deputy 1: {np.mean(rmse_deputy1_values)} m")
    print(f"    - Deputy 2: {np.mean(rmse_deputy2_values)} m")
    print(f"    - Deputy 3: {np.mean(rmse_deputy3_values)} m")

    # Save data to pickle file
    save_simulation_data(args, X_true, X_est_all)
