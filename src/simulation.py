import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from lm import LM
from fcekf import FCEKF
from hcmci import HCMCI
from ccekf import EKF, CCEKF
from newton import Newton
from cnkkt import CNKKT
from unkkt import UNKKT
from approxh_newton import approxH_Newton
from mm_newton import MM_Newton
from dynamics import Dynamics
from utils import (
    rmse,
    save_simulation_data,
    save_propagation_data,
    get_form_initial_conditions,
)
from config import SimulationConfig as config


def run_tudat_propagation(args):
    return print("Tudat propagation is not implemented yet.")


def run_propagation(args):
    # Initial conditions
    X_initial = get_form_initial_conditions(args.formation)

    # Propagation
    dynamics_propagator = Dynamics()
    X = np.zeros((config.n, 1, config.K))
    X[:, :, 0] = X_initial
    for k in range(config.K - 1):
        X[:, :, k + 1] = dynamics_propagator.f(config.dt, X[:, :, k])

    # Save propagation data to pickle file
    save_propagation_data(args, config.dt, config.K, X)


def run_simulation(args):
    # Simulation parameters
    M = args.monte_carlo_sims  # Number of Monte-Carlo simulations

    # Initial conditions and true state vectors (from Tudat)
    X_initial = get_form_initial_conditions(args.formation)
    with open(
        f"data/tudatpy_form{args.formation}_ts_{int(config.dt)}.pkl", "rb"
    ) as file:
        X_true = pickle.load(file)

    # Process noise covariance matrix estimation
    dynamics_propagator = Dynamics()
    if args.algorithm == "lm" or args.algorithm == "fcekf" or args.algorithm == "hcmci" or args.algorithm == "ccekf":
        K_true = X_true.shape[2]
        X_dynamics_propagator = np.zeros((config.n, 1, K_true))
        X_dynamics_propagator[:, :, 0] = X_initial
        X_tudat_propagator = X_true.transpose(2, 0, 1).reshape(K_true, config.n)
        for k, X_k in enumerate(X_tudat_propagator[:-1, :]):
            X_dynamics_propagator[:, :, k + 1] = dynamics_propagator.f(
                config.dt, X_k.reshape(config.n, 1)
            )
        X_dynamics_propagator = X_dynamics_propagator.transpose(2, 0, 1).reshape(
            K_true, config.n
        )
        diff = X_tudat_propagator - X_dynamics_propagator
        Q = np.cov(diff.T)
        print("Estimated process noise covariance matrix :\n", pd.DataFrame(Q))
        Q_chief = Q[: config.n_x, : config.n_x]
        Q_deputies = Q[config.n_x :, config.n_x :]
    X_true = X_true[:, :, : config.K] # Truncate the true state vector to the simulation duration

    # Simulation
    X_est_all = []
    if args.algorithm == "lm":
        fcekf = FCEKF(Q, config.R)
        lm = LM(Q, config.R)
        for m in tqdm(range(M)):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est[:, :, 0] = X_initial + initial_dev
            for t in range(1, config.K):
                X_est[:, :, t] = dynamics_propagator.f(config.dt, X_est[:, :, t - 1])
            X_est = lm.apply(config.dt, X_est, Y)
            X_est_all.append(X_est)
    if args.algorithm == "fcekf":
        fcekf = FCEKF(Q, config.R)
        for m in tqdm(range(M)):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est[:, :, 0] = X_initial + initial_dev
            P = np.diag(initial_dev.reshape(-1) ** 2)
            for t in range(1, config.K):
                X_est[:, :, t], P = fcekf.apply(
                    config.dt, X_est[:, :, t - 1], P, Y[:, :, t]
                )
            X_est_all.append(X_est)
    elif args.algorithm == "hcmci":
        Q = np.diag(np.diag(Q))
        fcekf = FCEKF(Q, config.R)
        chief_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(config.R))
        deputy1_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(config.R))
        deputy2_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(config.R))
        deputy3_hcmci = HCMCI(np.linalg.inv(Q), np.linalg.inv(config.R))
        for m in tqdm(range(M)):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            X_est_chief = np.zeros_like(X_true)
            X_est_deputy1 = np.zeros_like(X_true)
            X_est_deputy2 = np.zeros_like(X_true)
            X_est_deputy3 = np.zeros_like(X_true)
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
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

            for t in range(1, config.K):
                (
                    q_chief,
                    Omega_chief,
                    delta_q_chief,
                    delta_Omega_chief,
                ) = chief_hcmci.prediction(
                    config.dt, X_est_chief[:, :, t - 1], Omega_chief, Y[:, :, t]
                )
                (
                    q_deputy1,
                    Omega_deputy1,
                    delta_q_deputy1,
                    delta_Omega_deputy1,
                ) = deputy1_hcmci.prediction(
                    config.dt, X_est_deputy1[:, :, t - 1], Omega_deputy1, Y[:, :, t]
                )
                (
                    q_deputy2,
                    Omega_deputy2,
                    delta_q_deputy2,
                    delta_Omega_deputy2,
                ) = deputy2_hcmci.prediction(
                    config.dt, X_est_deputy2[:, :, t - 1], Omega_deputy2, Y[:, :, t]
                )
                (
                    q_deputy3,
                    Omega_deputy3,
                    delta_q_deputy3,
                    delta_Omega_deputy3,
                ) = deputy3_hcmci.prediction(
                    config.dt, X_est_deputy3[:, :, t - 1], Omega_deputy3, Y[:, :, t]
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

                for l in range(1, config.L + 1):
                    delta_q_vec_chief[:, :, l] = config.pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy1[:, :, l] = config.pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy2[:, :, l] = config.pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )
                    delta_q_vec_deputy3[:, :, l] = config.pi * (
                        delta_q_vec_chief[:, :, l - 1]
                        + delta_q_vec_deputy1[:, :, l - 1]
                        + delta_q_vec_deputy2[:, :, l - 1]
                        + delta_q_vec_deputy3[:, :, l - 1]
                    )

                    delta_Omega_vec_chief[:, :, l] = config.pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy1[:, :, l] = config.pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy2[:, :, l] = config.pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )
                    delta_Omega_vec_deputy3[:, :, l] = config.pi * (
                        delta_Omega_vec_chief[:, :, l - 1]
                        + delta_Omega_vec_deputy1[:, :, l - 1]
                        + delta_Omega_vec_deputy2[:, :, l - 1]
                        + delta_Omega_vec_deputy3[:, :, l - 1]
                    )

                    q_vec_chief[:, :, l] = config.pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy1[:, :, l] = config.pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy2[:, :, l] = config.pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )
                    q_vec_deputy3[:, :, l] = config.pi * (
                        q_vec_chief[:, :, l - 1]
                        + q_vec_deputy1[:, :, l - 1]
                        + q_vec_deputy2[:, :, l - 1]
                        + q_vec_deputy3[:, :, l - 1]
                    )

                    Omega_vec_chief[:, :, l] = config.pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy1[:, :, l] = config.pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy2[:, :, l] = config.pi * (
                        Omega_vec_chief[:, :, l - 1]
                        + Omega_vec_deputy1[:, :, l - 1]
                        + Omega_vec_deputy2[:, :, l - 1]
                        + Omega_vec_deputy3[:, :, l - 1]
                    )
                    Omega_vec_deputy3[:, :, l] = config.pi * (
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
                    config.gamma,
                )
                X_est_deputy1[:, :, t], Omega_deputy1 = deputy1_hcmci.correction(
                    delta_q_vec_deputy1,
                    delta_Omega_vec_deputy1,
                    q_vec_deputy1,
                    Omega_vec_deputy1,
                    config.gamma,
                )
                X_est_deputy2[:, :, t], Omega_deputy2 = deputy2_hcmci.correction(
                    delta_q_vec_deputy2,
                    delta_Omega_vec_deputy2,
                    q_vec_deputy2,
                    Omega_vec_deputy2,
                    config.gamma,
                )
                X_est_deputy3[:, :, t], Omega_deputy3 = deputy3_hcmci.correction(
                    delta_q_vec_deputy3,
                    delta_Omega_vec_deputy3,
                    q_vec_deputy3,
                    Omega_vec_deputy3,
                    config.gamma,
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
        fcekf = FCEKF(Q, config.R)
        chief_ekf = EKF(Q_chief, config.R_chief)
        deputy1_ccekf = CCEKF(Q_deputies, config.R_deputies)
        deputy2_ccekf = CCEKF(Q_deputies, config.R_deputies)
        deputy3_ccekf = CCEKF(Q_deputies, config.R_deputies)
        for m in tqdm(range(M)):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            X_est = np.zeros_like(X_true)
            X_est_chief = np.zeros_like(X_true[:6])
            X_est_deputy1 = np.zeros_like(X_true[6:])
            X_est_deputy2 = np.zeros_like(X_true[6:])
            X_est_deputy3 = np.zeros_like(X_true[6:])
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
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

            for t in range(1, config.K):
                X_est_chief[:, :, t], P_chief = chief_ekf.apply(
                    config.dt, X_est_chief[:, :, t - 1], P_chief, Y[:3, :, t]
                )
                (
                    X_est_deputy1[:, :, t],
                    P_deputy1,
                    P_deputy1_chief,
                ) = deputy1_ccekf.apply(
                    config.dt,
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
                    config.dt,
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
                    config.dt,
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
    elif args.algorithm == "newton":
        newton = Newton()
        for m in tqdm(range(M), desc="MC runs", leave=True):
            # Set the random seed for reproducibility
            if config.seed is not None:
                np.random.seed(config.seed)
            
            # Generate observations
            # Y = np.zeros((config.o, 1, config.K))
            # for k in range(config.K):
            #     Y[:, :, k] = newton.h(X_true[:, :, k]) + np.random.multivariate_normal(np.zeros(config.o), config.R).reshape((config.o, 1))
            Y = np.zeros((config.o_tree, 1, config.K))
            for k in range(config.K):
                Y[:, :, k] = newton.h(X_true[:, :, k]) + np.random.multivariate_normal(np.zeros(config.o_tree), config.R_tree).reshape((config.o_tree, 1))

            # Initial guess for the state vector
            X_est = np.zeros_like(X_true)
            # for k in range(config.H):
            #     X_est[:, :, k] = None
            x_init = X_initial + np.random.multivariate_normal(np.zeros(config.n), config.P_0).reshape((config.n, 1))
            
            # Run the framework
            for k in tqdm(range(config.H - 1, config.K), desc="Windows", leave=False):
                x_init, x_est_k = newton.solve_MHE_problem(k, Y, x_init, X_true[:, :, k - config.H + 1],  X_true[:, :, k])
                X_est[:, :, k] = x_est_k
                # Warm-start
                x_init = dynamics_propagator.f(config.dt, x_init)
            X_est_all.append(X_est)
    elif args.algorithm == "cnkkt":
        fcekf = FCEKF(Q, config.R)
        cnkkt = CNKKT(config.H, config.R_chief, config.r_deputy_pos)
        for m in tqdm(range(M), desc="MC runs", leave=True):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est = cnkkt.apply(config.dt, X_initial + initial_dev, Y)
            X_est_all.append(X_est)
        X_true = X_true[:, :, : config.K - config.H + 1]
    elif args.algorithm == "unkkt":
        fcekf = FCEKF(Q, config.R)
        unkkt = UNKKT(config.H, config.R_chief, config.r_deputy_pos)
        for m in tqdm(range(M), desc="MC runs", leave=True):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        fcekf.h_function_chief(X_true[:, :, t]),
                        fcekf.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est = unkkt.apply(config.dt, X_initial + initial_dev, Y)
            X_est_all.append(X_est)
        X_true = X_true[:, :, : config.K - config.H + 1]
    elif args.algorithm == "approxh-newton":
        approxh_newton = approxH_Newton(config.H, config.R_chief, config.r_deputy_pos)
        for m in tqdm(range(M), desc="MC runs", leave=True):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        approxh_newton.h_function_chief(X_true[:, :, t]),
                        approxh_newton.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est = approxh_newton.apply(config.dt, X_initial + initial_dev, Y, X_true)
            X_est_all.append(X_est)
        X_true = X_true[:, :, : config.K - config.H + 1]
    elif args.algorithm == "mm-newton":
        mm_newton = MM_Newton(config.H, config.R_chief, config.r_deputy_pos)
        for m in tqdm(range(M), desc="MC runs", leave=True):
            # Observations
            Y = np.zeros((9, 1, config.K))
            for t in range(config.K):
                Y[:, :, t] = np.concatenate(
                    (
                        mm_newton.h_function_chief(X_true[:, :, t]),
                        mm_newton.h_function_deputy(X_true[:, :, t]),
                    ),
                    axis=0,
                ) + np.random.normal(
                    0, np.sqrt(np.diag(config.R)).reshape((9, 1)), size=(9, 1)
                )

            # Initial state vector and state covariance estimate
            initial_dev = np.concatenate(
                (
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                    config.p_pos_initial * np.random.randn(3, 1),
                    config.p_vel_initial * np.random.randn(3, 1),
                )
            )
            X_est = mm_newton.apply(config.dt, X_initial + initial_dev, Y, X_true)
            X_est_all.append(X_est)
        X_true = X_true[:, :, : config.K - config.H + 1]

    # Compute average RMSE
    rmse_chief_values = []
    rmse_deputy1_values = []
    rmse_deputy2_values = []
    rmse_deputy3_values = []
    for m in range(M):
        rmse_chief = rmse(
            X_est_all[m][:6, :, config.K_RMSE :], X_true[:6, :, config.K_RMSE :]
        )
        rmse_deputy1 = rmse(
            X_est_all[m][6:12, :, config.K_RMSE :], X_true[6:12, :, config.K_RMSE :]
        )
        rmse_deputy2 = rmse(
            X_est_all[m][12:18, :, config.K_RMSE :], X_true[12:18, :, config.K_RMSE :]
        )
        rmse_deputy3 = rmse(
            X_est_all[m][18:24, :, config.K_RMSE :], X_true[18:24, :, config.K_RMSE :]
        )
        if (
            rmse_chief < config.invalid_rmse
            and rmse_deputy1 < config.invalid_rmse
            and rmse_deputy2 < config.invalid_rmse
            and rmse_deputy3 < config.invalid_rmse
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
