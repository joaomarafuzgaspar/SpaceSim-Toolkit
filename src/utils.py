import pickle
import numpy as np

from dynamics import coe2rv
from datetime import datetime


def get_form_initial_conditions(formation):
    """
    Get the initial conditions for the formation specified by the input argument.

    Parameters
    formation (int): The formation to get the initial conditions for.

    Returns
    np.ndarray: The initial conditions for the specified formation.
    """
    if formation == 1:
        # Formation Setup for REx-Lab Mission
        x_initial_chief = coe2rv(
            semi_major_axis=6903.50e3,
            eccentricity=0.0011,
            inclination=np.deg2rad(97.49),
            argument_of_periapsis=np.deg2rad(0.0),
            longitude_of_ascending_node=np.deg2rad(0.0),
            true_anomaly=np.deg2rad(0.0),
        )
        x_initial_deputy1 = coe2rv(
            semi_major_axis=6903.98e3,
            eccentricity=0.0012,
            inclination=np.deg2rad(97.49),
            argument_of_periapsis=np.deg2rad(9.23),
            longitude_of_ascending_node=np.deg2rad(0.0),
            true_anomaly=np.deg2rad(350.76),
        )
        x_initial_deputy2 = coe2rv(
            semi_major_axis=6902.67e3,
            eccentricity=0.0012,
            inclination=np.deg2rad(97.47),
            argument_of_periapsis=np.deg2rad(327.27),
            longitude_of_ascending_node=np.deg2rad(0.0),
            true_anomaly=np.deg2rad(32.72),
        )
        x_initial_deputy3 = coe2rv(
            semi_major_axis=6904.34e3,
            eccentricity=0.0014,
            inclination=np.deg2rad(97.52),
            argument_of_periapsis=np.deg2rad(330.47),
            longitude_of_ascending_node=np.deg2rad(0.0),
            true_anomaly=np.deg2rad(29.52),
        )
    elif formation == 2:
        # Formation Setup for Higher-Orbit Difference
        x_initial_chief = coe2rv(
            semi_major_axis=6978e3,
            eccentricity=2.6e-6,
            inclination=np.deg2rad(97.79),
            longitude_of_ascending_node=np.deg2rad(1.5e-5),
            argument_of_periapsis=np.deg2rad(303.34),
            true_anomaly=np.deg2rad(157.36),
        )
        x_initial_deputy1 = coe2rv(
            semi_major_axis=6978e3,
            eccentricity=6.48e-3,
            inclination=np.deg2rad(97.26),
            argument_of_periapsis=np.deg2rad(281.15),
            longitude_of_ascending_node=np.deg2rad(272.80),
            true_anomaly=np.deg2rad(269.52),
        )
        x_initial_deputy2 = coe2rv(
            semi_major_axis=6978e3,
            eccentricity=6.6e-5,
            inclination=np.deg2rad(97.79),
            argument_of_periapsis=np.deg2rad(104.07),
            longitude_of_ascending_node=np.deg2rad(149.99),
            true_anomaly=np.deg2rad(206.00),
        )
        x_initial_deputy3 = coe2rv(
            semi_major_axis=6978e3,
            eccentricity=1.3e-5,
            inclination=np.deg2rad(97.79),
            argument_of_periapsis=np.deg2rad(257.43),
            longitude_of_ascending_node=np.deg2rad(70.00),
            true_anomaly=np.deg2rad(332.57),
        )
    return np.concatenate(
        (x_initial_chief, x_initial_deputy1, x_initial_deputy2, x_initial_deputy3)
    ).reshape(24, 1)


def rmse(X_est, X_true):
    return np.sqrt(
        np.mean(np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0) ** 2)
    )


def save_data(args, X_true, X_est_all):
    # Fill the dictionary
    data = {}
    data["true"] = X_true
    for m in range(args.monte_carlo_sims):
        data[m] = X_est_all[m]

    # Save to pickle file
    with open(
        f'data/{args.algorithm}_form{args.formation}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl',
        "wb",
    ) as file:
        pickle.dump(data, file)
