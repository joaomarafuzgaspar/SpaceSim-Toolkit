import numpy as np

from scipy.optimize import minimize
from dynamics import SatelliteDynamics


class WLSTSQ:
    """
    This class implements the (constrained) Weighted Least Squares (WLSTSQ).
    """

    def __init__(self, Q, R):
        """
        Initialize the WLSTSQ class.

        Parameters:
        Q (np.array): Process noise covariance.
        R (np.array): Measurement noise covariance.
        SatelliteDynamics (class): Satellite dynamics model.
        """
        # Define noise covariances
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

        self.dynamic_model = SatelliteDynamics()

    def h_function_chief(self, x_vec):
        """
        Computes the measurement vector based on the current state vector.
        The measurement vector includes position components.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).
        """
        return x_vec[0:3]

    def h_function_deputy(self, x_vec):
        """
        Computes the measurement vector based on the current state vector.

        Parameters:
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        y (np.array): The measurement vector of the satellite (range [km]).
        """
        r_chief = x_vec[:3]
        r_deputy1 = x_vec[6:9]
        r_deputy2 = x_vec[12:15]
        r_deputy3 = x_vec[18:21]

        range_deputy1_chief = np.linalg.norm(r_deputy1 - r_chief)
        range_deputy1_deputy2 = np.linalg.norm(r_deputy1 - r_deputy2)
        range_deputy1_deputy3 = np.linalg.norm(r_deputy1 - r_deputy3)
        range_deputy2_chief = np.linalg.norm(r_deputy2 - r_chief)
        range_deputy2_deputy3 = np.linalg.norm(r_deputy2 - r_deputy3)
        range_deputy3_chief = np.linalg.norm(r_deputy3 - r_chief)

        return np.array(
            [
                [range_deputy1_chief],
                [range_deputy1_deputy2],
                [range_deputy1_deputy3],
                [range_deputy2_chief],
                [range_deputy2_deputy3],
                [range_deputy3_chief],
            ]
        )

    def J_function(self, vars, T):
        V = vars[: 9 * T].reshape((9, 1, T))
        W = vars[9 * T :].reshape((24, 1, T - 1))

        obj_fun = 0
        for t in range(T):
            obj_fun += np.dot(V[:, :, t].T, np.dot(np.linalg.inv(self.R), V[:, :, t]))
        for t in range(T - 1):
            obj_fun += np.dot(W[:, :, t].T, np.dot(np.linalg.inv(self.Q), W[:, :, t]))
        return obj_fun

    def constraints(self, vars, T, dt, x_initial, Y):
        V = vars[: 9 * T].reshape((9, 1, T))
        W = vars[9 * T :].reshape((24, 1, T - 1))
        X = np.zeros((24, 1, T))
        X[:, :, 0] = x_initial

        con_eqs = []
        for t in range(T):
            con_eqs.append(
                V[:, :, t]
                - (
                    Y[:, :, t]
                    - np.concatenate(
                        (
                            self.h_function_chief(X[:, :, t]),
                            self.h_function_deputy(X[:, :, t]),
                        )
                    )
                )
            )
            if t != T - 1:
                con_eqs.append(
                    W[:, :, t]
                    - (X[:, :, t + 1] - self.dynamic_model.x_new(dt, X[:, :, t]))
                )
                X[:, :, t + 1] = X[:, :, t] + W[:, :, t]

        return np.concatenate([eq.flatten() for eq in con_eqs])

    def apply(self, dt, x_initial, Y):
        # Initializations
        T = Y.shape[2]
        X_est = np.zeros((24, 1, T))

        # Initial guess
        V_initial = np.zeros((9, 1, T))
        W_initial = np.zeros((24, 1, T - 1))
        V_initial[:, :, 0] = Y[:, :, 0] - np.concatenate((self.h_function_chief(x_initial), self.h_function_deputy(x_initial)))
        vars_initial = np.concatenate((V_initial.flatten(), W_initial.flatten()))

        # Optimization
        solution = minimize(
            fun=self.J_function,
            x0=vars_initial,
            args=(T,),
            method="SLSQP",
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda vars: self.constraints(vars, T, dt, x_initial, Y),
                }
            ],
            options={"maxiter": 1000, "disp": True},
        )

        # Extract the solution
        if solution.success:
            vars_est = solution.x
            W_est = vars_est[9 * T :].reshape((24, 1, T - 1))

            # Update X_est based on the optimized V and W
            X_est[:, :, 0] = x_initial
            for t in range(T - 1):
                X_est[:, :, t + 1] = X_est[:, :, t] + W_est[:, :, t]

            print("Optimization successful")
        else:
            print("Optimization failed")

        return X_est
