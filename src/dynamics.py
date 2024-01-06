import numpy as np

from earth import Earth, AtmosphericModel


class SatelliteDynamics:
    """
    This class models the dynamics of a satellite in a LEO orbit.
    It accounts for Earth's gravitational force, Earth's oblateness (J2 effect), and atmospheric drag.
    """

    def __init__(self):
        """
        Initialize satellite parameters.
        """
        self.C_drag = 2.22  # Drag coefficient
        self.A_drag = 0.01 * 1e-6  # Drag area [m^2] -> [km^2]
        self.m = 1.0  # Mass [kg]

        self.earth = Earth()  # Earth model
        self.atm = AtmosphericModel()  # Atmospheric model

    def f_function(self, x_vec):
        """
        Computes the dynamics of the satellite based on the current state vector.
        The state vector includes position and velocity components.

        Parameters:
        t (float): Time.
        x_vec (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        x_dot_vec (np.array): The derivative of the state vector.
        """
        x_dot_vec = np.zeros_like(x_vec)
        for i in range(int(x_vec.shape[0] / 6)):
            r_vec = x_vec[i * 6 : i * 6 + 3]  # Satellite position vector [km]
            x, y, z = r_vec  # Satellite position components [km]
            r = np.linalg.norm(r_vec)  # Satellite position magnitude [km]
            r_dot_vec = x_vec[
                i * 6 + 3 : i * 6 + 6
            ]  # Satellite velocity vector [km / s]

            # Compute contributions from Earth's gravitational force
            r_ddot_vec_grav = (
                -self.earth.mu * r_vec / r**3
            )  # Acceleration due to gravity [km / s^2]

            # Compute contributions from Earth's oblateness (J2 effect)
            r_ddot_vec_J2 = (
                -(3 * self.earth.J_2 * self.earth.mu * self.earth.R**2)
                / (2 * r**5)
                * np.array(
                    [
                        (1 - 5 * z**2 / r**2) * x,
                        (1 - 5 * z**2 / r**2) * y,
                        (3 - 5 * z**2 / r**2) * z,
                    ]
                )
            )  # Acceleration due to J2 effect [km / s^2]

            # Compute contributions from atmospheric drag
            h = r - self.earth.R  # Altitude [km]
            rho_atm = self.atm.get_rho(h)  # Atmospheric density [kg / km^3]

            # print(self.earth.omega_vec.shape)
            # print(r_vec.shape)
            # exit()
            r_dot_vec_rel = r_dot_vec - np.cross(
                self.earth.omega_vec, r_vec.reshape(-1)
            ).reshape(
                (3, 1)
            )  # Relative velocity vector [km / s]
            r_dot_rel = np.linalg.norm(
                r_dot_vec_rel
            )  # Relative velocity magnitude [km / s]

            r_ddot_vec_drag = (
                -0.5
                * self.C_drag
                * self.A_drag
                / self.m
                * rho_atm
                * r_dot_rel
                * r_dot_vec_rel
            )  # Acceleration due to atmospheric drag [km / s^2]

            # Superposition of all contributions
            x_dot_vec[i * 6 : i * 6 + 6] = np.concatenate(
                (r_dot_vec, r_ddot_vec_grav + r_ddot_vec_J2 + r_ddot_vec_drag)
            )

        return x_dot_vec

    def F_jacobian(self, x_vec):
        F = np.zeros((x_vec.shape[0], x_vec.shape[0]))
        for i in range(int(x_vec.shape[0] / 6)):
            r_vec = x_vec[i * 6 : i * 6 + 3]  # Satellite position vector [km]
            x, y, z = r_vec  # Satellite position components [km]
            r = np.linalg.norm(r_vec)  # Satellite position magnitude [km]
            r_dot_vec = x_vec[
                i * 6 + 3 : i * 6 + 6
            ]  # Satellite velocity vector [km / s]

            # Compute contributions from Earth's gravitational force
            dr_ddot_vec_grav_dr_vec = -self.earth.mu * (
                np.eye(3) / r**3 - 3 * r_vec * r_vec.T / r**5
            )  # Jacobian of acceleration due to gravity w.r.t. position [s^{-2}]

            # Compute contributions from Earth's oblateness (J2 effect)
            dr_ddot_vec_J2_dr_vec = (
                -(3 * self.earth.J_2 * self.earth.mu * self.earth.R**2)
                / (2 * r**9)
                * np.array(
                    [
                        [
                            -4 * x**4
                            - 3 * x**2 * y**2
                            + 27 * x**2 * z**2
                            + y**4
                            - 3 * y**2 * z**2
                            - 4 * z**4,
                            -5 * x**3 * y - 5 * x * y**3 + 30 * x * y * z**2,
                            -15 * x**3 * z - 15 * x * y**2 * z + 20 * x * z**3,
                        ],
                        [
                            -5 * x**3 * y - 5 * x * y**3 + 30 * x * y * z**2,
                            x**4
                            - 3 * x**2 * y**2
                            - 3 * x**2 * z**2
                            - 4 * y**4
                            + 27 * y**2 * z**2
                            - 4 * z**4,
                            -15 * x**2 * y * z - 15 * y**3 * z + 20 * y * z**3,
                        ],
                        [
                            -15 * x**3 * z - 15 * x * y**2 * z + 20 * x * z**3,
                            -15 * x**2 * y * z - 15 * y**3 * z + 20 * y * z**3,
                            3 * x**4
                            + 6 * x**2 * y**2
                            - 24 * x**2 * z**2
                            + 3 * y**4
                            - 24 * y**2 * z**2
                            + 8 * z**4,
                        ],
                    ]
                )
            ).reshape(
                (3, 3)
            )  # Jacobian of acceleration due to J2 effect w.r.t. position [s^{-2}]

            # Compute contributions from atmospheric drag
            h = r - self.earth.R  # Altitude [km]
            rho_atm = self.atm.get_rho(h)  # Atmospheric density [kg / km^3]
            drho_atm_dr_vec = (
                -rho_atm / self.atm.get_H(h) * r_vec.T / r
            )  # Jacobian of atmospheric density w.r.t. position [kg / km^4]

            r_dot_vec_rel = r_dot_vec - np.cross(
                self.earth.omega_vec, r_vec.reshape(-1)
            ).reshape(
                (3, 1)
            )  # Relative velocity vector [km / s]
            r_dot_rel = np.linalg.norm(
                r_dot_vec_rel
            )  # Relative velocity magnitude [km / s]
            dr_dot_vec_rel_dr_vec = np.array(
                [[0, self.earth.omega, 0], [-self.earth.omega, 0, 0], [0, 0, 0]]
            )  # Jacobian of relative velocity w.r.t. position [s^{-1}]

            dr_ddot_vec_drag_dr_vec = (
                -0.5
                * self.C_drag
                * self.A_drag
                / self.m
                * (
                    r_dot_rel * r_dot_vec_rel * drho_atm_dr_vec
                    + rho_atm
                    * (
                        r_dot_vec_rel * r_dot_vec_rel.T / r_dot_rel
                        + r_dot_rel * np.eye(3)
                    )
                    * dr_dot_vec_rel_dr_vec
                )
            )  # Jacobian of acceleration due to atmospheric drag w.r.t. position [s^{-2}]
            dr_ddot_vec_drag_dr_dot_vec = (
                -0.5
                * self.C_drag
                * self.A_drag
                / self.m
                * rho_atm
                * (r_dot_vec_rel * r_dot_vec_rel.T / r_dot_rel + r_dot_rel * np.eye(3))
            )  # Jacobian of acceleration due to atmospheric drag w.r.t. velocity [s^{-1}]

            # Superposition of all contributions
            F[i * 6 : i * 6 + 3, i * 6 + 3 : i * 6 + 6] = np.eye(3)
            F[i * 6 + 3 : i * 6 + 6, i * 6 : i * 6 + 3] = (
                dr_ddot_vec_grav_dr_vec
                + dr_ddot_vec_J2_dr_vec
                + dr_ddot_vec_drag_dr_vec
            )
            F[
                i * 6 + 3 : i * 6 + 6, i * 6 + 3 : i * 6 + 6
            ] = dr_ddot_vec_drag_dr_dot_vec

        return F

    def x_new(self, dt, x_old):
        """
        Computes the new state vector based on the current state vector and the time step.

        Parameters:
        dt (float): Time step.
        x_old (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        x_new (np.array): The new state vector of the satellite (position [km] and velocity [km / s]).
        """
        k1 = self.f_function(x_old)
        k2 = self.f_function(x_old + dt / 2 * k1)
        k3 = self.f_function(x_old + dt / 2 * k2)
        k4 = self.f_function(x_old + dt * k3)
        return x_old + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def x_new_and_F(self, dt, x_old):
        """
        Computes the new state vector based on the current state vector and the time step.

        Parameters:
        dt (float): Time step.
        x_old (np.array): The current state vector of the satellite (position [km] and velocity [km / s]).

        Returns:
        x_new (np.array): The new state vector of the satellite (position [km] and velocity [km / s]).
        """
        # New state calculation
        k1 = self.f_function(x_old)
        k2 = self.f_function(x_old + dt / 2 * k1)
        k3 = self.f_function(x_old + dt / 2 * k2)
        k4 = self.f_function(x_old + dt * k3)
        x_new = x_old + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Jacobian calculation
        K1 = self.F_jacobian(x_old)
        K2 = self.F_jacobian(x_old + dt / 2 * k1)
        K3 = self.F_jacobian(x_old + dt / 2 * k2)
        K4 = self.F_jacobian(x_old + dt * k3)
        dk1_dx = K1
        dk2_dx = K2 @ (np.eye(len(K1)) + dt / 2 * dk1_dx)
        dk3_dx = K3 @ (np.eye(len(K1)) + dt / 2 * dk2_dx)
        dk4_dx = K4 @ (np.eye(len(K1)) + dt * dk3_dx)
        F = np.eye(len(K1)) + dt / 6 * (dk1_dx + 2 * dk2_dx + 2 * dk3_dx + dk4_dx)

        return x_new, F
