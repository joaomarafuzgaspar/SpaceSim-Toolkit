import numpy as np

from earth import Earth, AtmosphericModel
from tudatpy.astro import element_conversion


def rotation_matrix_x(alpha):
    """Returns the 3D rotation matrix around the X-axis for a given angle alpha."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )


def rotation_matrix_y(beta):
    """Returns the 3D rotation matrix around the Y-axis for a given angle beta."""
    return np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )


def rotation_matrix_z(gamma):
    """Returns the 3D rotation matrix around the Z-axis for a given angle gamma."""
    return np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )


def coe2rv(
    semi_major_axis,
    eccentricity,
    inclination,
    argument_of_periapsis,
    longitude_of_ascending_node,
    true_anomaly,
):
    """
    Converts classical orbital elements to position and velocity vectors.

    Parameters:
    semi_major_axis (float): Semi-major axis [m].
    eccentricity (float): Eccentricity.
    inclination (float): Inclination [rad].
    argument_of_periapsis (float): Argument of periapsis [rad].
    longitude_ascending_node (float): Longitude of the ascending node [rad].
    true_anomaly (float): True anomaly [rad].

    Returns:
    np.array: Combined position and velocity vector in the geocentric equatorial frame [m, m / s].
    """
    earth = Earth()

    r = (
        semi_major_axis
        * (1 - eccentricity**2)
        / (1 + eccentricity * np.cos(true_anomaly))
    )
    r_vec_perifocal = r * np.array([np.cos(true_anomaly), np.sin(true_anomaly), 0])
    v_vec_perifocal = np.sqrt(
        earth.mu / (semi_major_axis * (1 - eccentricity**2))
    ) * np.array(
        [
            -np.sin(true_anomaly),
            eccentricity + np.cos(true_anomaly),
            0,
        ]
    )

    rotation_matrix_perifocal_to_geocentric_equatorial = (
        rotation_matrix_z(longitude_of_ascending_node)
        @ rotation_matrix_x(inclination)
        @ rotation_matrix_z(argument_of_periapsis)
    )
    r_vec_geocentric_equatorial = (
        rotation_matrix_perifocal_to_geocentric_equatorial @ r_vec_perifocal
    )
    v_vec_geocentric_equatorial = (
        rotation_matrix_perifocal_to_geocentric_equatorial @ v_vec_perifocal
    )

    rv_geocentric_equatorial_vec = np.concatenate(
        (r_vec_geocentric_equatorial, v_vec_geocentric_equatorial)
    )

    if rv_geocentric_equatorial_vec is None:
        return element_conversion.keplerian_to_cartesian_elementwise(
            semi_major_axis=semi_major_axis,
            eccentricity=eccentricity,
            inclination=inclination,
            argument_of_periapsis=argument_of_periapsis,
            longitude_of_ascending_node=longitude_of_ascending_node,
            true_anomaly=true_anomaly,
            gravitational_parameter=earth.mu,
        )
    return rv_geocentric_equatorial_vec


def rv2coe(rv_geocentric_equatorial_vec):
    """
    Converts position and velocity vectors to classical orbital elements.

    Parameters:
    rv_geocentric_equatorial_vec (np.array): Combined position and velocity vector in the geocentric equatorial frame [m, m / s].

    Returns:
    semi_major_axis (float): Semi-major axis [m].
    eccentricity (float): Eccentricity.
    inclination (float): Inclination [rad].
    argument_of_periapsis (float): Argument of periapsis [rad].
    longitude_of_ascending_node (float): Longitude of the ascending node [rad].
    true_anomaly (float): True anomaly [rad].
    """
    earth = Earth()

    r_vec = rv_geocentric_equatorial_vec[:3].flatten()
    v_vec = rv_geocentric_equatorial_vec[3:].flatten()
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    semi_major_axis = 1 / (2 / r - v**2 / earth.mu)

    h_vec = np.cross(r_vec, v_vec)
    e_vec = np.cross(v_vec, h_vec) / earth.mu - r_vec / r
    eccentricity = np.linalg.norm(e_vec)

    h = np.linalg.norm(h_vec)
    inclination = np.arccos(h_vec[2] / h)

    e_z_vec = np.array([0, 0, 1])
    n_vec = np.cross(e_z_vec, h_vec)
    n = np.linalg.norm(n_vec)
    longitude_of_ascending_node = np.arccos(n_vec[0] / n)
    if n_vec[1] < 0:
        longitude_of_ascending_node = 2 * np.pi - longitude_of_ascending_node

    argument_of_periapsis = np.arccos(np.dot(n_vec, e_vec) / (n * eccentricity))
    if e_vec[2] < 0:
        argument_of_periapsis = 2 * np.pi - argument_of_periapsis

    true_anomaly = np.arccos(np.dot(e_vec, r_vec) / (eccentricity * r))
    v_r = np.dot(v_vec, r_vec) / r
    if v_r < 0:
        true_anomaly = 2 * np.pi - true_anomaly

    if (
        semi_major_axis is None
        or eccentricity is None
        or inclination is None
        or argument_of_periapsis is None
        or longitude_of_ascending_node is None
        or true_anomaly is None
    ):
        return element_conversion.cartesian_to_keplerian(
            cartesian_elements=rv_geocentric_equatorial_vec,
            gravitational_parameter=earth.mu,
        )
    return np.array(
        [
            semi_major_axis,
            eccentricity,
            inclination,
            argument_of_periapsis,
            longitude_of_ascending_node,
            true_anomaly,
        ]
    )


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
        self.A_drag = 0.01  # Drag area [m^2]
        self.m = 1.0  # Mass [kg]

        self.earth = Earth()  # Earth model
        self.atm = AtmosphericModel()  # Atmospheric model

    def f_function(self, x_vec):
        """
        Computes the dynamics of the satellite based on the current state vector.
        The state vector includes position and velocity components.

        Parameters:
        t (float): Time.
        x_vec (np.array): The current state vector of the satellite (position [m] and velocity [m / s]).

        Returns:
        x_dot_vec (np.array): The derivative of the state vector.
        """
        x_dot_vec = np.zeros_like(x_vec)
        for i in range(int(x_vec.shape[0] / 6)):
            r_vec = x_vec[i * 6 : i * 6 + 3]  # Satellite position vector [m]
            x, y, z = r_vec  # Satellite position components [m]
            r = np.linalg.norm(r_vec)  # Satellite position magnitude [m]
            r_dot_vec = x_vec[
                i * 6 + 3 : i * 6 + 6
            ]  # Satellite velocity vector [m / s]

            # Compute contributions from Earth's gravitational force
            r_ddot_vec_grav = (
                -self.earth.mu * r_vec / r**3
            )  # Acceleration due to gravity [m / s^2]

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
            )  # Acceleration due to J2 effect [m / s^2]

            # Compute contributions from atmospheric drag
            h = r - self.earth.R  # Altitude [m]
            rho_atm = self.atm.get_rho(h)  # Atmospheric density [kg / m^3]

            r_dot_vec_rel = r_dot_vec - np.cross(
                self.earth.omega_vec, r_vec.reshape(-1)
            ).reshape(
                (3, 1)
            )  # Relative velocity vector [m / s]
            r_dot_rel = np.linalg.norm(
                r_dot_vec_rel
            )  # Relative velocity magnitude [m / s]

            r_ddot_vec_drag = (
                -0.5
                * self.C_drag
                * self.A_drag
                / self.m
                * rho_atm
                * r_dot_rel
                * r_dot_vec_rel
            )  # Acceleration due to atmospheric drag [m / s^2]

            # Superposition of all contributions
            x_dot_vec[i * 6 : i * 6 + 6] = np.concatenate(
                (r_dot_vec, r_ddot_vec_grav + r_ddot_vec_J2 + r_ddot_vec_drag)
            )

        return x_dot_vec

    def F_jacobian(self, x_vec):
        F = np.zeros((x_vec.shape[0], x_vec.shape[0]))
        for i in range(int(x_vec.shape[0] / 6)):
            r_vec = x_vec[i * 6 : i * 6 + 3]  # Satellite position vector [m]
            x, y, z = r_vec  # Satellite position components [m]
            r = np.linalg.norm(r_vec)  # Satellite position magnitude [m]
            r_dot_vec = x_vec[
                i * 6 + 3 : i * 6 + 6
            ]  # Satellite velocity vector [m / s]

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
            h = r - self.earth.R  # Altitude [m]
            rho_atm = self.atm.get_rho(h)  # Atmospheric density [kg / m^3]
            drho_atm_dr_vec = (
                -rho_atm / self.atm.get_H(h) * r_vec.T / r
            )  # Jacobian of atmospheric density w.r.t. position [kg / m^4]

            r_dot_vec_rel = r_dot_vec - np.cross(
                self.earth.omega_vec, r_vec.reshape(-1)
            ).reshape(
                (3, 1)
            )  # Relative velocity vector [m / s]
            r_dot_rel = np.linalg.norm(
                r_dot_vec_rel
            )  # Relative velocity magnitude [m / s]
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
            F[i * 6 + 3 : i * 6 + 6, i * 6 + 3 : i * 6 + 6] = (
                dr_ddot_vec_drag_dr_dot_vec
            )

        return F

    def x_new(self, dt, x_old):
        """
        Computes the new state vector based on the current state vector and the time step.

        Parameters:
        dt (float): Time step.
        x_old (np.array): The current state vector of the satellite (position [m] and velocity [m / s]).

        Returns:
        x_new (np.array): The new state vector of the satellite (position [m] and velocity [m / s]).
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
        x_old (np.array): The current state vector of the satellite (position [m] and velocity [m / s]).

        Returns:
        x_new (np.array): The new state vector of the satellite (position [m] and velocity [m / s]).
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

    def dPhidt(self, x, Phi):
        """
        Computes the time derivative of the state transition matrix.

        Parameters:
        x (np.array): The current state vector (position [m] and velocity [m / s]).
        Phi (np.array): The state transition matrix.

        Returns:
        dPhi_dt (np.array): The time derivative of the state transition matrix.
        """
        return self.F_jacobian(x) @ Phi

    def Phi(self, dt, x_old):
        """
        Computes the state transition matrix.

        Parameters:
        dt (float): Time step.
        x_old (np.array): The current state vector (position [m] and velocity [m / s]).

        Returns:
        Phi (np.array): The state transition matrix.
        """
        k1 = self.f_function(x_old)
        k2 = self.f_function(x_old + dt / 2 * k1)
        k3 = self.f_function(x_old + dt / 2 * k2)

        # State transition matrix calculation
        K1 = self.dPhidt(x_old, np.eye(len(x_old)))
        K2 = self.dPhidt(x_old + dt / 2 * k1, np.eye(len(x_old)) + dt / 2 * K1)
        K3 = self.dPhidt(x_old + dt / 2 * k2, np.eye(len(x_old)) + dt / 2 * K2)
        K4 = self.dPhidt(x_old + dt * k3, np.eye(len(x_old)) + dt * K3)
        Phi = np.eye(len(x_old)) + dt / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

        return Phi
