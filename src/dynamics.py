import numpy as np

from earth import Earth, AtmosphericModel
from config import SimulationConfig as config

# Load tudatpy modules
try:
    from tudatpy.interface import spice
    from tudatpy.util import result2array
    from tudatpy import numerical_simulation
    from tudatpy.numerical_simulation import (
        environment_setup,
        propagation_setup,
        estimation_setup,
    )

    TUDATPY_AVAILABLE = True
except ModuleNotFoundError:
    TUDATPY_AVAILABLE = False


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


def rv2mean_argument_of_latitude(rv_geocentric_equatorial_vec):
    _, eccentricity, _, argument_of_periapsis, _, true_anomaly = rv2coe(
        rv_geocentric_equatorial_vec
    )
    eccentric_anomaly = 2 * np.arctan(
        np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(true_anomaly / 2)
    )
    mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
    mean_argument_of_latitude = mean_anomaly + argument_of_periapsis
    return mean_argument_of_latitude


class SatelliteDynamics:
    """
    This class models the dynamics of a satellite in a LEO orbit.
    It accounts for Earth's gravitational force, Earth's oblateness (J2 effect), and atmospheric drag.
    """

    def __init__(self):
        """
        Initialize satellite parameters.
        """
        self.C_drag = 2.2  # Drag coefficient
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


class Propagator:
    def __init__(
        self,
        simulation_start_epoch,
        simulation_end_epoch,
        fixed_step_size,
        initial_conditions,
    ):
        # Load SPICE kernels
        spice.load_standard_kernels()

        # Create default body settings
        bodies_to_create = ["Sun", "Moon", "Venus", "Earth", "Mars", "Jupiter"]

        # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation
        )
        body_settings.get("Earth").atmosphere_settings = (
            environment_setup.atmosphere.nrlmsise00()
        )

        # Create system of bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Add satellite bodies to the system and set their mass
        mass = config.mass
        spacecrafts = ["Chief", "Deputy1", "Deputy2", "Deputy3"]
        for spacecraft in spacecrafts:
            bodies.create_empty_body(spacecraft)
            bodies.get(spacecraft).mass = mass

        # Add the aerodynamic interface to the environment
        reference_area = config.A_drag
        drag_coefficient = config.C_drag
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area, [drag_coefficient, 0.0, 0.0]
        )
        for spacecraft in spacecrafts:
            environment_setup.add_aerodynamic_coefficient_interface(
                bodies, spacecraft, aero_coefficient_settings
            )

        # Define radiation pressure settings
        reference_area_radiation = config.A_SRP
        radiation_pressure_coefficient = config.C_SRP
        occulting_bodies_dict = dict()
        occulting_bodies_dict["Sun"] = ["Earth"]

        # Create and add the radiation pressure interface to the environment
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun",
            reference_area_radiation,
            radiation_pressure_coefficient,
            occulting_bodies=occulting_bodies_dict["Sun"],
        )
        for spacecraft in spacecrafts:
            environment_setup.add_radiation_pressure_interface(
                bodies, spacecraft, radiation_pressure_settings
            )

        # Define accelerations acting on each vehicle
        #   1. Earth’s gravity field EGM96 spherical harmonic expansion up to degree and order 24
        #   2. Atmospheric drag NRLMSISE-00 model
        #   3. Cannon ball solar radiation pressure, assuming constant reflectivity coefficient and radiation area
        #   4. Third-body perturbations of the Sun, Moon, Venus, Mars and Jupiter
        accelerations_settings = dict(
            Sun=[
                propagation_setup.acceleration.point_mass_gravity(),
                propagation_setup.acceleration.cannonball_radiation_pressure(),
            ],
            Moon=[propagation_setup.acceleration.point_mass_gravity()],
            Venus=[propagation_setup.acceleration.point_mass_gravity()],
            Earth=[
                propagation_setup.acceleration.spherical_harmonic_gravity(24, 24),
                propagation_setup.acceleration.aerodynamic(),
            ],
            Mars=[propagation_setup.acceleration.point_mass_gravity()],
            Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        )
        acceleration_settings = {}
        bodies_to_propagate = []
        central_bodies = []
        for spacecraft in spacecrafts:
            acceleration_settings[spacecraft] = accelerations_settings
            bodies_to_propagate.append(spacecraft)
            central_bodies.append("Earth")

        # Create acceleration models
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )

        # Create numerical integrator settings
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            fixed_step_size
        )

        # Create termination settings
        termination_condition = propagation_setup.propagator.time_termination(
            simulation_end_epoch
        )

        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_conditions,
            simulation_start_epoch,
            integrator_settings,
            termination_condition,
        )

        # Setup parameters settings to propagate the state transition matrix
        parameter_settings = estimation_setup.parameter.initial_states(
            propagator_settings, bodies
        )

        # Create the parameters that will be estimated
        parameters_to_estimate = estimation_setup.create_parameter_set(
            parameter_settings, bodies
        )

        # Create the variational equation solver and propagate the dynamics
        self.variational_equations_solver = (
            numerical_simulation.create_variational_equations_solver(
                bodies,
                propagator_settings,
                parameters_to_estimate,
                simulate_dynamics_on_creation=True,
            )
        )

    def run(self):
        # Extract the resulting state history
        states = self.variational_equations_solver.state_history
        states = result2array(states)[:, 1:]

        return states


class Dynamics:
    def __init__(self):
        self.earth = Earth()
        self.atm = AtmosphericModel()

    def a_grav(self, x_vec):
        p_vec = x_vec[:3]
        p_norm = np.linalg.norm(p_vec)
        return -self.earth.mu * p_vec / p_norm**3

    def da_grav_dp_vec(self, x_vec):
        p_vec = x_vec[:3]
        p_norm = np.linalg.norm(p_vec)
        return -self.earth.mu * (
            np.eye(config.n_p) / p_norm**3 - 3 * np.outer(p_vec, p_vec) / p_norm**5
        )

    def d2a_grav_dp_vec_dp_vecT(self, x_vec):
        p_vec = x_vec[:3]
        p_norm = np.linalg.norm(p_vec)
        term1_der = -3 / p_norm**5 * np.kron(np.eye(config.n_p), p_vec)
        term2_der = 1 / p_norm**5 * (
            np.kron(np.eye(config.n_p).reshape(-1, 1), p_vec.T)
            + np.kron(p_vec, np.eye(config.n_p))
        ) - 5 / p_norm**7 * np.kron(p_vec, np.outer(p_vec, p_vec))
        return -self.earth.mu * (term1_der - 3 * term2_der)

    def a_J2(self, x_vec):
        p_vec = x_vec[:3]
        x, y, z = p_vec
        p_norm = np.linalg.norm(p_vec)
        return (
            -3
            * self.earth.J_2
            * self.earth.mu
            * self.earth.R**2
            / (2 * p_norm**5)
            * np.array(
                [
                    (1 - 5 * z**2 / p_norm**2) * x,
                    (1 - 5 * z**2 / p_norm**2) * y,
                    (3 - 5 * z**2 / p_norm**2) * z,
                ]
            )
        )

    def da_J2_dp_vec(self, x_vec):
        p_vec = x_vec[:3]
        x, y, z = p_vec
        p_norm = np.linalg.norm(p_vec)
        result = (
            -3
            * self.earth.J_2
            * self.earth.mu
            * self.earth.R**2
            / (2 * p_norm**9)
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
        )
        return np.squeeze(result)

    def d2a_J2_dp_vec_dp_vecT(self, x_vec):
        p_vec = x_vec[:3]
        x, y, z = p_vec
        p_norm = np.linalg.norm(p_vec)
        result = (
            -3
            * self.earth.J_2
            * self.earth.mu
            * self.earth.R**2
            / (2 * p_norm**11)
            * np.array(
                [
                    [
                        20 * x**5
                        + 5 * x**3 * y**2
                        - 205 * x**3 * z**2
                        - 15 * x * y**4
                        + 75 * x * y**2 * z**2
                        + 90 * x * z**4,
                        30 * x**4 * y
                        + 25 * x**2 * y**3
                        - 255 * x**2 * y * z**2
                        - 5 * y**5
                        + 25 * y**3 * z**2
                        + 30 * y * z**4,
                        90 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        - 205 * x**2 * z**3
                        - 15 * y**4 * z
                        + 5 * y**2 * z**3
                        + 20 * z**5,
                    ],
                    [
                        30 * x**4 * y
                        + 25 * x**2 * y**3
                        - 255 * x**2 * y * z**2
                        - 5 * y**5
                        + 25 * y**3 * z**2
                        + 30 * y * z**4,
                        -5 * x**5
                        + 25 * x**3 * y**2
                        + 25 * x**3 * z**2
                        + 30 * x * y**4
                        - 255 * x * y**2 * z**2
                        + 30 * x * z**4,
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                    ],
                    [
                        90 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        - 205 * x**2 * z**3
                        - 15 * y**4 * z
                        + 5 * y**2 * z**3
                        + 20 * z**5,
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                        -15 * x**5
                        - 30 * x**3 * y**2
                        + 180 * x**3 * z**2
                        - 15 * x * y**4
                        + 180 * x * y**2 * z**2
                        - 120 * x * z**4,
                    ],
                    [
                        30 * x**4 * y
                        + 25 * x**2 * y**3
                        - 255 * x**2 * y * z**2
                        - 5 * y**5
                        + 25 * y**3 * z**2
                        + 30 * y * z**4,
                        -5 * x**5
                        + 25 * x**3 * y**2
                        + 25 * x**3 * z**2
                        + 30 * x * y**4
                        - 255 * x * y**2 * z**2
                        + 30 * x * z**4,
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                    ],
                    [
                        -5 * x**5
                        + 25 * x**3 * y**2
                        + 25 * x**3 * z**2
                        + 30 * x * y**4
                        - 255 * x * y**2 * z**2
                        + 30 * x * z**4,
                        -15 * x**4 * y
                        + 5 * x**2 * y**3
                        + 75 * x**2 * y * z**2
                        + 20 * y**5
                        - 205 * y**3 * z**2
                        + 90 * y * z**4,
                        -15 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        + 5 * x**2 * z**3
                        + 90 * y**4 * z
                        - 205 * y**2 * z**3
                        + 20 * z**5,
                    ],
                    [
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                        -15 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        + 5 * x**2 * z**3
                        + 90 * y**4 * z
                        - 205 * y**2 * z**3
                        + 20 * z**5,
                        -15 * x**4 * y
                        - 30 * x**2 * y**3
                        + 180 * x**2 * y * z**2
                        - 15 * y**5
                        + 180 * y**3 * z**2
                        - 120 * y * z**4,
                    ],
                    [
                        90 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        - 205 * x**2 * z**3
                        - 15 * y**4 * z
                        + 5 * y**2 * z**3
                        + 20 * z**5,
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                        -15 * x**5
                        - 30 * x**3 * y**2
                        + 180 * x**3 * z**2
                        - 15 * x * y**4
                        + 180 * x * y**2 * z**2
                        - 120 * x * z**4,
                    ],
                    [
                        105 * x**3 * y * z + 105 * x * y**3 * z - 210 * x * y * z**3,
                        -15 * x**4 * z
                        + 75 * x**2 * y**2 * z
                        + 5 * x**2 * z**3
                        + 90 * y**4 * z
                        - 205 * y**2 * z**3
                        + 20 * z**5,
                        -15 * x**4 * y
                        - 30 * x**2 * y**3
                        + 180 * x**2 * y * z**2
                        - 15 * y**5
                        + 180 * y**3 * z**2
                        - 120 * y * z**4,
                    ],
                    [
                        -15 * x**5
                        - 30 * x**3 * y**2
                        + 180 * x**3 * z**2
                        - 15 * x * y**4
                        + 180 * x * y**2 * z**2
                        - 120 * x * z**4,
                        -15 * x**4 * y
                        - 30 * x**2 * y**3
                        + 180 * x**2 * y * z**2
                        - 15 * y**5
                        + 180 * y**3 * z**2
                        - 120 * y * z**4,
                        -75 * x**4 * z
                        - 150 * x**2 * y**2 * z
                        + 200 * x**2 * z**3
                        - 75 * y**4 * z
                        + 200 * y**2 * z**3
                        - 40 * z**5,
                    ],
                ]
            )
        )

        return np.squeeze(result)

    def a_drag(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        return (
            -1
            / 2
            * config.C_drag
            * config.A_drag
            / config.mass
            * rho
            * v_rel_norm
            * v_vec_rel
        )

    def da_drag_dp_vec(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        drho_dp_vec = -rho / self.atm.get_H(h) * p_vec / p_norm
        dv_vec_rel_dp_vec = np.array(
            [[0, self.earth.omega, 0], [-self.earth.omega, 0, 0], [0, 0, 0]]
        )
        return (
            -1
            / 2
            * config.C_drag
            * config.A_drag
            / config.mass
            * (
                rho
                * (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)
                @ dv_vec_rel_dp_vec
                + v_rel_norm * v_vec_rel * drho_dp_vec.T
            )
        )

    def da_drag_dv_vec(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        return (
            -1
            / 2
            * config.C_drag
            * config.A_drag
            / config.mass
            * rho
            * (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)
        )

    def d2a_drag_dp_vec_dp_vecT(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        H = self.atm.get_H(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        drho_dp_vec = -rho / H * p_vec / p_norm
        d2rho_dp_vec2 = rho / H**2 * p_vec * p_vec.T / p_norm**2 - rho / H * (
            np.eye(3) / p_norm - p_vec * p_vec.T / p_norm**3
        )
        dv_vec_rel_dp_vec = np.array(
            [[0, self.earth.omega, 0], [-self.earth.omega, 0, 0], [0, 0, 0]]
        )
        term11 = (
            rho
            * np.kron(np.eye(3), dv_vec_rel_dp_vec).T
            @ (
                1
                / v_rel_norm
                * (
                    np.kron(np.eye(3), v_vec_rel)
                    + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T)
                    + np.kron(v_vec_rel, np.eye(3))
                )
                - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))
            )
            @ dv_vec_rel_dp_vec
        )
        term12 = np.kron(np.eye(3), drho_dp_vec) @ (
            (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)
            @ dv_vec_rel_dp_vec
        )
        term1 = term11 + term12
        term21 = np.kron(
            (
                (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)
                @ dv_vec_rel_dp_vec
            ).reshape(-1, 1),
            drho_dp_vec.T,
        )
        term22 = np.kron(v_rel_norm * v_vec_rel, np.eye(3)) @ d2rho_dp_vec2
        term2 = term21 + term22
        return -1 / 2 * config.C_drag * config.A_drag / config.mass * (term1 + term2)

    def d2a_drag_dv_vec_dp_vecT(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        drho_dp_vec = -rho / self.atm.get_H(h) * p_vec / p_norm
        dv_vec_rel_dp_vec = np.array(
            [[0, self.earth.omega, 0], [-self.earth.omega, 0, 0], [0, 0, 0]]
        )
        term1 = (
            rho
            * (
                1
                / v_rel_norm
                * (
                    np.kron(np.eye(3), v_vec_rel)
                    + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T)
                    + np.kron(v_vec_rel, np.eye(3))
                )
                - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))
            )
            @ dv_vec_rel_dp_vec
        )
        term2 = np.kron(
            (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm).reshape(
                -1, 1
            ),
            drho_dp_vec.T,
        )
        return -1 / 2 * config.C_drag * config.A_drag / config.mass * (term1 + term2)

    def d2a_drag_dp_vec_dv_vecT(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        drho_dp_vec = -rho / self.atm.get_H(h) * p_vec / p_norm
        dv_vec_rel_dp_vec = np.array(
            [[0, self.earth.omega, 0], [-self.earth.omega, 0, 0], [0, 0, 0]]
        )
        term1 = (
            rho
            * np.kron(np.eye(3), dv_vec_rel_dp_vec).T
            @ (
                1
                / v_rel_norm
                * (
                    np.kron(np.eye(3), v_vec_rel)
                    + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T)
                    + np.kron(v_vec_rel, np.eye(3))
                )
                - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))
            )
        )
        term2 = np.kron(np.eye(3), drho_dp_vec) @ (
            v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm
        )
        return -1 / 2 * config.C_drag * config.A_drag / config.mass * (term1 + term2)

    def d2a_drag_dv_vec_dv_vecT(self, x_vec):
        p_vec = x_vec[:3]
        v_vec = x_vec[3:]
        p_norm = np.linalg.norm(p_vec)
        h = p_norm - self.earth.R
        rho = self.atm.get_rho(h)
        omega_vec = np.array([[0], [0], [self.earth.omega]])
        v_vec_rel = v_vec - np.cross(
            omega_vec.reshape(
                3,
            ),
            p_vec.reshape(
                3,
            ),
        ).reshape(3, 1)
        v_rel_norm = np.linalg.norm(v_vec_rel)
        return (
            -1
            / 2
            * config.C_drag
            * config.A_drag
            / config.mass
            * rho
            * (
                1
                / v_rel_norm
                * (
                    np.kron(np.eye(3), v_vec_rel)
                    + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T)
                    + np.kron(v_vec_rel, np.eye(3))
                )
                - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))
            )
        )

    def diff_eq(self, x_vec):
        x_dot_vec = np.zeros_like(x_vec)
        for i in range(int(x_vec.shape[0] / config.n_x)):
            x_vec_i = x_vec[i * config.n_x : i * config.n_x + config.n_x]
            v_vec_i = x_vec_i[config.n_p : config.n_x]
            x_dot_vec[i * config.n_x : i * config.n_x + config.n_x] = np.concatenate(
                (
                    v_vec_i,
                    self.a_grav(x_vec_i) + self.a_J2(x_vec_i) + self.a_drag(x_vec_i),
                )
            )
        return x_dot_vec

    def Ddiff_eq(self, x_vec):
        first_order_der = np.zeros((config.n, config.n))
        for i in range(int(x_vec.shape[0] / config.n_x)):
            x_vec_i = x_vec[i * config.n_x : i * config.n_x + config.n_x]
            first_order_der[
                i * config.n_x : i * config.n_x + config.n_p,
                i * config.n_x + config.n_p : i * config.n_x + config.n_x,
            ] = np.eye(config.n_p)
            first_order_der[
                i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                i * config.n_x : i * config.n_x + config.n_p,
            ] = (
                self.da_grav_dp_vec(x_vec_i)
                + self.da_J2_dp_vec(x_vec_i)
                + self.da_drag_dp_vec(x_vec_i)
            )
            first_order_der[
                i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                i * config.n_x + config.n_p : i * config.n_x + config.n_x,
            ] = self.da_drag_dv_vec(x_vec_i)
        return first_order_der

    def Hdiff_eq(self, x_vec):
        second_order_der = np.zeros((config.n, config.n, config.n))
        for i in range(int(x_vec.shape[0] / config.n_x)):
            x_vec_i = x_vec[i * config.n_x : i * config.n_x + config.n_x]
            aux_pp = (
                self.d2a_grav_dp_vec_dp_vecT(x_vec_i)
                + self.d2a_J2_dp_vec_dp_vecT(x_vec_i)
                + self.d2a_drag_dp_vec_dp_vecT(x_vec_i)
            ).reshape((config.n_p, config.n_p, config.n_p))
            aux_pv = self.d2a_drag_dp_vec_dv_vecT(x_vec_i).reshape(
                (config.n_p, config.n_p, config.n_p)
            )
            aux_vp = self.d2a_drag_dv_vec_dp_vecT(x_vec_i).reshape(
                (config.n_p, config.n_p, config.n_p)
            )
            aux_vv = self.d2a_drag_dv_vec_dv_vecT(x_vec_i).reshape(
                (config.n_p, config.n_p, config.n_p)
            )
            for j in range(config.n_p):
                second_order_der[
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x : i * config.n_x + config.n_p,
                    i * config.n_x + j,
                ] = aux_pp[:, :, j]
                second_order_der[
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x : i * config.n_x + config.n_p,
                    i * config.n_x + j + config.n_p,
                ] = aux_pv[:, :, j]
                second_order_der[
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x + j,
                ] = aux_vp[:, :, j]
                second_order_der[
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x + config.n_p : i * config.n_x + config.n_x,
                    i * config.n_x + j + config.n_p,
                ] = aux_vv[:, :, j]
        return second_order_der.reshape((config.n * config.n, config.n))

    def f(self, dt, x_old):
        k1 = self.diff_eq(x_old)
        k2 = self.diff_eq(x_old + dt / 2 * k1)
        k3 = self.diff_eq(x_old + dt / 2 * k2)
        k4 = self.diff_eq(x_old + dt * k3)
        return x_old + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def Df(self, dt, x_old):
        k1 = self.diff_eq(x_old)
        k2 = self.diff_eq(x_old + dt / 2 * k1)
        k3 = self.diff_eq(x_old + dt / 2 * k2)

        Dk1 = self.Ddiff_eq(x_old)
        Dk2 = self.Ddiff_eq(x_old + dt / 2 * k1) @ (np.eye(config.n) + dt / 2 * Dk1)
        Dk3 = self.Ddiff_eq(x_old + dt / 2 * k2) @ (np.eye(config.n) + dt / 2 * Dk2)
        Dk4 = self.Ddiff_eq(x_old + dt * k3) @ (np.eye(config.n) + dt * Dk3)
        return np.eye(config.n) + dt / 6 * (Dk1 + 2 * Dk2 + 2 * Dk3 + Dk4)

    def f_and_Df(self, dt, x_old):  # more efficient
        k1 = self.diff_eq(x_old)
        k2 = self.diff_eq(x_old + dt / 2 * k1)
        k3 = self.diff_eq(x_old + dt / 2 * k2)
        k4 = self.diff_eq(x_old + dt * k3)

        Dk1 = self.Ddiff_eq(x_old)
        Dk2 = self.Ddiff_eq(x_old + dt / 2 * k1) @ (np.eye(config.n) + dt / 2 * Dk1)
        Dk3 = self.Ddiff_eq(x_old + dt / 2 * k2) @ (np.eye(config.n) + dt / 2 * Dk2)
        Dk4 = self.Ddiff_eq(x_old + dt * k3) @ (np.eye(config.n) + dt * Dk3)
        return x_old + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), np.eye(
            config.n
        ) + dt / 6 * (Dk1 + 2 * Dk2 + 2 * Dk3 + Dk4)

    def Hf(self, dt, x_old):
        k1 = self.diff_eq(x_old)
        k2 = self.diff_eq(x_old + dt / 2 * k1)
        k3 = self.diff_eq(x_old + dt / 2 * k2)

        Dk1 = self.Ddiff_eq(x_old)
        Dk2 = self.Ddiff_eq(x_old + dt / 2 * k1) @ (np.eye(config.n) + dt / 2 * Dk1)
        Dk3 = self.Ddiff_eq(x_old + dt / 2 * k2) @ (np.eye(config.n) + dt / 2 * Dk2)

        Hk1 = self.Hdiff_eq(x_old)
        Hk2 = np.kron(
            np.eye(config.n), np.eye(config.n) + dt / 2 * Dk1
        ).T @ self.Hdiff_eq(x_old + dt / 2 * k1) @ (
            np.eye(config.n) + dt / 2 * Dk1
        ) + np.kron(
            self.Ddiff_eq(x_old + dt / 2 * k1), np.eye(config.n)
        ) @ (
            dt / 2 * Hk1
        )
        Hk3 = np.kron(
            np.eye(config.n), np.eye(config.n) + dt / 2 * Dk2
        ).T @ self.Hdiff_eq(x_old + dt / 2 * k2) @ (
            np.eye(config.n) + dt / 2 * Dk2
        ) + np.kron(
            self.Ddiff_eq(x_old + dt / 2 * k2), np.eye(config.n)
        ) @ (
            dt / 2 * Hk2
        )
        Hk4 = np.kron(np.eye(config.n), np.eye(config.n) + dt * Dk3).T @ self.Hdiff_eq(
            x_old + dt * k3
        ) @ (np.eye(config.n) + dt * Dk3) + np.kron(
            self.Ddiff_eq(x_old + dt * k3), np.eye(config.n)
        ) @ (
            dt * Hk3
        )
        return dt / 6 * (Hk1 + 2 * Hk2 + 2 * Hk3 + Hk4)
