import numpy as np


class Earth:
    """
    This class models the Earth.
    """

    def __init__(self):
        """
        Initialize Earth parameters.
        """
        self.R = 6378.1363e3  # Earth's radius [m]
        self.mu = 3.986004418e14  # Earth's gravitational parameter [m^3 / s^2] | Note: TudatPy uses 398600441500000.0 = 3.986004415e14 [m^3 / s^2]
        self.J_2 = 0.00108262545  # Earth's oblateness term [adimensional]

        self.sidereal_day = (
            23 * 3600 + 56 * 60 + 4.09
        )  # Duration of one sidereal day [s / sidereal day]
        self.omega = 2 * np.pi / self.sidereal_day  # Earth's angular velocity [rad / s]
        self.omega_vec = np.array(
            [0, 0, self.omega]
        )  # Earth's angular velocity vector [rad / s]


class AtmosphericModel:
    """
    This class models the Earth's atmosphere as an exponential model.
    """

    def __init__(self):
        """
        Initialize atmospheric model parameters.
        """
        self.h_0 = (
            np.array(
                [
                    0,
                    25,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                    110,
                    120,
                    130,
                    140,
                    150,
                    180,
                    200,
                    250,
                    300,
                    350,
                    400,
                    450,
                    500,
                    600,
                    700,
                    800,
                    900,
                    1000,
                ]
            )
            * 1e3
        )  # Reference altitude [km] -> [m]

        self.rho_0 = np.array(
            [
                1.225,
                3.899e-2,
                1.774e-2,
                3.972e-3,
                1.057e-3,
                3.206e-4,
                8.770e-5,
                1.905e-5,
                3.396e-6,
                5.297e-7,
                9.661e-8,
                2.438e-8,
                8.484e-9,
                3.845e-9,
                2.070e-9,
                5.464e-10,
                2.789e-10,
                7.248e-11,
                2.418e-11,
                9.518e-12,
                3.725e-12,
                1.585e-12,
                6.967e-13,
                1.454e-13,
                3.614e-14,
                1.170e-14,
                5.245e-15,
                3.019e-15,
            ]
        )  # Reference density [kg / m^3]

        self.H = (
            np.array(
                [
                    7.249,
                    6.349,
                    6.682,
                    7.554,
                    8.382,
                    7.714,
                    6.549,
                    5.799,
                    5.382,
                    5.877,
                    7.263,
                    9.473,
                    12.636,
                    16.149,
                    22.523,
                    29.740,
                    37.105,
                    45.546,
                    53.628,
                    53.298,
                    58.515,
                    60.828,
                    63.822,
                    71.835,
                    88.667,
                    124.64,
                    181.05,
                    268.00,
                ]
            )
            * 1e3
        )  # Scale height [km] -> [m]

    def get_rho(self, h):
        """
        Returns the atmospheric density at the given altitude.

        Parameters:
        h (float): The altitude in m.

        Returns:
        float: The atmospheric density in kg / m^3.
        """
        # Find the index of the closest below altitude
        idx = 27 if h >= 1000e3 else np.argmax(h < self.h_0) - 2

        # Compute the density
        return self.rho_0[idx] * np.exp(-(h - self.h_0[idx]) / self.H[idx])

    def get_H(self, h):
        """
        Returns the atmospheric scale height at the given altitude.

        Parameters:
        h (float): The altitude in m.

        Returns:
        float: The atmospheric scale height in m.
        """
        # Find the index of the closest below altitude
        idx = 27 if h >= 1000e3 else np.argmax(h < self.h_0) - 2

        # Compute the scale height
        return self.H[idx]
