import numpy as np

from dynamics import SatelliteDynamics


class HCMCI:
    def __init__(self, W, V):
        self.W = W
        self.V = V

        self.dynamic_model = SatelliteDynamics()

    def h_function_chief(self, x_vec):
        return x_vec[0:3]

    def H_jacobian_chief(self):
        H = np.zeros((3, 24))
        H[0:3, 0:3] = np.eye(3)
        return H

    def h_function_deputy(self, x_vec):
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

    def H_jacobian_deputy(self, x_vec):
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

        H = np.zeros((6, 24))
        H[0, 0:3] = -(r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[0, 6:9] = (r_deputy1 - r_chief).reshape(-1) / range_deputy1_chief
        H[1, 6:9] = (r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[1, 12:15] = -(r_deputy1 - r_deputy2).reshape(-1) / range_deputy1_deputy2
        H[2, 6:9] = (r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[2, 18:21] = -(r_deputy1 - r_deputy3).reshape(-1) / range_deputy1_deputy3
        H[3, 0:3] = -(r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[3, 12:15] = (r_deputy2 - r_chief).reshape(-1) / range_deputy2_chief
        H[4, 12:15] = (r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[4, 18:21] = -(r_deputy2 - r_deputy3).reshape(-1) / range_deputy2_deputy3
        H[5, 0:3] = -(r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        H[5, 18:21] = (r_deputy3 - r_chief).reshape(-1) / range_deputy3_chief
        return H

    def prediction(self, dt, x_est_old, Omega_est_old, y):
        x_pred, F = self.dynamic_model.x_new_and_F(dt, x_est_old)
        Omega_pred = (
            self.W
            - self.W
            @ F
            @ np.linalg.inv(Omega_est_old + F.T @ self.W @ F)
            @ F.T
            @ self.W
        )
        q_pred = Omega_pred @ x_pred

        H = np.vstack((self.H_jacobian_chief(), self.H_jacobian_deputy(x_pred)))
        y_est = np.concatenate(
            (self.h_function_chief(x_pred), self.h_function_deputy(x_pred))
        )
        delta_q_est_new = H.T @ self.V @ (y - y_est + H @ x_pred)
        delta_Omega_est_new = H.T @ self.V @ H

        return q_pred, Omega_pred, delta_q_est_new, delta_Omega_est_new

    def init_consensus(self, delta_q, delta_Omega, q, Omega, L):
        delta_q_vec = np.zeros((24, 1, L + 1))
        delta_q_vec[:, :, 0] = delta_q
        delta_Omega_vec = np.zeros((24, 24, L + 1))
        delta_Omega_vec[:, :, 0] = delta_Omega
        q_vec = np.zeros((24, 1, L + 1))
        q_vec[:, :, 0] = q
        Omega_vec = np.zeros((24, 24, L + 1))
        Omega_vec[:, :, 0] = Omega

        return delta_q_vec, delta_Omega_vec, q_vec, Omega_vec

    def correction(self, delta_q_vec, delta_Omega_vec, q_vec, Omega_vec, gamma):
        q_est_new = q_vec[:, :, -1] + gamma * delta_q_vec[:, :, -1]
        Omega_est_new = Omega_vec[:, :, -1] + gamma * delta_Omega_vec[:, :, -1]
        x_est_new = np.linalg.inv(Omega_est_new) @ q_est_new

        return x_est_new, Omega_est_new
