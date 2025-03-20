import numpy as np

n_p = 3
n_x = 6
# mu = Earth().mu
mu = 1.0
J_2 = 1.0
# J_2 = Earth().J_2
R = 1.0
# R = Earth().R
C_drag = 1.0
A_drag = 1.0
m = 1.0
rho_0 = 1.0
h_0 = 1.0
H = 1.0
omega = 1.0

def a_grav(p_vec):
    p_norm = np.linalg.norm(p_vec)
    return -mu * p_vec / p_norm**3

def da_grav_dp_vec(p_vec):
    n = len(p_vec)
    p_norm = np.linalg.norm(p_vec)
    return -mu * (np.eye(n) / p_norm**3 - 3 * np.outer(p_vec, p_vec) / p_norm**5)

def d2a_grav_dp_vec_dp_vecT(p_vec):
    n = len(p_vec)
    p_norm = np.linalg.norm(p_vec)
    term1_der = -3 / p_norm**5 * np.kron(np.eye(n), p_vec)
    term2_der = 1 / p_norm**5 * (np.kron(np.eye(n).reshape(-1, 1), p_vec.T) + np.kron(p_vec, np.eye(n))) - 5 / p_norm**7 * np.kron(p_vec, np.outer(p_vec, p_vec))
    return -mu * (term1_der - 3 * term2_der)

def a_J2(p_vec):
    x, y, z = p_vec
    p_norm = np.linalg.norm(p_vec)
    return -3 * J_2 * mu * R**2 / (2 * p_norm**5) * np.array([(1 - 5 * z**2 / p_norm**2) * x, 
                                                              (1 - 5 * z**2 / p_norm**2) * y, 
                                                              (3 - 5 * z**2 / p_norm**2) * z])

def da_J2_dp_vec(p_vec):
    x, y, z = p_vec
    p_norm = np.linalg.norm(p_vec)
    result = -3 * J_2 * mu * R**2 / (2 * p_norm**9) * np.array([[-4 * x**4 - 3 * x**2 * y**2 + 27 * x**2 * z**2 + y**4 - 3 * y**2 * z**2 - 4 * z**4, -5 * x**3 * y - 5 * x * y**3 + 30 * x * y * z**2, -15 * x**3 * z - 15 * x * y**2 * z + 20 * x * z**3],
                                                                [-5 * x**3 * y - 5 * x * y**3 + 30 * x * y * z**2, x**4 - 3 * x**2 * y**2 - 3 * x**2 * z**2 - 4 * y**4 + 27 * y**2 * z**2 - 4 * z**4, -15 * x**2 * y * z - 15 * y**3 * z + 20 * y * z**3],
                                                                [-15 * x**3 * z - 15 * x * y**2 * z + 20 * x * z**3, -15 * x**2 * y * z - 15 * y**3 * z + 20 * y * z**3, 3 * x**4+ 6 * x**2 * y**2- 24 * x**2 * z**2+ 3 * y**4- 24 * y**2 * z**2+ 8 * z**4]])
    return np.squeeze(result)

def d2a_J2_dp_vec_dp_vecT(p_vec):
    x, y, z = p_vec
    p_norm = np.linalg.norm(p_vec)
    result = -3 * J_2 * mu * R**2 / (2 * p_norm**11) * np.array([[20*x**5 + 5*x**3*y**2 - 205*x**3*z**2 - 15*x*y**4 + 75*x*y**2*z**2 + 90*x*z**4, 30*x**4*y + 25*x**2*y**3 - 255*x**2*y*z**2 - 5*y**5 + 25*y**3*z**2 + 30*y*z**4, 90*x**4*z + 75*x**2*y**2*z - 205*x**2*z**3 - 15*y**4*z + 5*y**2*z**3 + 20*z**5],
                                                                 [30*x**4*y + 25*x**2*y**3 - 255*x**2*y*z**2 - 5*y**5 + 25*y**3*z**2 + 30*y*z**4, -5*x**5 + 25*x**3*y**2 + 25*x**3*z**2 + 30*x*y**4 - 255*x*y**2*z**2 + 30*x*z**4, 105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3],
                                                                 [90*x**4*z + 75*x**2*y**2*z - 205*x**2*z**3 - 15*y**4*z + 5*y**2*z**3 + 20*z**5, 105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3, -15*x**5 - 30*x**3*y**2 + 180*x**3*z**2 - 15*x*y**4 + 180*x*y**2*z**2 - 120*x*z**4],
                                                                 [30*x**4*y + 25*x**2*y**3 - 255*x**2*y*z**2 - 5*y**5 + 25*y**3*z**2 + 30*y*z**4, -5*x**5 + 25*x**3*y**2 + 25*x**3*z**2 + 30*x*y**4 - 255*x*y**2*z**2 + 30*x*z**4, 105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3],
                                                                 [-5*x**5 + 25*x**3*y**2 + 25*x**3*z**2 + 30*x*y**4 - 255*x*y**2*z**2 + 30*x*z**4, -15*x**4*y + 5*x**2*y**3 + 75*x**2*y*z**2 + 20*y**5 - 205*y**3*z**2 + 90*y*z**4, -15*x**4*z + 75*x**2*y**2*z + 5*x**2*z**3 + 90*y**4*z - 205*y**2*z**3 + 20*z**5],
                                                                 [105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3, -15*x**4*z + 75*x**2*y**2*z + 5*x**2*z**3 + 90*y**4*z - 205*y**2*z**3 + 20*z**5, -15*x**4*y - 30*x**2*y**3 + 180*x**2*y*z**2 - 15*y**5 + 180*y**3*z**2 - 120*y*z**4],
                                                                 [90*x**4*z + 75*x**2*y**2*z - 205*x**2*z**3 - 15*y**4*z + 5*y**2*z**3 + 20*z**5, 105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3, -15*x**5 - 30*x**3*y**2 + 180*x**3*z**2 - 15*x*y**4 + 180*x*y**2*z**2 - 120*x*z**4],
                                                                 [105*x**3*y*z + 105*x*y**3*z - 210*x*y*z**3, -15*x**4*z + 75*x**2*y**2*z + 5*x**2*z**3 + 90*y**4*z - 205*y**2*z**3 + 20*z**5, -15*x**4*y - 30*x**2*y**3 + 180*x**2*y*z**2 - 15*y**5 + 180*y**3*z**2 - 120*y*z**4],
                                                                 [-15*x**5 - 30*x**3*y**2 + 180*x**3*z**2 - 15*x*y**4 + 180*x*y**2*z**2 - 120*x*z**4, -15*x**4*y - 30*x**2*y**3 + 180*x**2*y*z**2 - 15*y**5 + 180*y**3*z**2 - 120*y*z**4, -75*x**4*z - 150*x**2*y**2*z + 200*x**2*z**3 - 75*y**4*z + 200*y**2*z**3 - 40*z**5]])
    
    return np.squeeze(result)

def a_drag(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    return -1 / 2 * C_drag * A_drag / m * rho * v_rel_norm * v_vec_rel

def da_drag_dp_vec(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    drho_dp_vec = -rho_0 / H * np.exp(-(h - h_0) / H) * p_vec / p_norm
    dv_vec_rel_dp_vec = np.array([[0, omega, 0], [-omega, 0, 0], [0, 0, 0]])
    return -1 / 2 * C_drag * A_drag / m * (rho * (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm) @ dv_vec_rel_dp_vec + v_rel_norm * v_vec_rel * drho_dp_vec.T)

def da_drag_dv_vec(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    return -1 / 2 * C_drag * A_drag / m * rho * (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)

def d2a_drag_dp_vec_dp_vecT(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    drho_dp_vec = -rho_0 / H * np.exp(-(h - h_0) / H) * p_vec / p_norm
    d2rho_dp_vec2 = rho_0 / H**2 * np.exp(-(h - h_0) / H) * p_vec * p_vec.T / p_norm**2 - rho_0 / H * np.exp(-(h - h_0) / H) * (np.eye(3) / p_norm - p_vec * p_vec.T / p_norm**3)
    dv_vec_rel_dp_vec = np.array([[0, omega, 0], [-omega, 0, 0], [0, 0, 0]])
    term11 = rho * np.kron(np.eye(3), dv_vec_rel_dp_vec).T @ (1 / v_rel_norm * (np.kron(np.eye(3), v_vec_rel) + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T) + np.kron(v_vec_rel, np.eye(3))) - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))) @ dv_vec_rel_dp_vec
    term12 = np.kron(np.eye(3), drho_dp_vec) @ ((v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm) @ dv_vec_rel_dp_vec)
    term1 = term11 + term12
    term21 = np.kron(((v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm) @ dv_vec_rel_dp_vec).reshape(-1, 1), drho_dp_vec.T)
    term22 = np.kron(v_rel_norm * v_vec_rel, np.eye(3)) @ d2rho_dp_vec2
    term2 = term21 + term22
    return -1 / 2 * C_drag * A_drag / m * (term1 + term2)

def d2a_drag_dv_vec_dp_vecT(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    drho_dp_vec = -rho_0 / H * np.exp(-(h - h_0) / H) * p_vec / p_norm
    dv_vec_rel_dp_vec = np.array([[0, omega, 0], [-omega, 0, 0], [0, 0, 0]])
    term1 = rho * (1 / v_rel_norm * (np.kron(np.eye(3), v_vec_rel) + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T) + np.kron(v_vec_rel, np.eye(3))) - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel))) @ dv_vec_rel_dp_vec
    term2 = np.kron((v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm).reshape(-1, 1), drho_dp_vec.T)
    return -1 / 2 * C_drag * A_drag / m * (term1 + term2)

def d2a_drag_dp_vec_dv_vecT(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    drho_dp_vec = -rho_0 / H * np.exp(-(h - h_0) / H) * p_vec / p_norm
    dv_vec_rel_dp_vec = np.array([[0, omega, 0], [-omega, 0, 0], [0, 0, 0]])
    term1 = rho * np.kron(np.eye(3), dv_vec_rel_dp_vec).T @ (1 / v_rel_norm * (np.kron(np.eye(3), v_vec_rel) + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T) + np.kron(v_vec_rel, np.eye(3))) - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel)))
    term2 = np.kron(np.eye(3), drho_dp_vec) @ (v_rel_norm * np.eye(3) + v_vec_rel * v_vec_rel.T / v_rel_norm)
    return -1 / 2 * C_drag * A_drag / m * (term1 + term2)

def d2a_drag_dv_vec_dv_vecT(x_vec):
    p_vec = x_vec[:3]
    v_vec = x_vec[3:]
    p_norm = np.linalg.norm(p_vec)
    h = p_norm - R
    rho = rho_0 * np.exp(-(h - h_0) / H)
    omega_vec = np.array([[0], [0], [omega]]) 
    v_vec_rel = v_vec - np.cross(omega_vec.reshape(3,), p_vec.reshape(3,)).reshape(3, 1)
    v_rel_norm = np.linalg.norm(v_vec_rel)
    return -1 / 2 * C_drag * A_drag / m * rho * (1 / v_rel_norm * (np.kron(np.eye(3), v_vec_rel) + np.kron(np.eye(3).reshape(-1, 1), v_vec_rel.T) + np.kron(v_vec_rel, np.eye(3))) - 1 / v_rel_norm**3 * np.kron(v_vec_rel, np.outer(v_vec_rel, v_vec_rel)))