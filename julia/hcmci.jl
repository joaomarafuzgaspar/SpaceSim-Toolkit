using Random
using Debugger
using Statistics
using ProgressMeter
using BlockDiagonals
using CSV, DataFrames
using SatelliteDynamics
using LinearAlgebra, MATLAB, StaticArrays
using SparseArrays, IterativeSolvers, Infiltrator


struct Spacecraft
    X_est::Matrix{Any}
    X_true::Matrix{Any}
    Omega::Matrix{Any}
end


function dynamics(x_vec)
    # Earth
    R = 6378.1363                             # Earth's radius [km]
    mu = 3.986004418e5                        # Earth's gravitational parameter [km^3 / s^2]
    J_2 = 0.00108262545                       # Earth's oblateness term [adimensional]
    sidereal_day = 23 * 3600 + 56 * 60 + 4.09 # Duration of one sidereal day [s / sidereal day]
    omega = 2 * pi / sidereal_day             # Earth's angular velocity [rad / s]
    omega_vec = [0; 0; omega]                 # Earth's angular velocity vector [rad / s]

    # Satellite 
    C_drag = 2.22         # Drag coefficient
    A_drag = 0.01 * 1e-6  # Drag area [m^2] -> [km^2]
    m = 1.0               # Mass [kg]

    nx = size(x_vec, 1)
    x_dot_vec = zeros(nx, size(x_vec, 2))

    F = zeros(nx, nx)
    for i = 1:nx/6
        i = round(Int, i)
        r_vec = x_vec[((i-1)*6+1):((i-1)*6+3)] # Satellite position vector [km]
        x = r_vec[1] # Satellite x coordinate [km]
        y = r_vec[2] # Satellite y coordinate [km]
        z = r_vec[3] # Satellite z coordinate [km]
        r = norm(r_vec) # Satellite position magnitude [km]
        r_dot_vec = x_vec[((i-1)*6+4):((i-1)*6+6)] # Satellite velocity vector [km / s]

        # Compute contributions from Earth's gravitational force
        r_ddot_vec_grav = -mu * r_vec / r^3 # Acceleration due to gravity [km / s^2]
        dr_ddot_vec_grav_dr_vec = -mu * (I(3) / r^3 - 3 * r_vec * r_vec' / r^5)  # Jacobian of acceleration due to gravity w.r.t. position [s^{-2}]

        # Compute contributions from Earth's oblateness (J2 effect)
        r_ddot_vec_J2 = -(3 * J_2 * mu * R^2) / (2 * r^5) * [(1 - 5 * z^2 / r^2) * x; (1 - 5 * z^2 / r^2) * y; (3 - 5 * z^2 / r^2) * z] # Acceleration due to J2 effect [km / s^2]
        cmp_xx = -4 * x^4 - 3 * x^2 * y^2 + 27 * x^2 * z^2 + y^4 - 3 * y^2 * z^2 - 4 * z^4
        cmp_xy = -5 * x^3 * y - 5 * x * y^3 + 30 * x * y * z^2
        cmp_xz = -15 * x^3 * z - 15 * x * y^2 * z + 20 * x * z^3
        cmp_yx = -5 * x^3 * y - 5 * x * y^3 + 30 * x * y * z^2
        cmp_yy = x^4 - 3 * x^2 * y^2 - 3 * x^2 * z^2 - 4 * y^4 + 27 * y^2 * z^2 - 4 * z^4
        cmp_yz = -15 * x^2 * y * z - 15 * y^3 * z + 20 * y * z^3
        cmp_zx = -15 * x^3 * z - 15 * x * y^2 * z + 20 * x * z^3
        cmp_zy = -15 * x^2 * y * z - 15 * y^3 * z + 20 * y * z^3
        cmp_zz = 3 * x^4 + 6 * x^2 * y^2 - 24 * x^2 * z^2 + 3 * y^4 - 24 * y^2 * z^2 + 8 * z^4
        dr_ddot_vec_J2_dr_vec = -(3 * J_2 * mu * R^2) / (2 * r^9) * [cmp_xx cmp_xy cmp_xz; cmp_yx cmp_yy cmp_yz; cmp_zx cmp_zy cmp_zz] # Jacobian of acceleration due to J2 effect w.r.t. position [s^{-2}]

        # Compute contributions from atmospheric drag
        h_0_vec = [0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
            130, 140, 150, 180, 200, 250, 300, 350, 400, 450,
            500, 600, 700, 800, 900, 1000] # Reference altitude [km]
        rho_0_vec = [1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3,
            3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7,
            9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9,
            5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11,
            9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
            1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15,
            3.019e-15] * 1e9 # Reference density [kg / m^3] -> [kg / km^3]
        H_vec = [7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549,
            5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
            22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
            58.515, 60.828, 63.822, 71.835, 88.667, 124.64,
            181.05, 268.00] # Scale height [km]
        h = r - R # Altitude [km]
        idx = (h >= 1000) ? 28 : findfirst((x_vec) -> x_vec > 0, h_0_vec .- h) - 1
        h_0 = h_0_vec[idx-1] # Reference altitude [km]
        rho_0 = rho_0_vec[idx-1] # Reference density [kg / km^3]
        H = H_vec[idx-1] # Scale height [km]
        rho_atm = rho_0 * exp(-(h - h_0) / H) # Atmospheric density [kg / km^3]
        drho_atm_dr_vec = -rho_atm / H * r_vec' / r # Jacobian of atmospheric density w.r.t. position [kg / km^4]

        r_dot_vec_rel = r_dot_vec - cross(omega_vec, r_vec) # Relative velocity vector [km / s]
        r_dot_rel = norm(r_dot_vec_rel) # Relative velocity magnitude [km / s]
        dr_dot_vec_rel_dr_vec = [0 omega 0; -omega 0 0; 0 0 0] # Jacobian of relative velocity w.r.t. position [s^{-1}]

        r_ddot_vec_drag = -0.5 * C_drag * A_drag / m * rho_atm * r_dot_rel * r_dot_vec_rel # Acceleration due to atmospheric drag [km / s^2]
        dr_ddot_vec_drag_dr_vec = -0.5 * C_drag * A_drag / m * (r_dot_rel * r_dot_vec_rel * drho_atm_dr_vec + rho_atm * (r_dot_vec_rel * r_dot_vec_rel' / r_dot_rel + r_dot_rel * I(3)) * dr_dot_vec_rel_dr_vec) # Jacobian of acceleration due to atmospheric drag w.r.t. position [s^{-2}]
        dr_ddot_vec_drag_dr_dot_vec = -0.5 * C_drag * A_drag / m * rho_atm * (r_dot_vec_rel * r_dot_vec_rel' / r_dot_rel + r_dot_rel * I(3)) # Jacobian of acceleration due to atmospheric drag w.r.t. velocity [s^{-1}]

        # Superposition of all contributions
        x_dot_vec[(i-1)*6+1:(i-1)*6+6] = [r_dot_vec; r_ddot_vec_grav + r_ddot_vec_J2 + r_ddot_vec_drag]
        F[((i-1)*6+1):((i-1)*6+3), ((i-1)*6+4):((i-1)*6+6)] = I(3)
        F[((i-1)*6+4):((i-1)*6+6), ((i-1)*6+1):((i-1)*6+3)] = dr_ddot_vec_grav_dr_vec + dr_ddot_vec_J2_dr_vec + dr_ddot_vec_drag_dr_vec
        F[((i-1)*6+4):((i-1)*6+6), ((i-1)*6+4):((i-1)*6+6)] = dr_ddot_vec_drag_dr_dot_vec
    end

    return x_dot_vec, F
end

function x_new_and_F(dt, x_old)
    # New state calculation
    k1, K1 = dynamics(x_old)
    k2, K2 = dynamics(x_old + dt / 2 * k1)
    k3, K3 = dynamics(x_old + dt / 2 * k2)
    k4, K4 = dynamics(x_old + dt * k3)
    x_new = x_old + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Jacobian calculation
    dk1_dx = K1
    dk2_dx = K2 * (I(size(K1, 1)) + dt / 2 * dk1_dx)
    dk3_dx = K3 * (I(size(K1, 1)) + dt / 2 * dk2_dx)
    dk4_dx = K4 * (I(size(K1, 1)) + dt * dk3_dx)
    F = I(size(K1, 1)) + dt / 6 * (dk1_dx + 2 * dk2_dx + 2 * dk3_dx + dk4_dx)

    return x_new, F
end

function measurement_chief(x_chief)
    return x_chief[1:3]
end

function measurement_deputies(x_chief, x_deputies)
    r_chief = x_chief[1:3]
    r_deputy1 = x_deputies[1:3]
    r_deputy2 = x_deputies[7:9]
    r_deputy3 = x_deputies[13:15]

    return [norm(r_deputy1 - r_chief); norm(r_deputy1 - r_deputy2); norm(r_deputy1 - r_deputy3); norm(r_deputy2 - r_chief); norm(r_deputy2 - r_deputy3); norm(r_deputy3 - r_chief)]
end

function apply_FCEKF(dt, x_est_old, P_est_old, y, Q, R)
    # Prediction Step
    x_pred, F = x_new_and_F(dt, x_est_old)
    P_pred = F * P_est_old * F' + Q
    # display(inv(P_pred))

    # Kalman Gain
    H = zeros(9, 24)
    H[1:3, 1:3] = I(3)
    H[4, 1:3] = -1 * (x_pred[7:9] - x_pred[1:3]) ./ norm(x_pred[7:9, :] - x_pred[1:3, :])
    H[4, 7:9] = (x_pred[7:9] - x_pred[1:3]) ./ norm(x_pred[7:9, :] - x_pred[1:3, :])
    H[5, 7:9] = (x_pred[7:9] - x_pred[13:15]) ./ norm(x_pred[7:9, :] - x_pred[13:15, :])
    H[5, 13:15] = -1 * (x_pred[7:9] - x_pred[13:15]) ./ norm(x_pred[7:9, :] - x_pred[13:15, :])
    H[6, 7:9] = (x_pred[7:9] - x_pred[19:21]) ./ norm(x_pred[7:9] - x_pred[19:21])
    H[6, 19:21] = -1 * (x_pred[7:9] - x_pred[19:21]) ./ norm(x_pred[7:9] - x_pred[19:21])
    H[7, 1:3] = -1 * (x_pred[13:15] - x_pred[1:3]) ./ norm(x_pred[13:15] - x_pred[1:3])
    H[7, 13:15] = (x_pred[13:15] - x_pred[1:3]) ./ norm(x_pred[13:15] - x_pred[1:3])
    H[8, 13:15] = (x_pred[13:15] - x_pred[19:21]) ./ norm(x_pred[13:15] - x_pred[19:21])
    H[8, 19:21] = -1 * (x_pred[13:15] - x_pred[19:21]) ./ norm(x_pred[13:15] - x_pred[19:21])
    H[9, 1:3] = -1 * (x_pred[19:21] - x_pred[1:3]) ./ norm(x_pred[19:21] - x_pred[1:3])
    H[9, 19:21] = (x_pred[19:21] - x_pred[1:3]) ./ norm(x_pred[19:21] - x_pred[1:3])
    K = P_pred * H' / (H * P_pred * H' + R)

    # Update Step
    y_est = zeros(9, 1)
    y_est[1:3] = x_pred[1:3]
    y_est[4] = norm(x_pred[7:9, :] - x_pred[1:3, :])
    y_est[5] = norm(x_pred[7:9, :] - x_pred[13:15, :])
    y_est[6] = norm(x_pred[7:9, :] - x_pred[19:21, :])
    y_est[7] = norm(x_pred[13:15, :] - x_pred[1:3, :])
    y_est[8] = norm(x_pred[13:15, :] - x_pred[19:21, :])
    y_est[9] = norm(x_pred[19:21, :] - x_pred[1:3, :])
    x_est_new = x_pred + K * (y - y_est)
    P_est_new = P_pred - K * H * P_pred

    return x_est_new, P_est_new
end

function apply_EKF(dt, x_est_old, Omega_est_old, y, W, V)
    # Prediction Step
    x_pred, F = x_new_and_F(dt, x_est_old)
    Omega_pred = W - W * F * inv(Omega_est_old + F' * W * F) * F' * W
    q_pred = Omega_pred * x_pred
    # display(Omega_pred)

    # Observation Jacobian computation
    H = zeros(9, 24)
    H[1:3, 1:3] = I(3)
    H[4, 1:3] = -1 * (x_pred[7:9] - x_pred[1:3]) ./ norm(x_pred[7:9, :] - x_pred[1:3, :])
    H[4, 7:9] = (x_pred[7:9] - x_pred[1:3]) ./ norm(x_pred[7:9, :] - x_pred[1:3, :])
    H[5, 7:9] = (x_pred[7:9] - x_pred[13:15]) ./ norm(x_pred[7:9, :] - x_pred[13:15, :])
    H[5, 13:15] = -1 * (x_pred[7:9] - x_pred[13:15]) ./ norm(x_pred[7:9, :] - x_pred[13:15, :])
    H[6, 7:9] = (x_pred[7:9] - x_pred[19:21]) ./ norm(x_pred[7:9] - x_pred[19:21])
    H[6, 19:21] = -1 * (x_pred[7:9] - x_pred[19:21]) ./ norm(x_pred[7:9] - x_pred[19:21])
    H[7, 1:3] = -1 * (x_pred[13:15] - x_pred[1:3]) ./ norm(x_pred[13:15] - x_pred[1:3])
    H[7, 13:15] = (x_pred[13:15] - x_pred[1:3]) ./ norm(x_pred[13:15] - x_pred[1:3])
    H[8, 13:15] = (x_pred[13:15] - x_pred[19:21]) ./ norm(x_pred[13:15] - x_pred[19:21])
    H[8, 19:21] = -1 * (x_pred[13:15] - x_pred[19:21]) ./ norm(x_pred[13:15] - x_pred[19:21])
    H[9, 1:3] = -1 * (x_pred[19:21] - x_pred[1:3]) ./ norm(x_pred[19:21] - x_pred[1:3])
    H[9, 19:21] = (x_pred[19:21] - x_pred[1:3]) ./ norm(x_pred[19:21] - x_pred[1:3])

    # Update Step
    y_est = zeros(9, 1)
    y_est[1:3] = x_pred[1:3]
    y_est[4] = norm(x_pred[7:9, :] - x_pred[1:3, :])
    y_est[5] = norm(x_pred[7:9, :] - x_pred[13:15, :])
    y_est[6] = norm(x_pred[7:9, :] - x_pred[19:21, :])
    y_est[7] = norm(x_pred[13:15, :] - x_pred[1:3, :])
    y_est[8] = norm(x_pred[13:15, :] - x_pred[19:21, :])
    y_est[9] = norm(x_pred[19:21, :] - x_pred[1:3, :])
    delta_q_est_new = H' * V * (y - y_est + H * x_pred)
    delta_Omega_est_new = H' * V * H

    return q_pred, Omega_pred, delta_q_est_new, delta_Omega_est_new
end

function init_Consensus(delta_q, delta_Omega, q, Omega, L)
    delta_q_vec = zeros(24, L + 1)
    delta_q_vec[:, 1] = delta_q
    delta_Omega_vec = zeros(24, 24, L + 1)
    delta_Omega_vec[:, :, 1] = delta_Omega
    q_vec = zeros(24, L + 1)
    q_vec[:, 1] = q
    Omega_vec = zeros(24, 24, L + 1)
    Omega_vec[:, :, 1] = Omega

    return delta_q_vec, delta_Omega_vec, q_vec, Omega_vec
end

function end_Consensus(delta_q, delta_Omega, q, Omega, gamma)
    q_est_new = q + gamma * delta_q
    Omega_est_new = Omega + gamma * delta_Omega
    x_est_new = inv(Omega_est_new) * q_est_new

    return x_est_new, Omega_est_new
end

# Simulation parameters
dt = 60.0  # Time step [s]
T = 3950  # Duration [min]
T_RMSE = 301 # Index from which the RMSE is calculated
n_simulations = 1  # Number of Monte-Carlo simulations
L = 1  # Number of consensus iterations
N = 4  # Number of satellites
gamma = N  # Consensus gain 1
pi = 1 / N  # Consensus gain 2

# Initial state vector and state covariance
# Formation Setup for REx-Lab Mission
x_initial = permutedims([6895.6 0 0 0 -0.99164 7.5424 6895.6 3e-05 1e-05 -0.0015 -0.99214 7.5426 6895.6 1e-05 3e-06 0.005 -0.98964 7.5422 6895.6 -2e-05 4e-06 0.00545 -0.99594 7.5423])

# Formation Setup for Higher-Orbit Difference
# x_initial = permutedims([-1295.9 -929.67 6793.4 -7.4264 0.1903 -1.3906 171.77 6857.3 -1281.5 1.0068 -1.3998 -7.3585 -4300.5 1654.8 -5241 -4.6264 3.4437 4.8838 -2197.8 -3973.4 -5298.7 1.377 5.6601 -4.8155])
p_pos_initial = 1e-1  # [km]
p_vel_initial = 1e-5  # [km / s]

# Process noise
q_chief_pos = 1e-4  # [km]
q_chief_vel = 1e-5  # [km / s]
Q_chief = Diagonal([q_chief_pos * ones(3); q_chief_vel * ones(3)] .^ 2)
q_deputy_pos = 1e-3  # [km]
q_deputy_vel = 1e-5  # [km / s]
Q_deputy = Diagonal([q_deputy_pos * ones(3); q_deputy_vel * ones(3)] .^ 2)
Q_deputies = Matrix(BlockDiagonal([Q_deputy, Q_deputy, Q_deputy]))
Q = Matrix(BlockDiagonal([Q_chief, Q_deputies]))
W = inv(Q)

# Observation noise
r_chief_pos = 1e-4  # [km]
R_chief = Matrix((r_chief_pos^2)I, 3, 3)
r_deputy_pos = 1e-3  # [km]
R_deputies = Matrix((r_deputy_pos^2)I, 6, 6)
R = Matrix(BlockDiagonal([R_chief, R_deputies]))
V = inv(R)

# Get the true state vectors
X_chief = fill(zeros(6, 1), T)
X_deputies = fill(zeros(18, 1), T)
X = vcat(X_chief, X_deputies)
X_chief[1] = x_initial[1:6, :]
X_deputies[1] = x_initial[7:end, :]
X[1] = vcat(X_chief[1], X_deputies[1])
for i = 1:T-1
    X_chief[i+1], _ = x_new_and_F(dt, X_chief[i])
    X_deputies[i+1], _ = x_new_and_F(dt, X_deputies[i])
    X[i+1] = vcat(X_chief[i+1], X_deputies[i+1])
end

rmse_chief_values = Vector{Any}(undef, n_simulations)
rmse_deputy1_values = Vector{Any}(undef, n_simulations)
rmse_deputy2_values = Vector{Any}(undef, n_simulations)
rmse_deputy3_values = Vector{Any}(undef, n_simulations)

@showprogress 1 "Running simulations..." for i in 1:n_simulations
    # Observations
    Y_chief = fill(zeros(3, 1), T)
    Y_deputies = fill(zeros(6, 1), T)
    Y = vcat(Y_chief, Y_deputies)
    for i = 1:T
        Y_chief[i] = measurement_chief(X_chief[i]) + sqrt(R_chief) * randn(size(R_chief, 1), 1)
        Y_deputies[i] = measurement_deputies(X_chief[i], X_deputies[i]) + sqrt(R_deputies) * randn(size(R_deputies, 1), 1)
        Y[i] = vcat(Y_chief[i], Y_deputies[i])
    end

    # Initial state vector and state covariance estimate
    initial_dev = [p_pos_initial * randn(3, 1); p_vel_initial * randn(3, 1); p_pos_initial * randn(3, 1); p_vel_initial * randn(3, 1); p_pos_initial * randn(3, 1); p_vel_initial * randn(3, 1); p_pos_initial * randn(3, 1); p_vel_initial * randn(3, 1)]
    X_est = x_initial + initial_dev

    # P = Matrix(Diagonal(vec(initial_dev .^ 2)))
    # Spacecrafts = Spacecraft(X_est, x_initial, P)
    # Formation = [Spacecrafts, Spacecrafts, Spacecrafts, Spacecrafts]
    
    Omega = inv(Matrix(Diagonal(vec(initial_dev .^ 2))))
    SpacecraftChief = Spacecraft(X_est, x_initial, Omega)
    SpacecraftDep1 = Spacecraft(X_est, x_initial, Omega)
    SpacecraftDep2 = Spacecraft(X_est, x_initial, Omega)
    SpacecraftDep3 = Spacecraft(X_est, x_initial, Omega)
    Formation = [SpacecraftChief, SpacecraftDep1, SpacecraftDep2, SpacecraftDep3]

    for t = 2:T
        # x_new, P = apply_FCEKF(dt, Formation[1, t-1].X_est, Matrix(Diagonal(vec(initial_dev .^ 2))), Y[t], Q, R)
        # Spacecrafts = Spacecraft(x_new, X[t], P)
        # Formation = hcat(Formation, [Spacecrafts, Spacecrafts, Spacecrafts, Spacecrafts])

        # println("t = $t")
        q_chief, Omega_chief, delta_q_chief, delta_Omega_chief = apply_EKF(dt, Formation[1, t-1].X_est, Formation[1, t-1].Omega, Y[t], W, V)
        q_deputy1, Omega_deputy1, delta_q_deputy1, delta_Omega_deputy1 = apply_EKF(dt, Formation[2, t-1].X_est, Formation[2, t-1].Omega, Y[t], W, V)
        q_deputy2, Omega_deputy2, delta_q_deputy2, delta_Omega_deputy2 = apply_EKF(dt, Formation[3, t-1].X_est, Formation[3, t-1].Omega, Y[t], W, V)
        q_deputy3, Omega_deputy3, delta_q_deputy3, delta_Omega_deputy3 = apply_EKF(dt, Formation[4, t-1].X_est, Formation[4, t-1].Omega, Y[t], W, V)

        # Consensus
        delta_q_vec_chief, delta_Omega_vec_chief, q_vec_chief, Omega_vec_chief = init_Consensus(delta_q_chief, delta_Omega_chief, q_chief, Omega_chief, L)
        delta_q_vec_deputy1, delta_Omega_vec_deputy1, q_vec_deputy1, Omega_vec_deputy1 = init_Consensus(delta_q_deputy1, delta_Omega_deputy1, q_deputy1, Omega_deputy1, L)
        delta_q_vec_deputy2, delta_Omega_vec_deputy2, q_vec_deputy2, Omega_vec_deputy2 = init_Consensus(delta_q_deputy2, delta_Omega_deputy2, q_deputy2, Omega_deputy2, L)
        delta_q_vec_deputy3, delta_Omega_vec_deputy3, q_vec_deputy3, Omega_vec_deputy3 = init_Consensus(delta_q_deputy3, delta_Omega_deputy3, q_deputy3, Omega_deputy3, L)

        for l = 2:L+1
            delta_q_vec_chief[:, l] = pi * (delta_q_vec_chief[:, l-1] + delta_q_vec_deputy1[:, l-1] + delta_q_vec_deputy2[:, l-1] + delta_q_vec_deputy3[:, l-1])
            delta_q_vec_deputy1[:, l] = pi * (delta_q_vec_chief[:, l-1] + delta_q_vec_deputy1[:, l-1] + delta_q_vec_deputy2[:, l-1] + delta_q_vec_deputy3[:, l-1])
            delta_q_vec_deputy2[:, l] = pi * (delta_q_vec_chief[:, l-1] + delta_q_vec_deputy1[:, l-1] + delta_q_vec_deputy2[:, l-1] + delta_q_vec_deputy3[:, l-1])
            delta_q_vec_deputy3[:, l] = pi * (delta_q_vec_chief[:, l-1] + delta_q_vec_deputy1[:, l-1] + delta_q_vec_deputy2[:, l-1] + delta_q_vec_deputy3[:, l-1])

            delta_Omega_vec_chief[:, :, l] = pi * (delta_Omega_vec_chief[:, :, l-1] + delta_Omega_vec_deputy1[:, :, l-1] + delta_Omega_vec_deputy2[:, :, l-1] + delta_Omega_vec_deputy3[:, :, l-1])
            delta_Omega_vec_deputy1[:, :, l] = pi * (delta_Omega_vec_chief[:, :, l-1] + delta_Omega_vec_deputy1[:, :, l-1] + delta_Omega_vec_deputy2[:, :, l-1] + delta_Omega_vec_deputy3[:, :, l-1])
            delta_Omega_vec_deputy2[:, :, l] = pi * (delta_Omega_vec_chief[:, :, l-1] + delta_Omega_vec_deputy1[:, :, l-1] + delta_Omega_vec_deputy2[:, :, l-1] + delta_Omega_vec_deputy3[:, :, l-1])
            delta_Omega_vec_deputy3[:, :, l] = pi * (delta_Omega_vec_chief[:, :, l-1] + delta_Omega_vec_deputy1[:, :, l-1] + delta_Omega_vec_deputy2[:, :, l-1] + delta_Omega_vec_deputy3[:, :, l-1])

            q_vec_chief[:, l] = pi * (q_vec_chief[:, l-1] + q_vec_deputy1[:, l-1] + q_vec_deputy2[:, l-1] + q_vec_deputy3[:, l-1])
            q_vec_deputy1[:, l] = pi * (q_vec_chief[:, l-1] + q_vec_deputy1[:, l-1] + q_vec_deputy2[:, l-1] + q_vec_deputy3[:, l-1])
            q_vec_deputy2[:, l] = pi * (q_vec_chief[:, l-1] + q_vec_deputy1[:, l-1] + q_vec_deputy2[:, l-1] + q_vec_deputy3[:, l-1])
            q_vec_deputy3[:, l] = pi * (q_vec_chief[:, l-1] + q_vec_deputy1[:, l-1] + q_vec_deputy2[:, l-1] + q_vec_deputy3[:, l-1])

            Omega_vec_chief[:, :, l] = pi * (Omega_vec_chief[:, :, l-1] + Omega_vec_deputy1[:, :, l-1] + Omega_vec_deputy2[:, :, l-1] + Omega_vec_deputy3[:, :, l-1])
            Omega_vec_deputy1[:, :, l] = pi * (Omega_vec_chief[:, :, l-1] + Omega_vec_deputy1[:, :, l-1] + Omega_vec_deputy2[:, :, l-1] + Omega_vec_deputy3[:, :, l-1])
            Omega_vec_deputy2[:, :, l] = pi * (Omega_vec_chief[:, :, l-1] + Omega_vec_deputy1[:, :, l-1] + Omega_vec_deputy2[:, :, l-1] + Omega_vec_deputy3[:, :, l-1])
            Omega_vec_deputy3[:, :, l] = pi * (Omega_vec_chief[:, :, l-1] + Omega_vec_deputy1[:, :, l-1] + Omega_vec_deputy2[:, :, l-1] + Omega_vec_deputy3[:, :, l-1])
        end

        x_est_new_chief, Omega_chief = end_Consensus(delta_q_vec_chief[:, end], delta_Omega_vec_chief[:, :, end], q_vec_chief[:, end], Omega_vec_chief[:, :, end], gamma)
        x_est_new_deputy1, Omega_deputy1 = end_Consensus(delta_q_vec_deputy1[:, end], delta_Omega_vec_deputy1[:, :, end], q_vec_deputy1[:, end], Omega_vec_deputy1[:, :, end], gamma)
        x_est_new_deputy2, Omega_deputy2 = end_Consensus(delta_q_vec_deputy2[:, end], delta_Omega_vec_deputy2[:, :, end], q_vec_deputy2[:, end], Omega_vec_deputy2[:, :, end], gamma)
        x_est_new_deputy3, Omega_deputy3 = end_Consensus(delta_q_vec_deputy3[:, end], delta_Omega_vec_deputy3[:, :, end], q_vec_deputy3[:, end], Omega_vec_deputy3[:, :, end], gamma)

        SpacecraftChief = Spacecraft(reshape(x_est_new_chief, (length(x_est_new_chief), 1)), X[t], Omega_chief)
        SpacecraftDep1 = Spacecraft(reshape(x_est_new_deputy1, (length(x_est_new_deputy1), 1)), X[t], Omega_deputy1)
        SpacecraftDep2 = Spacecraft(reshape(x_est_new_deputy2, (length(x_est_new_deputy2), 1)), X[t], Omega_deputy2)
        SpacecraftDep3 = Spacecraft(reshape(x_est_new_deputy3, (length(x_est_new_deputy3), 1)), X[t], Omega_deputy3)

        Formation = hcat(Formation, [SpacecraftChief, SpacecraftDep1, SpacecraftDep2, SpacecraftDep3])
    end

    # Initialize arrays to store the squared differences
    rmse_chief_squared_diffs = []
    rmse_deputy1_squared_diffs = []
    rmse_deputy2_squared_diffs = []
    rmse_deputy3_squared_diffs = []

    # Calculate squared differences for each time step
    for t in T_RMSE:T
        push!(rmse_chief_squared_diffs, norm(Formation[1, t].X_est[1:3, :] - Formation[1, t].X_true[1:3, :])^2)
        push!(rmse_deputy1_squared_diffs, norm(Formation[2, t].X_est[7:9, :] - Formation[2, t].X_true[7:9, :])^2)
        push!(rmse_deputy2_squared_diffs, norm(Formation[3, t].X_est[13:15, :] - Formation[3, t].X_true[13:15, :])^2)
        push!(rmse_deputy3_squared_diffs, norm(Formation[4, t].X_est[19:21, :] - Formation[4, t].X_true[19:21, :])^2)
    end

    # Calculate the RMSE
    rmse_chief_values[i] = sqrt(mean(rmse_chief_squared_diffs))
    rmse_deputy1_values[i] = sqrt(mean(rmse_deputy1_squared_diffs))
    rmse_deputy2_values[i] = sqrt(mean(rmse_deputy2_squared_diffs))
    rmse_deputy3_values[i] = sqrt(mean(rmse_deputy3_squared_diffs))

    if i == n_simulations
        EKF_dev_min1 = zeros(T, 1)
        EKF_dev_min2 = zeros(T, 1)
        EKF_dev_min3 = zeros(T, 1)
        EKF_dev_min4 = zeros(T, 1)

        df = DataFrame(t=Int[], EKF_dev_min1=Float64[], EKF_dev_min2=Float64[], EKF_dev_min3=Float64[], EKF_dev_min4=Float64[])

        for t = 1:T
            dev_min1 = norm(Formation[1, t].X_est[1:3, :] - Formation[1, t].X_true[1:3, :])
            dev_min2 = norm(Formation[2, t].X_est[7:9, :] - Formation[2, t].X_true[7:9, :])
            dev_min3 = norm(Formation[3, t].X_est[13:15, :] - Formation[3, t].X_true[13:15, :])
            dev_min4 = norm(Formation[4, t].X_est[19:21, :] - Formation[4, t].X_true[19:21, :])

            EKF_dev_min1[t] = dev_min1
            EKF_dev_min2[t] = dev_min2
            EKF_dev_min3[t] = dev_min3
            EKF_dev_min4[t] = dev_min4

            push!(df, (t, dev_min1, dev_min2, dev_min3, dev_min4))
        end

        CSV.write("data.csv", df)
    end
end

# Calculate average RMSE
rmse_chief_average = mean(rmse_chief_values)
rmse_deputy1_average = mean(rmse_deputy1_values)
rmse_deputy2_average = mean(rmse_deputy2_values)
rmse_deputy3_average = mean(rmse_deputy3_values)

println("Average RMSE for chief: $(rmse_chief_average * 1e3) m")
println("Average RMSE for deputy 1: $(rmse_deputy1_average * 1e3) m")
println("Average RMSE for deputy 2: $(rmse_deputy2_average * 1e3) m")
println("Average RMSE for deputy 3: $(rmse_deputy3_average * 1e3) m")
