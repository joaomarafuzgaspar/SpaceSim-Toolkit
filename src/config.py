# src/config.py

class SimulationConfig:
    # Simulation parameters
    dt = 60.0  # Time step [s]
    K = 395  # Duration [min]
    M = 1  # Number of Monte-Carlo simulations
    N = 4  # Number of spacecrafts
    formation = 1 
    n_x = 6 
    n_p = 3  
    n = N * n_x
    o = 3 + 3 + 2 + 1 
    H = 15 # Window size [min]
    invalid_rmse = 1e2 # [m]
    
    # Spacecraft parameters
    C_drag = 2.22  # Drag coefficient
    A_drag = 0.01  # Drag area [m^2]
    m = 1.0  # Mass [kg]
    
    # Newton's method-based algorithms parameters
    grad_norm_order_mag = True
    grad_norm_tol = 1e-6
    max_iterations = 20
