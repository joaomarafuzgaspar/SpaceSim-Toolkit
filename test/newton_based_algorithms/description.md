## Last updated: 09/02/2025
# Files Description


- `approxh_newton.ipynb`: Uses np.random.seed(42). FIXME
- `approxh_newton2.ipynb`: Uses np.random.seed(43). FIXME
- `centralized_newton_with_dynamics` (09/02/2025): Include dynamics in the finite-horizon cost function. The true states are propagated using TudatPy and matrix Q is computed using our propagators behaviour compared to the true states. The algorithm is struggling to converge to the optimal solution, as the hessian formulation/computation seems not to be as accurate as possible because possibly second-order derivative terms are missing.
- `cnkkt.ipynb`: FIXME
- `comparison_newton_based.ipynb` (09/02/2025): Cost function and gradient norm visualization for the Newton-based frameworks: centralized Newton (unkkt), approximated Hessian Newton (approxh_newton) and Majorization-Minimization outer loop with Newton's method inner loop (mm_newton). Gradient norm stopping condition and maximum number of iterations can be tweaked for testing purposes.
- `mm_newton.ipynb`: FIXME
- `mm_newton2.ipynb`: FIXME
- `mm_newton_all_range_surrogate.ipynb`: FIXME
- `mm_newton_different_params.ipynb`: FIXME
- `mm_newton_redundant_obs.ipynb `: FIXME
- `mm_newton_updated_params.ipynb`: FIXME
- `unkkt_bearings.ipynb`: FIXME
- `unkkt_bearings_new_network.ipynb`: FIXME
- `unkkt_new_network.ipynb`: FIXME
- `unkkt.ipynb`: FIXME
- `unkkt2.ipynb`: FIXME
- `visualizer_nkkt.ipynb`: FIXME


# References
[1] - Pedroso, Leonardo, and Pedro Batista. "Distributed decentralized EKF for very large-scale networks with application to satellite mega-constellations navigation." Control Engineering Practice 135 (2023): 105509.