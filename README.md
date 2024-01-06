# SpaceSim-Toolkit

## Overview
SpaceSim-Toolkit is an open-source simulation framework designed for space enthusiasts, researchers, and engineers. It provides tools for simulating spacecraft dynamics, orbital mechanics, and navigation algorithms in various space missions.

## Features
- **Spacecraft Dynamics:** Detailed simulation of spacecraft motion and behavior.
- **Orbital Mechanics:** Tools for analyzing and simulating orbital paths and maneuvers.
- **Navigation Algorithms:** Advanced algorithms for space navigation and trajectory planning.

## Getting Started
1. **Installation:** Instructions on how to install and setup the toolkit.
2. **Documentation:** Link to the full documentation for detailed guidance.
3. **Examples:** Sample projects or scripts demonstrating the toolkit's capabilities.

## Usage
To use the SpaceSim-Toolkit, follow these steps:

1. **Run Simulation**: 
```
python src/main.py
```
2. **Command Line Arguments**:
    - `-m` or `--matlab`: To display MATLAB figure.
    - `-f` or `--formation`: Choose formation type (1 for VREx mission, 2 for higher-orbit).
    - `-a` or `--algorithm`: Select navigation algorithm (fcekf, hcmci, or ccekf).
    - `-M` or `--monte-carlo-sims`: Set number of Monte-Carlo simulations to run (integer >= 1).

3. **Example Command**:
```bash
# This command runs 10 Monte-Carlo simulations using the fcekf algorithm for the VREx mission formation, with MATLAB visualization enabled.
python script_name.py -m -f 1 -a fcekf -M 10
```

## Acknowledgements
Credits to contributors, supporting organizations, or any other acknowledgements.

## Contact
Information on how to get in touch with the project maintainers or contributors.

---

For more information and the latest updates, visit the [SpaceSim-Toolkit GitHub repository](https://github.com/joaogaspar2001/SpaceSim-Toolkit).
