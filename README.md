# 🛰 SpaceSim-Toolkit

## 🎯 Features
- Simulate **spacecraft dynamics** with **high-fidelity** numerical propagator from [TUDAT](https://docs.tudat.space/en/latest/) using **Python**
- Configure **spacecraft** and **multi-spacecraft missions** with easy-to-edit custom configuration files
- Run and compare different legacy and advanced **estimation algorithms**
- Visualize **orbits** and **ground tracks** using MATLAB and **results** using Monte-Carlo analysis
- Easy to implement estimation strategies in **high-level** Python language

## 🚀 Index
- 💡 [Description](#-description)
- 💿 [Installation](#-installation)
- 📖 [Documentation](#-documentation)
- 🦆 [Examples](#-examples)
- 📞 [Contact](#-contact)
- 💥 [References](#-references)

## 💡 Description
**SpaceSim-Toolkit** is an open-source simulation framework designed for space enthusiasts, researchers, and engineers. It provides a comprehensive suite of tools for simulating spacecraft dynamics, orbital mechanics, and advanced estimation algorithms in various space missions.

This repository implements the research work presented in [[1]](#ref-1), providing a complete framework for decentralized state estimation in multi-agent systems, with specific applications to low Earth orbit satellite formations.

## 💿 Installation
To setup the virtual environment for **SpaceSim-Toolkit** follow these steps:
```bash
# Clone the repository
$ git clone git@github.com:joaomarafuzgaspar/SpaceSim-Toolkit.git
$ cd SpaceSim-Toolkit

# Create the virtual environment and activate it
$ conda env create -f environment.yml
$ conda activate spacesim-toolkit

# Install matlab engine for MATLAB release R2022b
$ python -m pip install matlabengine==9.13.11
````

> [!IMPORTANT]
> You need to make sure you have `MATLAB` installed.
> To know which version of `matlabengine` you need to install, refer to [this link](https://pypi.org/project/matlabengine/).

## 📖 Documentation
To run simulations, execute `python src/main.py` with the following command-line arguments:

| Option                         | Description |
|--------------------------------|-------------|
| `-v`, `--visualize`            | Display results. Use `all_deviations` to show all estimators’ deviations; `orbits` to show propagated orbits. |
| `-f`, `--formation`            | Select the satellite formation. `1` for V-R3x mission, `2` for higher-orbit formation. |
| `-c`, `--config`               | Specify the configuration file. Default is `config_files/_config_default.py`. |
| `-a`, `--algorithm`            | Choose the estimation algorithm. See the list below. |
| `-M`, `--monte-carlo-sims`     | Number of Monte Carlo simulations to run (integer ≥ 1). |
| `-p`, `--propagate`            | Propagate spacecraft dynamics using the configured model. |

### Available Estimation Algorithms (`-a`)
- `lm`: Levenberg–Marquardt (nonlinear least squares)
- `fcekf`: Fully centralized Extended Kalman Filter (EKF)
- `hcmci`: Hybrid consensus on measurements and consensus on information [[2]](#ref-2)
- `ccekf`: Consider Covariance EKF [[3]](#ref-3)
- `newton`: Newton’s method
- `gauss-newton`: Gauss–Newton method
- `tree-newton`: Newton variant using tree-structured Hessian
- `approxA-newton`: Newton variant using an approximate Hessian (tree + residual diagonals)
- `mm-newton`: Newton variant using Majorization–Minimization
- `inewton`: Iterative Newton’s method
- `pcg-newton`: Newton variant with Preconditioned Conjugate Gradient solver

For detailed explanations of the Newton-based algorithms, refer to [[1]](#ref-1).

## 🦆 Examples
### Run propagation
This command propagates the dynamics for the V-R3x mission formation and saves the evolution of the state vectors.
```bash
$ python src/main.py -f 1 -p
```

### Run simulation with visualization
This command runs 10 Monte-Carlo simulations using the `fcekf` algorithm for the V-R3x mission formation, with visualization enabled.
```bash
$ python src/main.py -v -f 1 -a fcekf -M 10
```

### Orbits of the available formations
#### Formation I - V-R3x Mission

<div align="center">

| Satellite                  | $i \ [^\circ]$ | $\Omega \ [^\circ]$ | $\varpi \ [^\circ]$ | $a \ [\mathrm{km}]$ | $e$      | $\theta_0 \ [^\circ]$ |
| :------------------------- | :------------: | :-----------------: | :-----------------: | :-----------------: | :------: | :-------------------: |
| Chief 1 $(\mathcal{S}_1)$  | $97.79$        | $0$                 | $0$                 | $6903.50$           | $0.0011$ | $0$                   |
| Deputy 1 $(\mathcal{S}_2)$ | $97.26$        | $0$                 | $9.23$              | $6903.98$           | $0.0012$ | $350.76$              |
| Deputy 2 $(\mathcal{S}_3)$ | $97.79$        | $0$                 | $327.27$            | $6902.67$           | $0.0012$ | $32.72$               |
| Deputy 3 $(\mathcal{S}_4)$ | $97.79$        | $0$                 | $330.47$            | $6904.34$           | $0.0014$ | $29.52$               |
</div>

<p align="center">
    <img src="/gifs/orbits_form1.gif" width="600">
</p>

#### Formation II - Higher Orbit Difference

<div align="center">

| Satellite                  | $i \ [^\circ]$ | $\Omega \ [^\circ]$  | $\varpi \ [^\circ]$ | $a \ [\mathrm{km}]$ | $e$                   | $\theta_0 \ [^\circ]$ |
| :------------------------- | :------------: | :------------------: | :-----------------: | :-----------------: | :-------------------: | :-------------------: |
| Chief 1 $(\mathcal{S}_1)$  | $97.49$        | $1.5 \times 10^{-5}$ | $303.34$            | $6978$              | $2.6 \times 10^{-6}$  | $157.36$              |
| Deputy 1 $(\mathcal{S}_2)$ | $97.49$        | $272.80$             | $281.15$            | $6978$              | $6.48 \times 10^{-3}$ | $269.52$              |
| Deputy 2 $(\mathcal{S}_3)$ | $97.47$        | $149.99$             | $204.07$            | $6978$              | $6.6 \times 10^{-5}$  | $206.00$              |
| Deputy 3 $(\mathcal{S}_4)$ | $97.52$        | $70$                 | $257.43$            | $6978$              | $1.3 \times 10^{-5}$  | $332.57$              |

</div>

<p align="center">
    <img src="/gifs/orbits_form2.gif" width="600">
</p>

## 📞 Contact
**SpaceSim-Toolkit** is currently maintained by João Marafuz Gaspar ([joao.marafuz.gaspar@tecnico.ulisboa.pt](mailto:joao.marafuz.gaspar@tecnico.ulisboa.pt)).

## 💥 References
<a id="ref-1">[1]</a> J. Marafuz Gaspar. *Decentralized estimator for dynamical multi-agent systems with an application to LEO satellite formations*. Master's thesis, Instituto Superior Técnico, University of Lisbon,
2025. [online] Available [here](https://scholar.tecnico.ulisboa.pt/records/y9bKGa4tgKYVZ6Z6T7yLt-jjeGr21ssJVQrr).

<a id="ref-2">[2]</a> G. Battistelli and L. Chisci. *Stability of consensus extended Kalman filter for distributed state estimation*. Automatica, 68:169–178, 2016.

<a id="ref-3">[3]</a> R. Cordeiro. *A low-communication distributed state-estimation framework for satellite formations*. Master's thesis, Instituto Superior Técnico, University of Lisbon, 2022.