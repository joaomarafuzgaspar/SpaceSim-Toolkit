# 🛰 SpaceSim-Toolkit

## 🎯 Features
- **Spacecraft Dynamics:** Detailed simulation of spacecraft motion and behavior.
- **Orbital Mechanics:** Tools for analyzing and simulating orbital paths and maneuvers.
- **Navigation Algorithms:** Advanced algorithms for space navigation and trajectory planning.

## 🚀 Index
- 💡 [Description](#-description)
- 💿 [Installation](#-installation)
- 📖 [Documentation](#-documentation)
- 🦆 [Example](#-example)
- 📞 [Contact](#-contact)
- 💥 [References](#-references)

## 💡 Description
**SpaceSim-Toolkit** is an open-source simulation framework designed for space enthusiasts, researchers, and engineers. It provides tools for simulating spacecraft dynamics, orbital mechanics, and navigation algorithms in various space missions. This code is based on <a href="#ref-1">[1]</a>.

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
Alongside `python src/main.py` several command line arguments can be used as follows:
- `-m` or `--matlab`: To display MATLAB figure(s). If `None` it displays last applied algorithm deviations, `all_deviations` displays all applied algorithms deviations and `orbits` displays last propagated orbits.
- `-f` or `--formation`: Choose formation type (1 for V-R3x mission, 2 for higher-orbit).
- `-a` or `--algorithm`: Select navigation algorithm (fcekf, hcmci, or ccekf).
- `-M` or `--monte-carlo-sims`: Set number of Monte-Carlo simulations to run (integer >= 1).
- `-p` or `--propagate`: Propagate the spacecraft dynamics using the dynamics model.

## 🦆 Examples
### Run propagation
This command propagates the dynamics for the V-R3x mission formation and saves the evolution of the state vectors and Jacobians.
```bash
$ python src/main.py -f 1 -p
```

### Run simulation
This command runs 10 Monte-Carlo simulations using the `fcekf` algorithm for the V-R3x mission formation, with MATLAB visualization enabled.
```bash
$ python src/main.py -m -f 1 -a fcekf -M 10
```

### Orbits of the available formations
#### Formation I - V-R3x Mission

<div align="center">

| Satellite                  | $i \ [^\circ]$ | $\Omega \ [^\circ]$ | $\varpi \ [^\circ]$ | $a \ [\mathrm{km}]$ | $e$      | $\theta_0 \ [^\circ]$ |
| :------------------------- | :------------: | :-----------------: | :-----------------: | :-----------------: | :------: | :-------------------: |
| Chief 1 $(\mathcal{C}_1)$  | $97.79$        | $0$                 | $0$                 | $6903.50$           | $0.0011$ | $0$                   |
| Deputy 1 $(\mathcal{D}_1)$ | $97.26$        | $0$                 | $9.23$              | $6903.98$           | $0.0012$ | $350.76$              |
| Deputy 2 $(\mathcal{D}_2)$ | $97.79$        | $0$                 | $327.27$            | $6902.67$           | $0.0012$ | $32.72$               |
| Deputy 3 $(\mathcal{D}_3)$ | $97.79$        | $0$                 | $330.47$            | $6904.34$           | $0.0014$ | $29.52$               |
</div>

<p align="center">
    <img src="/gifs/orbits_form1.gif" width="600">
</p>

#### Formation II - Higher Orbit Difference

<div align="center">

| Satellite                  | $i \ [^\circ]$ | $\Omega \ [^\circ]$  | $\varpi \ [^\circ]$ | $a \ [\mathrm{km}]$ | $e$                   | $\theta_0 \ [^\circ]$ |
| :------------------------- | :------------: | :------------------: | :-----------------: | :-----------------: | :-------------------: | :-------------------: |
| Chief 1 $(\mathcal{C}_1)$  | $97.49$        | $1.5 \times 10^{-5}$ | $303.34$            | $6978$              | $2.6 \times 10^{-6}$  | $157.36$              |
| Deputy 1 $(\mathcal{D}_1)$ | $97.49$        | $272.80$             | $281.15$            | $6978$              | $6.48 \times 10^{-3}$ | $269.52$              |
| Deputy 2 $(\mathcal{D}_2)$ | $97.47$        | $149.99$             | $204.07$            | $6978$              | $6.6 \times 10^{-5}$  | $206.00$              |
| Deputy 3 $(\mathcal{D}_3)$ | $97.52$        | $70$                 | $257.43$            | $6978$              | $1.3 \times 10^{-5}$  | $332.57$              |

</div>

<p align="center">
    <img src="/gifs/orbits_form2.gif" width="600">
</p>

## 📞 Contact
**SpaceSim-Toolkit** is currently maintained by João Marafuz Gaspar ([joao.marafuz.gaspar@tecnico.ulisboa.pt](mailto:joao.marafuz.gaspar@tecnico.ulisboa.pt)).

## 💥 References
<a id="ref-1">[1]</a> João Marafuz Gaspar. 2024. *Collaborative Localization for Satellite Formations*. [online] Available [here](https://web.tecnico.ulisboa.pt/ist196240/thesis/JoaoMarafuzGaspar-PIC2-Report.pdf).