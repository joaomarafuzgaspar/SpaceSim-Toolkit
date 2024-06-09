# 🛰 SpaceSim-Toolkit

## 🎯 Features
- **Spacecraft Dynamics:** Detailed simulation of spacecraft motion and behavior.
- **Orbital Mechanics:** Tools for analyzing and simulating orbital paths and maneuvers.
- **Navigation Algorithms:** Advanced algorithms for space navigation and trajectory planning.

## 🚀 Index
- 💡 [Description](#-description)
- 💿 [Usage](#-usage)
- 📖 [Documentation](#-documentation)
- 🦆 [Example](#-example)
- 📞 [Contact](#-contact)
- 💥 [References](#-references)

## 💡 Description
**SpaceSim-Toolkit** is an open-source simulation framework designed for space enthusiasts, researchers, and engineers. It provides tools for simulating spacecraft dynamics, orbital mechanics, and navigation algorithms in various space missions. This code is based on <a href="#ref-1">[1]</a>.

## 💿 Usage
To use the **SpaceSim-Toolkit**, follow these steps:
```bash
git clone git@github.com:joaomarafuzgaspar/SpaceSim-Toolkit.git

cd SpaceSim-Toolkit

python src/main.py
```

## 📖 Documentation
Alongside `python src/main.py` several command line arguments can be used as follows:
- `-m` or `--matlab`: To display MATLAB figure(s).
- `-f` or `--formation`: Choose formation type (1 for VREx mission, 2 for higher-orbit).
- `-a` or `--algorithm`: Select navigation algorithm (fcekf, hcmci, or ccekf).
- `-M` or `--monte-carlo-sims`: Set number of Monte-Carlo simulations to run (integer >= 1).
- `-p` or `--propagate`: Propagate the spacecraft dynamics using the dynamics model.

## 🦆 Examples
### Run propagation
This command propagates the dynamics for the VREx mission formation.
```bash
python src/main.py -f 1 -p
```

### Run simulation
This command runs 10 Monte-Carlo simulations using the `fcekf` algorithm for the VREx mission formation, with MATLAB visualization enabled.
```bash
python src/main.py -m -f 1 -a fcekf -M 10
```

## 📞 Contact
**SpaceSim-Toolkit** is currently maintained by João Marafuz Gaspar ([joao.marafuz.gaspar@tecnico.ulisboa.pt](mailto:joao.marafuz.gaspar@tecnico.ulisboa.pt)).

## 💥 References
<a id="ref-1">[1]</a> João Marafuz Gaspar. 2024. *Collaborative Localization for Satellite Formations*. [online] Available [here](https://web.tecnico.ulisboa.pt/ist196240/thesis/JoaoMarafuzGaspar-PIC2-Report.pdf).