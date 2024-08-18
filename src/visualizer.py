import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matlab import engine


plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman"}
)
# MATLAB default color cycle
matlab_default_colors = [
    [0, 0.4470, 0.7410],  # blue
    [0.8500, 0.3250, 0.0980],  # orange
    [0.9290, 0.6940, 0.1250],  # yellow
    [0.4940, 0.1840, 0.5560],  # purple
    [0.4660, 0.6740, 0.1880],  # green
    [0.3010, 0.7450, 0.9330],  # light blue
    [0.6350, 0.0780, 0.1840],  # red
]
# Set the color cycle in Matplotlib to match MATLAB's default
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=matlab_default_colors)


# Function to get the latest file based on a prefix
def get_latest_file(prefix, directory="data/"):
    files = os.listdir(directory)
    filtered_files = [f for f in files if f.startswith(prefix) and f.endswith(".csv")]
    filtered_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True
    )
    return os.path.join(directory, filtered_files[0]) if filtered_files else None


def visualizer_devs():
    # Load the latest files into DataFrames
    devs_filepath = get_latest_file("devs_")
    algorithm = re.search(r"devs_(.*?)_", devs_filepath).group(1).upper()
    formation = re.search(r"form(\d+)", devs_filepath).group(1)
    devs = pd.read_csv(devs_filepath)

    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = len(devs["dev_chief"])  # Duration [min]
    t = np.arange(0, dt * (T + 1), dt)  # Time vector [s]

    # Plot positions based on screen size
    fig_width = 2 * 6.4  # in inches
    fig_height = 2 * 4.8  # in inches

    # Create a 2 by 2 figure
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Plot 1: Chief
    axs[0, 0].plot(t[:T] / (dt * dt), devs["dev_chief"], ".-", label=algorithm)
    axs[0, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{C}_1} - \\mathbf{r}_k^{\\mathcal{C}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 0].set_ylim([1e-5, 1e0])
    axs[0, 0].tick_params(axis="both", which="both", direction="in")
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].set_title("Chief")

    # Plot 2: Deputy 1
    axs[0, 1].plot(t[:T] / (dt * dt), devs["dev_deputy1"], ".-", label=algorithm)
    axs[0, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 1].grid(True, which="both")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_1} - \\mathbf{r}_k^{\\mathcal{D}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 1].set_ylim([1e-4, 1e0])
    axs[0, 1].legend(fontsize=12)
    axs[0, 1].set_title("Deputy 1")

    # Plot 3: Deputy 2
    axs[1, 0].plot(
        t[:T] / (dt * dt), devs["dev_deputy2"], ".-", linewidth=2, label=algorithm
    )
    axs[1, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_2} - \\mathbf{r}_k^{\\mathcal{D}_2}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 0].set_ylim([1e-4, 1e0])
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].set_title("Deputy 2")

    # Plot 4: Deputy 3
    axs[1, 1].plot(
        t[:T] / (dt * dt), devs["dev_deputy3"], ".-", linewidth=2, label=algorithm
    )
    axs[1, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 1].grid(True, which="both")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_3} - \\mathbf{r}_k^{\\mathcal{D}_3}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 1].set_ylim([1e-4, 1e0])
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].set_title("Deputy 3")

    fig.suptitle(f"Formation {formation}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    return


def visualizer_all_devs(formation=1):
    # Load the latest files into DataFrames
    fcekf_devs = pd.read_csv(get_latest_file(f"devs_fcekf_form{formation}_"))
    hcmci_devs = pd.read_csv(get_latest_file(f"devs_hcmci_form{formation}_"))
    ccekf_devs = pd.read_csv(get_latest_file(f"devs_ccekf_form{formation}_"))

    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = len(fcekf_devs["dev_chief"])  # Duration [min]
    t = np.arange(0, dt * (T + 1), dt)  # Time vector [s]

    # Plot positions based on screen size
    fig_width = 2 * 6.4  # in inches
    fig_height = 2 * 4.8  # in inches

    # Create a 2 by 2 figure
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Plot 1: Chief
    axs[0, 0].plot(t[:T] / (dt * dt), fcekf_devs["dev_chief"], ".-", label="FCEKF")
    axs[0, 0].plot(t[:T] / (dt * dt), hcmci_devs["dev_chief"], ".-", label="HCMCI")
    axs[0, 0].plot(t[:T] / (dt * dt), ccekf_devs["dev_chief"], ".-", label="CCEKF")
    axs[0, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{C}_1} - \\mathbf{r}_k^{\\mathcal{C}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 0].set_ylim([1e-5, 1e0])
    axs[0, 0].tick_params(axis="both", which="both", direction="in")
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].set_title("Chief")

    # Plot 2: Deputy 1
    axs[0, 1].plot(t[:T] / (dt * dt), fcekf_devs["dev_deputy1"], ".-", label="FCEKF")
    axs[0, 1].plot(t[:T] / (dt * dt), hcmci_devs["dev_deputy1"], ".-", label="HCMCI")
    axs[0, 1].plot(t[:T] / (dt * dt), ccekf_devs["dev_deputy1"], ".-", label="CCEKF")
    axs[0, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 1].grid(True, which="both")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_1} - \\mathbf{r}_k^{\\mathcal{D}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 1].set_ylim([1e-4, 1e0])
    axs[0, 1].legend(fontsize=12)
    axs[0, 1].set_title("Deputy 1")

    # Plot 3: Deputy 2
    axs[1, 0].plot(
        t[:T] / (dt * dt), fcekf_devs["dev_deputy2"], ".-", linewidth=2, label="FCEKF"
    )
    axs[1, 0].plot(
        t[:T] / (dt * dt), hcmci_devs["dev_deputy2"], ".-", linewidth=2, label="HCMCI"
    )
    axs[1, 0].plot(
        t[:T] / (dt * dt), ccekf_devs["dev_deputy2"], ".-", linewidth=2, label="CCEKF"
    )
    axs[1, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_2} - \\mathbf{r}_k^{\\mathcal{D}_2}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 0].set_ylim([1e-4, 1e0])
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].set_title("Deputy 2")

    # Plot 4: Deputy 3
    axs[1, 1].plot(
        t[:T] / (dt * dt), fcekf_devs["dev_deputy3"], ".-", linewidth=2, label="FCEKF"
    )
    axs[1, 1].plot(
        t[:T] / (dt * dt), hcmci_devs["dev_deputy3"], ".-", linewidth=2, label="HCMCI"
    )
    axs[1, 1].plot(
        t[:T] / (dt * dt), ccekf_devs["dev_deputy3"], ".-", linewidth=2, label="CCEKF"
    )
    axs[1, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 1].grid(True, which="both")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_3} - \\mathbf{r}_k^{\\mathcal{D}_3}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 1].set_ylim([1e-4, 1e0])
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].set_title("Deputy 3")

    fig.suptitle(f"Formation {formation}", fontsize=16)
    plt.tight_layout()
    plt.show()
    return


def run_visualizer(args):
    # Call the visualizer script
    if args.matlab == "orbits":
        eng = engine.start_matlab()
        matlab_functions_path = os.path.abspath("../SpaceSim-Toolkit/src")
        eng.addpath(matlab_functions_path, nargout=0)
        print("Displaying MATLAB figure with the orbits...")
        try:
            eng.visualizer_orbits(nargout=0)
        except engine.MatlabExecutionError as e:
            print(f"An error occurred: {e}")
        eng.quit()
    elif args.matlab == "all_deviations":
        print(
            f"Displaying all algorithms estimates deviations for Formation {'I' if args.formation == 1 else 'II' if args.formation == 2 else 'Unknown'}..."
        )
        visualizer_all_devs(args.formation)
    else:
        print("Displaying deviation for latest applied algorithm...")
        visualizer_devs()
    return
