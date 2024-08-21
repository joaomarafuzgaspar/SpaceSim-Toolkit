import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import rmse
from matlab import engine
from datetime import datetime


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
def get_latest_file(prefix="fcekf", suffix=".pkl", directory="data/"):
    files = os.listdir(directory)
    # Filter files by prefix and suffix
    filtered_files = [f for f in files if f.startswith(prefix) and f.endswith(suffix)]

    # Extract date from filename and sort files by this date
    def extract_date(file_name):
        # Assuming the date is always in the format 'YYYYMMDD_HHMMSS' and is at a fixed position
        # Adjust the slice indices based on your actual filename structure
        date_str = file_name.split("_")[-2] + file_name.split("_")[-1].split(".")[0]
        return datetime.strptime(date_str, "%Y%m%d%H%M%S")

    filtered_files.sort(key=lambda x: extract_date(x), reverse=True)

    return os.path.join(directory, filtered_files[0]) if filtered_files else None


def metrics(args, M, T_RMSE, data):
    dev_chief_values = []
    dev_deputy1_values = []
    dev_deputy2_values = []
    dev_deputy3_values = []
    rmse_chief_values = []
    rmse_deputy1_values = []
    rmse_deputy2_values = []
    rmse_deputy3_values = []
    X_true = data["true"]
    for m in range(M):
        X_est = data[m]

        # Compute relative deviations
        dev_chief = np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0)
        dev_deputy1 = np.linalg.norm(X_est[6:9, :, :] - X_true[6:9, :, :], axis=0)
        dev_deputy2 = np.linalg.norm(X_est[12:15, :, :] - X_true[12:15, :, :], axis=0)
        dev_deputy3 = np.linalg.norm(X_est[18:21, :, :] - X_true[18:21, :, :], axis=0)

        # Compute RMSEs
        rmse_chief = rmse(X_est[:6, :, T_RMSE:], X_true[:6, :, T_RMSE:])
        rmse_deputy1 = rmse(X_est[6:12, :, T_RMSE:], X_true[6:12, :, T_RMSE:])
        rmse_deputy2 = rmse(X_est[12:18, :, T_RMSE:], X_true[12:18, :, T_RMSE:])
        rmse_deputy3 = rmse(X_est[18:24, :, T_RMSE:], X_true[18:24, :, T_RMSE:])

        # Only keep the valid values
        invalid_rmse = 1e2
        if (
            rmse_chief < invalid_rmse
            and rmse_deputy1 < invalid_rmse
            and rmse_deputy2 < invalid_rmse
            and rmse_deputy3 < invalid_rmse
        ):
            dev_chief_values.append(dev_chief)
            dev_deputy1_values.append(dev_deputy1)
            dev_deputy2_values.append(dev_deputy2)
            dev_deputy3_values.append(dev_deputy3)
            rmse_chief_values.append(rmse_chief)
            rmse_deputy1_values.append(rmse_deputy1)
            rmse_deputy2_values.append(rmse_deputy2)
            rmse_deputy3_values.append(rmse_deputy3)
            # print(f'For Monte Carlo Run #{m + 1} the RMSEs are:')
            # print(f'    - Chief: {rmse_chief} m')
            # print(f'    - Deputy 1: {rmse_deputy1} m')
            # print(f'    - Deputy 2: {rmse_deputy2} m')
            # print(f'    - Deputy 3: {rmse_deputy3} m')
        else:
            print(
                f"(!!) For Monte Carlo Run #{m + 1} the algorithm diverged with RMSEs:"
            )
            print(f"    - Chief: {rmse_chief} m")
            print(f"    - Deputy 1: {rmse_deputy1} m")
            print(f"    - Deputy 2: {rmse_deputy2} m")
            print(f"    - Deputy 3: {rmse_deputy3} m")

    # Compute averages
    dev_chief_avg = np.mean(dev_chief_values, axis=0).reshape(-1, 1)
    dev_deputy1_avg = np.mean(dev_deputy1_values, axis=0).reshape(-1, 1)
    dev_deputy2_avg = np.mean(dev_deputy2_values, axis=0).reshape(-1, 1)
    dev_deputy3_avg = np.mean(dev_deputy3_values, axis=0).reshape(-1, 1)
    print(f"Average RMSEs:")
    print(f"    - Chief: {np.mean(rmse_chief_values)} m")
    print(f"    - Deputy 1: {np.mean(rmse_deputy1_values)} m")
    print(f"    - Deputy 2: {np.mean(rmse_deputy2_values)} m")
    print(f"    - Deputy 3: {np.mean(rmse_deputy3_values)} m")
    return dev_chief_avg, dev_deputy1_avg, dev_deputy2_avg, dev_deputy3_avg


def visualizer_devs(args):
    # Load the latest files
    if args.algorithm and args.formation:
        data_filepath = get_latest_file(
            prefix=f"{args.algorithm}_form{args.formation}_"
        )
    else:
        data_filepath = get_latest_file()
    algorithm = re.search(r"/(.*?)_", data_filepath).group(1).upper()
    formation = re.search(r"form(\d+)", data_filepath).group(1)
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
    print(f'Plotting data from "{data_filepath}"...')

    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = np.shape(data["true"])[2]  # Duration [min]
    T_RMSE = T - 95  # Duration for RMSE calculation [min]
    time = np.arange(0, T) / dt  # Time vector [h]
    M = len(data) - 1  # Number of Monte Carlo simulations

    # Get data to plot
    dev_chief, dev_deputy1, dev_deputy2, dev_deputy3 = metrics(args, M, T_RMSE, data)

    # Plot positions based on screen size
    fig_width = 2 * 6.4  # in inches
    fig_height = 2 * 4.8  # in inches

    # Create a 2 by 2 figure
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Plot 1: Chief
    axs[0, 0].plot(time, dev_chief, ".-", label=algorithm)
    axs[0, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{C}_1} - \\mathbf{r}_k^{\\mathcal{C}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 0].set_ylim([1e-2, 1e3])
    axs[0, 0].tick_params(axis="both", which="both", direction="in")
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].set_title("Chief")

    # Plot 2: Deputy 1
    axs[0, 1].plot(time, dev_deputy1, ".-", label=algorithm)
    axs[0, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 1].grid(True, which="both")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_1} - \\mathbf{r}_k^{\\mathcal{D}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 1].set_ylim([1e-1, 1e3])
    axs[0, 1].tick_params(axis="both", which="both", direction="in")
    axs[0, 1].legend(fontsize=12)
    axs[0, 1].set_title("Deputy 1")

    # Plot 3: Deputy 2
    axs[1, 0].plot(time, dev_deputy2, ".-", label=algorithm)
    axs[1, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_2} - \\mathbf{r}_k^{\\mathcal{D}_2}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 0].set_ylim([1e-1, 1e3])
    axs[1, 0].tick_params(axis="both", which="both", direction="in")
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].set_title("Deputy 2")

    # Plot 4: Deputy 3
    axs[1, 1].plot(time, dev_deputy3, ".-", label=algorithm)
    axs[1, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 1].grid(True, which="both")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_3} - \\mathbf{r}_k^{\\mathcal{D}_3}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 1].set_ylim([1e-1, 1e3])
    axs[1, 1].tick_params(axis="both", which="both", direction="in")
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].set_title("Deputy 3")

    fig.suptitle(f"Formation {formation}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    return


def visualizer_all_devs(args):
    # Load the latest files
    data_filepath_fcekf = get_latest_file(prefix=f"fcekf_form{args.formation}_")
    data_filepath_hcmci = get_latest_file(prefix=f"hcmci_form{args.formation}_")
    data_filepath_ccekf = get_latest_file(prefix=f"ccekf_form{args.formation}_")
    with open(data_filepath_fcekf, "rb") as f:
        data_fcekf = pickle.load(f)
    with open(data_filepath_hcmci, "rb") as f:
        data_hcmci = pickle.load(f)
    with open(data_filepath_ccekf, "rb") as f:
        data_ccekf = pickle.load(f)
    print(
        f'Plotting data from "{data_filepath_fcekf}", "{data_filepath_hcmci}" and "{data_filepath_ccekf}"...'
    )

    # Simulation parameters
    dt = 60.0  # Time step [s]
    T = np.shape(data_fcekf["true"])[2]  # Duration [min]
    T_RMSE = T - 95  # Duration for RMSE calculation [min]
    time = np.arange(0, T) / dt  # Time vector [h]
    M = len(data_fcekf) - 1  # Number of Monte Carlo simulations

    # Get data to plot
    dev_chief_fcekf, dev_deputy1_fcekf, dev_deputy2_fcekf, dev_deputy3_fcekf = metrics(
        args, M, T_RMSE, data_fcekf
    )
    dev_chief_hcmci, dev_deputy1_hcmci, dev_deputy2_hcmci, dev_deputy3_hcmci = metrics(
        args, M, T_RMSE, data_hcmci
    )
    dev_chief_ccekf, dev_deputy1_ccekf, dev_deputy2_ccekf, dev_deputy3_ccekf = metrics(
        args, M, T_RMSE, data_ccekf
    )

    # Plot positions based on screen size
    fig_width = 2 * 6.4  # in inches
    fig_height = 2 * 4.8  # in inches

    # Create a 2 by 2 figure
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))

    # Plot 1: Chief
    axs[0, 0].plot(time, dev_chief_fcekf, ".-", label="FCEKF")
    axs[0, 0].plot(time, dev_chief_hcmci, ".-", label="HCMCI")
    axs[0, 0].plot(time, dev_chief_ccekf, ".-", label="CCEKF")
    axs[0, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{C}_1} - \\mathbf{r}_k^{\\mathcal{C}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 0].set_ylim([1e-2, 1e3])
    axs[0, 0].tick_params(axis="both", which="both", direction="in")
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].set_title("Chief")

    # Plot 2: Deputy 1
    axs[0, 1].plot(time, dev_deputy1_fcekf, ".-", label="FCEKF")
    axs[0, 1].plot(time, dev_deputy1_hcmci, ".-", label="HCMCI")
    axs[0, 1].plot(time, dev_deputy1_ccekf, ".-", label="CCEKF")
    axs[0, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[0, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[0, 1].grid(True, which="both")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[0, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_1} - \\mathbf{r}_k^{\\mathcal{D}_1}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[0, 1].set_ylim([1e-1, 1e3])
    axs[0, 1].tick_params(axis="both", which="both", direction="in")
    axs[0, 1].legend(fontsize=12)
    axs[0, 1].set_title("Deputy 1")

    # Plot 3: Deputy 2
    axs[1, 0].plot(time, dev_deputy2_fcekf, ".-", label="FCEKF")
    axs[1, 0].plot(time, dev_deputy2_hcmci, ".-", label="HCMCI")
    axs[1, 0].plot(time, dev_deputy2_ccekf, ".-", label="CCEKF")
    axs[1, 0].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 0].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 0].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_2} - \\mathbf{r}_k^{\\mathcal{D}_2}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 0].set_ylim([1e-1, 1e3])
    axs[1, 0].tick_params(axis="both", which="both", direction="in")
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].set_title("Deputy 2")

    # Plot 4: Deputy 3
    axs[1, 1].plot(time, dev_deputy3_fcekf, ".-", label="FCEKF")
    axs[1, 1].plot(time, dev_deputy3_hcmci, ".-", label="HCMCI")
    axs[1, 1].plot(time, dev_deputy3_ccekf, ".-", label="CCEKF")
    axs[1, 1].grid(which="major", color="#DDDDDD", zorder=1)
    axs[1, 1].grid(which="minor", color="#DDDDDD", linestyle=(0, (1, 3)), zorder=2)
    axs[1, 1].grid(True, which="both")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$t$ [h]", fontsize=12)
    axs[1, 1].set_ylabel(
        "$\\left\\|\\hat{\\mathbf{r}}_k^{\\mathcal{D}_3} - \\mathbf{r}_k^{\\mathcal{D}_3}\\right\\|_\\mathrm{av}$ [km]",
        fontsize=12,
    )
    axs[1, 1].set_ylim([1e-1, 1e3])
    axs[1, 1].tick_params(axis="both", which="both", direction="in")
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].set_title("Deputy 3")

    fig.suptitle(f"Formation {args.formation}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
    return


def run_visualizer(args):
    # Call the visualizer script
    if args.visualize == "orbits":
        eng = engine.start_matlab()
        matlab_functions_path = os.path.abspath("../SpaceSim-Toolkit/src")
        eng.addpath(matlab_functions_path, nargout=0)
        print("Displaying MATLAB figure with the orbits...")
        try:
            if args.formation:
                data_filepath = get_latest_file(
                    prefix=f"prop_form{args.formation}", suffix=".mat"
                )
                formation = args.formation
            else:
                data_filepath = get_latest_file(prefix="prop", suffix=".mat")
                formation = re.search(r"form(\d+)", data_filepath).group(1)
            eng.visualizer_orbits(data_filepath, formation, nargout=0)
        except engine.MatlabExecutionError as e:
            print(f"An error occurred: {e}")
        finally:
            eng.quit()
    elif args.visualize == "all_deviations":
        print(
            f"Displaying all algorithms estimates deviations for Formation {'I' if args.formation == 1 else 'II' if args.formation == 2 else 'Unknown'}..."
        )
        visualizer_all_devs(args)
    else:
        print("Displaying deviation for latest applied algorithm...")
        visualizer_devs(args)
    return
