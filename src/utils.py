import os
import numpy as np
import matlab.engine


def rmse(X_est, X_true):
    return np.sqrt(
        np.mean(np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0) ** 2)
    )


def run_visualizer(to_display):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Get the absolute path to the directory containing visualizer.m
    matlab_functions_path = os.path.abspath("../SpaceSim-Toolkit/src")

    # Add the directory to the MATLAB path
    eng.addpath(matlab_functions_path, nargout=0)

    # Call the visualizer script
    if to_display == "orbits":
        try:
            eng.visualizer_orbits(nargout=0)
        except matlab.engine.MatlabExecutionError as e:
            print(f"An error occurred: {e}")
    elif to_display == "all_deviations":
        try:
            eng.visualizer_all(nargout=0)
        except matlab.engine.MatlabExecutionError as e:
            print(f"An error occurred: {e}")
    else:
        try:
            eng.visualizer(nargout=0)
        except matlab.engine.MatlabExecutionError as e:
            print(f"An error occurred: {e}")

    # Quit the MATLAB engine
    eng.quit()
