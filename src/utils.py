import numpy as np
import matlab.engine

def rmse(X_est, X_true):
    return np.sqrt(
        np.mean(np.linalg.norm(X_est[:3, :, :] - X_true[:3, :, :], axis=0) ** 2)
    )


def run_visualizer():
    # Run MATLAB
    eng = matlab.engine.start_matlab()
    eng.visualizer(nargout=0)
    eng.quit()
