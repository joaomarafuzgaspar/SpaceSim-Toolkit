import argparse

from utils import run_visualizer
from simulation import run_simulation


def check_positive(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="SpaceSim-Toolkit is an advanced, open-source simulation framework designed for space enthusiasts, researchers, and engineers. This versatile toolkit focuses on providing a robust platform for simulating spacecraft dynamics, orbital mechanics, and navigation algorithms in various space missions and scenarios."
    )
    parser.add_argument(
        "-m", "--matlab", action="store_true", help="Display MATLAB figure"
    )
    parser.add_argument(
        "-f",
        "--formation",
        type=int,
        choices=[1, 2],
        default=1,
        help="Formation to simulate (1 for VREx mission formation or 2 for higher-orbit difference formation)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["fcekf", "hcmci", "ccekf"],
        default="fcekf",
        help="Navigation algorithm to simulate (fcekf for FCEKF, hcmci for HCMCI, or ccekf for CCEKF)",
    )
    parser.add_argument(
        "-M",
        "--monte-carlo-sims",
        type=check_positive,
        default=1,
        help="Number of Monte-Carlo simulations to run (must be an integer >= 1)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run simulation
    print(
        f"Running {args.monte_carlo_sims} Monte-Carlo simulations for Formation {'I' if args.formation == 1 else 'II' if args.formation == 2 else 'Unknown'} with Algorithm {args.algorithm}..."
    )
    run_simulation(args)

    # Check if the MATLAB flag is set
    if args.matlab:
        print("Displaying MATLAB figure...")
        run_visualizer()


if __name__ == "__main__":
    main()
