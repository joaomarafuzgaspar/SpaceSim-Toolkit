import argparse

from visualizer import run_visualizer
from simulation import run_simulation, run_propagation


class MatlabAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            # If no value is provided, set a flag (e.g., True)
            setattr(namespace, self.dest, True)
        else:
            # If a value is provided, use it
            setattr(namespace, self.dest, values)


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
        "-v",
        "--visualize",
        nargs="?",  # '?' allows the argument to be optional
        default=None,
        const=True,  # If the argument is provided without a value, set it to True
        choices=[None, "all_deviations", "orbits"],
        action=MatlabAction,
        help="Display figure(s); use 'all_deviations' to display all algorithms estimates deviations and 'orbits' to display the orbits.",
    )
    parser.add_argument(
        "-f",
        "--formation",
        type=int,
        choices=[1, 2],
        help="Formation to simulate (1 for VREx mission formation or 2 for higher-orbit difference formation)",
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        choices=["wlstsq-lm", "fcekf", "hcmci", "ccekf", "cnkkt", "unkkt"],
        help="Navigation algorithm to simulate (wlstsq-lm for WLSTSQ-LM, fcekf for FCEKF, hcmci for HCMCI, ccekf for CCEKF, cnkkt for CNKKT, or unkkt for UNKKT)",
    )
    parser.add_argument(
        "-M",
        "--monte-carlo-sims",
        default=1,
        type=check_positive,
        help="Number of Monte-Carlo simulations to run (must be an integer >= 1)",
    )
    parser.add_argument(
        "-p",
        "--propagate",
        action="store_true",
        help="Propagate the spacecraft dynamics using the dynamics model",
    )

    # Parse arguments
    args = parser.parse_args()

    # Run propagation
    if args.propagate and args.formation:
        print(
            f"Propagating spacecraft dynamics for Formation {'I' if args.formation == 1 else 'II' if args.formation == 2 else 'Unknown'}..."
        )
        run_propagation(args)

    # Run simulation
    if args.monte_carlo_sims and args.formation and args.algorithm:
        print(
            f"Running {args.monte_carlo_sims} Monte-Carlo simulations for Formation {'I' if args.formation == 1 else 'II' if args.formation == 2 else 'Unknown'} with Algorithm {args.algorithm}..."
        )
        run_simulation(args)

    # Check if the MATLAB flag is set
    if args.visualize:
        run_visualizer(args)


if __name__ == "__main__":
    main()
