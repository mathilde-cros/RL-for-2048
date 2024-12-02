
import argparse

description = "Imitation learning"
algo_list = ["random", "heuristic", "policy_gradient"]
heuristic_list = ["empty-cells", "snake",
                  "monotonic", "smoothness", "merge-potential", "corner-max-tile", "combined"]


def get_args():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--algo", choices=algo_list, required=True, help="algorithm"
    )

    parser.add_argument(
        "--delay", type=int, default=100, help="delay between moves"
    )

    parser.add_argument('--gui', action='store_true',
                        help='Enable GUI', default=False)
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for simulation')

    parser.add_argument('--heuristic', choices=heuristic_list,
                        required=False, default=heuristic_list[0])

    parser.set_defaults(**get_default_args() or {})

    args = parser.parse_args()
    return args


def get_default_args() -> dict:
    pass
