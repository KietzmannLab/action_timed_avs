# params.py
import argparse
import os

# ==============================
# Default Parameters
# ==============================
QUANTILES = 160
NUM_ALPHAS = 40
EVENT_TYPE = "fixation"
EVENT_COMPARISON_ANALYSIS = "mixing_factor_analysis"  # default
EVENT_TYPE_IC = 'scene'
USE_ICA_DATA = False # False for Fig. 1 E; True for Fig. 2
DISCARDED_ICS_DIPOLE_LOCATION = {
    "as01": [31, 45, 46, 56, 57], # 5/80 = 6.3%
    "as02": [17, 19, 23, 28, 36, 43, 44, 66, 67, 71, 72, 73, 75, 78], # 14/80 = 17.5%
    "as03": [7, 8, 23, 39, 40, 60], # 6/80 = 7.5%
    "as04": [4, 24, 28, 41, 64, 70], # 6/80 = 7.5%
    "as05": [38, 47, 52], # 3/80 = 3.8%
}

# ==============================
# Argument Parsing / Overrides
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="Event Comparison Parameters")

    parser.add_argument(
        "--event_comparison",
        type=str,
        help="Event comparison override, e.g., 'pso', 'motion_energy', etc."
    )
    parser.add_argument(
        "--use_ica_data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=160,
    )

    args, _ = parser.parse_known_args()
    return args

args = parse_args()

# ==============================
# Environment fallback (optional)
# ==============================
ENV_EVENT_COMPARISON = os.environ.get("EVENT_COMPARISON_ANALYSIS")
ENV_USE_ICA_DATA = os.environ.get("USE_ICA_DATA", "false").lower() == "true"

# ==============================
# Apply overrides
# ==============================
EVENT_COMPARISON_ANALYSIS = args.event_comparison or ENV_EVENT_COMPARISON or EVENT_COMPARISON_ANALYSIS
USE_ICA_DATA = args.use_ica_data or ENV_USE_ICA_DATA or USE_ICA_DATA
if USE_ICA_DATA:
    QUANTILES = args.quantiles if args.quantiles is not None else 15


