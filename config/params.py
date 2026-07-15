from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_FILE = ROOT_DIR / "data" / "raw" / "question1_3.xlsx"
FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
TABLE_DIR = ROOT_DIR / "outputs" / "tables"

FORMATION_RADIUS = 100.0
FORMATION_COUNT = 9
CENTER_ID = 0
FIXED_TRANSMITTER_ID = 1
OUTER_IDS = tuple(range(1, FORMATION_COUNT + 1))
MAX_EXTRA_TRANSMITTERS = 2

CONVERGENCE_TOL = 1e-4
MAX_ROUNDS = 60
GEOMETRY_TOL = 1e-8

FIGURE_DPI = 300
SCATTER_SIZE = 36
IDEAL_COLOR = "#3b6fb6"
ACTUAL_COLOR = "#c0392b"
TRACK_COLOR = "#2f855a"
GRID_COLOR = "#b0b0b0"

Q2_DRONE_IDS = tuple(range(1, 16))
Q2_EDGE_LENGTH = 50.0
Q2_LOSS_TOL = 1e-4
Q2_MAX_ROUNDS = 200
Q2_RANDOM_SEED = 20260714
Q2_RADIUS_PERTURB = 15.0
Q2_ANGLE_PERTURB_DEG = 0.5
Q2_STEP_CANDIDATES = (0.15, 0.3, 0.5, 0.7, 1.0)
Q2_FIXED_CENTER_ID = 5
Q2_IDEAL_MARKER_COLOR = "#3b6fb6"
Q2_ACTUAL_MARKER_COLOR = "#c0392b"
Q2_TEMPLATE_COLOR = "#7f8c8d"
Q2_GROUP_COLORS = {
    "hex_05": "#1f77b4",
    "hex_08": "#2ca02c",
    "hex_09": "#d62728",
    "boundary_top": "#9467bd",
    "boundary_bottom": "#ff7f0e",
    "boundary_left": "#8c564b",
}
