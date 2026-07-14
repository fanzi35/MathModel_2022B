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
