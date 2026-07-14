from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).parent.parent
FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
TABLE_DIR = ROOT_DIR / "outputs" / "tables"

FORMATION_RADIUS = 10.0
FORMATION_COUNT = 9
FIGURE_SIZE = (8, 8)
PLOT_LIMIT = 16.0
DRONE_SIZE = 0.45
VERIFY_TOL = 1e-7

# 示例编号与位置：FY0i=FY01，FY0j=FY04，FY0k≈FY07
ANGLE_I = 0.0
ANGLE_J = 2 * np.pi / 3
ANGLE_K = np.deg2rad(238.0)
RADIUS_K = 10.7

MAIN_COLOR = "#3b6fb6"
SECONDARY_COLOR = "#2b8a3e"
RECEIVER_COLOR = "#c0392b"
FORMATION_COLOR = "#444444"
HELPER_COLOR = "#888888"
