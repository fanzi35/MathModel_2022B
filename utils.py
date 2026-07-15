from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.params import CENTER_ID, FIXED_TRANSMITTER_ID, FORMATION_COUNT, FORMATION_RADIUS, GEOMETRY_TOL, Q2_FIXED_CENTER_ID


def configure_matplotlib() -> None:
    """设置中文字体。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_directories(paths: list[Path]) -> None:
    """创建输出目录。"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def parse_polar_text(value: str) -> tuple[float, float]:
    """解析形如 '(98,40.10)' 的极坐标文本。"""
    clean = str(value).strip().replace("(", "").replace(")", "")
    radius_text, angle_text = [item.strip() for item in clean.split(",")]
    return float(radius_text), float(angle_text)


def polar_to_cartesian(radius: float, angle_deg: float) -> np.ndarray:
    """极坐标转直角坐标。"""
    angle_rad = np.deg2rad(angle_deg)
    return np.array([radius * np.cos(angle_rad), radius * np.sin(angle_rad)], dtype=float)


def cartesian_to_polar(point: np.ndarray) -> tuple[float, float]:
    """直角坐标转极坐标。"""
    x_coord, y_coord = map(float, point)
    radius = float(np.hypot(x_coord, y_coord))
    angle_deg = float(np.degrees(np.arctan2(y_coord, x_coord)))
    if angle_deg < 0:
        angle_deg += 360.0
    return radius, angle_deg


def load_initial_positions(excel_path: Path) -> pd.DataFrame:
    """读取初始位置表并补充直角坐标。"""
    raw_df = pd.read_excel(excel_path, sheet_name="Sheet1")
    parsed = raw_df["极坐标(m,°)"].map(parse_polar_text)
    result = raw_df.copy()
    result["radius"] = parsed.map(lambda item: item[0])
    result["theta_deg"] = parsed.map(lambda item: item[1])
    coords = np.vstack([polar_to_cartesian(row.radius, row.theta_deg) for row in result.itertuples()])
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    result = result.rename(columns={"无人机编号": "drone_id"})
    result["drone_id"] = result["drone_id"].astype(int)
    return result[["drone_id", "radius", "theta_deg", "x", "y"]]


def build_ideal_positions() -> pd.DataFrame:
    """生成理想编队位置表。"""
    rows: list[dict[str, float | int]] = [
        {"drone_id": CENTER_ID, "radius": 0.0, "theta_deg": 0.0, "x": 0.0, "y": 0.0}
    ]
    for drone_id in range(1, FORMATION_COUNT + 1):
        theta_deg = 360.0 * (drone_id - 1) / FORMATION_COUNT
        x_coord, y_coord = polar_to_cartesian(FORMATION_RADIUS, theta_deg)
        rows.append(
            {
                "drone_id": drone_id,
                "radius": FORMATION_RADIUS,
                "theta_deg": theta_deg,
                "x": x_coord,
                "y": y_coord,
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_position_dict(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """把位置表转成坐标字典。"""
    return {
        int(row.drone_id): np.array([row.x, row.y], dtype=float)
        for row in df.itertuples(index=False)
    }


def angle_at_vertex(point_a: np.ndarray, vertex: np.ndarray, point_b: np.ndarray, degrees: bool = False) -> float:
    """计算顶点处夹角。"""
    vector_a = np.asarray(point_a, dtype=float) - np.asarray(vertex, dtype=float)
    vector_b = np.asarray(point_b, dtype=float) - np.asarray(vertex, dtype=float)
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < GEOMETRY_TOL or norm_b < GEOMETRY_TOL:
        return 0.0
    cosine_value = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine_value = float(np.clip(cosine_value, -1.0, 1.0))
    angle_rad = float(np.arccos(cosine_value))
    if degrees:
        return float(np.degrees(angle_rad))
    return angle_rad


def build_angle_lookup_table(ideal_positions: dict[int, np.ndarray]) -> pd.DataFrame:
    """预计算角度识别表。"""
    rows: list[dict[str, float | int]] = []
    for receiver_id in range(2, FORMATION_COUNT + 1):
        candidates: list[tuple[int, float]] = []
        for sender_id in range(2, FORMATION_COUNT + 1):
            if sender_id == receiver_id:
                continue
            feature_angle = angle_at_vertex(
                ideal_positions[FIXED_TRANSMITTER_ID],
                ideal_positions[receiver_id],
                ideal_positions[sender_id],
                degrees=True,
            )
            candidates.append((sender_id, feature_angle))
        candidates.sort(key=lambda item: item[1])
        for index, (sender_id, feature_angle) in enumerate(candidates):
            lower = 0.0 if index == 0 else 0.5 * (candidates[index - 1][1] + feature_angle)
            upper = 180.0 if index == len(candidates) - 1 else 0.5 * (feature_angle + candidates[index + 1][1])
            rows.append(
                {
                    "接收无人机": receiver_id,
                    "候选发射无人机": sender_id,
                    "理论角度(度)": feature_angle,
                    "角度下界(度)": lower,
                    "角度上界(度)": upper,
                }
            )
    return pd.DataFrame(rows)


def identify_sender_id(receiver_id: int, observed_angle_deg: float, lookup_table: pd.DataFrame) -> int:
    """根据角度识别辅助发射机编号。"""
    subset = lookup_table[lookup_table["接收无人机"] == receiver_id]
    matched = subset[
        (subset["角度下界(度)"] <= observed_angle_deg)
        & (subset["角度上界(度)"] >= observed_angle_deg)
    ]
    if not matched.empty:
        best_row = matched.iloc[(matched["理论角度(度)"] - observed_angle_deg).abs().argmin()]
        return int(best_row["候选发射无人机"])
    best_row = subset.iloc[(subset["理论角度(度)"] - observed_angle_deg).abs().argmin()]
    return int(best_row["候选发射无人机"])


def constant_angle_circles(point_a: np.ndarray, point_b: np.ndarray, angle_rad: float) -> list[tuple[np.ndarray, float]]:
    """由弦和圆周角构造两个定角圆。"""
    if angle_rad <= GEOMETRY_TOL or abs(np.pi - angle_rad) <= GEOMETRY_TOL:
        raise ValueError("角度过小或过大，无法稳定构造定角圆。")
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    chord = point_b - point_a
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= GEOMETRY_TOL:
        raise ValueError("弦长过小，无法构造定角圆。")
    midpoint = 0.5 * (point_a + point_b)
    normal = np.array([-chord[1], chord[0]], dtype=float) / chord_length
    radius = chord_length / (2.0 * np.sin(angle_rad))
    offset = chord_length / (2.0 * np.tan(angle_rad))
    centers = [midpoint + offset * normal, midpoint - offset * normal]
    return [(center, radius) for center in centers]


def circle_intersections(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
) -> list[np.ndarray]:
    """计算两圆交点。"""
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    distance = float(np.linalg.norm(center_b - center_a))
    if distance <= GEOMETRY_TOL:
        return []
    if distance > radius_a + radius_b + GEOMETRY_TOL:
        return []
    if distance < abs(radius_a - radius_b) - GEOMETRY_TOL:
        return []

    x_value = (radius_a**2 - radius_b**2 + distance**2) / (2.0 * distance)
    y_square = radius_a**2 - x_value**2
    if y_square < -GEOMETRY_TOL:
        return []
    y_value = float(np.sqrt(max(y_square, 0.0)))

    direction = (center_b - center_a) / distance
    base = center_a + x_value * direction
    normal = np.array([-direction[1], direction[0]], dtype=float)

    point_1 = base + y_value * normal
    point_2 = base - y_value * normal
    if np.linalg.norm(point_1 - point_2) <= GEOMETRY_TOL:
        return [point_1]
    return [point_1, point_2]


def unique_points(points: list[np.ndarray], tol: float = 1e-6) -> list[np.ndarray]:
    """去除重复交点。"""
    result: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tol for existing in result):
            result.append(point)
    return result


def l2_loss(positions: dict[int, np.ndarray], ideal_positions: dict[int, np.ndarray]) -> float:
    """计算外圈无人机 L2 损失。"""
    return float(
        sum(np.linalg.norm(positions[drone_id] - ideal_positions[drone_id]) ** 2 for drone_id in range(1, FORMATION_COUNT + 1))
    )


def max_position_error(positions: dict[int, np.ndarray], ideal_positions: dict[int, np.ndarray]) -> float:
    """计算最大位置误差。"""
    return float(
        max(np.linalg.norm(positions[drone_id] - ideal_positions[drone_id]) for drone_id in range(1, FORMATION_COUNT + 1))
    )


def make_result_dataframe(final_positions: dict[int, np.ndarray]) -> pd.DataFrame:
    """整理与初始表同结构的最终结果表。"""
    rows: list[dict[str, float | int | str]] = []
    for drone_id in range(0, FORMATION_COUNT + 1):
        final_radius, final_theta = cartesian_to_polar(final_positions[drone_id])
        rows.append(
            {
                "无人机编号": drone_id,
                "极坐标(m,°)": f"({final_radius:.6f},{final_theta:.6f})",
            }
        )
    return pd.DataFrame(rows)


def build_q2_ideal_positions(edge_length: float) -> pd.DataFrame:
    """生成第二问锥形编队的理想坐标。"""
    sqrt3 = np.sqrt(3.0)
    rows = [
        {"drone_id": 1, "x": 1.5 * edge_length, "y": -0.5 * sqrt3 * edge_length},
        {"drone_id": 2, "x": 1.0 * edge_length, "y": 0.0 * edge_length},
        {"drone_id": 3, "x": 0.5 * edge_length, "y": -0.5 * sqrt3 * edge_length},
        {"drone_id": 4, "x": 0.5 * edge_length, "y": 0.5 * sqrt3 * edge_length},
        {"drone_id": 5, "x": 0.0, "y": 0.0},
        {"drone_id": 6, "x": -0.5 * edge_length, "y": -0.5 * sqrt3 * edge_length},
        {"drone_id": 7, "x": 0.0, "y": 1.0 * sqrt3 * edge_length},
        {"drone_id": 8, "x": -0.5 * edge_length, "y": 0.5 * sqrt3 * edge_length},
        {"drone_id": 9, "x": -1.0 * edge_length, "y": 0.0 * edge_length},
        {"drone_id": 10, "x": -1.5 * edge_length, "y": -0.5 * sqrt3 * edge_length},
        {"drone_id": 11, "x": -0.5 * edge_length, "y": 1.5 * sqrt3 * edge_length},
        {"drone_id": 12, "x": -1.0 * edge_length, "y": 1.0 * sqrt3 * edge_length},
        {"drone_id": 13, "x": -1.5 * edge_length, "y": 0.5 * sqrt3 * edge_length},
        {"drone_id": 14, "x": -2.0 * edge_length, "y": 0.0 * edge_length},
        {"drone_id": 15, "x": -2.5 * edge_length, "y": -0.5 * sqrt3 * edge_length},
    ]
    result = pd.DataFrame(rows)
    polar = result.apply(lambda row: cartesian_to_polar(np.array([row["x"], row["y"]], dtype=float)), axis=1)
    result["radius"] = polar.map(lambda item: item[0])
    result["theta_deg"] = polar.map(lambda item: item[1])
    return result[["drone_id", "radius", "theta_deg", "x", "y"]]


def build_q2_groups() -> list[dict[str, object]]:
    """构造第二问的三个局部圆与三条边界分组。"""
    return [
        {"name": "hex_05", "kind": "hex", "members": [5, 2, 3, 6, 9, 8, 4]},
        {"name": "hex_08", "kind": "hex", "members": [8, 5, 4, 7, 12, 13, 9]},
        {"name": "hex_09", "kind": "hex", "members": [9, 5, 8, 13, 14, 10, 6]},
        {"name": "boundary_top", "kind": "boundary", "members": [11, 7, 4, 2, 1]},
        {"name": "boundary_bottom", "kind": "boundary", "members": [15, 10, 6, 3, 1]},
        {"name": "boundary_left", "kind": "boundary", "members": [11, 12, 13, 14, 15]},
    ]


def build_q2_edge_targets(edge_length: float) -> dict[tuple[int, int], float]:
    """生成第二问总损失函数对应的目标边集。"""
    edge_targets: dict[tuple[int, int], float] = {}
    groups = build_q2_groups()
    for group in groups:
        members = list(group["members"])
        if group["kind"] == "hex":
            center = members[0]
            ring = members[1:]
            for node_id in ring:
                edge_targets[tuple(sorted((center, node_id)))] = edge_length
            for left, right in zip(ring, ring[1:] + ring[:1]):
                edge_targets[tuple(sorted((left, right)))] = edge_length
        else:
            for start_index, left in enumerate(members):
                for right_index in range(start_index + 1, len(members)):
                    right = members[right_index]
                    edge_targets[tuple(sorted((left, right)))] = edge_length * (right_index - start_index)
    return edge_targets


def q2_total_loss(positions: dict[int, np.ndarray], edge_targets: dict[tuple[int, int], float]) -> float:
    """计算第二问总损失函数。"""
    return float(
        sum((np.linalg.norm(positions[i] - positions[j]) - distance) ** 2 for (i, j), distance in edge_targets.items())
    )


def generate_q2_initial_positions(
    ideal_positions: dict[int, np.ndarray],
    radius_perturb: float,
    angle_perturb_deg: float,
    seed: int,
) -> dict[int, np.ndarray]:
    """在给定偏差范围内随机生成第二问初始状态。"""
    rng = np.random.default_rng(seed)
    positions: dict[int, np.ndarray] = {}
    for drone_id, point in ideal_positions.items():
        radius, theta_deg = cartesian_to_polar(point)
        if radius <= GEOMETRY_TOL:
            sampled_radius = float(rng.uniform(0.0, radius_perturb))
            sampled_theta = float(rng.uniform(0.0, 360.0))
        else:
            sampled_radius = max(0.0, radius + float(rng.uniform(-radius_perturb, radius_perturb)))
            sampled_theta = theta_deg + float(rng.uniform(-angle_perturb_deg, angle_perturb_deg))
        positions[drone_id] = polar_to_cartesian(sampled_radius, sampled_theta)
    return positions


def make_polar_state_table(positions: dict[int, np.ndarray]) -> pd.DataFrame:
    """把位置字典整理成编号、极径、极角表。"""
    rows: list[dict[str, float | int]] = []
    for drone_id in sorted(positions):
        radius, theta_deg = cartesian_to_polar(positions[drone_id])
        rows.append({"无人机编号": drone_id, "极径": radius, "极角(度)": theta_deg})
    return pd.DataFrame(rows)


def fit_rigid_template(current_points: np.ndarray, template_points: np.ndarray) -> np.ndarray:
    """将理想模板刚体配准到当前点集。"""
    current_center = current_points.mean(axis=0)
    template_center = template_points.mean(axis=0)
    current_shift = current_points - current_center
    template_shift = template_points - template_center
    covariance = template_shift.T @ current_shift
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    transformed = template_shift @ rotation.T + current_center
    return transformed


def generate_q2_initial_positions_strict(
    ideal_positions: dict[int, np.ndarray],
    radius_perturb: float,
    angle_perturb_deg: float,
    seed: int,
) -> dict[int, np.ndarray]:
    """生成第二问随机初始状态，并固定模板中心节点。"""
    rng = np.random.default_rng(seed)
    positions: dict[int, np.ndarray] = {}
    for drone_id, point in ideal_positions.items():
        if drone_id == Q2_FIXED_CENTER_ID:
            positions[drone_id] = point.copy()
            continue

        radius, theta_deg = cartesian_to_polar(point)
        sampled_radius = max(0.0, radius + float(rng.uniform(-radius_perturb, radius_perturb)))
        sampled_theta = theta_deg + float(rng.uniform(-angle_perturb_deg, angle_perturb_deg))
        positions[drone_id] = polar_to_cartesian(sampled_radius, sampled_theta)
    return positions


def make_q2_state_table(positions: dict[int, np.ndarray]) -> pd.DataFrame:
    """整理第二问状态表，只输出编号、极径、极角。"""
    rows: list[dict[str, float | int]] = []
    for drone_id in sorted(positions):
        radius, theta_deg = cartesian_to_polar(positions[drone_id])
        rows.append({"无人机编号": drone_id, "极径": round(radius, 6), "极角": round(theta_deg, 6)})
    return pd.DataFrame(rows)


def minimal_angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """计算两个角度的最小圆周差。"""
    diff = abs(float(angle_a) - float(angle_b)) % 360.0
    return min(diff, 360.0 - diff)


def copy_positions(positions: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """复制位置字典。"""
    return {drone_id: point.copy() for drone_id, point in positions.items()}


def apply_group_move(
    positions: dict[int, np.ndarray],
    member_ids: list[int],
    target_points: np.ndarray,
    step_size: float,
) -> dict[int, np.ndarray]:
    """按给定步长将一个分组向局部目标移动。"""
    updated = copy_positions(positions)
    for index, drone_id in enumerate(member_ids):
        updated[drone_id] = positions[drone_id] + step_size * (target_points[index] - positions[drone_id])
    return updated
