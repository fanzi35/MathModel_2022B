from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon


def configure_matplotlib() -> None:
    """设置中文字体与负号显示。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def polar_to_cartesian(radius: float, angle: float) -> np.ndarray:
    """极坐标转直角坐标。"""
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)


def build_formation_dataframe(radius: float, count: int) -> pd.DataFrame:
    """生成圆形编队的理想位置表。"""
    indices = np.arange(1, count + 1)
    angles = 2 * np.pi * (indices - 1) / count
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    return pd.DataFrame(
        {
            "code": [f"FY{idx:02d}" for idx in indices],
            "angle": angles,
            "x": xs,
            "y": ys,
        }
    )


def angle_of_line(p1: np.ndarray, p2: np.ndarray) -> float:
    """返回连线方向角。"""
    delta = np.asarray(p2) - np.asarray(p1)
    return float(np.arctan2(delta[1], delta[0]))


def circle_from_three_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[np.ndarray, float]:
    """由三点确定圆心和半径。"""
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)
    det = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(det) < 1e-10:
        raise ValueError("三点近乎共线，无法确定圆。")

    ux = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    ) / det
    uy = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    ) / det
    center = np.array([ux, uy], dtype=float)
    radius = float(np.linalg.norm(center - np.asarray(p1)))
    return center, radius


def point_to_circle_residual(
    point: np.ndarray, center: np.ndarray, radius: float
) -> float:
    """计算点到圆的残差。"""
    return float(abs(np.linalg.norm(np.asarray(point) - np.asarray(center)) - radius))


def ensure_directories(paths: list[Path]) -> None:
    """创建所需目录。"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def draw_circle(
    ax,
    center: np.ndarray,
    radius: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None = None,
) -> None:
    """绘制圆。"""
    patch = Circle(
        tuple(center),
        radius,
        fill=False,
        edgecolor=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(patch)


def draw_drone(
    ax,
    center: np.ndarray,
    heading: float,
    size: float,
    color: str,
    edgecolor: str = "white",
    zorder: int = 5,
) -> None:
    """绘制一个简化无人机图标。"""
    base = np.array(
        [
            [1.1, 0.0],
            [0.15, 0.28],
            [0.05, 0.65],
            [-0.10, 0.20],
            [-0.75, 0.28],
            [-0.35, 0.0],
            [-0.75, -0.28],
            [-0.10, -0.20],
            [0.05, -0.65],
            [0.15, -0.28],
        ],
        dtype=float,
    )
    rotation = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )
    points = base @ rotation.T * size + np.asarray(center)
    drone = Polygon(
        points,
        closed=True,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=0.7,
        zorder=zorder,
    )
    ax.add_patch(drone)


def draw_angle_arc(
    ax,
    center: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    color: str,
    text: str,
    text_scale: float = 1.2,
) -> None:
    """在顶点位置绘制夹角弧线。"""
    angle1 = np.degrees(angle_of_line(center, p1))
    angle2 = np.degrees(angle_of_line(center, p2))
    start, end = sorted([angle1, angle2])
    if end - start > 180:
        start, end = end, start + 360
    arc = Arc(
        tuple(center),
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=start,
        theta2=end,
        color=color,
        linewidth=1.2,
        zorder=4,
    )
    ax.add_patch(arc)
    mid = np.radians((start + end) / 2)
    text_point = np.asarray(center) + radius * text_scale * np.array([np.cos(mid), np.sin(mid)])
    ax.text(text_point[0], text_point[1], text, color=color, fontsize=11)


def annotate_point(
    ax,
    label: str,
    point: np.ndarray,
    point_offset: tuple[float, float],
    text_offset: tuple[float, float] | None = None,
    text: str | None = None,
) -> None:
    """标注几何点和对应编号。"""
    ax.scatter(point[0], point[1], s=24, c="black", zorder=6)
    ax.text(point[0] + point_offset[0], point[1] + point_offset[1], label, fontsize=11)
    if text:
        tx, ty = text_offset if text_offset is not None else point_offset
        ax.text(point[0] + tx, point[1] + ty, text, fontsize=10, color="#333333")


def setup_axes(ax, limit: float) -> None:
    """统一坐标轴样式。"""
    ax.set_aspect("equal")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("横坐标 x")
    ax.set_ylabel("纵坐标 y")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

