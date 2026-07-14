from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.params import (
    ANGLE_I,
    ANGLE_J,
    ANGLE_K,
    DRONE_SIZE,
    FIGURE_DIR,
    FIGURE_SIZE,
    FORMATION_COLOR,
    FORMATION_COUNT,
    FORMATION_RADIUS,
    HELPER_COLOR,
    MAIN_COLOR,
    PLOT_LIMIT,
    RADIUS_K,
    RECEIVER_COLOR,
    SECONDARY_COLOR,
    TABLE_DIR,
    VERIFY_TOL,
)
from utils import (
    angle_of_line,
    annotate_point,
    build_formation_dataframe,
    circle_from_three_points,
    configure_matplotlib,
    draw_angle_arc,
    draw_circle,
    draw_drone,
    ensure_directories,
    point_to_circle_residual,
    polar_to_cartesian,
    setup_axes,
)


def build_scene() -> dict[str, np.ndarray | pd.DataFrame | float]:
    """构造示意图所需的关键点。"""
    formation = build_formation_dataframe(FORMATION_RADIUS, FORMATION_COUNT)
    point_o = np.array([0.0, 0.0])
    point_a = polar_to_cartesian(FORMATION_RADIUS, ANGLE_I)
    point_b = polar_to_cartesian(FORMATION_RADIUS, ANGLE_J)
    point_p = polar_to_cartesian(RADIUS_K, ANGLE_K)

    center_1, radius_1 = circle_from_three_points(point_o, point_a, point_p)
    center_2, radius_2 = circle_from_three_points(point_o, point_b, point_p)

    return {
        "formation": formation,
        "O": point_o,
        "A": point_a,
        "B": point_b,
        "P": point_p,
        "C1": center_1,
        "R1": radius_1,
        "C2": center_2,
        "R2": radius_2,
    }


def verify_scene(scene: dict[str, np.ndarray | pd.DataFrame | float]) -> None:
    """运行几何自检，确保生成的图形关系正确。"""
    formation = scene["formation"]
    point_o = scene["O"]
    point_a = scene["A"]
    point_b = scene["B"]
    point_p = scene["P"]
    center_1 = scene["C1"]
    center_2 = scene["C2"]
    radius_1 = scene["R1"]
    radius_2 = scene["R2"]

    outer_residual = np.sqrt(formation["x"] ** 2 + formation["y"] ** 2) - FORMATION_RADIUS
    assert np.max(np.abs(outer_residual)) < VERIFY_TOL, "圆形编队坐标不在同一圆周上。"

    assert point_to_circle_residual(point_o, center_1, radius_1) < VERIFY_TOL
    assert point_to_circle_residual(point_a, center_1, radius_1) < VERIFY_TOL
    assert point_to_circle_residual(point_p, center_1, radius_1) < VERIFY_TOL
    assert point_to_circle_residual(point_o, center_2, radius_2) < VERIFY_TOL
    assert point_to_circle_residual(point_b, center_2, radius_2) < VERIFY_TOL
    assert point_to_circle_residual(point_p, center_2, radius_2) < VERIFY_TOL


def draw_formation(ax, formation: pd.DataFrame, scene: dict[str, np.ndarray | pd.DataFrame | float]) -> None:
    """绘制圆形编队与飞机图标。"""
    draw_circle(
        ax,
        np.array([0.0, 0.0]),
        FORMATION_RADIUS,
        color=FORMATION_COLOR,
        linestyle="-",
        linewidth=1.6,
    )

    for row in formation.itertuples(index=False):
        point = np.array([row.x, row.y], dtype=float)
        heading = angle_of_line(np.array([0.0, 0.0]), point)
        color = HELPER_COLOR
        if np.allclose(point, scene["A"]):
            color = MAIN_COLOR
        elif np.allclose(point, scene["B"]):
            color = SECONDARY_COLOR
        draw_drone(ax, point, heading, DRONE_SIZE, color=color)

    draw_drone(ax, scene["O"], np.pi / 2, DRONE_SIZE, color=MAIN_COLOR)
    draw_drone(ax, scene["P"], angle_of_line(scene["O"], scene["P"]), DRONE_SIZE, color=RECEIVER_COLOR)


def save_constant_angle_circle(scene: dict[str, np.ndarray | pd.DataFrame | float], output_path: Path) -> None:
    """生成单个定角圆示意图。"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    draw_formation(ax, scene["formation"], scene)

    draw_circle(
        ax,
        scene["C1"],
        scene["R1"],
        color=MAIN_COLOR,
        linestyle="--",
        linewidth=1.8,
    )

    ax.plot(
        [scene["O"][0], scene["A"][0], scene["P"][0], scene["O"][0]],
        [scene["O"][1], scene["A"][1], scene["P"][1], scene["O"][1]],
        color=HELPER_COLOR,
        linewidth=1.2,
    )
    draw_angle_arc(ax, scene["P"], scene["O"], scene["A"], 1.4, MAIN_COLOR, "定角")

    annotate_point(ax, "O", scene["O"], (-0.8, -0.8), text_offset=(0.3, -1.2), text="FY00")
    annotate_point(ax, "A", scene["A"], (0.4, -0.6), text_offset=(0.6, -1.5), text="FY0i")
    annotate_point(ax, "P", scene["P"], (0.4, 0.4), text_offset=(0.6, -0.8), text="FY0k")
    ax.text(scene["B"][0] - 1.1, scene["B"][1] + 0.7, "FY0j", fontsize=10, color="#333333")
    ax.text(scene["C1"][0] + 0.3, scene["C1"][1] + 0.5, "定角圆", color=MAIN_COLOR, fontsize=11)

    setup_axes(ax, PLOT_LIMIT)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_intersection_figure(scene: dict[str, np.ndarray | pd.DataFrame | float], output_path: Path) -> None:
    """生成两个定角圆相交示意图。"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    draw_formation(ax, scene["formation"], scene)

    draw_circle(
        ax,
        scene["C1"],
        scene["R1"],
        color=MAIN_COLOR,
        linestyle="--",
        linewidth=1.8,
    )
    draw_circle(
        ax,
        scene["C2"],
        scene["R2"],
        color=SECONDARY_COLOR,
        linestyle="--",
        linewidth=1.8,
    )

    ax.plot(
        [scene["O"][0], scene["A"][0], scene["P"][0], scene["O"][0]],
        [scene["O"][1], scene["A"][1], scene["P"][1], scene["O"][1]],
        color=MAIN_COLOR,
        linewidth=1.0,
        alpha=0.65,
    )
    ax.plot(
        [scene["O"][0], scene["B"][0], scene["P"][0], scene["O"][0]],
        [scene["O"][1], scene["B"][1], scene["P"][1], scene["O"][1]],
        color=SECONDARY_COLOR,
        linewidth=1.0,
        alpha=0.65,
    )
    draw_angle_arc(ax, scene["P"], scene["O"], scene["A"], 1.3, MAIN_COLOR, "角1")
    draw_angle_arc(ax, scene["P"], scene["B"], scene["O"], 2.1, SECONDARY_COLOR, "角2", text_scale=1.1)

    annotate_point(ax, "O", scene["O"], (-0.8, -0.8), text_offset=(0.3, -1.2), text="FY00")
    annotate_point(ax, "A", scene["A"], (0.4, -0.6), text_offset=(0.6, -1.5), text="FY0i")
    annotate_point(ax, "B", scene["B"], (-1.2, 0.4), text_offset=(-2.0, 1.1), text="FY0j")
    annotate_point(ax, "P", scene["P"], (0.4, 0.4), text_offset=(0.6, -0.8), text="FY0k")
    ax.text(scene["C1"][0] + 0.3, scene["C1"][1] + 0.5, "定角圆1", color=MAIN_COLOR, fontsize=11)
    ax.text(scene["C2"][0] - 1.6, scene["C2"][1] + 0.3, "定角圆2", color=SECONDARY_COLOR, fontsize=11)

    setup_axes(ax, PLOT_LIMIT)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """主函数：自检并生成两张示意图。"""
    configure_matplotlib()
    ensure_directories([FIGURE_DIR, TABLE_DIR])
    scene = build_scene()
    verify_scene(scene)

    constant_path = FIGURE_DIR / "constant_angle_circle.png"
    intersect_path = FIGURE_DIR / "intersecting_constant_angle_circles.png"

    save_constant_angle_circle(scene, constant_path)
    save_intersection_figure(scene, intersect_path)

    assert constant_path.exists() and constant_path.stat().st_size > 0
    assert intersect_path.exists() and intersect_path.stat().st_size > 0

    print(f"已生成图片: {constant_path}")
    print(f"已生成图片: {intersect_path}")


if __name__ == "__main__":
    main()
