from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.params import (
    ACTUAL_COLOR,
    CENTER_ID,
    CONVERGENCE_TOL,
    FIGURE_DIR,
    FIGURE_DPI,
    FIXED_TRANSMITTER_ID,
    FORMATION_COUNT,
    FORMATION_RADIUS,
    GRID_COLOR,
    IDEAL_COLOR,
    MAX_EXTRA_TRANSMITTERS,
    MAX_ROUNDS,
    OUTER_IDS,
    RAW_DATA_FILE,
    SCATTER_SIZE,
    TABLE_DIR,
    TRACK_COLOR,
)
from utils import (
    angle_at_vertex,
    build_angle_lookup_table,
    build_ideal_positions,
    cartesian_to_polar,
    circle_intersections,
    configure_matplotlib,
    constant_angle_circles,
    dataframe_to_position_dict,
    ensure_directories,
    identify_sender_id,
    l2_loss,
    load_initial_positions,
    make_result_dataframe,
    max_position_error,
    unique_points,
)


def solve_receiver_estimate(
    base_point: np.ndarray,
    anchor_a: np.ndarray,
    angle_a: float,
    anchor_b: np.ndarray,
    angle_b: float,
    reference_point: np.ndarray,
) -> np.ndarray:
    """用两个定角圆求接收机位置估计。"""
    candidates: list[np.ndarray] = []
    circles_a = constant_angle_circles(base_point, anchor_a, angle_a)
    circles_b = constant_angle_circles(base_point, anchor_b, angle_b)

    for center_a, radius_a in circles_a:
        for center_b, radius_b in circles_b:
            for point in circle_intersections(center_a, radius_a, center_b, radius_b):
                if np.linalg.norm(point - base_point) > 1e-6:
                    candidates.append(point)

    candidates = unique_points(candidates)
    if not candidates:
        return np.asarray(reference_point, dtype=float).copy()
    return min(candidates, key=lambda point: np.linalg.norm(point - reference_point))


def estimate_with_single_sender(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    receiver_id: int,
    actual_sender_id: int,
    lookup_table: pd.DataFrame,
) -> tuple[np.ndarray, int, float]:
    """使用一个辅助发射机生成当前位置估计。"""
    observed_identity_angle = angle_at_vertex(
        positions[FIXED_TRANSMITTER_ID],
        positions[receiver_id],
        positions[actual_sender_id],
        degrees=True,
    )
    pseudo_sender_id = identify_sender_id(receiver_id, observed_identity_angle, lookup_table)

    angle_01 = angle_at_vertex(
        positions[CENTER_ID],
        positions[receiver_id],
        positions[FIXED_TRANSMITTER_ID],
        degrees=False,
    )
    angle_0s = angle_at_vertex(
        positions[CENTER_ID],
        positions[receiver_id],
        positions[actual_sender_id],
        degrees=False,
    )
    estimate = solve_receiver_estimate(
        positions[CENTER_ID],
        ideal_positions[FIXED_TRANSMITTER_ID],
        angle_01,
        ideal_positions[pseudo_sender_id],
        angle_0s,
        ideal_positions[receiver_id],
    )
    return estimate, pseudo_sender_id, observed_identity_angle


def estimate_receiver_position(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    receiver_id: int,
    extra_senders: tuple[int, ...],
    lookup_table: pd.DataFrame,
) -> tuple[np.ndarray, list[int]]:
    """使用一个或两个辅助发射机估计接收机当前位置。"""
    estimates: list[np.ndarray] = []
    matched_ids: list[int] = []
    for sender_id in extra_senders:
        estimate, pseudo_sender_id, _ = estimate_with_single_sender(
            positions,
            ideal_positions,
            receiver_id,
            sender_id,
            lookup_table,
        )
        estimates.append(estimate)
        matched_ids.append(pseudo_sender_id)
    return np.mean(estimates, axis=0), matched_ids


def apply_round(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    extra_senders: tuple[int, ...],
    lookup_table: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """对一个候选发射机方案执行一轮调整。"""
    new_positions = {drone_id: point.copy() for drone_id, point in positions.items()}
    transmitters = {FIXED_TRANSMITTER_ID, *extra_senders}
    records: list[dict[str, object]] = []

    for receiver_id in OUTER_IDS:
        if receiver_id in transmitters:
            continue
        estimated_position, matched_ids = estimate_receiver_position(
            positions,
            ideal_positions,
            receiver_id,
            extra_senders,
            lookup_table,
        )
        move_vector = ideal_positions[receiver_id] - estimated_position
        new_positions[receiver_id] = positions[receiver_id] + move_vector
        final_error = float(np.linalg.norm(new_positions[receiver_id] - ideal_positions[receiver_id]))
        records.append(
            {
                "接收无人机": receiver_id,
                "估计位置x": float(estimated_position[0]),
                "估计位置y": float(estimated_position[1]),
                "匹配发射编号": ",".join(f"FY{sender_id:02d}" for sender_id in matched_ids),
                "移动向量x": float(move_vector[0]),
                "移动向量y": float(move_vector[1]),
                "更新后误差": final_error,
            }
        )
    return new_positions, pd.DataFrame(records)


def choose_best_round(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    lookup_table: pd.DataFrame,
) -> tuple[tuple[int, ...], dict[int, np.ndarray], pd.DataFrame]:
    """单轮贪心：选择下一轮 L2 最小的发射机组合。"""
    best_choice: tuple[int, ...] | None = None
    best_positions: dict[int, np.ndarray] | None = None
    best_records: pd.DataFrame | None = None
    best_loss = np.inf

    candidate_ids = tuple(drone_id for drone_id in OUTER_IDS if drone_id != FIXED_TRANSMITTER_ID)
    for sender_count in range(1, MAX_EXTRA_TRANSMITTERS + 1):
        for extra_senders in combinations(candidate_ids, sender_count):
            candidate_positions, candidate_records = apply_round(
                positions,
                ideal_positions,
                extra_senders,
                lookup_table,
            )
            candidate_loss = l2_loss(candidate_positions, ideal_positions)
            if candidate_loss < best_loss:
                best_choice = extra_senders
                best_positions = candidate_positions
                best_records = candidate_records
                best_loss = candidate_loss

    assert best_choice is not None
    assert best_positions is not None
    assert best_records is not None
    return best_choice, best_positions, best_records


def run_greedy_adjustment(
    initial_positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    lookup_table: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict[int, list[np.ndarray]], list[float]]:
    """运行单轮贪心迭代。"""
    positions = {drone_id: point.copy() for drone_id, point in initial_positions.items()}
    trajectories: dict[int, list[np.ndarray]] = {
        drone_id: [point.copy()] for drone_id, point in positions.items()
    }
    round_rows: list[dict[str, object]] = []
    loss_history = [l2_loss(positions, ideal_positions)]

    for round_index in range(1, MAX_ROUNDS + 1):
        loss_before = l2_loss(positions, ideal_positions)
        if loss_before <= CONVERGENCE_TOL:
            break

        extra_senders, next_positions, receiver_records = choose_best_round(
            positions,
            ideal_positions,
            lookup_table,
        )
        loss_after = l2_loss(next_positions, ideal_positions)
        max_error_after = max_position_error(next_positions, ideal_positions)

        round_rows.append(
            {
                "轮次": round_index,
                "发射机": ",".join(["FY00", f"FY{FIXED_TRANSMITTER_ID:02d}", *[f"FY{sender:02d}" for sender in extra_senders]]),
                "接收机数量": int(len(receiver_records)),
                "轮前L2": loss_before,
                "轮后L2": loss_after,
                "轮后最大误差": max_error_after,
            }
        )

        positions = next_positions
        for drone_id in positions:
            trajectories[drone_id].append(positions[drone_id].copy())
        loss_history.append(loss_after)

        if loss_after <= CONVERGENCE_TOL:
            break

    return positions, pd.DataFrame(round_rows), trajectories, loss_history


def plot_formation(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """绘制编队位置示意图。"""
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(FORMATION_RADIUS * np.cos(theta), FORMATION_RADIUS * np.sin(theta), color=IDEAL_COLOR, linewidth=1.4)

    ideal_points = np.vstack([ideal_positions[drone_id] for drone_id in OUTER_IDS])
    actual_points = np.vstack([positions[drone_id] for drone_id in OUTER_IDS])
    ax.scatter(ideal_points[:, 0], ideal_points[:, 1], s=SCATTER_SIZE, color=IDEAL_COLOR, label="理想位置")
    ax.scatter(actual_points[:, 0], actual_points[:, 1], s=SCATTER_SIZE, color=ACTUAL_COLOR, label="当前/最终位置")
    ax.scatter([0.0], [0.0], s=SCATTER_SIZE, color="black")
    ax.text(3.0, -4.0, "FY00", fontsize=10)

    for drone_id in OUTER_IDS:
        ax.text(positions[drone_id][0] + 1.5, positions[drone_id][1] + 1.5, f"FY{drone_id:02d}", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("横坐标 x")
    ax.set_ylabel("纵坐标 y")
    ax.grid(True, linestyle=":", linewidth=0.6, color=GRID_COLOR, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    trajectories: dict[int, list[np.ndarray]],
    ideal_positions: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """绘制九架外圈无人机的轨迹子图。"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for subplot, drone_id in zip(axes.flat, OUTER_IDS):
        track = np.vstack(trajectories[drone_id])
        subplot.plot(FORMATION_RADIUS * np.cos(theta), FORMATION_RADIUS * np.sin(theta), color=IDEAL_COLOR, linewidth=1.0)
        subplot.plot(track[:, 0], track[:, 1], color=TRACK_COLOR, linewidth=1.2, marker="o", markersize=2.8)
        subplot.scatter(
            [ideal_positions[drone_id][0]],
            [ideal_positions[drone_id][1]],
            color=IDEAL_COLOR,
            s=SCATTER_SIZE,
        )
        subplot.scatter([track[0, 0]], [track[0, 1]], color=ACTUAL_COLOR, s=SCATTER_SIZE)
        subplot.scatter([track[-1, 0]], [track[-1, 1]], color="black", s=SCATTER_SIZE * 0.75)
        subplot.text(ideal_positions[drone_id][0] + 2.0, ideal_positions[drone_id][1] + 2.0, f"FY{drone_id:02d}", fontsize=9)
        subplot.set_aspect("equal")
        subplot.set_xlabel("横坐标 x")
        subplot.set_ylabel("纵坐标 y")
        subplot.grid(True, linestyle=":", linewidth=0.6, color=GRID_COLOR, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history: list[float], output_path: Path) -> None:
    """绘制 L2 收敛曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = np.arange(len(loss_history))
    ax.plot(rounds, loss_history, color=TRACK_COLOR, linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("轮次")
    ax.set_ylabel("L2 loss")
    ax.grid(True, linestyle=":", linewidth=0.6, color=GRID_COLOR, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def save_tables(
    angle_lookup_table: pd.DataFrame,
    round_table: pd.DataFrame,
    result_table: pd.DataFrame,
) -> None:
    """保存结果表格。"""
    angle_lookup_table.to_excel(TABLE_DIR / "angle_lookup_table.xlsx", index=False)
    round_table.to_excel(TABLE_DIR / "greedy_round_selection.xlsx", index=False)
    result_table.to_excel(TABLE_DIR / "final_result_table.xlsx", index=False)


def verify_outputs(
    final_positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    round_table: pd.DataFrame,
    loss_history: list[float],
) -> None:
    """在结束前做结果校验。"""
    final_loss = l2_loss(final_positions, ideal_positions)
    assert final_loss <= CONVERGENCE_TOL, f"算法未收敛到阈值内，最终 L2 为 {final_loss:.6e}"
    assert round_table.shape[0] > 0, "没有生成有效轮次记录。"
    assert all(loss_history[index + 1] <= loss_history[index] + 1e-9 for index in range(len(loss_history) - 1)), "L2 未单调下降。"


def main() -> None:
    """运行第一问第(3)问的单轮贪心调整。"""
    configure_matplotlib()
    ensure_directories([FIGURE_DIR, TABLE_DIR])

    initial_df = load_initial_positions(RAW_DATA_FILE)
    ideal_df = build_ideal_positions()
    initial_positions = dataframe_to_position_dict(initial_df)
    ideal_positions = dataframe_to_position_dict(ideal_df)
    angle_lookup_table = build_angle_lookup_table(ideal_positions)

    final_positions, round_table, trajectories, loss_history = run_greedy_adjustment(
        initial_positions,
        ideal_positions,
        angle_lookup_table,
    )
    result_table = make_result_dataframe(final_positions)

    save_tables(angle_lookup_table, round_table, result_table)
    plot_formation(initial_positions, ideal_positions, FIGURE_DIR / "q1_initial_formation.png")
    plot_trajectories(trajectories, ideal_positions, FIGURE_DIR / "q1_trajectory_grid.png")
    plot_loss_curve(loss_history, FIGURE_DIR / "q1_l2_loss_curve.png")
    plot_formation(final_positions, ideal_positions, FIGURE_DIR / "q1_final_formation.png")

    verify_outputs(final_positions, ideal_positions, round_table, loss_history)

    print(f"已完成轮次: {len(round_table)}")
    print(f"最终 L2 loss: {loss_history[-1]:.8e}")
    print(f"最终最大误差: {max_position_error(final_positions, ideal_positions):.8e}")


if __name__ == "__main__":
    main()
