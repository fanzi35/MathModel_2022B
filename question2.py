from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config.params import (
    FIGURE_DIR,
    FIGURE_DPI,
    GRID_COLOR,
    Q2_ACTUAL_MARKER_COLOR,
    Q2_ANGLE_PERTURB_DEG,
    Q2_DRONE_IDS,
    Q2_EDGE_LENGTH,
    Q2_GROUP_COLORS,
    Q2_IDEAL_MARKER_COLOR,
    Q2_LOSS_TOL,
    Q2_MAX_ROUNDS,
    Q2_RADIUS_PERTURB,
    Q2_RANDOM_SEED,
    Q2_STEP_CANDIDATES,
    Q2_TEMPLATE_COLOR,
    SCATTER_SIZE,
    TABLE_DIR,
)
from utils import (
    angle_at_vertex,
    apply_group_move,
    build_q2_edge_targets,
    build_q2_groups,
    build_q2_ideal_positions,
    cartesian_to_polar,
    circle_intersections,
    configure_matplotlib,
    constant_angle_circles,
    copy_positions,
    dataframe_to_position_dict,
    ensure_directories,
    fit_rigid_template,
    generate_q2_initial_positions_strict,
    make_q2_state_table,
    minimal_angle_difference_deg,
    q2_total_loss,
    unique_points,
)


def build_group_target_points(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    member_ids: list[int],
) -> np.ndarray:
    """将理想模板刚体配准到当前分组。"""
    current_points = np.vstack([positions[drone_id] for drone_id in member_ids])
    template_points = np.vstack([ideal_positions[drone_id] for drone_id in member_ids])
    return fit_rigid_template(current_points, template_points)


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


def update_hex05_group(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    group: dict[str, object],
    edge_targets: dict[tuple[int, int], float],
) -> tuple[dict[int, np.ndarray], dict[str, float | int | str]]:
    """以 FY05 为圆心的局部圆，使用两架圆周飞机进行计算。"""
    center_id = 5
    ring_ids = list(group["members"])[1:]
    loss_before = q2_total_loss(positions, edge_targets)
    best_positions = copy_positions(positions)
    best_loss = loss_before
    best_pair: tuple[int, int] | None = tuple(ring_ids[:2])

    for sender_pair in combinations(ring_ids, 2):
        candidate_positions = copy_positions(positions)
        for receiver_id in ring_ids:
            if receiver_id in sender_pair:
                continue
            estimate = solve_receiver_estimate(
                base_point=positions[center_id],
                anchor_a=positions[sender_pair[0]],
                angle_a=angle_at_vertex(positions[center_id], positions[receiver_id], positions[sender_pair[0]], degrees=False),
                anchor_b=positions[sender_pair[1]],
                angle_b=angle_at_vertex(positions[center_id], positions[receiver_id], positions[sender_pair[1]], degrees=False),
                reference_point=ideal_positions[receiver_id],
            )
            move_vector = ideal_positions[receiver_id] - estimate
            candidate_positions[receiver_id] = positions[receiver_id] + move_vector

        candidate_loss = q2_total_loss(candidate_positions, edge_targets)
        if candidate_loss < best_loss - 1e-12:
            best_positions = candidate_positions
            best_loss = candidate_loss
            best_pair = sender_pair

    assert best_pair is not None, "hex_05 未找到有效的两机发射组合。"
    if best_loss >= loss_before - 1e-12:
        fallback_positions, fallback_record = update_group_with_rigid_template(
            positions,
            ideal_positions,
            group,
            edge_targets,
            fixed_ids=(5,),
        )
        fallback_loss = q2_total_loss(fallback_positions, edge_targets)
        if fallback_loss < best_loss - 1e-12:
            best_positions = fallback_positions
            best_loss = fallback_loss
            record = {
                "轮次局部分组": str(group["name"]),
                "发射机选择": "FY05," + ",".join(f"FY{sender:02d}" for sender in best_pair) + " -> 模板回退",
                "更新前损失": float(loss_before),
                "更新后损失": float(best_loss),
                "更新节点数": int(len(ring_ids) - len(best_pair)),
            }
            return best_positions, record

    assert len(best_pair) == 2, "hex_05 发射机数量应为 2。"
    record = {
        "轮次局部分组": str(group["name"]),
        "发射机选择": "FY05," + ",".join(f"FY{sender:02d}" for sender in best_pair),
        "更新前损失": float(loss_before),
        "更新后损失": float(best_loss),
        "更新节点数": int(len(ring_ids) - len(best_pair)),
    }
    return best_positions, record


def update_hex_with_fy05_reference(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    group: dict[str, object],
    edge_targets: dict[tuple[int, int], float],
) -> tuple[dict[int, np.ndarray], dict[str, float | int | str]]:
    """当 FY05 位于圆周上时，模仿问题一第三小问更新局部圆。"""
    members = list(group["members"])
    center_id = members[0]
    perimeter_ids = members[1:]
    assert 5 in perimeter_ids, "局部圆中必须包含 FY05。"

    candidate_sender_sets: list[tuple[int, ...]] = []
    for sender_count in (2, 3):
        for sender_set in combinations(perimeter_ids, sender_count):
            if 5 in sender_set:
                candidate_sender_sets.append(sender_set)

    loss_before = q2_total_loss(positions, edge_targets)
    best_positions = copy_positions(positions)
    best_loss = loss_before
    best_sender_set: tuple[int, ...] | None = candidate_sender_sets[0] if candidate_sender_sets else None

    for sender_set in candidate_sender_sets:
        extra_senders = tuple(drone_id for drone_id in sender_set if drone_id != 5)
        candidate_positions = copy_positions(positions)
        for receiver_id in perimeter_ids:
            if receiver_id in sender_set:
                continue

            estimates: list[np.ndarray] = []
            for sender_id in extra_senders:
                estimate = solve_receiver_estimate(
                    base_point=positions[center_id],
                    anchor_a=positions[5],
                    angle_a=angle_at_vertex(positions[center_id], positions[receiver_id], positions[5], degrees=False),
                    anchor_b=ideal_positions[sender_id],
                    angle_b=angle_at_vertex(positions[center_id], positions[receiver_id], positions[sender_id], degrees=False),
                    reference_point=ideal_positions[receiver_id],
                )
                estimates.append(estimate)

            mean_estimate = np.mean(estimates, axis=0)
            move_vector = ideal_positions[receiver_id] - mean_estimate
            candidate_positions[receiver_id] = positions[receiver_id] + move_vector

        candidate_loss = q2_total_loss(candidate_positions, edge_targets)
        if candidate_loss < best_loss - 1e-12:
            best_positions = candidate_positions
            best_loss = candidate_loss
            best_sender_set = sender_set

    assert best_sender_set is not None, f"{group['name']} 未找到有效的发射机组合。"
    if best_loss >= loss_before - 1e-12:
        fallback_positions, fallback_record = update_group_with_rigid_template(
            positions,
            ideal_positions,
            group,
            edge_targets,
            fixed_ids=(5, center_id),
        )
        fallback_loss = q2_total_loss(fallback_positions, edge_targets)
        if fallback_loss < best_loss - 1e-12:
            best_positions = fallback_positions
            best_loss = fallback_loss
            record = {
                "轮次局部分组": str(group["name"]),
                "发射机选择": ",".join(f"FY{sender:02d}" for sender in best_sender_set) + " -> 模板回退",
                "更新前损失": float(loss_before),
                "更新后损失": float(best_loss),
                "更新节点数": int(len(perimeter_ids) - len(best_sender_set)),
            }
            return best_positions, record

    assert 5 in best_sender_set and len(best_sender_set) in (2, 3), "参考圆发射机数量应为 2 或 3 且必须包含 FY05。"
    record = {
        "轮次局部分组": str(group["name"]),
        "发射机选择": ",".join(f"FY{sender:02d}" for sender in best_sender_set),
        "更新前损失": float(loss_before),
        "更新后损失": float(best_loss),
        "更新节点数": int(len(perimeter_ids) - len(best_sender_set)),
    }
    return best_positions, record


def update_group_with_rigid_template(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    group: dict[str, object],
    edge_targets: dict[tuple[int, int], float],
    fixed_ids: tuple[int, ...] = (),
) -> tuple[dict[int, np.ndarray], dict[str, float | int | str]]:
    """使用局部刚体模板做一次贪心更新，可固定部分节点不动。"""
    member_ids = list(group["members"])
    target_points = build_group_target_points(positions, ideal_positions, member_ids)

    loss_before = q2_total_loss(positions, edge_targets)
    best_positions = copy_positions(positions)
    best_loss = loss_before
    best_step = 0.0

    for step_size in (0.0, *Q2_STEP_CANDIDATES):
        movable_ids = [drone_id for drone_id in member_ids if drone_id not in fixed_ids]
        movable_targets = np.vstack(
            [target_points[member_ids.index(drone_id)] for drone_id in movable_ids]
        ) if movable_ids else np.empty((0, 2))
        candidate_positions = apply_group_move(positions, movable_ids, movable_targets, step_size)
        candidate_loss = q2_total_loss(candidate_positions, edge_targets)
        if candidate_loss < best_loss - 1e-12:
            best_positions = candidate_positions
            best_loss = candidate_loss
            best_step = step_size

    record = {
        "轮次局部分组": str(group["name"]),
        "发射机选择": "边界刚体更新",
        "更新步长": float(best_step),
        "更新前损失": float(loss_before),
        "更新后损失": float(best_loss),
        "更新节点数": int(len(member_ids)),
    }
    return best_positions, record


def update_boundary_group(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    group: dict[str, object],
    edge_targets: dict[tuple[int, int], float],
) -> tuple[dict[int, np.ndarray], dict[str, float | int | str]]:
    """边界分组继续使用刚体模板贪心更新。"""
    return update_group_with_rigid_template(positions, ideal_positions, group, edge_targets, fixed_ids=())


def update_single_group(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    group: dict[str, object],
    edge_targets: dict[tuple[int, int], float],
) -> tuple[dict[int, np.ndarray], dict[str, float | int | str]]:
    """按分组类型执行一次更新。"""
    if group["kind"] == "boundary":
        return update_boundary_group(positions, ideal_positions, group, edge_targets)
    if str(group["name"]) == "hex_05":
        return update_hex05_group(positions, ideal_positions, group, edge_targets)
    return update_hex_with_fy05_reference(positions, ideal_positions, group, edge_targets)


def run_q2_greedy_iteration(
    initial_positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    groups: list[dict[str, object]],
    edge_targets: dict[tuple[int, int], float],
) -> tuple[dict[int, np.ndarray], list[float], pd.DataFrame]:
    """每轮依次更新三个局部圆和三条边界。"""
    positions = copy_positions(initial_positions)
    loss_history = [q2_total_loss(positions, edge_targets)]
    round_records: list[dict[str, float | int | str]] = []

    for round_index in range(1, Q2_MAX_ROUNDS + 1):
        loss_before_round = q2_total_loss(positions, edge_targets)
        if loss_before_round < Q2_LOSS_TOL:
            break

        for group in groups:
            previous_fy05 = positions[5].copy()
            positions, local_record = update_single_group(
                positions=positions,
                ideal_positions=ideal_positions,
                group=group,
                edge_targets=edge_targets,
            )
            assert np.linalg.norm(positions[5] - previous_fy05) <= 1e-12, "FY05 位置不应发生移动。"
            local_record["轮次"] = round_index
            round_records.append(local_record)

        loss_after_round = q2_total_loss(positions, edge_targets)
        assert loss_after_round <= loss_before_round + 1e-9, "第二问单轮损失未保持非增。"
        loss_history.append(loss_after_round)

        if loss_after_round < Q2_LOSS_TOL:
            break

    return positions, loss_history, pd.DataFrame(round_records)


def orient_q2_positions_for_plot(positions: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """统一第二问绘图朝向，保证分布与题目示意图一致。"""
    oriented = copy_positions(positions)

    rotate_deg = 30.0
    rotate_rad = np.deg2rad(rotate_deg)
    rotation = np.array(
        [
            [np.cos(rotate_rad), -np.sin(rotate_rad)],
            [np.sin(rotate_rad), np.cos(rotate_rad)],
        ],
        dtype=float,
    )
    for drone_id in oriented:
        oriented[drone_id] = rotation @ oriented[drone_id]

    apex_x = oriented[1][0]
    base_center_x = 0.5 * (oriented[11][0] + oriented[15][0])
    if apex_x < base_center_x:
        for drone_id in oriented:
            oriented[drone_id] = np.array([-oriented[drone_id][0], oriented[drone_id][1]], dtype=float)

    if oriented[11][1] < oriented[15][1]:
        for drone_id in oriented:
            oriented[drone_id] = np.array([oriented[drone_id][0], -oriented[drone_id][1]], dtype=float)

    return oriented


def plot_q2_grouping(
    ideal_positions: dict[int, np.ndarray],
    groups: list[dict[str, object]],
    output_path: Path,
) -> None:
    """绘制锥形队列分组示意图。"""
    ideal_positions = orient_q2_positions_for_plot(ideal_positions)
    fig, ax = plt.subplots(figsize=(10, 8))

    for group in groups:
        color = Q2_GROUP_COLORS[str(group["name"])]
        members = list(group["members"])
        if group["kind"] == "hex":
            center = ideal_positions[members[0]]
            ring = np.vstack([ideal_positions[drone_id] for drone_id in members[1:]])
            closed_ring = np.vstack([ring, ring[0]])
            ax.plot(closed_ring[:, 0], closed_ring[:, 1], color=color, linewidth=1.8, label=str(group["name"]))
            for drone_id in members[1:]:
                point = ideal_positions[drone_id]
                ax.plot([center[0], point[0]], [center[1], point[1]], color=color, linewidth=1.1, alpha=0.85)
        else:
            points = np.vstack([ideal_positions[drone_id] for drone_id in members])
            ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.2, label=str(group["name"]))

    ideal_points = np.vstack([ideal_positions[drone_id] for drone_id in Q2_DRONE_IDS])
    ax.scatter(ideal_points[:, 0], ideal_points[:, 1], s=SCATTER_SIZE, color=Q2_TEMPLATE_COLOR, zorder=3)
    for drone_id in Q2_DRONE_IDS:
        point = ideal_positions[drone_id]
        ax.text(point[0] + 2.0, point[1] + 2.0, f"FY{drone_id:02d}", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("横坐标 x / m")
    ax.set_ylabel("纵坐标 y / m")
    ax.grid(True, linestyle=":", linewidth=0.6, color=GRID_COLOR, alpha=0.7)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_q2_state(
    positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """绘制第二问状态示意图。"""
    positions = orient_q2_positions_for_plot(positions)
    ideal_positions = orient_q2_positions_for_plot(ideal_positions)
    fig, ax = plt.subplots(figsize=(10, 8))

    ideal_points = np.vstack([ideal_positions[drone_id] for drone_id in Q2_DRONE_IDS])
    actual_points = np.vstack([positions[drone_id] for drone_id in Q2_DRONE_IDS])

    ax.scatter(
        ideal_points[:, 0],
        ideal_points[:, 1],
        s=SCATTER_SIZE * 1.2,
        facecolors="none",
        edgecolors=Q2_IDEAL_MARKER_COLOR,
        linewidths=1.5,
        label="理想位置",
    )
    ax.scatter(
        actual_points[:, 0],
        actual_points[:, 1],
        s=SCATTER_SIZE,
        color=Q2_ACTUAL_MARKER_COLOR,
        label="当前位置",
        zorder=3,
    )

    for drone_id in Q2_DRONE_IDS:
        point = positions[drone_id]
        ideal_point = ideal_positions[drone_id]
        ax.plot(
            [ideal_point[0], point[0]],
            [ideal_point[1], point[1]],
            linestyle="--",
            linewidth=0.8,
            color=Q2_TEMPLATE_COLOR,
            alpha=0.7,
        )
        ax.text(point[0] + 2.0, point[1] + 2.0, f"FY{drone_id:02d}", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("横坐标 x / m")
    ax.set_ylabel("纵坐标 y / m")
    ax.grid(True, linestyle=":", linewidth=0.6, color=GRID_COLOR, alpha=0.7)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def verify_q2_template(
    ideal_positions: dict[int, np.ndarray],
    edge_targets: dict[tuple[int, int], float],
) -> None:
    """验证理想模板本身满足总损失为零。"""
    template_loss = q2_total_loss(ideal_positions, edge_targets)
    assert template_loss <= 1e-10, f"第二问理想模板损失不为零：{template_loss:.6e}"


def verify_q2_initial_state(
    initial_positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
) -> None:
    """验证随机初始状态满足扰动范围要求。"""
    for drone_id in Q2_DRONE_IDS:
        initial_radius, initial_angle = cartesian_to_polar(initial_positions[drone_id])
        ideal_radius, ideal_angle = cartesian_to_polar(ideal_positions[drone_id])
        if drone_id == 5:
            assert np.linalg.norm(initial_positions[drone_id] - ideal_positions[drone_id]) <= 1e-12
            continue
        assert abs(initial_radius - ideal_radius) <= Q2_RADIUS_PERTURB + 1e-9, f"FY{drone_id:02d} 极径扰动越界。"
        assert (
            minimal_angle_difference_deg(initial_angle, ideal_angle) <= Q2_ANGLE_PERTURB_DEG + 1e-9
        ), f"FY{drone_id:02d} 极角扰动越界。"


def verify_q2_outputs(
    final_positions: dict[int, np.ndarray],
    ideal_positions: dict[int, np.ndarray],
    edge_targets: dict[tuple[int, int], float],
    loss_history: list[float],
    output_paths: list[Path],
) -> None:
    """结束前统一校验收敛与文件输出。"""
    final_loss = q2_total_loss(final_positions, edge_targets)
    assert final_loss < Q2_LOSS_TOL, f"第二问未收敛到阈值内，最终损失为 {final_loss:.6e}"
    assert np.linalg.norm(final_positions[5] - ideal_positions[5]) <= 1e-12, "FY05 最终位置不应变化。"
    assert len(loss_history) >= 2, "第二问未产生有效迭代。"
    assert all(loss_history[index + 1] <= loss_history[index] + 1e-9 for index in range(len(loss_history) - 1)), "第二问损失未保持单调不增。"
    for path in output_paths:
        assert path.exists() and path.stat().st_size > 0, f"输出文件缺失：{path}"


def save_q2_outputs(
    initial_table: pd.DataFrame,
    final_table: pd.DataFrame,
    initial_path: Path,
    final_path: Path,
) -> None:
    """保存第二问表格输出。"""
    initial_table.to_csv(initial_path, index=False, encoding="utf-8-sig")
    final_table.to_csv(final_path, index=False, encoding="utf-8-sig")


def main() -> None:
    """运行第二问的局部模板贪心迭代。"""
    configure_matplotlib()
    ensure_directories([FIGURE_DIR, TABLE_DIR])

    ideal_df = build_q2_ideal_positions(Q2_EDGE_LENGTH)
    ideal_positions = dataframe_to_position_dict(ideal_df)
    groups = build_q2_groups()
    edge_targets = build_q2_edge_targets(Q2_EDGE_LENGTH)

    verify_q2_template(ideal_positions, edge_targets)

    initial_positions = generate_q2_initial_positions_strict(
        ideal_positions=ideal_positions,
        radius_perturb=Q2_RADIUS_PERTURB,
        angle_perturb_deg=Q2_ANGLE_PERTURB_DEG,
        seed=Q2_RANDOM_SEED,
    )
    verify_q2_initial_state(initial_positions, ideal_positions)

    final_positions, loss_history, _ = run_q2_greedy_iteration(
        initial_positions=initial_positions,
        ideal_positions=ideal_positions,
        groups=groups,
        edge_targets=edge_targets,
    )

    initial_table = make_q2_state_table(initial_positions)
    final_table = make_q2_state_table(final_positions)

    initial_table_path = TABLE_DIR / "q2_initial_state.csv"
    final_table_path = TABLE_DIR / "q2_final_state.csv"
    group_figure_path = FIGURE_DIR / "q2_grouping.png"
    initial_figure_path = FIGURE_DIR / "q2_initial_state.png"
    final_figure_path = FIGURE_DIR / "q2_final_state.png"

    save_q2_outputs(initial_table, final_table, initial_table_path, final_table_path)
    plot_q2_grouping(ideal_positions, groups, group_figure_path)
    plot_q2_state(initial_positions, ideal_positions, initial_figure_path)
    plot_q2_state(final_positions, ideal_positions, final_figure_path)

    verify_q2_outputs(
        final_positions=final_positions,
        ideal_positions=ideal_positions,
        edge_targets=edge_targets,
        loss_history=loss_history,
        output_paths=[
            initial_table_path,
            final_table_path,
            group_figure_path,
            initial_figure_path,
            final_figure_path,
        ],
    )

    print(f"第二问迭代轮次: {len(loss_history) - 1}")
    print(f"第二问最终损失: {loss_history[-1]:.8e}")


if __name__ == "__main__":
    main()
