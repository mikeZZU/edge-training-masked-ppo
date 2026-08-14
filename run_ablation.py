"""Train and evaluate the controlled PPO ablations."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
from train_ppo import (
    PPOPolicy,
    generate_dynamic_trace,
    train_one_seed,
)


VARIANTS = [
    {
        "name": "Full Masked-PPO",
        "tag": "full",
        "use_action_mask": True,
        "state_mode": "full",
        "reward_mode": "regret",
    },
    {
        "name": "w/o Dynamic State",
        "tag": "no_dynamic_state",
        "use_action_mask": True,
        "state_mode": "static_resource",
        "reward_mode": "regret",
    },
    {
        "name": "w/o Memory Mask",
        "tag": "no_memory_mask",
        "use_action_mask": False,
        "state_mode": "full",
        "reward_mode": "regret",
    },
    {
        "name": "w/o Regret Reward",
        "tag": "no_regret_reward",
        "use_action_mask": True,
        "state_mode": "full",
        "reward_mode": "absolute",
    },
]


def load_ablation_profile():
    """Use the paper profile already used by the published outputs."""
    candidates = [
        cfg.OUTPUT_DIR / f"profile_{cfg.MODE_NAME}.csv",
        cfg.WORKSPACE / "figure2" / "profile_paper.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "No paper profile found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_model(model_path):
    model = PPOPolicy(
        3 + cfg.NUM_SPLIT_POINTS,
        cfg.NUM_SPLIT_POINTS,
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
        )
    )
    model.eval()
    return model


@torch.inference_mode()
def simulate_variant(
    variant,
    seed,
    profile_df,
    accuracy_df,
    model,
):
    accuracy_seed = (
        accuracy_df[accuracy_df["seed"] == seed]
        .sort_values("successful_updates")
        .reset_index(drop=True)
    )
    if len(accuracy_seed) == 0:
        raise RuntimeError(
            f"No accuracy history for seed {seed}."
        )

    total_updates = int(
        accuracy_seed["successful_updates"].max()
    )
    trace = generate_dynamic_trace(
        profile_df,
        length=total_updates,
        seed=500000 + seed,
    )
    profile_memory = profile_df["memory_mb"].to_numpy(
        dtype=np.float64
    )
    bandwidth = trace["bandwidth_mbps"].to_numpy(dtype=np.float64)
    available_memory = trace["memory_mb"].to_numpy(dtype=np.float64)
    alpha = trace["alpha"].to_numpy(dtype=np.float64)
    rtt = trace["rtt_ms"].to_numpy(dtype=np.float64)
    device_ms = profile_df["device_ms"].to_numpy(dtype=np.float64)
    server_ms = profile_df["server_ms"].to_numpy(dtype=np.float64)
    feature_bytes = profile_df["feature_bytes"].to_numpy(dtype=np.float64)

    latency_matrix = (
        alpha[:, None] * device_ms[None, :]
        + server_ms[None, :]
        + 2.0
        * feature_bytes[None, :]
        / (bandwidth[:, None] * 1e6 / 8.0)
        * 1000.0
        + rtt[:, None]
    )
    feasible_matrix = (
        profile_memory[None, :] <= available_memory[:, None]
    )
    feasible_latencies = np.where(
        feasible_matrix,
        latency_matrix,
        np.inf,
    )
    oracle_actions = np.argmin(feasible_latencies, axis=1)
    best_feasible_latencies = np.min(feasible_latencies, axis=1)

    resource_state = np.column_stack(
        [
            np.clip((bandwidth - 5.0) / 125.0, 0.0, 1.0),
            np.clip(
                (available_memory - profile_memory.min())
                / max(1.10 * profile_memory.max() - profile_memory.min(), 1e-8),
                0.0,
                1.0,
            ),
            np.clip((alpha - 0.75) / 0.80, 0.0, 1.0),
        ]
    ).astype(np.float32)
    if variant["state_mode"] == "static_resource":
        static_state = np.median(resource_state, axis=0)
        resource_state[:] = static_state

    torch.set_num_threads(1)
    resource_tensor = torch.from_numpy(resource_state)
    feasible_tensor = torch.from_numpy(feasible_matrix)
    state_tensor = torch.zeros(
        (1, 3 + cfg.NUM_SPLIT_POINTS),
        dtype=torch.float32,
    )

    safe_action = int(np.argmin(profile_memory))
    previous_action = safe_action

    elapsed_system_ms = 0.0
    oom_count = 0
    switch_count = 0
    low_attempts = 0
    low_successes = 0
    regrets = []
    oracle_matches = 0
    decision_times_us = []
    checkpoint_index = 0
    time_to_target_s = np.nan

    for update_index in range(1, total_updates + 1):
        row_index = update_index - 1
        current_memory = float(available_memory[row_index])
        feasible_mask = feasible_matrix[row_index]
        feasible_count = int(feasible_mask.sum())
        is_low_memory = feasible_count == 1
        if is_low_memory:
            low_attempts += 1

        decision_start = time.perf_counter_ns()
        state_tensor[0, :3] = resource_tensor[row_index]
        state_tensor[0, 3:].zero_()
        state_tensor[0, 3 + previous_action] = 1.0
        logits, _ = model(
            state_tensor
        )
        current_logits = logits[0]
        if variant["use_action_mask"]:
            current_logits = current_logits.masked_fill(
                ~feasible_tensor[row_index],
                -1e9,
            )
        action = int(torch.argmax(current_logits).item())
        decision_times_us.append(
            (time.perf_counter_ns() - decision_start) / 1000.0
        )

        selected_latency_ms = float(
            latency_matrix[row_index, action]
        )
        best_feasible_latency = float(
            best_feasible_latencies[row_index]
        )
        regrets.append(
            100.0
            * float(
                np.clip(
                    (selected_latency_ms - best_feasible_latency)
                    / max(best_feasible_latency, 1e-8),
                    0.0,
                    1.0,
                )
            )
        )

        oracle_action = int(oracle_actions[row_index])
        if feasible_mask[action] and action == oracle_action:
            oracle_matches += 1

        required_memory = float(
            profile_df.iloc[action]["memory_mb"]
        )
        feasible = required_memory <= current_memory
        switched = action != previous_action
        switch_count += int(switched)

        if feasible:
            elapsed_system_ms += (
                selected_latency_ms
                + (cfg.SWITCH_COST_MS if switched else 0.0)
                + decision_times_us[-1] / 1000.0
            )
            if is_low_memory:
                low_successes += 1
            previous_action = action
        else:
            oom_count += 1
            failed_compute_ms = (
                cfg.OOM_FAILED_RATIO * selected_latency_ms
            )
            safe_latency_ms = float(
                latency_matrix[row_index, safe_action]
            )
            recovery_switch_ms = (
                cfg.SWITCH_COST_MS
                if previous_action != safe_action
                else 0.0
            )
            elapsed_system_ms += (
                decision_times_us[-1] / 1000.0
                + failed_compute_ms
                + cfg.OOM_RECOVERY_MS
                + recovery_switch_ms
                + safe_latency_ms
            )
            previous_action = safe_action

        while (
            checkpoint_index < len(accuracy_seed)
            and update_index
            >= int(
                accuracy_seed.iloc[checkpoint_index][
                    "successful_updates"
                ]
            )
        ):
            accuracy_value = float(
                accuracy_seed.iloc[checkpoint_index]["test_accuracy"]
            )
            if (
                np.isnan(time_to_target_s)
                and accuracy_value >= cfg.TARGET_ACCURACY
            ):
                time_to_target_s = elapsed_system_ms / 1000.0
            checkpoint_index += 1

    return {
        "variant": variant["name"],
        "seed": seed,
        "time_to_target_s": float(time_to_target_s),
        "final_system_time_s": float(elapsed_system_ms / 1000.0),
        "oom_count": int(oom_count),
        "low_memory_attempts": int(low_attempts),
        "low_memory_survival_rate": float(
            100.0 * low_successes / max(low_attempts, 1)
        ),
        "switch_count": int(switch_count),
        "mean_regret_percent": float(np.mean(regrets)),
        "oracle_match_rate": float(
            100.0 * oracle_matches / max(total_updates, 1)
        ),
        "decision_median_us": float(np.median(decision_times_us)),
        "decision_p95_us": float(
            np.percentile(decision_times_us, 95)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain-full",
        action="store_true",
        help="Retrain the full model instead of reusing existing paper models.",
    )
    args = parser.parse_args()

    cfg.prepare_output_dir()
    profile_df = load_ablation_profile()
    accuracy_df = pd.read_csv(
        cfg.OUTPUT_DIR
        / f"accuracy_history_{cfg.MODE_NAME}.csv"
    )
    ablation_root = cfg.OUTPUT_DIR / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in VARIANTS:
        variant_dir = ablation_root / variant["tag"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("=" * 70)
        print(variant["name"])
        print("=" * 70)

        for seed in cfg.SEEDS:
            existing_full = (
                cfg.OUTPUT_DIR
                / f"ppo_seed_{seed}_{cfg.MODE_NAME}.pt"
            )
            if (
                variant["tag"] == "full"
                and existing_full.exists()
                and not args.retrain_full
            ):
                model_path = existing_full
                print(f"seed={seed}: reuse {model_path}")
            else:
                result = train_one_seed(
                    seed,
                    profile_df,
                    use_action_mask=variant["use_action_mask"],
                    state_mode=variant["state_mode"],
                    reward_mode=variant["reward_mode"],
                    output_dir=variant_dir,
                    run_name=variant["tag"],
                )
                model_path = Path(result["model_path"])

            model = load_model(model_path)
            row = simulate_variant(
                variant,
                seed,
                profile_df,
                accuracy_df,
                model,
            )
            rows.append(row)
            print(
                f"seed={seed} | "
                f"time={row['time_to_target_s']:.3f}s | "
                f"OOM={row['oom_count']} | "
                f"low={row['low_memory_survival_rate']:.2f}% | "
                f"switch={row['switch_count']}"
            )

    per_seed_df = pd.DataFrame(rows)
    per_seed_path = (
        ablation_root / f"ablation_per_seed_{cfg.MODE_NAME}.csv"
    )
    per_seed_df.to_csv(per_seed_path, index=False, encoding="utf-8-sig")

    numeric_columns = [
        "time_to_target_s",
        "final_system_time_s",
        "oom_count",
        "low_memory_survival_rate",
        "switch_count",
        "mean_regret_percent",
        "oracle_match_rate",
        "decision_median_us",
        "decision_p95_us",
    ]
    summary_rows = []
    for variant_name, group in per_seed_df.groupby("variant", sort=False):
        summary = {"variant": variant_name}
        for column in numeric_columns:
            values = group[column].to_numpy(dtype=np.float64)
            summary[f"{column}_mean"] = float(np.nanmean(values))
            summary[f"{column}_std"] = float(
                np.nanstd(values, ddof=1)
                if len(values) > 1
                else 0.0
            )
        summary_rows.append(summary)

    summary_path = (
        ablation_root / f"ablation_summary_{cfg.MODE_NAME}.csv"
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Per-seed results: {per_seed_path}")
    print(f"Summary results : {summary_path}")


if __name__ == "__main__":
    main()
