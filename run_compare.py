import time

import numpy as np
import pandas as pd
import torch

import config as cfg

from train_ppo import (
    PPOPolicy,
    build_state,
    calculate_latency_ms,
    generate_dynamic_trace,
    load_profile,
)


METHODS = [
    "Delay-only Search",
    "Memory-Constrained Search",
    "PPO",
]


# ============================================================
# 1. 读取真实 accuracy-successful updates 曲线
# ============================================================

def load_accuracy_history():

    path = (
        cfg.OUTPUT_DIR
        / f"accuracy_history_{cfg.MODE_NAME}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Accuracy history does not exist: {path}"
        )

    df = pd.read_csv(
        path
    )

    return df


# ============================================================
# 2. 加载指定 seed 的 PPO
# ============================================================

def load_ppo_model(seed):

    state_dimension = (
        3
        + cfg.NUM_SPLIT_POINTS
    )

    model = PPOPolicy(
        state_dimension,
        cfg.NUM_SPLIT_POINTS,
    )

    model_path = (
        cfg.OUTPUT_DIR
        / (
            f"ppo_seed_{seed}_"
            f"{cfg.MODE_NAME}.pt"
        )
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"PPO model does not exist: {model_path}"
        )

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


# ============================================================
# 3. Delay-only
# ============================================================

def choose_delay_only(
    profile_df,
    state_row,
):

    latencies = np.asarray(
        [
            calculate_latency_ms(
                profile_df,
                action,
                state_row,
            )
            for action in range(
                cfg.NUM_SPLIT_POINTS
            )
        ],
        dtype=np.float64,
    )

    action = int(
        np.argmin(
            latencies
        )
    )

    return action


# ============================================================
# 4. Memory-Constrained Search
# ============================================================

def choose_memory_constrained(
    profile_df,
    state_row,
):

    available_memory = float(
        state_row[
            "memory_mb"
        ]
    )

    memory = (
        profile_df[
            "memory_mb"
        ].to_numpy(
            dtype=np.float64
        )
    )

    feasible = (
        memory
        <= available_memory
    )

    if not feasible.any():

        raise RuntimeError(
            "No feasible split action."
        )

    latencies = np.asarray(
        [
            calculate_latency_ms(
                profile_df,
                action,
                state_row,
            )
            for action in range(
                cfg.NUM_SPLIT_POINTS
            )
        ],
        dtype=np.float64,
    )

    latencies[
        ~feasible
    ] = np.inf

    action = int(
        np.argmin(
            latencies
        )
    )

    return action


# ============================================================
# 5. PPO
# ============================================================

def choose_ppo(
    model,
    profile_df,
    state_row,
    previous_action,
):

    state = build_state(
        profile_df,
        state_row,
        previous_action,
    )

    available_memory = float(
        state_row[
            "memory_mb"
        ]
    )

    memory_required = (
        profile_df[
            "memory_mb"
        ].to_numpy(
            dtype=np.float64
        )
    )

    action_mask = (
        memory_required
        <= available_memory
    )

    if not action_mask.any():

        raise RuntimeError(
            "No feasible PPO action."
        )

    (
        action,
        _,
        _,
        _,
    ) = model.choose_action(
        state,
        deterministic=True,
        action_mask=action_mask,
    )

    return int(
        action
    )


# ============================================================
# 6. 单方法、单seed系统仿真
# ============================================================

def simulate_method(
    method,
    seed,
    profile_df,
    accuracy_df,
    ppo_model,
):

    accuracy_seed = (
        accuracy_df[
            accuracy_df["seed"]
            == seed
        ]
        .sort_values(
            "successful_updates"
        )
        .reset_index(
            drop=True
        )
    )

    if len(accuracy_seed) == 0:

        raise RuntimeError(
            f"No accuracy history for seed {seed}"
        )

    total_updates = int(
        accuracy_seed[
            "successful_updates"
        ].max()
    )

    trace = generate_dynamic_trace(
        profile_df,
        length=total_updates,
        seed=500000 + seed,
    )

    profile_memory = (
        profile_df[
            "memory_mb"
        ].to_numpy(
            dtype=np.float64
        )
    )

    safe_action = int(
        np.argmin(
            profile_memory
        )
    )

    previous_action = (
        safe_action
    )

    successful_updates = 0

    elapsed_system_ms = 0.0

    oom_count = 0

    switch_count = 0

    low_attempts = 0

    low_successes = 0

    action_counts = np.zeros(
        cfg.NUM_SPLIT_POINTS,
        dtype=np.int64,
    )

    accuracy_records = []

    checkpoint_index = 0

    while (
        successful_updates
        < total_updates
    ):

        state_row = trace.iloc[
            successful_updates
        ]

        available_memory = float(
            state_row[
                "memory_mb"
            ]
        )

        feasible_count = int(
            (
                profile_memory
                <= available_memory
            ).sum()
        )

        is_low_memory = (
            feasible_count
            == 1
        )

        if is_low_memory:

            low_attempts += 1

        # ----------------------------------------------------
        # 在线决策
        # ----------------------------------------------------

        decision_start = (
            time.perf_counter_ns()
        )

        if (
            method
            == "Delay-only Search"
        ):

            action = (
                choose_delay_only(
                    profile_df,
                    state_row,
                )
            )

        elif (
            method
            == "Memory-Constrained Search"
        ):

            action = (
                choose_memory_constrained(
                    profile_df,
                    state_row,
                )
            )

        elif (
            method
            == "PPO"
        ):

            action = choose_ppo(
                ppo_model,
                profile_df,
                state_row,
                previous_action,
            )

        else:

            raise ValueError(
                f"Unknown method: {method}"
            )

        decision_ms = (
            time.perf_counter_ns()
            - decision_start
        ) / 1e6

        action_counts[
            action
        ] += 1

        selected_latency_ms = (
            calculate_latency_ms(
                profile_df,
                action,
                state_row,
            )
        )

        required_memory = float(
            profile_df.iloc[
                action
            ][
                "memory_mb"
            ]
        )

        feasible = (
            required_memory
            <= available_memory
        )

        switched = (
            action
            != previous_action
        )

        if switched:

            switch_count += 1

        switch_ms = (
            cfg.SWITCH_COST_MS
            if switched
            else 0.0
        )

        # ----------------------------------------------------
        # 正常执行
        # ----------------------------------------------------

        if feasible:

            elapsed_system_ms += (
                selected_latency_ms
                + decision_ms
                + switch_ms
            )

            if is_low_memory:

                low_successes += 1

            previous_action = (
                action
            )

        # ----------------------------------------------------
        # OOM + 恢复 + 安全切分点重算
        # ----------------------------------------------------

        else:

            oom_count += 1

            failed_compute_ms = (
                cfg.OOM_FAILED_RATIO
                * selected_latency_ms
            )

            safe_latency_ms = (
                calculate_latency_ms(
                    profile_df,
                    safe_action,
                    state_row,
                )
            )

            recovery_switch_ms = (
                cfg.SWITCH_COST_MS
                if previous_action
                != safe_action
                else 0.0
            )

            elapsed_system_ms += (
                decision_ms
                + failed_compute_ms
                + cfg.OOM_RECOVERY_MS
                + recovery_switch_ms
                + safe_latency_ms
            )

            previous_action = (
                safe_action
            )

        # OOM 后安全重算同样成功完成当前 SGD update。
        successful_updates += 1

        # ----------------------------------------------------
        # 到达真实训练曲线 checkpoint 时记录准确率
        # ----------------------------------------------------

        while (
            checkpoint_index
            < len(
                accuracy_seed
            )
            and successful_updates
            >= int(
                accuracy_seed.iloc[
                    checkpoint_index
                ][
                    "successful_updates"
                ]
            )
        ):

            row = (
                accuracy_seed.iloc[
                    checkpoint_index
                ]
            )

            accuracy_records.append(
                {
                    "method":
                        method,

                    "seed":
                        seed,

                    "epoch":
                        int(
                            row[
                                "epoch"
                            ]
                        ),

                    "successful_updates":
                        int(
                            row[
                                "successful_updates"
                            ]
                        ),

                    "test_accuracy":
                        float(
                            row[
                                "test_accuracy"
                            ]
                        ),

                    "elapsed_system_s":
                        (
                            elapsed_system_ms
                            / 1000.0
                        ),
                }
            )

            checkpoint_index += 1

    # --------------------------------------------------------
    # Time to Target
    # --------------------------------------------------------

    target_records = [
        row
        for row in accuracy_records
        if row[
            "test_accuracy"
        ]
        >= cfg.TARGET_ACCURACY
    ]

    if target_records:

        reached_target = True

        time_to_target_s = float(
            target_records[
                0
            ][
                "elapsed_system_s"
            ]
        )

    else:

        reached_target = False

        time_to_target_s = (
            np.nan
        )

    low_survival_rate = (
        100.0
        * low_successes
        / max(
            low_attempts,
            1,
        )
    )

    summary = {
        "method":
            method,

        "seed":
            seed,

        "reached_target":
            reached_target,

        "time_to_target_s":
            time_to_target_s,

        "final_system_time_s":
            (
                elapsed_system_ms
                / 1000.0
            ),

        "oom_count":
            int(
                oom_count
            ),

        "low_memory_attempts":
            int(
                low_attempts
            ),

        "low_memory_successes":
            int(
                low_successes
            ),

        "low_memory_survival_rate":
            float(
                low_survival_rate
            ),

        "switch_count":
            int(
                switch_count
            ),
    }

    action_rows = []

    for action in range(
        cfg.NUM_SPLIT_POINTS
    ):

        action_rows.append(
            {
                "method":
                    method,

                "seed":
                    seed,

                "action":
                    action,

                "count":
                    int(
                        action_counts[
                            action
                        ]
                    ),
            }
        )

    return (
        accuracy_records,
        summary,
        action_rows,
    )


# ============================================================
# 7. 决策时间
# ============================================================

def benchmark_decision_time(
    profile_df,
    ppo_model,
):

    # 减少 CPU 多线程波动。
    torch.set_num_threads(
        1
    )

    total_states = (
        cfg.DECISION_WARMUP
        + cfg.DECISION_RUNS
    )

    trace = generate_dynamic_trace(
        profile_df,
        length=total_states,
        seed=777777,
    )

    results = []

    for method in METHODS:

        times_us = []

        previous_action = 0

        for index in range(
            total_states
        ):

            state_row = (
                trace.iloc[
                    index
                ]
            )

            start = (
                time.perf_counter_ns()
            )

            if (
                method
                == "Delay-only Search"
            ):

                action = (
                    choose_delay_only(
                        profile_df,
                        state_row,
                    )
                )

            elif (
                method
                == "Memory-Constrained Search"
            ):

                action = (
                    choose_memory_constrained(
                        profile_df,
                        state_row,
                    )
                )

            elif (
                method
                == "PPO"
            ):

                action = (
                    choose_ppo(
                        ppo_model,
                        profile_df,
                        state_row,
                        previous_action,
                    )
                )

            else:

                raise ValueError(
                    method
                )

            elapsed_us = (
                time.perf_counter_ns()
                - start
            ) / 1000.0

            previous_action = (
                action
            )

            if (
                index
                >= cfg.DECISION_WARMUP
            ):

                times_us.append(
                    elapsed_us
                )

        times_us = np.asarray(
            times_us,
            dtype=np.float64,
        )

        results.append(
            {
                "method":
                    method,

                "median_us":
                    float(
                        np.median(
                            times_us
                        )
                    ),

                "p95_us":
                    float(
                        np.percentile(
                            times_us,
                            95,
                        )
                    ),
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    cfg.prepare_output_dir()

    print()
    print("=" * 70)
    print("Three-method comparison")
    print("=" * 70)

    print(
        f"Mode: {cfg.MODE_NAME}"
    )

    profile_df = (
        load_profile()
    )

    accuracy_df = (
        load_accuracy_history()
    )

    all_accuracy_records = []

    all_summary_rows = []

    all_action_rows = []

    first_ppo_model = None

    for seed in cfg.SEEDS:

        print()
        print(
            f"Seed = {seed}"
        )

        ppo_model = (
            load_ppo_model(
                seed
            )
        )

        if first_ppo_model is None:

            first_ppo_model = (
                ppo_model
            )

        for method in METHODS:

            print(
                f"Running {method}"
            )

            (
                accuracy_records,
                summary,
                action_rows,
            ) = simulate_method(
                method=method,
                seed=seed,
                profile_df=profile_df,
                accuracy_df=accuracy_df,
                ppo_model=ppo_model,
            )

            all_accuracy_records.extend(
                accuracy_records
            )

            all_summary_rows.append(
                summary
            )

            all_action_rows.extend(
                action_rows
            )

            print(
                f"  final_time="
                f"{summary['final_system_time_s']:.3f} s | "
                f"OOM={summary['oom_count']} | "
                f"low_survival="
                f"{summary['low_memory_survival_rate']:.2f}%"
            )

    time_accuracy_df = pd.DataFrame(
        all_accuracy_records
    )

    summary_df = pd.DataFrame(
        all_summary_rows
    )

    action_df = pd.DataFrame(
        all_action_rows
    )

    decision_df = (
        benchmark_decision_time(
            profile_df,
            first_ppo_model,
        )
    )

    time_accuracy_path = (
        cfg.OUTPUT_DIR
        / f"time_accuracy_{cfg.MODE_NAME}.csv"
    )

    summary_path = (
        cfg.OUTPUT_DIR
        / f"comparison_summary_{cfg.MODE_NAME}.csv"
    )

    action_path = (
        cfg.OUTPUT_DIR
        / f"action_counts_{cfg.MODE_NAME}.csv"
    )

    decision_path = (
        cfg.OUTPUT_DIR
        / f"decision_time_{cfg.MODE_NAME}.csv"
    )

    time_accuracy_df.to_csv(
        time_accuracy_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    action_df.to_csv(
        action_path,
        index=False,
        encoding="utf-8-sig",
    )

    decision_df.to_csv(
        decision_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("Comparison finished")
    print("=" * 70)

    print()
    print("Summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("Decision time:")
    print(
        decision_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {time_accuracy_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    print(
        f"Saved: {action_path}"
    )

    print(
        f"Saved: {decision_path}"
    )


if __name__ == "__main__":
    main()
