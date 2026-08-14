import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

import config as cfg


# ============================================================
# 1. PPO 固定在 CPU
# ============================================================

# PPO 网络本身很小。
# 后续在线决策时间也需要在 CPU 上测试，
# 因此这里直接使用 CPU，避免 GPU 调度时间干扰。
PPO_DEVICE = torch.device("cpu")


# ============================================================
# 2. 读取离线 Profile
# ============================================================

def load_profile():
    profile_path = (
        cfg.OUTPUT_DIR
        / f"profile_{cfg.MODE_NAME}.csv"
    )

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile file does not exist: "
            f"{profile_path}"
        )

    profile_df = pd.read_csv(
        profile_path
    )

    required_columns = {
        "action",
        "device_ms",
        "server_ms",
        "feature_bytes",
        "memory_mb",
    }

    missing = (
        required_columns
        - set(profile_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Profile missing columns: {missing}"
        )

    if len(profile_df) != cfg.NUM_SPLIT_POINTS:
        raise RuntimeError(
            "Profile does not contain 18 split points."
        )

    return profile_df


# ============================================================
# 3. 分段动态轨迹生成
# ============================================================

def make_segment_trace(
    length,
    rng,
    values,
    min_length,
    max_length,
    probabilities=None,
):
    result = []

    while len(result) < length:

        segment_length = int(
            rng.integers(
                min_length,
                max_length + 1,
            )
        )

        selected = rng.choice(
            values,
            p=probabilities,
        )

        result.extend(
            [float(selected)]
            * segment_length
        )

    return np.asarray(
        result[:length],
        dtype=np.float32,
    )


def generate_dynamic_trace(
    profile_df,
    length,
    seed,
):
    """
    当前 EI 会议版先使用可重复的合成动态环境。

    动态状态包括：

    bandwidth_mbps
    memory_mb
    alpha
    rtt_ms

    显存按照：
        30% high
        40% medium
        30% low

    进行分段变化。

    后续如果获得 Lumos5G 原始轨迹，
    再由 ChatGPT 决定是否替换 bandwidth 部分。
    """

    rng = np.random.default_rng(
        seed
    )

    memory_profile = (
        profile_df["memory_mb"]
        .to_numpy(
            dtype=np.float64
        )
    )

    memory_min = float(
        np.min(memory_profile)
    )

    memory_max = float(
        np.max(memory_profile)
    )

    memory_median = float(
        np.median(memory_profile)
    )

    memory_high = (
        1.10
        * memory_max
    )

    memory_medium = (
        memory_median
    )

    memory_low = (
        1.05
        * memory_min
    )

    memory_mb = make_segment_trace(
        length=length,
        rng=rng,
        values=[
            memory_high,
            memory_medium,
            memory_low,
        ],
        min_length=20,
        max_length=50,
        probabilities=[
            0.30,
            0.40,
            0.30,
        ],
    )

    # 加入很小的随机扰动。
    memory_mb = (
        memory_mb
        * rng.uniform(
            0.98,
            1.02,
            size=length,
        )
    )

    # 必须保证至少存在一个可执行动作。
    memory_mb = np.maximum(
        memory_mb,
        memory_min * 1.01,
    )

    bandwidth_mbps = make_segment_trace(
        length=length,
        rng=rng,
        values=[
            12.0,
            25.0,
            60.0,
            120.0,
        ],
        min_length=15,
        max_length=45,
        probabilities=[
            0.20,
            0.30,
            0.30,
            0.20,
        ],
    )

    bandwidth_mbps = (
        bandwidth_mbps
        * rng.uniform(
            0.90,
            1.10,
            size=length,
        )
    )

    bandwidth_mbps = np.clip(
        bandwidth_mbps,
        5.0,
        None,
    )

    alpha = make_segment_trace(
        length=length,
        rng=rng,
        values=[
            0.80,
            1.00,
            1.25,
            1.45,
        ],
        min_length=20,
        max_length=60,
        probabilities=[
            0.20,
            0.35,
            0.30,
            0.15,
        ],
    )

    alpha = (
        alpha
        * rng.uniform(
            0.97,
            1.03,
            size=length,
        )
    )

    rtt_ms = make_segment_trace(
        length=length,
        rng=rng,
        values=[
            5.0,
            8.0,
            12.0,
            18.0,
        ],
        min_length=15,
        max_length=50,
        probabilities=[
            0.25,
            0.35,
            0.25,
            0.15,
        ],
    )

    rtt_ms = (
        rtt_ms
        + rng.normal(
            0.0,
            0.5,
            size=length,
        )
    )

    rtt_ms = np.clip(
        rtt_ms,
        1.0,
        None,
    )

    trace_df = pd.DataFrame(
        {
            "bandwidth_mbps":
                bandwidth_mbps,

            "memory_mb":
                memory_mb,

            "alpha":
                alpha,

            "rtt_ms":
                rtt_ms,
        }
    )

    return trace_df


# ============================================================
# 4. 时延模型
# ============================================================

def calculate_latency_ms(
    profile_df,
    action,
    state_row,
):
    """
    T(c) =

        alpha * D_dev[c]
        + D_srv[c]
        + upload communication
        + download communication
        + RTT

    EI 会议版采用：

        B_up = B_down

    并采用：

        S_grad = S_out
    """

    action = int(action)

    profile_row = profile_df.iloc[
        action
    ]

    bandwidth_mbps = max(
        float(
            state_row[
                "bandwidth_mbps"
            ]
        ),
        1e-6,
    )

    feature_bytes = float(
        profile_row[
            "feature_bytes"
        ]
    )

    # Mbps -> Byte/s
    bandwidth_bytes_per_second = (
        bandwidth_mbps
        * 1e6
        / 8.0
    )

    one_way_communication_ms = (
        feature_bytes
        / bandwidth_bytes_per_second
        * 1000.0
    )

    total_latency_ms = (
        float(
            state_row["alpha"]
        )
        * float(
            profile_row["device_ms"]
        )
        + float(
            profile_row["server_ms"]
        )
        + 2.0
        * one_way_communication_ms
        + float(
            state_row["rtt_ms"]
        )
    )

    return float(
        total_latency_ms
    )


# ============================================================
# 5. PPO 状态编码
# ============================================================

def build_state(
    profile_df,
    state_row,
    previous_action,
    state_mode="full",
    static_values=None,
):
    """
    State:

        normalized bandwidth
        normalized available memory
        normalized alpha
        one-hot(previous action)

    总状态维度：

        3 + 18 = 21
    """

    bandwidth = float(
        state_row[
            "bandwidth_mbps"
        ]
    )

    memory = float(
        state_row[
            "memory_mb"
        ]
    )

    alpha = float(
        state_row[
            "alpha"
        ]
    )

    if state_mode == "static_resource":
        if static_values is None:
            static_values = {
                "bandwidth_mbps": 67.5,
                "memory_mb": float(
                    profile_df["memory_mb"].median()
                ),
                "alpha": 1.0,
            }

        bandwidth = float(
            static_values["bandwidth_mbps"]
        )
        memory = float(
            static_values["memory_mb"]
        )
        alpha = float(
            static_values["alpha"]
        )
    elif state_mode != "full":
        raise ValueError(
            f"Unknown state mode: {state_mode}"
        )

    profile_memory = (
        profile_df["memory_mb"]
        .to_numpy(
            dtype=np.float64
        )
    )

    memory_min = float(
        np.min(profile_memory)
    )

    memory_high = (
        1.10
        * float(
            np.max(profile_memory)
        )
    )

    bandwidth_norm = (
        bandwidth - 5.0
    ) / (
        130.0 - 5.0
    )

    memory_norm = (
        memory - memory_min
    ) / max(
        memory_high - memory_min,
        1e-8,
    )

    alpha_norm = (
        alpha - 0.75
    ) / (
        1.55 - 0.75
    )

    bandwidth_norm = float(
        np.clip(
            bandwidth_norm,
            0.0,
            1.0,
        )
    )

    memory_norm = float(
        np.clip(
            memory_norm,
            0.0,
            1.0,
        )
    )

    alpha_norm = float(
        np.clip(
            alpha_norm,
            0.0,
            1.0,
        )
    )

    previous_action_onehot = np.zeros(
        cfg.NUM_SPLIT_POINTS,
        dtype=np.float32,
    )

    previous_action_onehot[
        int(previous_action)
    ] = 1.0

    state = np.concatenate(
        [
            np.asarray(
                [
                    bandwidth_norm,
                    memory_norm,
                    alpha_norm,
                ],
                dtype=np.float32,
            ),
            previous_action_onehot,
        ]
    )

    return state.astype(
        np.float32
    )


# ============================================================
# 6. 动态切分环境
# ============================================================

class DynamicSplitEnvironment:

    def __init__(
        self,
        profile_df,
        trace_df,
        use_action_mask=True,
        state_mode="full",
        reward_mode="regret",
    ):
        self.profile_df = (
            profile_df.reset_index(
                drop=True
            )
        )

        self.trace_df = (
            trace_df.reset_index(
                drop=True
            )
        )

        self.index = 0

        self.previous_action = 0

        self.use_action_mask = bool(
            use_action_mask
        )

        self.state_mode = state_mode

        self.reward_mode = reward_mode

        self.static_state_values = {
            "bandwidth_mbps": float(
                self.trace_df["bandwidth_mbps"].median()
            ),
            "memory_mb": float(
                self.trace_df["memory_mb"].median()
            ),
            "alpha": float(
                self.trace_df["alpha"].median()
            ),
        }

        # ----------------------------------------------------
        # 参考时延
        # ----------------------------------------------------

        reference_state = {
            "bandwidth_mbps":
                float(
                    self.trace_df[
                        "bandwidth_mbps"
                    ].median()
                ),

            "memory_mb":
                float(
                    self.trace_df[
                        "memory_mb"
                    ].median()
                ),

            "alpha":
                float(
                    self.trace_df[
                        "alpha"
                    ].median()
                ),

            "rtt_ms":
                float(
                    self.trace_df[
                        "rtt_ms"
                    ].median()
                ),
        }

        reference_latency = []

        for action in range(
            cfg.NUM_SPLIT_POINTS
        ):

            reference_latency.append(
                calculate_latency_ms(
                    self.profile_df,
                    action,
                    reference_state,
                )
            )

        self.reference_latency_ms = max(
            float(
                np.median(
                    reference_latency
                )
            ),
            1e-6,
        )

    def get_action_mask(self):

        if not self.use_action_mask:

            return np.ones(
                cfg.NUM_SPLIT_POINTS,
                dtype=bool,
            )

        row = self.trace_df.iloc[
            self.index
        ]

        available_memory = float(
            row[
                "memory_mb"
            ]
        )

        memory_required = (
            self.profile_df[
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
                "No feasible action in current state."
            )

        return action_mask

    def reset(self):

        self.index = 0

        self.previous_action = 0

        row = self.trace_df.iloc[
            self.index
        ]

        return build_state(
            self.profile_df,
            row,
            self.previous_action,
            state_mode=self.state_mode,
            static_values=self.static_state_values,
        )

    def step(
        self,
        action,
    ):

        action = int(action)

        row = self.trace_df.iloc[
            self.index
        ]

        required_memory = float(
            self.profile_df.iloc[
                action
            ][
                "memory_mb"
            ]
        )

        available_memory = float(
            row[
                "memory_mb"
            ]
        )

        oom = (
            required_memory
            > available_memory
        )

        latency_ms = (
            calculate_latency_ms(
                self.profile_df,
                action,
                row,
            )
        )

        if oom:

            # OOM 始终比任何正常可执行动作更差。
            reward = -2.0

        else:

            # ------------------------------------------------
            # 找出当前状态下所有显存可行的动作
            # ------------------------------------------------

            available_memory = float(
                row[
                    "memory_mb"
                ]
            )

            feasible_mask = (
                self.profile_df[
                    "memory_mb"
                ].to_numpy(
                    dtype=np.float64
                )
                <= available_memory
            )

            feasible_latencies = []

            for candidate_action in range(
                cfg.NUM_SPLIT_POINTS
            ):

                if feasible_mask[
                    candidate_action
                ]:

                    candidate_latency = (
                        calculate_latency_ms(
                            self.profile_df,
                            candidate_action,
                            row,
                        )
                    )

                    feasible_latencies.append(
                        candidate_latency
                    )

            if not feasible_latencies:

                raise RuntimeError(
                    "No feasible split action."
                )

            best_feasible_latency = float(
                min(
                    feasible_latencies
                )
            )

            # ------------------------------------------------
            # Relative regret
            #
            # 最优可行动作：
            # regret = 0
            # reward = +1
            #
            # 当前动作比最优慢50%：
            # regret = 0.5
            # reward = 0
            #
            # 当前动作达到最优动作2倍或更慢：
            # reward = -1
            #
            # OOM：
            # reward = -2
            # ------------------------------------------------

            relative_regret = (
                latency_ms
                - best_feasible_latency
            ) / max(
                best_feasible_latency,
                1e-8,
            )

            relative_regret = float(
                np.clip(
                    relative_regret,
                    0.0,
                    1.0,
                )
            )

            if self.reward_mode == "regret":

                reward = (
                    1.0
                    - 2.0
                    * relative_regret
                )

            elif self.reward_mode == "absolute":

                reward = (
                    1.0
                    - 2.0
                    * float(
                        np.clip(
                            latency_ms
                            / self.reference_latency_ms,
                            0.0,
                            1.0,
                        )
                    )
                )

            else:

                raise ValueError(
                    f"Unknown reward mode: {self.reward_mode}"
                )

            switched = (
                action
                != self.previous_action
            )

            if switched:

                reward -= (
                    cfg.SWITCH_PENALTY
                )

        self.previous_action = (
            action
        )

        self.index += 1

        done = (
            self.index
            >= len(
                self.trace_df
            )
        )

        if done:

            next_state = None

        else:

            next_row = (
                self.trace_df.iloc[
                    self.index
                ]
            )

            next_state = build_state(
                self.profile_df,
                next_row,
                self.previous_action,
                state_mode=self.state_mode,
                static_values=self.static_state_values,
            )

        info = {
            "oom":
                bool(oom),

            "latency_ms":
                float(
                    latency_ms
                ),

            "required_memory_mb":
                required_memory,

            "available_memory_mb":
                available_memory,
        }

        return (
            next_state,
            float(reward),
            done,
            info,
        )


# ============================================================
# 7. PPO-MLP Actor-Critic
# ============================================================

class PPOPolicy(nn.Module):

    def __init__(
        self,
        state_dimension,
        action_dimension,
    ):
        super().__init__()

        hidden_size = (
            cfg.GRU_HIDDEN_SIZE
        )

        self.feature = nn.Sequential(
            nn.Linear(
                state_dimension,
                hidden_size,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),
        )

        self.actor = nn.Linear(
            hidden_size,
            action_dimension,
        )

        self.critic = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x,
    ):

        feature = self.feature(
            x
        )

        logits = self.actor(
            feature
        )

        value = self.critic(
            feature
        ).squeeze(
            -1
        )

        return (
            logits,
            value,
        )

    @torch.no_grad()
    def choose_action(
        self,
        state,
        hidden=None,
        deterministic=False,
        action_mask=None,
    ):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=PPO_DEVICE,
        ).view(
            1,
            -1,
        )

        logits, value = self.forward(
            state_tensor
        )

        current_logits = logits[
            0
        ]

        if action_mask is not None:

            mask_tensor = torch.tensor(
                action_mask,
                dtype=torch.bool,
                device=PPO_DEVICE,
            )

            if not mask_tensor.any():

                raise RuntimeError(
                    "Action mask contains no feasible action."
                )

            current_logits = (
                current_logits.masked_fill(
                    ~mask_tensor,
                    -1e9,
                )
            )

        distribution = Categorical(
            logits=current_logits
        )

        if deterministic:

            action = torch.argmax(
                current_logits
            )

        else:

            action = (
                distribution.sample()
            )

        log_probability = (
            distribution.log_prob(
                action
            )
        )

        return (
            int(
                action.item()
            ),
            float(
                log_probability.item()
            ),
            float(
                value[
                    0
                ].item()
            ),
            None,
        )


# ============================================================
# 8. GAE
# ============================================================

def calculate_gae(
    rewards,
    values,
    dones,
):
    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    dones = np.asarray(
        dones,
        dtype=np.float32,
    )

    advantages = np.zeros_like(
        rewards
    )

    gae = 0.0

    for t in reversed(
        range(
            len(rewards)
        )
    ):

        if t == len(rewards) - 1:

            next_value = 0.0

        else:

            next_value = (
                values[
                    t + 1
                ]
            )

        non_terminal = (
            1.0
            - dones[t]
        )

        delta = (
            rewards[t]
            + cfg.GAMMA
            * next_value
            * non_terminal
            - values[t]
        )

        gae = (
            delta
            + cfg.GAMMA
            * cfg.GAE_LAMBDA
            * non_terminal
            * gae
        )

        advantages[t] = (
            gae
        )

    returns = (
        advantages
        + values
    )

    return (
        advantages,
        returns,
    )


# ============================================================
# 9. PPO 参数更新
# ============================================================

def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_log_probabilities,
    action_masks,
    advantages,
    returns,
):

    states_tensor = torch.tensor(
        np.asarray(
            states,
            dtype=np.float32,
        ),
        dtype=torch.float32,
        device=PPO_DEVICE,
    )

    actions_tensor = torch.tensor(
        actions,
        dtype=torch.long,
        device=PPO_DEVICE,
    )

    action_masks_tensor = torch.tensor(
        np.asarray(
            action_masks,
            dtype=np.bool_,
        ),
        dtype=torch.bool,
        device=PPO_DEVICE,
    )

    old_log_probability_tensor = (
        torch.tensor(
            old_log_probabilities,
            dtype=torch.float32,
            device=PPO_DEVICE,
        )
    )

    advantage_tensor = torch.tensor(
        advantages,
        dtype=torch.float32,
        device=PPO_DEVICE,
    )

    return_tensor = torch.tensor(
        returns,
        dtype=torch.float32,
        device=PPO_DEVICE,
    )

    advantage_tensor = (
        advantage_tensor
        - advantage_tensor.mean()
    ) / (
        advantage_tensor.std()
        + 1e-8
    )

    for _ in range(
        cfg.PPO_UPDATE_EPOCHS
    ):

        logits, values = model(
            states_tensor
        )

        logits = logits.masked_fill(
            ~action_masks_tensor,
            -1e9,
        )

        distribution = Categorical(
            logits=logits
        )

        new_log_probability = (
            distribution.log_prob(
                actions_tensor
            )
        )

        entropy = (
            distribution.entropy()
            .mean()
        )

        probability_ratio = torch.exp(
            new_log_probability
            - old_log_probability_tensor
        )

        surrogate_1 = (
            probability_ratio
            * advantage_tensor
        )

        surrogate_2 = (
            torch.clamp(
                probability_ratio,
                1.0 - cfg.PPO_CLIP,
                1.0 + cfg.PPO_CLIP,
            )
            * advantage_tensor
        )

        actor_loss = (
            - torch.min(
                surrogate_1,
                surrogate_2,
            ).mean()
        )

        critic_loss = (
            0.5
            * (
                return_tensor
                - values
            ).pow(
                2
            ).mean()
        )

        total_loss = (
            actor_loss
            + critic_loss
            - 0.01
            * entropy
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.5,
        )

        optimizer.step()


# ============================================================
# 10. 固定轨迹验证
# ============================================================

@torch.no_grad()
def evaluate_policy(
    model,
    profile_df,
    trace_df,
    use_action_mask=True,
    state_mode="full",
    reward_mode="regret",
):

    environment = (
        DynamicSplitEnvironment(
            profile_df,
            trace_df,
            use_action_mask=use_action_mask,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )
    )

    state = environment.reset()

    hidden = None

    total_reward = 0.0

    oom_count = 0

    action_counts = np.zeros(
        cfg.NUM_SPLIT_POINTS,
        dtype=np.int64,
    )

    while True:

        action_mask = (
            environment.get_action_mask()
        )

        action, _, _, hidden = (
            model.choose_action(
                state,
                hidden=hidden,
                deterministic=True,
                action_mask=action_mask,
            )
        )

        action_counts[
            action
        ] += 1

        next_state, reward, done, info = (
            environment.step(
                action
            )
        )

        total_reward += reward

        if info["oom"]:
            oom_count += 1

        if done:
            break

        state = next_state

    return (
        float(total_reward),
        int(oom_count),
        action_counts,
    )


# ============================================================
# 11. 单随机种子训练
# ============================================================

def train_one_seed(
    seed,
    profile_df,
    use_action_mask=True,
    state_mode="full",
    reward_mode="regret",
    output_dir=None,
    run_name=None,
):

    cfg.set_seed(
        seed
    )

    state_dimension = (
        3
        + cfg.NUM_SPLIT_POINTS
    )

    action_dimension = (
        cfg.NUM_SPLIT_POINTS
    )

    model = PPOPolicy(
        state_dimension,
        action_dimension,
    ).to(
        PPO_DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.PPO_LEARNING_RATE,
    )

    # 每个随机种子使用一条固定验证轨迹。
    validation_trace = (
        generate_dynamic_trace(
            profile_df,
            length=cfg.PPO_TRACE_LENGTH,
            seed=900000 + seed,
        )
    )

    best_validation_reward = (
        - float("inf")
    )

    best_validation_oom = None

    best_model_state = None

    training_history = []

    for episode in range(
        1,
        cfg.PPO_EPISODES + 1,
    ):

        training_trace = (
            generate_dynamic_trace(
                profile_df,
                length=cfg.PPO_TRACE_LENGTH,
                seed=(
                    seed * 10000
                    + episode
                ),
            )
        )

        environment = (
        DynamicSplitEnvironment(
            profile_df,
            training_trace,
            use_action_mask=use_action_mask,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )
        )

        state = (
            environment.reset()
        )

        hidden = None

        states = []

        actions = []

        old_log_probabilities = []

        action_masks = []

        values = []

        rewards = []

        dones = []

        training_oom = 0

        while True:

            action_mask = (
                environment.get_action_mask()
            )

            (
                action,
                log_probability,
                value,
                hidden,
            ) = model.choose_action(
                state,
                hidden=hidden,
                deterministic=False,
                action_mask=action_mask,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(
                action
            )

            states.append(
                state
            )

            actions.append(
                action
            )

            old_log_probabilities.append(
                log_probability
            )

            action_masks.append(
                action_mask.copy()
            )

            values.append(
                value
            )

            rewards.append(
                reward
            )

            dones.append(
                float(done)
            )

            if info["oom"]:
                training_oom += 1

            if done:
                break

            state = next_state

        (
            advantages,
            returns,
        ) = calculate_gae(
            rewards,
            values,
            dones,
        )

        ppo_update(
            model=model,
            optimizer=optimizer,
            states=states,
            actions=actions,
            old_log_probabilities=old_log_probabilities,
            action_masks=action_masks,
            advantages=advantages,
            returns=returns,
        )

        (
            validation_reward,
            validation_oom,
            validation_action_counts,
        ) = evaluate_policy(
            model,
            profile_df,
            validation_trace,
            use_action_mask=use_action_mask,
            state_mode=state_mode,
            reward_mode=reward_mode,
        )

        if (
            validation_reward
            > best_validation_reward
        ):

            best_validation_reward = (
                validation_reward
            )

            best_validation_oom = (
                validation_oom
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        training_history.append(
            {
                "seed":
                    seed,

                "episode":
                    episode,

                "train_reward":
                    float(
                        np.sum(
                            rewards
                        )
                    ),

                "train_oom":
                    int(
                        training_oom
                    ),

                "validation_reward":
                    float(
                        validation_reward
                    ),

                "validation_oom":
                    int(
                        validation_oom
                    ),
            }
        )

        print(
            f"seed={seed} | "
            f"episode={episode:03d}/{cfg.PPO_EPISODES} | "
            f"train_reward={np.sum(rewards):.2f} | "
            f"train_oom={training_oom} | "
            f"val_reward={validation_reward:.2f} | "
            f"val_oom={validation_oom}"
        )

    if best_model_state is None:
        raise RuntimeError(
            "No PPO model state was saved."
        )

    if output_dir is None:
        output_dir = cfg.OUTPUT_DIR

    output_dir = output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if run_name is None:
        model_filename = (
            f"ppo_seed_{seed}_"
            f"{cfg.MODE_NAME}.pt"
        )
    else:
        model_filename = (
            f"ppo_seed_{seed}_"
            f"{cfg.MODE_NAME}_{run_name}.pt"
        )

    seed_model_path = output_dir / model_filename

    torch.save(
        best_model_state,
        seed_model_path,
    )

    # 重新加载最佳模型进行一次最终验证。
    model.load_state_dict(
        best_model_state
    )

    (
        final_reward,
        final_oom,
        final_action_counts,
    ) = evaluate_policy(
        model,
        profile_df,
        validation_trace,
        use_action_mask=use_action_mask,
        state_mode=state_mode,
        reward_mode=reward_mode,
    )

    return {
        "seed":
            seed,

        "model_path":
            str(
                seed_model_path
            ),

        "best_validation_reward":
            float(
                best_validation_reward
            ),

        "best_validation_oom":
            int(
                best_validation_oom
            ),

        "final_validation_reward":
            float(
                final_reward
            ),

        "final_validation_oom":
            int(
                final_oom
            ),

        "final_action_counts":
            final_action_counts,

        "history":
            training_history,

        "model_state":
            best_model_state,
    }


# ============================================================
# 12. 主程序
# ============================================================

def main():

    cfg.prepare_output_dir()

    print()
    print("=" * 60)
    print("PPO training")
    print("=" * 60)

    print(
        f"Experiment mode : "
        f"{cfg.MODE_NAME}"
    )

    print(
        f"PPO device      : "
        f"{PPO_DEVICE}"
    )

    print(
        f"Episodes        : "
        f"{cfg.PPO_EPISODES}"
    )

    print(
        f"Trace length    : "
        f"{cfg.PPO_TRACE_LENGTH}"
    )

    print(
        f"MLP hidden size : "
        f"{cfg.GRU_HIDDEN_SIZE}"
    )

    profile_df = (
        load_profile()
    )

    all_history = []

    seed_results = []

    global_best_reward = (
        - float("inf")
    )

    global_best_state = None

    global_best_seed = None

    for seed in cfg.SEEDS:

        print()
        print("-" * 60)

        print(
            f"Training seed {seed}"
        )

        print("-" * 60)

        result = train_one_seed(
            seed,
            profile_df,
        )

        all_history.extend(
            result["history"]
        )

        seed_results.append(
            result
        )

        if (
            result[
                "best_validation_reward"
            ]
            > global_best_reward
        ):

            global_best_reward = (
                result[
                    "best_validation_reward"
                ]
            )

            global_best_state = copy.deepcopy(
                result[
                    "model_state"
                ]
            )

            global_best_seed = (
                seed
            )

        print()
        print(
            f"Seed {seed} best validation reward: "
            f"{result['best_validation_reward']:.3f}"
        )

        print(
            f"Seed {seed} best validation OOM: "
            f"{result['best_validation_oom']}"
        )

        print(
            "Best-model action counts:"
        )

        print(
            result[
                "final_action_counts"
            ]
        )

    # --------------------------------------------------------
    # 保存所有训练历史
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        all_history
    )

    history_path = (
        cfg.OUTPUT_DIR
        / f"ppo_training_{cfg.MODE_NAME}.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 保存所有 seed 的摘要
    # --------------------------------------------------------

    summary_rows = []

    for result in seed_results:

        summary_rows.append(
            {
                "seed":
                    result[
                        "seed"
                    ],

                "best_validation_reward":
                    result[
                        "best_validation_reward"
                    ],

                "best_validation_oom":
                    result[
                        "best_validation_oom"
                    ],

                "final_validation_reward":
                    result[
                        "final_validation_reward"
                    ],

                "final_validation_oom":
                    result[
                        "final_validation_oom"
                    ],

                "model_path":
                    result[
                        "model_path"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        cfg.OUTPUT_DIR
        / f"ppo_summary_{cfg.MODE_NAME}.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 保存当前模式下验证 Reward 最好的模型
    # --------------------------------------------------------

    if global_best_state is None:
        raise RuntimeError(
            "Global best PPO state is empty."
        )

    best_path = (
        cfg.OUTPUT_DIR
        / f"ppo_best_{cfg.MODE_NAME}.pt"
    )

    torch.save(
        global_best_state,
        best_path,
    )

    print()
    print("=" * 60)
    print("PPO training finished")
    print("=" * 60)

    print(
        f"Training history:"
    )

    print(
        history_path
    )

    print(
        f"PPO summary:"
    )

    print(
        summary_path
    )

    print(
        f"Best PPO model:"
    )

    print(
        best_path
    )

    print(
        f"Global best seed: "
        f"{global_best_seed}"
    )

    print(
        f"Global best validation reward: "
        f"{global_best_reward:.3f}"
    )


if __name__ == "__main__":
    main()
