# 实验设计文档

本文档解释本项目的完整实验设计：问题定义、方法建模、算法设计和实验方案。

---

## 1. 问题定义

### 1.1 背景

在边缘-云协同推理中，深度学习模型被切分为两部分：前半部分在设备端执行，后半部分卸载到服务器。切分点的选择直接影响：

- **端侧计算时延**（device_ms）：切分越深，设备端执行的前缀越长
- **服务器侧计算时延**（server_ms）：切分越深，服务器端剩余层数越少
- **中间特征通信时延**：取决于切分点输出的特征张量大小和当前网络带宽
- **端侧显存占用**：切分越深，设备端需同时驻留的参数和中间激活越多

### 1.2 核心挑战

切分点的最优选择**取决于时变的环境条件**：

- 网络带宽波动（如 5G/LTE 信号变化）
- 设备端可用显存波动（其他应用抢占）
- 设备端计算负载变化（α 系数）

**仅考虑时延而忽略显存的切分方案会在低显存时导致 OOM（Out-of-Memory）**，触发昂贵的故障恢复。

### 1.3 问题形式化

给定：
- $K = 18$ 个候选切分点 $\{0, 1, \dots, 17\}$
- 每个切分点 $c$ 的 Profile：`device_ms[c]`, `server_ms[c]`, `feature_bytes[c]`, `memory_mb[c]`
- 时变环境状态 $s_t$：$bandwidth_t$, $available\_memory_t$, $\alpha_t$, $rtt_t$

在每个时间步 $t$ 选择切分动作 $a_t \in \{0,\dots,17\}$，使得：

$$\text{minimize} \sum_t T(a_t, s_t)$$

满足显存约束：

$$memory\_mb[a_t] \leq available\_memory_t, \quad \forall t$$

其中 $T(a_t, s_t)$ 为总训练时延：

$$T = \alpha \cdot \text{device\_ms}[a] + \text{server\_ms}[a] + 2 \cdot \frac{\text{feature\_bytes}[a]}{\text{bandwidth}/8} + \text{RTT}$$

---

## 2. 实验模型：CIFAR-10 + ResNet50

### 2.1 为什么选 ResNet50

ResNet50 包含 16 个 Bottleneck block，加上开头 stem 和末尾 avgpool，恰好产生 18 个自然的切分候选点。不同切分点在计算量、中间特征大小和显存占用上有显著差异（见下表），构成有意义的决策空间。

### 2.2 18 个切分点

| action | 位置 | 特征 |
|--------|------|------|
| 0 | stem 后 | 最深 server，最浅 device，最大特征 (8MB)，最低显存 (147MB) |
| 1-3 | layer1 Bottleneck 后 | 特征 32MB，device 逐渐加重 |
| 4-7 | layer2 Bottleneck 后 | 特征 16MB，显存 597-730MB |
| 8-13 | layer3 Bottleneck 后 | 特征 8MB，显存 790-905MB |
| 14-16 | layer4 Bottleneck 后 | 特征 4MB，显存 961-1035MB |
| 17 | avgpool 后 | 最浅 server，最深 device，最小特征 (256KB)，最大显存 (1035MB) |

### 2.3 ResNet50 的 CIFAR-10 适配

CIFAR-10 输入为 32×32，而标准 ResNet50 为 224×224 设计。所做的调整：
- `conv1` 改为 3×3 kernel, stride=1, padding=1（替代 7×7, stride=2）
- 去掉 `maxpool`（替换为 `Identity`）
- `fc` 输出改为 10 类

---

## 3. 动态环境建模

### 3.1 四维环境状态

实验模拟的时变环境包括：

| 状态维度 | 范围 | 含义 |
|----------|------|------|
| bandwidth_mbps | 5-130 | 端-云通信带宽 |
| memory_mb | 151-1162 | 设备端当前可用显存 |
| alpha | 0.75-1.55 | 设备端计算速度倍率（>1 表示降频/负载）|
| rtt_ms | 1-20 | 往返时延 |

### 3.2 分段平稳轨迹

每个维度采用分段常数 + 随机扰动的方式生成，以模拟真实无线环境的准平稳特性：

- **memory**：三档（HIGH: 1139MB, MEDIUM: 794MB, LOW: 154MB），按 30%/40%/30% 比例分段，段长 20-50 步
- **bandwidth**：四档（12/25/60/120 Mbps），按 20%/30%/30%/20% 比例分段，段长 15-45 步
- **alpha**：四档（0.80/1.00/1.25/1.45），按 20%/35%/30%/15% 比例分段，段长 20-60 步
- **rtt**：四档（5/8/12/18 ms），按 25%/35%/25%/15% 比例分段，段长 15-50 步

各段内加入小幅随机扰动（±2-10%），确保轨迹不可简单记忆。

### 3.3 为什么用合成轨迹而非真实数据

- 实验设计阶段缺乏可直接使用的 5G/LTE 实测数据（如 Lumos5G）
- 合成轨迹提供了可控的分布覆盖（三种显存档位、四种带宽档位）
- 通过固定随机种子保证完全可复现
- 架构上预留了替换为真实轨迹的接口（`generate_dynamic_trace` 函数）

---

## 4. PPO 设计

### 4.1 状态空间

21 维向量：

| 维度 | 内容 |
|------|------|
| 0 | bandwidth 归一化 (5-130 → 0-1) |
| 1 | available_memory 归一化 (min-1.1×max → 0-1) |
| 2 | alpha 归一化 (0.75-1.55 → 0-1) |
| 3-20 | previous_action 的 one-hot 编码 (18 维) |

### 4.2 动作空间

18 个离散动作，对应 18 个切分点。

### 4.3 网络架构

```
21维 → Linear(21, 32) → Tanh → Linear(32, 32) → Tanh ─┬→ Linear(32, 18) → Actor
                                                         └→ Linear(32, 1)  → Critic
```

选择两层 32 维 MLP 而非 GRU/Transformer 的原因：
- 切分决策主要依赖当前环境状态而非长程历史
- MLP 推理速度更快（214μs vs GRU 的 ~195-267μs），满足在线决策的低延迟需求
- 实验中发现 GRU 版本的训练稳定性不如 MLP

### 4.4 Memory Action Mask

在每个时间步，根据 `available_memory` 计算可行动作掩码：

```
action_mask[c] = (memory_mb[c] <= available_memory)
```

不可行动作的 logit 被设为 `-1e9`，采样概率为 0。此设计保证 PPO 的训练和推理阶段**数学上不可能选择 OOM 动作**。

设计理由（演化过程）：
- 最初版本不包含 Action Mask，尝试通过 OOM = -100 的惩罚让 PPO "学会" 避免 OOM → 失败（PPO 收敛到仅选 action 0）
- 降低 OOM 惩罚至 -10，引入 latency clip → 失败（PPO 仍然仅选 action 0）
- 使用 action 0 作为 safe baseline 的相对收益奖励 → 部分成功（PPO 开始使用 2 个动作但 OOM = 0）
- 引入相对最优可行动作的 regret 奖励 → 成功（PPO 使用 2-4 个动作，0 OOM）
- 加入 Action Mask → **显著改善**（PPO 从 Episode 1 起就 0 OOM，动作空间探索效率大幅提升）

### 4.5 奖励函数

最终版本采用相对于当前状态最优可行动作的 **relative regret**：

```
feasible_latencies = [T(c, s) for c where action_mask[c] == True]
best_feasible_latency = min(feasible_latencies)

relative_regret = clip((T(action, s) - best_feasible_latency) / best_feasible_latency, 0, 1)
reward = 1.0 - 2.0 * relative_regret          # 范围 [-1, +1]
if action != previous_action: reward -= 0.05  # 切换惩罚

if OOM: reward = -2.0  # (Action Mask 下不会触发)
```

**设计理由：**
- 相比全局参考时延，regret 提供了更稳定的梯度信号：最优动作总是得到 +1，不论绝对时延如何变化
- clip(0, 1) 防止极端低带宽情况下单步奖励过负
- 切换惩罚 0.05 鼓励策略连续一致，减少无意义的切分点抖动

### 4.6 PPO 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Learning Rate | 3e-4 | Adam 优化器 |
| γ (discount) | 0.99 | 折扣因子 |
| λ (GAE) | 0.95 | GAE 平滑参数 |
| ε (clip) | 0.2 | PPO clip 范围 |
| Update Epochs | 4 | 每轮 rollout 的更新次数 |
| Entropy Coef | 0.01 | 鼓励探索 |
| Max Grad Norm | 0.5 | 梯度裁剪 |

---

## 5. 对比基线

### 5.1 Delay-only Search

每个时间步独立选择全局时延最小的切分点，**完全忽略显存约束**：

$$a_t = \arg\min_c T(c, s_t)$$

在带宽充足时几乎总是选择 action 17（特征最小），但在低显存时必定 OOM。

### 5.2 Memory-Constrained Search（Oracle 参考）

每个时间步独立在**当前可行动作集合**中选择时延最小的切分点：

$$a_t = \arg\min_{c: memory[c] \leq available\_memory_t} T(c, s_t)$$

这代表"已知完整时延模型时的最优可行决策"，作为 PPO 的性能上界参考。

### 5.3 三种方法对比

| 方法 | 显存约束 | 决策方式 | 是否需要 Profile | 推理延迟 |
|------|----------|----------|-----------------|----------|
| Delay-only | 无 | 18 次时延计算取 min | 是 | O(K) |
| Memory-Constrained | 显式过滤 | ≤18 次时延计算取 min | 是 | O(K) |
| **PPO (本方法)** | **Action Mask** | **1 次 MLP 前向** | **是** | **O(K + NN)** |

---

## 6. OOM 故障模型

当选择的切分点所需显存超过当前可用显存时：

1. 已消耗 `OOM_FAILED_RATIO × selected_latency_ms` 的无效计算时间（模拟部分前缀计算已完成的程度）
2. 触发 `OOM_RECOVERY_MS = 1500ms` 的恢复开销
3. 退回到最安全的切分点（action 0，显存需求最低）重新执行
4. 计入一次 OOM 和一次切换

此模型为简化的仿真假设。实际 OOM 恢复时间取决于具体框架（PyTorch/TensorFlow）的内存管理策略。

---

## 7. 系统仿真流程

对于每个 SGD update 步：

```
1. 读取当前环境状态 (bandwidth, memory, alpha, rtt)
2. 在线决策: action = policy.choose_action(state)
3. 计算: latency = T(action, state)
4. 检查显存: feasible = memory[action] <= available_memory
5a. 若 feasible: 累计 latency + 决策时间 + 切换开销
5b. 若 OOM:     累计 决策时间 + 部分计算 + 恢复开销 + action 0 重算时间
6. successful_updates += 1
7. 到达 accuracy checkpoint 时记录 (elapsed_system_time, test_accuracy)
```

OOM 后通过 action 0 重算保证该 SGD update 仍然成功完成，不破坏精度曲线的完整性。

---

## 8. 实验流程

### 阶段 1: Quick Mode

`QUICK_MODE = True`，用于快速验证代码正确性：

- CIFAR-10: 2 epoch, 5000 训练/1000 测试样本
- Profile: 3 warmup + 10 runs
- PPO: 100 episodes × 256 steps, 1 seed
- Decision benchmark: 100 warmup + 1000 runs

### 阶段 2: Paper Mode

`QUICK_MODE = False`，完整论文实验：

- CIFAR-10: 100 epoch, 50000 训练/10000 测试样本, 3 seeds
- Profile: 20 warmup + 50 runs
- PPO: 150 episodes × 512 steps, 3 seeds
- Decision benchmark: 1000 warmup + 9000 runs

### 阶段 3: 图表生成

从 CSV 文件读取数据，生成 4 组图表（Fig1-3 + Table I-II），分别对应系统框架、PPO 收敛、时间-精度曲线、核心性能柱状图。

---

## 9. 设计决策的演变

实验过程中经历了以下主要设计决策：

| 版本 | 关键改变 | 结果 |
|------|----------|------|
| v1 | OOM = -100, 无 Action Mask | PPO 仅选 action 0 |
| v2 | OOM = -10, latency clip = 5 | 同上 |
| v3 | 相对 action 0 的安全基线收益 | PPO 使用 2 动作 (0, 14), OOM=0 |
| v4 | 相对最优可行动作的 regret | PPO 使用 1 动作 (仅 0) |
| v5 | regret + Action Mask | **PPO 使用 4 动作 (0,8,9,17), OOM=0, 达 MC 水平** |

最终版本结合了 relative-regret 奖励和 Memory Action Mask，实现了最佳性能。
