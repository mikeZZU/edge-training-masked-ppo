from pathlib import Path
import random

import numpy as np
import torch


# ============================================================
# 1. 路径
# ============================================================

WORKSPACE = Path(__file__).resolve().parent

CIFAR_ROOT = WORKSPACE

CIFAR_FOLDER = WORKSPACE / "cifar-10-batches-py"

OUTPUT_DIR = WORKSPACE / "outputs"


# ============================================================
# 2. 实验模式
# ============================================================

# True：
# 先用少量数据快速检查整套代码是否能正常运行。
#
# False：
# 正式进行论文实验。
#
# 第一次运行必须保持 True。
QUICK_MODE = False

MODE_NAME = "quick" if QUICK_MODE else "paper"


# ============================================================
# 3. CIFAR-10 / ResNet50 训练参数
# ============================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

LEARNING_RATE = 0.1

MOMENTUM = 0.9

WEIGHT_DECAY = 5e-4

TARGET_ACCURACY = 90.0


if QUICK_MODE:
    EPOCHS = 2

    SEEDS = [1]

    QUICK_TRAIN_SAMPLES = 5000

    QUICK_TEST_SAMPLES = 1000

else:
    EPOCHS = 100

    SEEDS = [1, 2, 3]

    QUICK_TRAIN_SAMPLES = None

    QUICK_TEST_SAMPLES = None


# ============================================================
# 4. 离线 Profile 参数
# ============================================================

NUM_SPLIT_POINTS = 18


if QUICK_MODE:
    PROFILE_WARMUP = 3
    PROFILE_RUNS = 10

else:
    PROFILE_WARMUP = 20
    PROFILE_RUNS = 50


# ============================================================
# 5. PPO-GRU 参数
# ============================================================

GRU_HIDDEN_SIZE = 32

PPO_LEARNING_RATE = 3e-4

GAMMA = 0.99

GAE_LAMBDA = 0.95

PPO_CLIP = 0.2

PPO_UPDATE_EPOCHS = 4

SWITCH_PENALTY = 0.05


if QUICK_MODE:
    PPO_EPISODES = 100
    PPO_TRACE_LENGTH = 256

else:
    PPO_EPISODES = 150
    PPO_TRACE_LENGTH = 512


# ============================================================
# 6. 系统仿真参数
# ============================================================

# 切换切分点带来的固定额外开销，单位 ms。
SWITCH_COST_MS = 5.0

# OOM 后模型恢复的模拟固定开销，单位 ms。
#
# 后续如果完成真实 OOM 恢复时间测量，
# 再由 ChatGPT 决定是否替换该值。
OOM_RECOVERY_MS = 1500.0

# OOM 发生之前已经消耗的失败计算比例。
OOM_FAILED_RATIO = 0.35


# ============================================================
# 7. 决策时间测试参数
# ============================================================

if QUICK_MODE:
    DECISION_WARMUP = 100
    DECISION_RUNS = 1000

else:
    DECISION_WARMUP = 1000
    DECISION_RUNS = 9000


# ============================================================
# 8. 设备
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 9. 随机种子
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 10. 输出目录
# ============================================================

def prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 11. 基本环境检查
# ============================================================

def print_config() -> None:
    print("=" * 60)

    print("Experiment configuration")

    print("=" * 60)

    print(f"Workspace       : {WORKSPACE}")

    print(f"CIFAR folder    : {CIFAR_FOLDER}")

    print(f"Output folder   : {OUTPUT_DIR}")

    print(f"Mode            : {MODE_NAME}")

    print(f"Device          : {DEVICE}")

    print(f"Batch size      : {BATCH_SIZE}")

    print(f"Epochs          : {EPOCHS}")

    print(f"Split points    : {NUM_SPLIT_POINTS}")

    print(f"Seeds           : {SEEDS}")

    if torch.cuda.is_available():
        print(
            "GPU             : "
            + torch.cuda.get_device_name(0)
        )

        print(
            "CUDA version    : "
            + str(torch.version.cuda)
        )

    print("=" * 60)


if __name__ == "__main__":
    prepare_output_dir()

    print_config()
