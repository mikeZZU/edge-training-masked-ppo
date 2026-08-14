import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50

import config as cfg


# ============================================================
# 1. CIFAR-10 版 ResNet50
# ============================================================

class CifarResNet50(nn.Module):
    """
    将 torchvision ResNet50 调整为 CIFAR-10 输入。

    18 个候选切分动作：

    action 0:
        stem 后

    action 1 ~ 16:
        16 个 Bottleneck block 后

    action 17:
        average pooling 后

    这样得到 18 个互不重复的候选位置。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        base = resnet50(weights=None)

        # CIFAR-10 输入仅 32x32。
        # 使用 3x3, stride=1，并取消原 ResNet 的 maxpool。
        base.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        base.maxpool = nn.Identity()

        base.fc = nn.Linear(
            base.fc.in_features,
            num_classes,
        )

        stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
        )

        bottlenecks = []

        bottlenecks.extend(
            list(base.layer1.children())
        )

        bottlenecks.extend(
            list(base.layer2.children())
        )

        bottlenecks.extend(
            list(base.layer3.children())
        )

        bottlenecks.extend(
            list(base.layer4.children())
        )

        # 1 个 stem + 16 个 bottleneck + 1 个 avgpool
        self.stages = nn.ModuleList(
            [stem]
            + bottlenecks
            + [base.avgpool]
        )

        self.fc = base.fc

        if len(self.stages) != cfg.NUM_SPLIT_POINTS:
            raise RuntimeError(
                "Split point number is not 18."
            )

    def forward_to(
        self,
        x: torch.Tensor,
        action: int,
    ) -> torch.Tensor:
        """
        只运行到指定切分点。
        """

        if action < 0 or action >= len(self.stages):
            raise ValueError(
                f"Invalid split action: {action}"
            )

        for i in range(action + 1):
            x = self.stages[i](x)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        for stage in self.stages:
            x = stage(x)

        x = torch.flatten(
            x,
            1,
        )

        x = self.fc(x)

        return x


# ============================================================
# 2. CIFAR-10 DataLoader
# ============================================================

def build_dataloaders(seed: int):
    """
    构建训练集和测试集。

    QUICK_MODE:
        训练 5000 张
        测试 1000 张

    PAPER_MODE:
        使用完整 CIFAR-10
    """

    mean = (
        0.4914,
        0.4822,
        0.4465,
    )

    std = (
        0.2470,
        0.2435,
        0.2616,
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean,
                std,
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(cfg.CIFAR_ROOT),
        train=True,
        download=False,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(cfg.CIFAR_ROOT),
        train=False,
        download=False,
        transform=test_transform,
    )

    if cfg.QUICK_MODE:
        rng = np.random.default_rng(seed)

        train_indices = rng.choice(
            len(train_dataset),
            size=min(
                cfg.QUICK_TRAIN_SAMPLES,
                len(train_dataset),
            ),
            replace=False,
        )

        test_indices = rng.choice(
            len(test_dataset),
            size=min(
                cfg.QUICK_TEST_SAMPLES,
                len(test_dataset),
            ),
            replace=False,
        )

        train_dataset = Subset(
            train_dataset,
            train_indices.tolist(),
        )

        test_dataset = Subset(
            test_dataset,
            test_indices.tolist(),
        )

    generator = torch.Generator()

    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        test_loader,
    )


# ============================================================
# 3. GPU 同步
# ============================================================

def gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# 4. 单个切分点 Device Profile
# ============================================================

def profile_one_split(
    action: int,
):
    """
    对一个切分点测量：

    1. device-side training latency
    2. intermediate feature size
    3. device-side peak GPU memory

    这里的 device-side training 使用：

        prefix forward
        dummy gradient loss
        backward
        SGD update

    目的是测量不同模型前缀在设备侧的相对计算和显存需求。
    """

    cfg.set_seed(1234)

    model = CifarResNet50().to(
        cfg.DEVICE
    )

    model.train()

    prefix_parameters = []

    for i in range(action + 1):
        prefix_parameters.extend(
            list(
                model.stages[i].parameters()
            )
        )

    optimizer = torch.optim.SGD(
        prefix_parameters,
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    x = torch.randn(
        cfg.BATCH_SIZE,
        3,
        32,
        32,
        device=cfg.DEVICE,
    )

    # --------------------------------------------------------
    # 获取切分点输出大小
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        feature = model.forward_to(
            x,
            action,
        )

    feature_bytes = (
        feature.numel()
        * feature.element_size()
    )

    del feature

    model.train()

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(cfg.PROFILE_WARMUP):

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model.forward_to(
            x,
            action,
        )

        # 模拟服务器返回给切分点的梯度。
        # 对 prefix 的 backward 计算量进行实际测量。
        dummy_grad = torch.ones_like(
            out
        )

        out.backward(
            dummy_grad
        )

        optimizer.step()

    gpu_sync()

    # --------------------------------------------------------
    # 正式计时和显存测量
    # --------------------------------------------------------

    latency_list = []

    memory_list = []

    for _ in range(cfg.PROFILE_RUNS):

        optimizer.zero_grad(
            set_to_none=True
        )

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        gpu_sync()

        start = time.perf_counter()

        out = model.forward_to(
            x,
            action,
        )

        dummy_grad = torch.ones_like(
            out
        )

        out.backward(
            dummy_grad
        )

        optimizer.step()

        gpu_sync()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        latency_list.append(
            elapsed_ms
        )

        if torch.cuda.is_available():

            peak_mb = (
                torch.cuda.max_memory_allocated()
                / 1024.0
                / 1024.0
            )

            memory_list.append(
                peak_mb
            )

    device_ms = float(
        np.median(
            latency_list
        )
    )

    if memory_list:

        memory_mb = float(
            np.median(
                memory_list
            )
        )

    else:

        memory_mb = 0.0

    del optimizer
    del model
    del x

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        device_ms,
        int(feature_bytes),
        memory_mb,
    )


# ============================================================
# 5. 完整模型训练时间
# ============================================================

def profile_full_model():
    """
    测量完整 ResNet50 单 batch 的训练时间。

    后续使用：

        server_ms =
        full_train_ms - device_ms

    构造简单且可重复的服务器侧时延 Profile。
    """

    cfg.set_seed(1234)

    model = CifarResNet50().to(
        cfg.DEVICE
    )

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    x = torch.randn(
        cfg.BATCH_SIZE,
        3,
        32,
        32,
        device=cfg.DEVICE,
    )

    y = torch.randint(
        low=0,
        high=10,
        size=(cfg.BATCH_SIZE,),
        device=cfg.DEVICE,
    )

    for _ in range(cfg.PROFILE_WARMUP):

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(x)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

    gpu_sync()

    times = []

    memory_values = []

    for _ in range(cfg.PROFILE_RUNS):

        optimizer.zero_grad(
            set_to_none=True
        )

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        gpu_sync()

        start = time.perf_counter()

        logits = model(x)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        gpu_sync()

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        times.append(
            elapsed_ms
        )

        if torch.cuda.is_available():

            memory_values.append(
                torch.cuda.max_memory_allocated()
                / 1024.0
                / 1024.0
            )

    full_train_ms = float(
        np.median(
            times
        )
    )

    if memory_values:

        full_peak_mb = float(
            np.median(
                memory_values
            )
        )

    else:

        full_peak_mb = 0.0

    del optimizer
    del model
    del x
    del y

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        full_train_ms,
        full_peak_mb,
    )


# ============================================================
# 6. 生成完整的 18 点 Profile
# ============================================================

def create_profile():
    print()
    print("=" * 60)
    print("Creating split-point profile")
    print("=" * 60)

    full_train_ms, full_peak_mb = (
        profile_full_model()
    )

    print(
        f"Full-model train latency: "
        f"{full_train_ms:.3f} ms"
    )

    print(
        f"Full-model peak memory   : "
        f"{full_peak_mb:.3f} MB"
    )

    raw_rows = []

    for action in range(
        cfg.NUM_SPLIT_POINTS
    ):

        device_ms, feature_bytes, memory_mb = (
            profile_one_split(
                action
            )
        )

        raw_rows.append(
            {
                "action": action,
                "raw_device_ms": device_ms,
                "feature_bytes": feature_bytes,
                "raw_memory_mb": memory_mb,
            }
        )

        print(
            f"action={action:02d} | "
            f"device={device_ms:.3f} ms | "
            f"feature={feature_bytes / 1024.0:.1f} KB | "
            f"memory={memory_mb:.1f} MB"
        )

    raw_df = pd.DataFrame(
        raw_rows
    )

    # --------------------------------------------------------
    # 由于 GPU 微小计时波动可能破坏前缀计算的单调趋势，
    # 使用累计最大值消除测量噪声。
    # --------------------------------------------------------

    device_ms = np.maximum.accumulate(
        raw_df[
            "raw_device_ms"
        ].to_numpy(
            dtype=np.float64
        )
    )

    memory_mb = np.maximum.accumulate(
        raw_df[
            "raw_memory_mb"
        ].to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # device-side 最大时延不允许超过完整模型训练时间，
    # 避免 server_ms 变成负数。
    # --------------------------------------------------------

    max_device_allowed = (
        full_train_ms
        * 0.98
    )

    if device_ms[-1] > max_device_allowed:

        scale = (
            max_device_allowed
            / device_ms[-1]
        )

        device_ms = (
            device_ms
            * scale
        )

    server_ms = np.maximum(
        full_train_ms
        - device_ms,
        0.01,
    )

    profile_df = pd.DataFrame(
        {
            "action":
                raw_df["action"],

            "device_ms":
                device_ms,

            "server_ms":
                server_ms,

            "feature_bytes":
                raw_df["feature_bytes"],

            "memory_mb":
                memory_mb,
        }
    )

    output_path = (
        cfg.OUTPUT_DIR
        / f"profile_{cfg.MODE_NAME}.csv"
    )

    profile_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Profile saved to:"
    )

    print(
        output_path
    )

    print()
    print(profile_df)

    return profile_df


# ============================================================
# 7. 测试模型精度
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader,
):
    model.eval()

    correct = 0

    total = 0

    for images, labels in test_loader:

        images = images.to(
            cfg.DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            cfg.DEVICE,
            non_blocking=True,
        )

        logits = model(
            images
        )

        prediction = logits.argmax(
            dim=1
        )

        correct += int(
            (
                prediction
                == labels
            ).sum().item()
        )

        total += int(
            labels.numel()
        )

    accuracy = (
        100.0
        * correct
        / max(
            total,
            1,
        )
    )

    return accuracy


# ============================================================
# 8. 得到真实 successful updates - accuracy 曲线
# ============================================================

def create_accuracy_history():
    """
    所有三种切分策略后续共享这条真实训练精度曲线。

    原因：
    切分位置只改变系统时延和 OOM，
    不应该改变成功完成一次完整 SGD update 后的数学结果。

    因此这里真实训练 ResNet50，
    保存：

        successful_updates
        test_accuracy

    后续 run_compare.py 再将 successful_updates
    映射到不同算法各自的累计系统时间。
    """

    print()
    print("=" * 60)
    print("Training CIFAR-10 accuracy curve")
    print("=" * 60)

    history = []

    for seed in cfg.SEEDS:

        print()
        print(
            f"Seed = {seed}"
        )

        cfg.set_seed(
            seed
        )

        train_loader, test_loader = (
            build_dataloaders(
                seed
            )
        )

        model = CifarResNet50().to(
            cfg.DEVICE
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.EPOCHS,
            )
        )

        successful_updates = 0

        for epoch in range(
            1,
            cfg.EPOCHS + 1,
        ):

            model.train()

            loss_sum = 0.0

            batch_count = 0

            for images, labels in train_loader:

                images = images.to(
                    cfg.DEVICE,
                    non_blocking=True,
                )

                labels = labels.to(
                    cfg.DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels,
                )

                loss.backward()

                optimizer.step()

                successful_updates += 1

                loss_sum += float(
                    loss.item()
                )

                batch_count += 1

            scheduler.step()

            test_accuracy = evaluate(
                model,
                test_loader,
            )

            train_loss = (
                loss_sum
                / max(
                    batch_count,
                    1,
                )
            )

            history.append(
                {
                    "seed":
                        seed,

                    "epoch":
                        epoch,

                    "successful_updates":
                        successful_updates,

                    "test_accuracy":
                        test_accuracy,

                    "train_loss":
                        train_loss,
                }
            )

            print(
                f"epoch={epoch:03d}/{cfg.EPOCHS} | "
                f"updates={successful_updates} | "
                f"loss={train_loss:.4f} | "
                f"test_acc={test_accuracy:.2f}%"
            )

            # 正式模式达到目标精度后即可停止。
            if (
                not cfg.QUICK_MODE
                and test_accuracy
                >= cfg.TARGET_ACCURACY
            ):
                print(
                    "Target accuracy reached."
                )

                break

        del optimizer
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(
        history
    )

    output_path = (
        cfg.OUTPUT_DIR
        / f"accuracy_history_{cfg.MODE_NAME}.csv"
    )

    history_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Accuracy history saved to:"
    )

    print(
        output_path
    )

    print()
    print(
        history_df
    )

    return history_df


# ============================================================
# 9. 主程序
# ============================================================

def main():

    cfg.prepare_output_dir()

    cfg.print_config()

    if not cfg.CIFAR_FOLDER.exists():

        raise FileNotFoundError(
            f"CIFAR-10 folder does not exist: "
            f"{cfg.CIFAR_FOLDER}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "This experiment requires the GPU virtual environment."
        )

    torch.backends.cudnn.benchmark = True

    profile_df = create_profile()

    accuracy_df = create_accuracy_history()

    print()
    print("=" * 60)
    print("prepare_data.py finished")
    print("=" * 60)

    print(
        f"Profile rows: "
        f"{len(profile_df)}"
    )

    print(
        f"Accuracy-history rows: "
        f"{len(accuracy_df)}"
    )

    print(
        "No PPO training has been executed."
    )


if __name__ == "__main__":
    main()
