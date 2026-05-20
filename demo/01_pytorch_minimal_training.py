#!/usr/bin/env python3
"""Minimal PyTorch image classification training loop.

Use this script to understand the training skeleton before reading larger CV
repositories. It supports FakeData for a no-download smoke test and
Fashion-MNIST for a real small dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class SmallCNN(nn.Module):
    """简单的卷积神经网络，用于图像分类任务。
    
    架构包含：
    - 特征提取器：两个卷积块，每块包含卷积层、批归一化、ReLU激活和最大池化
    - 分类器：平均池化、展平层和全连接层
    """
    
    def __init__(self, in_channels: int, num_classes: int) -> None:
        """初始化模型架构。
        
        Args:
            in_channels: 输入图像的通道数（如灰度图为1，RGB图为3）
            num_classes: 分类的类别数
        """
        super().__init__()
        
        # 特征提取层：通过卷积、批归一化、激活和池化层提取图像特征
        self.features = nn.Sequential(
            # 第一个卷积块：输入通道 -> 32个过滤器
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 保持空间维度
            nn.BatchNorm2d(32),  # 批归一化稳定训练
            nn.ReLU(inplace=True),  # ReLU激活函数引入非线性
            nn.MaxPool2d(2),  # 2x2最大池化，空间尺寸减半
            
            # 第二个卷积块：32个通道 -> 64个过滤器
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 再次将空间尺寸减半
        )
        
        # 分类层：将特征转换为类别概率
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，输出(B, 64, 1, 1)
            nn.Flatten(),  # 展平为(B, 64)
            nn.Linear(64, num_classes),  # 全连接层进行分类
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为(batch_size, in_channels, height, width)
            
        Returns:
            分类逻辑（logits），形状为(batch_size, num_classes)
        """
        # 通过特征提取层获取图像特征
        x = self.features(x)
        # 通过分类层获得最终的分类结果
        return self.classifier(x)


def choose_device() -> torch.device:
    """选择计算设备（GPU或CPU）。
    
    优先级：CUDA GPU > Apple MPS > CPU
    
    Returns:
        torch.device: 选中的计算设备
    """
    # 优先使用CUDA GPU（NVIDIA显卡）
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 其次使用Apple Metal Performance Shaders（Mac M系列芯片）
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # 最后使用CPU作为后备
    return torch.device("cpu")


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
    """构建训练和验证数据加载器。
    
    支持两种数据集：
    1. FakeData：随机生成的数据，用于快速测试（不需要下载）
    2. Fashion-MNIST：真实的服装图像数据集（灰度图像）
    
    Args:
        args: 命令行参数对象，包含dataset、batch_size等配置
        
    Returns:
        tuple: (训练数据加载器, 验证数据加载器, 输入通道数, 类别数)
    """
    
    if args.dataset == "fashion-mnist":
        # 使用Fashion-MNIST数据集（灰度图，h=w=28）
        transform = transforms.Compose([transforms.ToTensor()])  # 转换为张量
        
        # 创建训练集
        train_set = datasets.FashionMNIST(
            root=args.data_root,  # 数据存储路径
            train=True,  # 加载训练子集
            transform=transform,
            download=True,  # 如果本地不存在则自动下载
        )
        
        # 创建验证集
        val_set = datasets.FashionMNIST(
            root=args.data_root,
            train=False,  # 加载测试子集
            transform=transform,
            download=True,
        )
        
        # Fashion-MNIST的配置
        in_channels = 1  # 灰度图像
        num_classes = 10  # 10个服装类别
    else:
        # 使用FakeData：生成随机数据用于开发和调试（无需网络下载）
        transform = transforms.Compose([transforms.ToTensor()])
        
        train_set = datasets.FakeData(
            size=args.fake_train_size,  # 训练样本数
            image_size=(3, 64, 64),  # RGB图像，64x64
            num_classes=args.num_classes,
            transform=transform,
        )
        
        val_set = datasets.FakeData(
            size=args.fake_val_size,  # 验证样本数
            image_size=(3, 64, 64),
            num_classes=args.num_classes,
            transform=transform,
        )
        
        # FakeData的配置
        in_channels = 3  # RGB图像
        num_classes = args.num_classes  # 自定义类别数

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,  # 批次大小
        shuffle=True,  # 随机打乱数据以增强泛化性
        num_workers=args.num_workers,  # 多线程加载数据
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=args.num_workers,
    )
    
    return train_loader, val_loader, in_channels, num_classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练模型一个完整的周期（epoch）。
    
    Args:
        model: 神经网络模型
        loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        tuple: (平均损失值, 准确率)
    """
    # 设置模型为训练模式（启用随机失活、批归一化等）
    model.train()
    
    # 初始化度量指标
    total_loss = 0.0  # 累积损失
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数

    # 遍历训练数据加载器中的所有批次
    for images, labels in tqdm(loader, desc="train", leave=False):
        # 将数据移到指定设备（GPU或CPU）
        images = images.to(device)
        labels = labels.to(device)

        # 清空梯度缓冲（防止梯度累积）
        optimizer.zero_grad(set_to_none=True)
        
        # 前向传播：计算模型预测
        logits = model(images)
        
        # 计算损失值（交叉熵损失）
        loss = criterion(logits, labels)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 优化步骤：根据梯度更新模型参数
        optimizer.step()

        # 计算该批次的统计信息
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size  # 累积损失
        # 比较预测结果与真实标签，统计正确的预测数
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    # 返回平均损失和准确率
    return total_loss / total, correct / total


@torch.no_grad()  # 装饰器：禁用梯度计算，节省内存和加快推理速度
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证模型在验证集上的性能。
    
    Args:
        model: 神经网络模型
        loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        tuple: (平均损失值, 准确率)
    """
    # 设置模型为评估模式（禁用随机失活、批归一化不计算运行统计）
    model.eval()
    
    # 初始化度量指标
    total_loss = 0.0
    correct = 0
    total = 0

    # 遍历验证数据集中的所有批次
    for images, labels in tqdm(loader, desc="eval", leave=False):
        # 将数据移到指定设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播（无梯度计算）
        logits = model(images)
        # 计算损失值
        loss = criterion(logits, labels)

        # 累积统计信息
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    # 返回平均损失和准确率
    return total_loss / total, correct / total


def write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    """将训练指标保存到CSV文件。
    
    Args:
        path: CSV文件的保存路径
        rows: 包含训练指标的字典列表（每行为一个epoch的指标）
    """
    # 创建输出目录（如果不存在）
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 打开文件并写入CSV格式的数据
    with path.open("w", newline="") as f:
        # 根据第一行的键创建CSV写入器
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        # 写入表头
        writer.writeheader()
        # 写入所有行数据
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        argparse.Namespace: 包含所有参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description="PyTorch最小化训练脚本")
    
    # 数据集相关参数
    parser.add_argument("--dataset", choices=["fake", "fashion-mnist"], default="fake",
                        help="选择数据集：fake(随机数据)或fashion-mnist(真实数据集)")
    parser.add_argument("--data-root", default="data",
                        help="数据存储的根目录")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=2,
                        help="训练的轮次数")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="数据加载的工作线程数")
    
    # FakeData相关参数
    parser.add_argument("--num-classes", type=int, default=10,
                        help="分类类别数（用于FakeData）")
    parser.add_argument("--fake-train-size", type=int, default=1024,
                        help="FakeData训练集大小")
    parser.add_argument("--fake-val-size", type=int, default=256,
                        help="FakeData验证集大小")
    
    # 输出相关参数
    parser.add_argument("--metrics-out", default="outputs/minimal_training_metrics.csv",
                        help="输出指标CSV文件的路径")
    
    return parser.parse_args()


def main() -> None:
    """主函数：执行完整的训练流程。
    
    流程：
    1. 解析命令行参数
    2. 选择计算设备
    3. 构建数据加载器
    4. 初始化模型、损失函数和优化器
    5. 循环训练多个epoch
    6. 每个epoch后评估验证集性能
    7. 保存训练指标到CSV文件
    """
    # 解析命令行参数
    args = parse_args()
    
    # 选择计算设备（CUDA > MPS > CPU）
    device = choose_device()
    print(f"Using device: {device}")

    # 构建训练和验证数据加载器
    train_loader, val_loader, in_channels, num_classes = build_loaders(args)
    
    # 初始化模型并移到指定设备
    model = SmallCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # 定义损失函数：交叉熵损失（用于多分类任务）
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器：AdamW（带权重衰减的Adam优化器）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 记录每个epoch的训练指标
    rows: list[dict[str, float | int]] = []
    
    # 循环训练多个epoch
    for epoch in range(1, args.epochs + 1):
        # 执行一个epoch的训练，返回训练损失和准确率
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        
        # 在验证集上评估模型性能
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 记录该epoch的四个关键指标
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        rows.append(row)
        
        # 打印该epoch的训练结果
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

    # 将所有epoch的指标保存到CSV文件
    write_metrics(Path(args.metrics_out), rows)
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()

