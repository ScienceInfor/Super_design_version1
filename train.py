import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from model import ProteinTransformer
from dataset import ProteinDataset, collate_fn
import time
import os  # 添加os模块用于创建目录


def compute_mask_and_targets(seq_onehot):
    # seq_onehot: (B, L, 21) 其中 21 是考虑了非标准氨基酸
    mask = torch.sum(seq_onehot, dim=-1) != 0  # (B, L) True for valid
    target_idx = torch.argmax(seq_onehot, dim=-1)  # (B, L)
    return mask, target_idx


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for coords, seq_onehot in dataloader:
        coords, seq_onehot = coords.to(device), seq_onehot.to(device)
        mask, targets = compute_mask_and_targets(seq_onehot)

        optimizer.zero_grad()
        logits = model(coords, padding_mask=~mask)  # mask=True表示忽略
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_valid = 0

    with torch.no_grad():
        for coords, seq_onehot in dataloader:
            coords, seq_onehot = coords.to(device), seq_onehot.to(device)
            mask, targets = compute_mask_and_targets(seq_onehot)

            logits = model(coords, padding_mask=~mask)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            total_correct += ((preds == targets) * mask).sum().item()
            total_valid += mask.sum().item()

    accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    return total_loss / len(dataloader), accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建绘图窗口
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Metrics')

    # 初始化图表
    ax1.set_title('Loss');
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss');
    ax1.grid(True)
    ax2.set_title('Accuracy');
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('Accuracy (%)');
    ax2.set_ylim(0, 100);
    ax2.grid(True)
    train_line, = ax1.plot([], [], 'r-', label='Train Loss')
    val_line, = ax1.plot([], [], 'b-', label='Val Loss')
    acc_line, = ax2.plot([], [], 'g-', label='Accuracy')
    ax1.legend();
    ax2.legend();
    plt.tight_layout()

    # 加载数据
    dataset = ProteinDataset('data1.pkl')
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 初始化模型、优化器、损失函数
    model = ProteinTransformer(max_len=2048).to(device)  # 这里确保传递max_len
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 日志变量
    num_epochs = 350
    epochs, train_losses, val_losses, accuracies = [], [], [], []
    best_val_loss = float('inf')

    # 创建模型保存目录
    os.makedirs('checkpoints', exist_ok=True)

    now_time = time.time()
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_loss)

        epochs.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(val_acc * 100)

        # 动态绘图
        train_line.set_data(epochs, train_losses)
        val_line.set_data(epochs, val_losses)
        acc_line.set_data(epochs, accuracies)
        ax1.relim();
        ax1.autoscale_view()
        ax2.relim();
        ax2.autoscale_view()
        fig.canvas.draw();
        fig.canvas.flush_events()

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc * 100:.2f}%")

        # 每两个epoch保存一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'checkpoints/model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # 如果验证集损失降低，保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_protein_transformer.pth')
            print(f"New best model saved with val loss: {val_loss:.4f}")

    end_time = time.time()
    print(f"Total training time: {end_time - now_time:.2f} seconds")

    # 保存最终模型
    torch.save(model.state_dict(), 'protein_transformer.pth')
    plt.ioff();
    plt.show()


if __name__ == '__main__':
    main()
