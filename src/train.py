import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Args
from seed import set_seed
from dataset import SpeechImageDataset
from model import CustomSwinTransformer


def build_transform(args):
    return transforms.Compose([
        transforms.Resize(args.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_model(args: Args):
    os.makedirs(os.path.dirname(args.best_ckpt_path), exist_ok=True)

    transform = build_transform(args)
    train_dataset = SpeechImageDataset(root_dir=args.train_root, transform=transform)
    val_dataset   = SpeechImageDataset(root_dir=args.val_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomSwinTransformer(num_classes=args.n_class).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.eta_min)

    best_val_acc = 0.0
    train_acc_list, train_loss_list = [], []
    val_acc_list, val_loss_list = [], []

    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")
        for inputs, labels, _ in train_loop:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if labels.dim() > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels

            loss = criterion(outputs, labels_idx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct += (preds == labels_idx).sum().item()
            total += labels_idx.size(0)

            train_loop.set_postfix(loss=running_loss/len(train_loader), mean_acc=100.*correct/total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                outputs = model(inputs)

                if labels.dim() > 1:
                    labels_idx = torch.argmax(labels, dim=1)
                else:
                    labels_idx = labels

                loss = criterion(outputs, labels_idx)
                val_loss += loss.item()

                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                val_correct += (preds == labels_idx).sum().item()
                val_total += labels_idx.size(0)

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epoch} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.best_ckpt_path)
            print(f"Saved best model -> {args.best_ckpt_path}")

        scheduler.step()

    epochs = range(len(train_acc_list))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc_list, label="Train Acc")
    plt.plot(epochs, val_acc_list, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.acc_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.plot(epochs, val_loss_list, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.loss_plot_path)
    plt.close()

    print(f"Training finished. Best Val Acc: {best_val_acc:.2f}%")
    return args.best_ckpt_path


if __name__ == "__main__":
    args = Args()
    set_seed(args.seed)
    train_model(args)
