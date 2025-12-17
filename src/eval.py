import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from config import Args
from seed import set_seed
from dataset import SpeechImageDataset
from model import CustomSwinTransformer
from metrics import save_confusion_matrix

@torch.no_grad()
def evaluate(args: Args):
    transform = transforms.Compose([
        transforms.Resize(args.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SpeechImageDataset(root_dir=args.test_root, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomSwinTransformer(num_classes=args.n_class).to(args.device)
    model.load_state_dict(torch.load(args.best_ckpt_path, map_location=args.device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    test_loss = 0.0
    correct, total = 0, 0
    class_correct = [0] * args.n_class
    class_total = [0] * args.n_class

    test_loop = tqdm(test_loader, desc="Testing")
    for inputs, labels, _ in test_loop:
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)

        if labels.dim() > 1:
            labels_idx = torch.argmax(labels, dim=1)
        else:
            labels_idx = labels

        loss = criterion(outputs, labels_idx)
        test_loss += loss.item()

        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_idx.cpu().numpy())

        correct += (preds == labels_idx).sum().item()
        total += labels_idx.size(0)

        for i in range(len(labels_idx)):
            lab = int(labels_idx[i].item())
            class_correct[lab] += int((preds[i] == labels_idx[i]).item())
            class_total[lab] += 1

        test_loop.set_postfix(loss=test_loss/len(test_loader), mean_acc=100.*correct/total)

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    print(f"\nTest Loss: {test_loss:.4f}, Mean Test Accuracy: {test_acc:.2f}%")

    print("\nClass-wise Accuracy:")
    class_names = test_dataset.classes
    for i in range(args.n_class):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            print(f"Class {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"Class {class_names[i]}: No samples")

    f1 = f1_score(all_labels, all_preds, average="weighted")
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")

    print(f"\nF1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    save_confusion_matrix(all_labels, all_preds, class_names, save_dir=args.cm_dir)

if __name__ == "__main__":
    args = Args()
    set_seed(args.seed)
    evaluate(args)
