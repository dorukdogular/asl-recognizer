import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "data/landmarks.npz"
BEST_PATH = "models/mlp_best.pth"
ONNX_PATH = "models/mlp_asl.onnx"
CLASSES_PATH = "models/mlp_classes.json"
IN_DIM = 63


class LandmarkMLP(nn.Module):
    def __init__(self, in_dim=IN_DIM, num_classes=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        self.num_classes = num_classes
        self.in_dim = in_dim

    def forward(self, x):
        return self.net(x)


def select_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss_sum += criterion(out, yb).item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


def export_onnx(model, path, in_dim, opset=17):
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        model_cpu, dummy, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train landmark MLP")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    data = np.load(args.data, allow_pickle=True)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    class_names = [str(c) for c in data["class_names"]]
    num_classes = len(class_names)
    print(f"Samples: {len(X)}  Features: {X.shape[1]}  Classes: {num_classes}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size, shuffle=False)

    model = LandmarkMLP(IN_DIM, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(BEST_PATH), exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names, "in_dim": IN_DIM,
                        "num_classes": num_classes}, BEST_PATH)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[e{epoch:03d}] lr={scheduler.get_last_lr()[0]:.2e} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)

    ckpt = torch.load(BEST_PATH, map_location="cpu")
    best_model = LandmarkMLP(IN_DIM, num_classes)
    best_model.load_state_dict(ckpt["model_state"])
    export_onnx(best_model, ONNX_PATH, IN_DIM)

    print("=" * 44)
    print(f"Best val_acc : {best_acc:.4f}  (target 0.95 "
          f"{'MET' if best_acc >= 0.95 else 'NOT MET'})")
    print(f"Saved: {BEST_PATH}")
    print(f"Saved: {ONNX_PATH}")
    print(f"Saved: {CLASSES_PATH}")
    print("=" * 44)


if __name__ == "__main__":
    main()
