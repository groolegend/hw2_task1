import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from test_dataloader import val_loader


CKPT_PATH = "ckpts/vit_ckpt/best_val_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=102):
    model = models.swin_t(weights=None)

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Testing Swin-T")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / total:.4f}",
            "acc": f"{correct / total:.4f}"
        })

    return total_loss / total, correct / total


def main():
    model = build_model(num_classes=102)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"Loaded checkpoint: {CKPT_PATH}")

    test_loss, test_acc = evaluate(model, val_loader)

    print("=" * 80)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()