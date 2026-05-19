import os
import json
import copy
import itertools
import yaml

import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models
import wandb
import timm

from test_dataloader import val_loader, test_loader
from test_dataloader import train_loader_augment as train_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name="swin_t", num_classes=102):
    if model_name == "swin_t":
        model = models.swin_t(
            weights=models.Swin_T_Weights.IMAGENET1K_V1
        )

        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif model_name == "vit_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model.to(device)


def build_optimizer(model, config):
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if (
            name.startswith("head")
            or name.startswith("fc")
            or name.startswith("classifier")
        ):
            head_params.append(param)
        else:
            backbone_params.append(param)

    params = [
        {"params": backbone_params, "lr": config["backbone_lr"]},
        {"params": head_params, "lr": config["head_lr"]},
    ]

    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(
            params,
            weight_decay=config["weight_decay"]
        )

    else:
        raise ValueError(config["optimizer"])


def build_scheduler(optimizer, config):
    scheduler_name = config["scheduler"]

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
            eta_min=config.get("eta_min", 0.0)
        )

    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 10),
            gamma=config.get("gamma", 0.3)
        )

    elif scheduler_name == "none":
        return None

    else:
        raise ValueError(scheduler_name)


@torch.no_grad()
def evaluate(model, loader, criterion, desc="Eval"):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def make_config_name(config):
    return (
        f"{config['model_name']}_"
        f"{config['optimizer']}_"
        f"ep{config['epochs']}_"
        f"bb{config['backbone_lr']}_"
        f"head{config['head_lr']}_"
        f"wd{config['weight_decay']}_"
        f"ls{config['label_smoothing']}_"
        f"{config['scheduler']}"
    )


def train_one_config(config, save_dir, wandb_project):
    wandb.init(
        project=wandb_project,
        name=config["name"],
        config=config,
        reinit=True,
    )

    model = build_model(
        model_name=config["model_name"],
        num_classes=102
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    best = {
        "best_val_acc": 0.0,
        "best_val_loss": None,
        "test_acc_at_best_val": 0.0,
        "test_loss_at_best_val": None,
        "best_epoch": -1,
        "config": copy.deepcopy(config)
    }

    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"{config['name']} | Epoch {epoch}/{config['epochs']}",
            leave=False
        )

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc": f"{correct / total:.4f}",
                "bb_lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "head_lr": f"{optimizer.param_groups[1]['lr']:.1e}",
            })

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            desc="Val"
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            desc="Test"
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }

        history.append(epoch_result)
        wandb.log(epoch_result, step=epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f} | "
            f"Test {test_acc:.4f}"
        )

        if val_acc > best["best_val_acc"]:
            best["best_val_acc"] = val_acc
            best["best_val_loss"] = val_loss
            best["test_acc_at_best_val"] = test_acc
            best["test_loss_at_best_val"] = test_loss
            best["best_epoch"] = epoch

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_val_model.pth")
            )

            with open(os.path.join(save_dir, "best_val.txt"), "w") as f:
                f.write(json.dumps(best, indent=4))

            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["test_acc_at_best_val"] = test_acc
            wandb.run.summary["test_loss_at_best_val"] = test_loss
            wandb.run.summary["best_epoch"] = epoch

        if scheduler is not None:
            scheduler.step()

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    wandb.finish()
    return best


def build_grid(cfg):
    search = cfg["search"]

    keys = [
        "model_name",
        "optimizer",
        "epochs",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "label_smoothing",
        "scheduler",
    ]

    values = [search[k] for k in keys]

    grid = []

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))

        config["eta_min"] = cfg.get("eta_min", 0.0)
        config["step_size"] = cfg.get("step_size", 10)
        config["gamma"] = cfg.get("gamma", 0.3)

        config["name"] = make_config_name(config)

        grid.append(config)

    return grid


def run_search(yaml_path):
    cfg = load_yaml(yaml_path)

    root_save_dir = cfg["root_save_dir"]
    os.makedirs(root_save_dir, exist_ok=True)

    grid = build_grid(cfg)

    global_best = {
        "best_val_acc": 0.0,
        "best_val_loss": None,
        "test_acc_at_best_val": 0.0,
        "test_loss_at_best_val": None,
        "best_epoch": -1,
        "config": None
    }

    all_results = []

    pbar = tqdm(
        enumerate(grid),
        total=len(grid),
        desc="Transformer Val Search"
    )

    for idx, config in pbar:
        print("=" * 100)
        print(f"Config {idx + 1}/{len(grid)}")
        print(json.dumps(config, indent=4))

        save_dir = os.path.join(root_save_dir, config["name"])
        os.makedirs(save_dir, exist_ok=True)

        best = train_one_config(
            config=config,
            save_dir=save_dir,
            wandb_project=cfg["wandb_project"]
        )

        all_results.append(best)

        if best["best_val_acc"] > global_best["best_val_acc"]:
            global_best = best

            with open(os.path.join(root_save_dir, "GLOBAL_BEST_VAL.txt"), "w") as f:
                f.write(json.dumps(global_best, indent=4))

        with open(os.path.join(root_save_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)

        pbar.set_postfix({
            "best_val": f"{global_best['best_val_acc']:.4f}",
            "test@bestval": f"{global_best['test_acc_at_best_val']:.4f}",
            "epoch": global_best["best_epoch"],
        })

    print("Search finished.")
    print(json.dumps(global_best, indent=4))


if __name__ == "__main__":
    run_search("train_transformer.yml")