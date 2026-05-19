from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Flowers102Dataset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_transform_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3)
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


root_dir = "/home/cheng/HW/data"

train_dataset = Flowers102Dataset(root_dir, split="train", transform=train_transform)
val_dataset = Flowers102Dataset(root_dir, split="val", transform=test_transform)
test_dataset = Flowers102Dataset(root_dir, split="test", transform=test_transform)

train_dataset_augment = Flowers102Dataset(root_dir, split="train", transform=train_transform_augment)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

train_loader_augment = DataLoader(
    train_dataset_augment,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4
)

