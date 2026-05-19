import os
from PIL import Image

import scipy.io
import torch
from torch.utils.data import Dataset


class Flowers102Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        root_dir: 数据集根目录，例如 "./data/flowers102"
        split: "train", "val", "test"
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "jpg")
        self.transform = transform

        labels_mat = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))
        setid_mat = scipy.io.loadmat(os.path.join(root_dir, "setid.mat"))

        labels = labels_mat["labels"][0]  # shape: (8189,)

        if split == "train":
            ids = setid_mat["trnid"][0]
        elif split == "val":
            ids = setid_mat["valid"][0]
        elif split == "test":
            ids = setid_mat["tstid"][0]
        else:
            raise ValueError("split must be one of: train, val, test")

        self.samples = []

        for img_id in ids:
            img_name = f"image_{img_id:05d}.jpg"
            img_path = os.path.join(self.image_dir, img_name)

            label = labels[img_id - 1] - 1  # 1~102 -> 0~101

            self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label