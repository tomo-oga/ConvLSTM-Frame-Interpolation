import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

random.seed(42)


class CustomTransform:
    def __init__(self, crop_size=(256, 256)):
        self.crop_size = crop_size

    def __call__(self, sequence):
        # Apply the same crop to all images in the sequence
        sequence = self.random_crop(sequence)

        # Randomly flip the entire sequence
        if random.random() > 0.5:
            sequence = [cv2.flip(img, 1) for img in sequence]

        # Randomly reverse the sequence
        if random.random() > 0.5:
            sequence.reverse()

        return sequence

    def random_crop(self, sequence):
        img_h, img_w, _ = sequence[0].shape
        crop_h, crop_w = self.crop_size
        top = random.randint(0, img_h - crop_h)
        left = random.randint(0, img_w - crop_w)
        return [img[top : top + crop_h, left : left + crop_w] for img in sequence]


class Vimeo90kDatasetTrain(Dataset):
    def __init__(self, transform=None, num_examples=None):
        self.root_dir = "vimeo_septuplet/sequences"
        self.transform = transform
        self.sequence_list = self.read_list_file(
            "vimeo_septuplet/sep_trainlist.txt", num_examples
        )

    def read_list_file(self, file_path, num_examples):
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Limit the number of examples if num_examples is specified
        if num_examples is not None and num_examples < len(lines):
            lines = lines[:num_examples]

        return [line.strip() for line in lines]

    def __len__(self):
        return len(self.sequence_list)

    def __getitem__(self, idx):
        sequence_path = os.path.join(self.root_dir, self.sequence_list[idx])
        sequence = [
            cv2.imread(os.path.join(sequence_path, f"im{i+1}.png")) for i in range(5)
        ]
        if self.transform is not None:
            sequence = self.transform(sequence)
        sequence = np.array(sequence).astype("float32")
        sequence = sequence / 255 * 2 - 1

        x_set = np.append(sequence[:2], sequence[3:], axis=0)
        y_set = sequence[2]
        return x_set, y_set
