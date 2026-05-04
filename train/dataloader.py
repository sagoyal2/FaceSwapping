from pathlib import Path
import torch
import cv2
import random
from torch.utils.data import Dataset, Sampler


def build_identity_index(root_dir):
    root = Path(root_dir)

    image_paths = [] # [path]
    identity_name_to_id = {} # (name, id)
    identity_to_indices = {} # (id, [indices]) -> indices of images for this identity

    folders = [p for p in sorted(root.iterdir()) if p.is_dir()]

    index_counter = 0
    for identity_id, folder in enumerate(folders):
        paths = sorted(folder.glob("*.jpg"))

        identity_name_to_id[folder.name] = identity_id
        identity_to_indices[identity_id] = []

        for path in paths:
            idx = index_counter

            image_paths.append(path)
            identity_to_indices[identity_id].append(idx)
            index_counter += 1

    return image_paths, identity_name_to_id, identity_to_indices


def _to_chw_tensor(img):
    """OpenCV/numpy images are HWC uint8; training expects float CHW in [0, 1]."""
    t = torch.from_numpy(img).float().div_(255.0)
    return t.permute(2, 0, 1)  # HWC -> CHW

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def normalize_imagenet_chw(img_chw: torch.Tensor) -> torch.Tensor:
    return (img_chw - IMAGENET_MEAN) / IMAGENET_STD

def unnormalize_imagenet_chw(img_chw: torch.Tensor) -> torch.Tensor:
    return img_chw * IMAGENET_STD + IMAGENET_MEAN

ARCFACE_MEAN = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)
ARCFACE_STD = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(3, 1, 1)

def normalize_arcface_chw(img_chw: torch.Tensor) -> torch.Tensor:
    return (img_chw - ARCFACE_MEAN) / ARCFACE_STD

def unnormalize_arcface_chw(img_chw: torch.Tensor) -> torch.Tensor:
    return img_chw * ARCFACE_STD + ARCFACE_MEAN


class FacePairDataset(Dataset):
    def __init__(self, root_dir):
        (
            self.image_paths,
            self.identity_name_to_id,
            self.identity_to_indices,
        ) = build_identity_index(root_dir)

        self.all_identities = self.identity_to_indices.keys()

        # Only use identities with at least 2 images
        self.positive_identities = [
            identity
            for identity, indices in self.identity_to_indices.items()
            if len(indices) >= 2
        ]

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __getitem__(self, pair_info):
        idx_a, idx_b, label = pair_info

        img_a = normalize_imagenet_chw(_to_chw_tensor(self.load_image(idx_a)))
        # img_b = normalize_imagenet_chw(_to_chw_tensor(self.load_image(idx_b)))
        img_b = normalize_arcface_chw(_to_chw_tensor(self.load_image(idx_b)))

        images = torch.stack([img_a, img_b], dim=0)
        return {
            "images": images,  # [2, C, H, W]
            "labels": torch.tensor(label, dtype=torch.float32),
        }



"""
Following the SimSwap Paper, Section 4. 
We want to alternate between positive (same identity) and negative (different identity) pairs for each batch.
"""
class AlternatingPairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_batches_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.identity_to_indices = dataset.identity_to_indices
        self.positive_identities = dataset.positive_identities
        self.all_identities = dataset.all_identities

    def __len__(self):
        return self.num_batches_per_epoch

    def sample_positive_pair(self):
        identity = random.choice(self.positive_identities)
        indices = self.identity_to_indices[identity]

        idx_a, idx_b = random.sample(indices, 2)

        return (idx_a, idx_b, 1)

    def sample_negative_pair(self):
        identity_a, identity_b = random.sample(self.all_identities, 2)

        idx_a = random.choice(self.identity_to_indices[identity_a])
        idx_b = random.choice(self.identity_to_indices[identity_b])

        return (idx_a, idx_b, 0)

    def __iter__(self):
        for batch_idx in range(self.num_batches_per_epoch):
            is_positive_batch = (batch_idx % 4 == 0)
            batch = []

            for _ in range(self.batch_size):
                if is_positive_batch:
                    pair = self.sample_positive_pair()
                else:
                    pair = self.sample_negative_pair()

                batch.append(pair)

            yield batch