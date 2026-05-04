from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from train.dataloader import _to_chw_tensor
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from typing import Tuple

from train.dataloader import normalize_imagenet_chw, unnormalize_imagenet_chw

def generate_image(simswap, target_image: torch.Tensor, source_image: torch.Tensor) -> torch.Tensor:
    """
    Generate a deepfake image from a target image and a source image.

    Args:
        simswap: The SimSwap model.
        target_image: The target image (B, C, H, W).
        source_image: The source image (B, C, H, W).

    Returns:
        target_fake: The generated image (B, C, H, W).
    """

    source_112 = F.interpolate(source_image, size=(112, 112), mode="bilinear")
    source_identity_embedding = simswap.arcface_model(source_112) # (B, 512)
    source_identity_embedding = F.normalize(source_identity_embedding, p=2, dim=1)
    target_fake = simswap.generator(target_image, source_identity_embedding) # (B, C, H, W)   

    return target_fake


def image_to_tensor(image: np.ndarray | str | Path) -> torch.Tensor:
    """
    Convert an image path or array to a tensor of size (3, 224, 224).

    Args:
        image: The image (H, W, 3).
        image_path: The path to the image.

    Returns:
        The image tensor (C, H, W).
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _to_chw_tensor(image)
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
    return image

# Initialize Models
# Face Detection Model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
def crop_and_align_face(image: np.ndarray | str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop and align a face from an image.

    Args:
        image: The image (H, W, 3).
        image_path: The path to the image.

    Returns:
        image_cropped_aligned_224: the cropped and aligned face (H, W, 3).
        M: the transformation matrix (3, 3).
    """
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
    faces = app.get(image)
    
    face = faces[0]
    kps = face.kps.astype("float32")
    image_cropped_aligned_224, M = face_align.norm_crop2(image, kps, image_size=224, mode="arcface")
    return image_cropped_aligned_224, M