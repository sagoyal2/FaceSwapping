"""
Inference for Image-to-Image (I2I) face swapping. 
WARNING: this is hack and only meant for some evaluation purposes, is not latest or up to date with I2I and I2V.

Example:
python inference/inference_timeline_I2I.py
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple
from models.simswap import SimSwap

from evaluation.utils import image_to_tensor, crop_and_align_face
from train.dataloader import normalize_imagenet_chw, unnormalize_imagenet_chw, normalize_arcface_chw

from tqdm import tqdm

def load_models(checkpoint_path: str, device: torch.device) -> SimSwap:
    simswap = SimSwap()

    state = torch.load(checkpoint_path, map_location=device)
    simswap.generator.load_state_dict(state)

    simswap.generator.eval()
    simswap.arcface_model.eval()

    return simswap


def composite_back(target_fake_cropped_aligned_224: torch.Tensor, target_image_path: str, target_M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite the generated image back to the original image, with some blending.

    Reference: https://github.com/deepinsight/insightface/blob/f8613d444c6c266e8ff2fb29676a0a1cba6ee7a1/python-package/insightface/model_zoo/inswapper.py#L46

    Args:
        target_fake_cropped_aligned_224: (224, 224, 3) uint8 BGR
        target_image_path: The path to the original image.
        target_M: (3, ) The transformation matrix.

    Returns:
        target_fake_unwarped: The unwarped generated image (H, W, 3)
        target_mask_unwarped: The mask of the unwarped generated image (H, W, 1)
        blended: The unwarped imaged composited into original frame(H, W, 3)
    """

    # Unwarp
    target_image = cv2.imread(target_image_path)
    H, W, _ = target_image.shape

    inverse_M = cv2.invertAffineTransform(target_M)
    target_fake_unwarped = cv2.warpAffine(target_fake_cropped_aligned_224, inverse_M, (W, H), borderValue=0.0) # (H, W, 3)

    # Blend (following from insightface inswapper paste_back)
    target_mask = np.full((224, 224), 255, dtype=np.float32)
    target_mask_unwarped = cv2.warpAffine(target_mask, inverse_M, (W, H), borderValue=0.0)

    target_mask_unwarped[target_mask_unwarped > 20] = 255 # create clean mask region

    ys, xs = np.where(target_mask_unwarped == 255)

    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = int(np.sqrt(mask_h * mask_w)) # get a proxy for face size

    erode_k = max(mask_size // 10, 10)
    target_mask_unwarped = cv2.erode(target_mask_unwarped, np.ones((erode_k, erode_k), np.uint8), iterations=1)

    blur_k = max(mask_size // 20, 5)
    blur_size = (2 * blur_k + 1, 2 * blur_k + 1)
    target_mask_unwarped = cv2.GaussianBlur(target_mask_unwarped, blur_size, 0)

    target_mask_unwarped = (target_mask_unwarped / 255.0)[..., None] # make mask float between [0,1] and shape (H, W, 1)

    blended = target_mask_unwarped * target_fake_unwarped.astype(np.float32) + (1.0 - target_mask_unwarped) * target_image.astype(np.float32)
    blended =  np.clip(blended, 0, 255).astype(np.uint8)

    return target_fake_unwarped, np.clip(target_mask_unwarped * 255.0, 0, 255).astype(np.uint8), blended


def save_image(
    checkpoint_path: Path,
    target: torch.Tensor,
    source: torch.Tensor,
    target_image_path: str,
    target_M: np.ndarray,
    out_dir: Path,
    suffix: str,
):
    device = torch.device("cuda")
    simswap = load_models(str(checkpoint_path), device)

    # Run Inference
    with torch.no_grad():

        # Normalize the target and source images
        target = normalize_imagenet_chw(target).unsqueeze(0).to(device) # (1, 3, 224, 224)
        # source = normalize_imagenet_chw(source).unsqueeze(0).to(device) # (1, 3, 224, 224)
        source = normalize_arcface_chw(source).unsqueeze(0).to(device) # (1, 3, 224, 224)
        
        # Generate the image
        source_112 = F.interpolate(source, size=(112, 112), mode="bilinear")
        emb = simswap.arcface_model(source_112)
        emb = F.normalize(emb, p=2, dim=1)
        target_fake_cropped_aligned_224 = simswap.generator(target, emb) # (1, 3, 224, 224)

        # Unnormalize the generated image
        target_fake_cropped_aligned_224 = target_fake_cropped_aligned_224.detach().cpu()
        target_fake_cropped_aligned_224 = unnormalize_imagenet_chw(target_fake_cropped_aligned_224)

    # Convert the generated image to uint8 BGR (required for cv2 warpAffine)
    target_fake_cropped_aligned_224 = target_fake_cropped_aligned_224.permute(0, 2, 3, 1)[0].numpy() # (224, 224, 3)
    target_fake_cropped_aligned_224 = np.clip(target_fake_cropped_aligned_224 * 255.0, 0, 255).astype(np.uint8) # [0,1] -> [0,255]
    target_fake_cropped_aligned_224 = cv2.cvtColor(target_fake_cropped_aligned_224, cv2.COLOR_RGB2BGR)
    
    # Warp the generated image back to the original image
    target_fake_unwarped, target_mask_unwarped, blended = composite_back(target_fake_cropped_aligned_224, target_image_path, target_M)

    cv2.imwrite(str(out_dir / f"target_fake_cropped_aligned_224_{suffix}.png"), target_fake_cropped_aligned_224)

if __name__ == "__main__":

    checkpoints_dir = Path("/home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233")

    # Sample a random source and target identity
    lfw_root = Path("/home/ubuntu/DemoFaceSwappingData/lfw_funneled")
    identity_dirs = [p for p in lfw_root.iterdir() if p.is_dir()]
    source_identity_dir1, source_identity_dir2, target_identity_dir1, target_identity_dir2 = random.sample(identity_dirs, 4)

    srs_dirs = [source_identity_dir1, source_identity_dir2]
    tgt_dirs = [target_identity_dir1, target_identity_dir2]

    for src_dir in srs_dirs:
        for tgt_dir in tgt_dirs:
            random_source_path = random.choice(list(src_dir.glob("*.jpg")))
            random_target_path = random.choice(list(tgt_dir.glob("*.jpg")))

            save_path = Path(f"/home/ubuntu/DemoFaceSwapping/scratch/test_{random_source_path.stem}_to_{random_target_path.stem}")
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving results to {save_path}")


            # Preprocess Images
            target_image_cropped_aligned_224, target_M = crop_and_align_face(str(random_target_path))
            target = image_to_tensor(target_image_cropped_aligned_224)
            cv2.imwrite(str(save_path / f"{random_target_path.stem}_target_image_cropped_aligned_224.png"), target_image_cropped_aligned_224)

            source_image_cropped_aligned_224, _ = crop_and_align_face(str(random_source_path))
            source = image_to_tensor(source_image_cropped_aligned_224)
            cv2.imwrite(str(save_path / f"{random_source_path.stem}_source_image_cropped_aligned_224.png"), source_image_cropped_aligned_224)

            # Run Inference on checkpoints
            for checkpoint in tqdm(sorted(checkpoints_dir.glob("*.pt"))):
                save_image(checkpoint, target, source, str(random_target_path), target_M, save_path, checkpoint.stem)
