import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple
from models.simswap import SimSwap


from preprocess.face_parsing import FaceParsingONNX

def load_models(checkpoint_path: str, device: torch.device) -> SimSwap:
    simswap = SimSwap()

    state = torch.load(checkpoint_path, map_location=device)
    simswap.generator.load_state_dict(state)

    simswap.generator.eval()
    simswap.arcface_model.eval()

    return simswap


# Face Parsing Model
SEGMENTATION_MODEL = Path("/home/ubuntu/DemoFaceSwappingData/segmentation_model/resnet18.onnx")
engine = FaceParsingONNX(SEGMENTATION_MODEL)

def get_inner_face_region_mask(image: np.ndarray) -> np.ndarray:
    """
    Get the inner face region mask of the image.

    Args:
        image: (H, W, 3) uint8 BGR

    Returns:
        face_region_mask: (H, W) [0, 1]
    """
    segmentation_mask = engine.predict(image)

    inner_face_region_classes = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
    inner_face_region_mask = np.isin(segmentation_mask, inner_face_region_classes)

    return inner_face_region_mask.astype(np.uint8)


def composite_back(target_fake_cropped_aligned_224: torch.Tensor, target_image: np.ndarray | str | Path, target_M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite the generated image back to the original image, with some blending.

    Reference: https://github.com/deepinsight/insightface/blob/f8613d444c6c266e8ff2fb29676a0a1cba6ee7a1/python-package/insightface/model_zoo/inswapper.py#L46

    Args:
        target_fake_cropped_aligned_224: (224, 224, 3) uint8 BGR
        target_image: The original image (H, W, 3) or the path to the original image.
        target_M: (3, ) The transformation matrix.

    Returns:
        target_fake_unwarped: The unwarped generated image (H, W, 3)
        target_mask_unwarped: The mask of the unwarped generated image (H, W, 1)
        square_mask_unwarped: The mask of the unwarped generated image (H, W, 1)
        blended_square: The unwarped imaged composited into original frame (H, W, 3)
        blended_selection: The unwarped imaged with inner face selection composited into original frame(H, W, 3)
        blended_selection_corrected: The unwarped imaged with inner face selection composited into original frame with color correction (H, W, 3)
    """

    # Unwarp
    if isinstance(target_image, (str, Path)):
        target_image = cv2.imread(str(target_image))
    H, W, _ = target_image.shape

    inverse_M = cv2.invertAffineTransform(target_M)
    target_fake_unwarped = cv2.warpAffine(target_fake_cropped_aligned_224, inverse_M, (W, H), borderValue=0.0) # (H, W, 3)

    # Create Mask for Blending (following from insightface inswapper paste_back)
    square_mask = np.full((224, 224), 255, dtype=np.float32)
    square_mask_unwarped = cv2.warpAffine(square_mask, inverse_M, (W, H), borderValue=0.0)

    square_mask_unwarped[square_mask_unwarped > 20] = 255 # create clean mask region
    inner_face_region_mask = get_inner_face_region_mask(target_image)

    # Only transfer over the the inner face region (intersection of two masks)
    target_mask_unwarped = square_mask_unwarped * inner_face_region_mask

    ys, xs = np.where(target_mask_unwarped == 255)

    mask_h = ys.max() - ys.min()
    mask_w = xs.max() - xs.min()
    mask_size = int(np.sqrt(mask_h * mask_w)) # get a proxy for face size

    erode_k = max(mask_size // 10, 10)
    target_mask_unwarped = cv2.erode(target_mask_unwarped, np.ones((erode_k, erode_k), np.uint8), iterations=1)

    blur_k = max(mask_size // 20, 5)
    blur_size = (2 * blur_k + 1, 2 * blur_k + 1)
    target_mask_unwarped = cv2.GaussianBlur(target_mask_unwarped, blur_size, 0)

    # Additional Color Correction (match means)
    target_mask_unwarped = (target_mask_unwarped / 255.0)[..., None] # make mask float between [0,1] and shape (H, W, 1)
    target_fake_selection = target_mask_unwarped * target_fake_unwarped.astype(np.float32)
    target_image_selection = target_mask_unwarped * target_image.astype(np.float32)

    w_sum = np.sum(target_mask_unwarped)
    new_mean = np.sum(target_fake_selection, axis=(0, 1)) / w_sum
    old_mean = np.sum(target_image_selection, axis=(0, 1)) / w_sum
    color_shift = old_mean - new_mean  # shift fake toward original region mean
    target_fake_selection_corrected = target_fake_selection + target_mask_unwarped * color_shift

    # Compositing 
    square_mask_unwarped = (square_mask_unwarped/255.0)[..., None] # make mask float between [0,1] and shape (H, W, 1)
    blended_square = square_mask_unwarped * target_fake_unwarped.astype(np.float32) + (1.0 - square_mask_unwarped) * target_image.astype(np.float32)
    blended_square = np.clip(blended_square, 0, 255).astype(np.uint8)

    blended_selection = target_fake_selection + (1.0 - target_mask_unwarped) * target_image.astype(np.float32)
    blended_selection =  np.clip(blended_selection, 0, 255).astype(np.uint8)

    blended_selection_corrected = target_fake_selection_corrected + (1.0 - target_mask_unwarped) * target_image.astype(np.float32)
    blended_selection_corrected = np.clip(blended_selection_corrected, 0, 255).astype(np.uint8)

    return {
        "target_fake_unwarped": target_fake_unwarped,
        "target_mask_unwarped": np.clip(target_mask_unwarped * 255.0, 0, 255).astype(np.uint8),
        "square_mask_unwarped": np.clip(square_mask_unwarped * 255.0, 0, 255).astype(np.uint8),
        "blended_square": blended_square,
        "blended_selection": blended_selection,
        "blended_selection_corrected": blended_selection_corrected
    }

