"""
Inference for Image-to-Image (I2I) face swapping.

Example:
python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--target_image_path /home/ubuntu/DemoFaceSwapping/evaluation/snl_images/snl_1.png \
--source_image_path /home/ubuntu/DemoFaceSwapping/evaluation/snl_images/snl_2.png \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_snl

python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt


python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--source_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Tamika_Catchings/Tamika_Catchings_0001.jpg \
--target_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Michelle_Hofland/Michelle_Hofland_0001.jpg \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_Tamika_Catchings_0001_to_Michelle_Hofland_0001


python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--source_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Prince_Charles/Prince_Charles_0002.jpg \
--target_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Hermando_Harton/Hermando_Harton_0001.jpg \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_Prince_Charles_0002_to_Hermando_Harton_0001

python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--source_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Jose_Miguel_Aleman/Jose_Miguel_Aleman_0001.jpg \
--target_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Hikmat_al-Azzawi/Hikmat_al-Azzawi_0001.jpg \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_Jose_Miguel_Aleman_0001_to_Hikmat_al-Azzawi_0001


python inference/inference_I2I.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--source_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Grady_Irvin_Jr/Grady_Irvin_Jr_0002.jpg \
--target_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Jennifer_Reilly/Jennifer_Reilly_0001.jpg \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_Grady_Irvin_Jr_0002_to_Jennifer_Reilly_0001

"""

import argparse
from pathlib import Path

import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from typing import Tuple
from inference.inference_base import load_models, composite_back

from evaluation.utils import image_to_tensor, crop_and_align_face
from train.dataloader import normalize_imagenet_chw, unnormalize_imagenet_chw, normalize_arcface_chw

def main(checkpoint_path: str, target_image_path: str, source_image_path: str, result_folder_path: str):
    
    # Load Models
    device = torch.device("cuda")
    simswap = load_models(checkpoint_path, device)

    # Preprocess Images
    target_image_cropped_aligned_224, target_M = crop_and_align_face(target_image_path)
    target = image_to_tensor(target_image_cropped_aligned_224)

    source_image_cropped_aligned_224, _ = crop_and_align_face(source_image_path)
    source = image_to_tensor(source_image_cropped_aligned_224)

    # Run Inference
    with torch.no_grad():

        # Normalize the target and source images
        target = normalize_imagenet_chw(target).unsqueeze(0).to(device) # (1, 3, 224, 224)
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
    composite_back_results = composite_back(target_fake_cropped_aligned_224, target_image_path, target_M)

    # Save all images
    cv2.imwrite(os.path.join(result_folder_path, "source_image_cropped_aligned_224.png"), source_image_cropped_aligned_224)
    cv2.imwrite(os.path.join(result_folder_path, "target_image_cropped_aligned_224.png"), target_image_cropped_aligned_224)
    cv2.imwrite(os.path.join(result_folder_path, "target_fake_cropped_aligned_224.png"), target_fake_cropped_aligned_224)
    cv2.imwrite(os.path.join(result_folder_path, "target_fake_unwarped.png"), composite_back_results["target_fake_unwarped"])
    cv2.imwrite(os.path.join(result_folder_path, "target_mask_unwarped.png"), composite_back_results["target_mask_unwarped"])
    cv2.imwrite(os.path.join(result_folder_path, "square_mask_unwarped.png"), composite_back_results["square_mask_unwarped"])
    cv2.imwrite(os.path.join(result_folder_path, "blended_square.png"), composite_back_results["blended_square"])
    cv2.imwrite(os.path.join(result_folder_path, "blended_selection.png"), composite_back_results["blended_selection"])
    cv2.imwrite(os.path.join(result_folder_path, "blended_selection_corrected.png"), composite_back_results["blended_selection_corrected"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target_image_path", type=str, required=False)
    parser.add_argument("--source_image_path", type=str, required=False)
    parser.add_argument("--result_folder_path", type=str, required=False)

    args = parser.parse_args()

    if args.target_image_path is None and args.source_image_path is None and args.result_folder_path is None:
        # Sample a random source and target identity
        lfw_root = Path("/home/ubuntu/DemoFaceSwappingData/lfw_funneled")
        identity_dirs = [p for p in lfw_root.iterdir() if p.is_dir()]
        source_identity_dir, target_identity_dir = random.sample(identity_dirs, 2)

        random_source_path = random.choice(list(source_identity_dir.glob("*.jpg")))
        random_target_path = random.choice(list(target_identity_dir.glob("*.jpg")))

        result_folder_path = Path(f"/home/ubuntu/DemoFaceSwapping/scratch/test_{random_source_path.stem}_to_{random_target_path.stem}")
        result_folder_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving results to {result_folder_path}")
        cv2.imwrite(os.path.join(result_folder_path, f"{random_source_path.stem}.jpg"), cv2.imread(str(random_source_path)))
        cv2.imwrite(os.path.join(result_folder_path, f"{random_target_path.stem}.jpg"), cv2.imread(str(random_target_path)))

        main(args.checkpoint, str(random_target_path), str(random_source_path), str(result_folder_path))
 
    else:
        os.makedirs(args.result_folder_path, exist_ok=True)
        cv2.imwrite(os.path.join(args.result_folder_path, f"{Path(args.source_image_path).stem}.jpg"), cv2.imread(args.source_image_path))
        cv2.imwrite(os.path.join(args.result_folder_path, f"{Path(args.target_image_path).stem}.jpg"), cv2.imread(args.target_image_path))
        main(args.checkpoint, args.target_image_path, args.source_image_path, args.result_folder_path)
