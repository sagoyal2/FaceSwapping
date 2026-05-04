"""
Inference for Image-to-Image (I2I) face swapping.

Example:

python inference/inference_I2V.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt

python inference/inference_I2V.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--target_video_path /home/ubuntu/DemoFaceSwapping/evaluation/vhfq_videos/vfhq_1_448_448.mov \
--source_image_path /home/ubuntu/DemoFaceSwappingData/lfw_funneled/Prince_Charles/Prince_Charles_0002.jpg \
--result_folder_path /home/ubuntu/DemoFaceSwapping/scratch/test_Prince_Charles_0002_to_vfhq_1_448_448

python inference/inference_I2V.py \
--checkpoint /home/ubuntu/DemoFaceSwapping/runs/simswap/checkpoints/run_20260504-030233/generator_iter_1649.pt \
--target_video_path /home/ubuntu/DemoFaceSwapping/evaluation/celebvhq_videos/celebvhq_3_448_448.mov

"""

import argparse
from pathlib import Path

import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from inference.inference_base import load_models, composite_back

from evaluation.utils import image_to_tensor, crop_and_align_face
from train.dataloader import normalize_imagenet_chw, unnormalize_imagenet_chw, normalize_arcface_chw

import imageio.v2 as imageio

def main(checkpoint_path: str, target_video_path: str, source_image_path: str, result_folder_path: str):
    device = torch.device("cuda")
    simswap = load_models(checkpoint_path, device)

    # Preprocess source image ONLY
    source_image_cropped_aligned_224, _ = crop_and_align_face(source_image_path)
    source = image_to_tensor(source_image_cropped_aligned_224)
    
    # Extract ArcFace embedding from source image
    with torch.no_grad():
        source = normalize_arcface_chw(source).unsqueeze(0).to(device)
        source_112 = F.interpolate(source, size=(112, 112), mode="bilinear")
        emb = simswap.arcface_model(source_112)
        emb = F.normalize(emb, p=2, dim=1)

    # Start processing the target video
    cap = cv2.VideoCapture(target_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames = []
    out_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess target frame
        raw_frames.append(frame)
        target_image_cropped_aligned_224, target_M = crop_and_align_face(frame)
        target = image_to_tensor(target_image_cropped_aligned_224)

        # Run Inference
        with torch.no_grad():

            # Normalize, Generate, Unnormalize target frame
            target = normalize_imagenet_chw(target).unsqueeze(0).to(device)
            target_fake_cropped_aligned_224 = simswap.generator(target, emb)
            target_fake_cropped_aligned_224 = target_fake_cropped_aligned_224.detach().cpu()
            target_fake_cropped_aligned_224 = unnormalize_imagenet_chw(target_fake_cropped_aligned_224)

        # Convert the generated image to uint8 BGR (required for cv2 warpAffine)
        target_fake_cropped_aligned_224 = target_fake_cropped_aligned_224.permute(0, 2, 3, 1)[0].numpy()
        target_fake_cropped_aligned_224 = np.clip(target_fake_cropped_aligned_224 * 255.0, 0, 255).astype(np.uint8)
        target_fake_cropped_aligned_224 = cv2.cvtColor(target_fake_cropped_aligned_224, cv2.COLOR_RGB2BGR)

        composite_back_results = composite_back(target_fake_cropped_aligned_224, frame, target_M)
        out_frames.append(composite_back_results["blended_selection_corrected"])

    cap.release()

    # Save the output video
    out_path = os.path.join(result_folder_path, "blended_selection_corrected_video.mp4")
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in out_frames:
        writer.write(f)
    writer.release()

    # Save the original and output as a GIF
    rgb_combined_frames = []
    for raw, out in zip(raw_frames, out_frames):
        raw = cv2.resize(raw, (250, 250))
        out = cv2.resize(out, (250, 250))
        rgb_combined_frames.append(cv2.hconcat([cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), cv2.cvtColor(out, cv2.COLOR_BGR2RGB)]))
    imageio.mimsave(os.path.join(result_folder_path, "original_and_output.gif"), rgb_combined_frames[::2], duration=0.04, loop=0) 

    print(f"Resolution: {w}x{h}")
    print(f"FPS: {fps}")
    print(f"Total number of frames: {len(out_frames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=False)
    parser.add_argument("--source_image_path", type=str, required=False)
    parser.add_argument("--result_folder_path", type=str, required=False)

    args = parser.parse_args()

    if args.target_video_path is None and args.source_image_path is None and args.result_folder_path is None:
        # Sample a random source 
        lfw_root = Path("/home/ubuntu/DemoFaceSwappingData/lfw_funneled")
        identity_dirs = [p for p in lfw_root.iterdir() if p.is_dir()]
        source_identity_dir = random.choice(identity_dirs)
        random_source_path = random.choice(list(source_identity_dir.glob("*.jpg")))

        # Load example target video
        target_video_path = Path("/home/ubuntu/DemoFaceSwapping/evaluation/vhfq_videos/vfhq_1_448_448.mov")

        result_folder_path = Path(f"/home/ubuntu/DemoFaceSwapping/scratch/test_{random_source_path.stem}_to_{target_video_path.stem}")
        result_folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving results to {result_folder_path}")
        cv2.imwrite(os.path.join(result_folder_path, f"{random_source_path.stem}.jpg"), cv2.imread(str(random_source_path)))

        main(args.checkpoint, str(target_video_path), str(random_source_path), str(result_folder_path))

    elif args.target_video_path is not None and args.source_image_path is None and args.result_folder_path is None:

        # Sample a random source 
        lfw_root = Path("/home/ubuntu/DemoFaceSwappingData/lfw_funneled")
        identity_dirs = [p for p in lfw_root.iterdir() if p.is_dir()]
        source_identity_dir = random.choice(identity_dirs)
        random_source_path = random.choice(list(source_identity_dir.glob("*.jpg")))

        target_video_path = Path(args.target_video_path)
        result_folder_path = Path(f"/home/ubuntu/DemoFaceSwapping/scratch/test_{random_source_path.stem}_to_{target_video_path.stem}")
        result_folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving results to {result_folder_path}")
        cv2.imwrite(os.path.join(result_folder_path, f"{random_source_path.stem}.jpg"), cv2.imread(str(random_source_path)))

        main(args.checkpoint, str(target_video_path), str(random_source_path), str(result_folder_path))
    else:
        result_folder_path = Path(args.result_folder_path)
        result_folder_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving results to {result_folder_path}")
        cv2.imwrite(os.path.join(args.result_folder_path, f"{Path(args.source_image_path).stem}.jpg"), cv2.imread(args.source_image_path))
        main(args.checkpoint, args.target_video_path, args.source_image_path, args.result_folder_path)

