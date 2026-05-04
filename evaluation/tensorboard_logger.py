from pathlib import Path
import random

import torch

from torchvision.utils import make_grid

from train.dataloader import normalize_imagenet_chw, unnormalize_imagenet_chw, normalize_arcface_chw, unnormalize_arcface_chw
from evaluation.utils import crop_and_align_face, generate_image, image_to_tensor

# Load Legend Image
LEGEND_IMAGE_PATH = Path("/home/ubuntu/DemoFaceSwapping/evaluation/source_results_target.png")
LEGEND_IMAGE = image_to_tensor(LEGEND_IMAGE_PATH)

# Load Evaluation from Dataset (indices into dataset.image_paths)
CUMULATIVE_SAME_IDENTITY_IMAGES = [6, 7]  
CUMULATIVE_DIFFERENT_IDENTITY_SOURCE = [0, 25]
CUMULATIVE_DIFFERENT_IDENTITY_TARGET = [200, 400]

# Load Evaluation from Disk Images
EVAL_LFW_FOLDER_PATH = Path("/home/ubuntu/DemoFaceSwapping/evaluation/lfw_images")
EVAL_LFW_IMAGES = [image_to_tensor(crop_and_align_face(image_path)[0]) for image_path in EVAL_LFW_FOLDER_PATH.glob("*.jpg")]

EVAL_CELEBVHQ_FOLDER_PATH = Path("/home/ubuntu/DemoFaceSwapping/evaluation/celebvhq_images")
EVAL_CELEBVHQ_IMAGES = [image_to_tensor(crop_and_align_face(image_path)[0]) for image_path in EVAL_CELEBVHQ_FOLDER_PATH.glob("*.png")]

EVAL_SNL_FOLDER_PATH = Path("/home/ubuntu/DemoFaceSwapping/evaluation/snl_images")
EVAL_SNL_IMAGES = [image_to_tensor(crop_and_align_face(image_path)[0]) for image_path in EVAL_SNL_FOLDER_PATH.glob("*.png")]



def get_grid(simswap, source_image: torch.Tensor | list, target_image: torch.Tensor | list, device: torch.device):
    """
    Return a tensorboard grid of the source and target images, along with the generated image.
    Args:
        simswap: The SimSwap model.
        source_image: The source image (B, C, H, W) as list
        target_image: The target image (B', C, H, W) as list

        NOTE: source and target are NOT necessarily the same number of images.

    Returns:
        The tensorboard grid 
    """
    source_image = torch.stack(source_image, dim=0)
    target_image = torch.stack(target_image, dim=0)

    num_sources = source_image.shape[0]
    nrow = 1 + num_sources  # legend + one column per source

    container = []

    # First row: Legend and Source Images
    container.append(LEGEND_IMAGE)
    container.extend(source_image)

    # Rest of the rows: Target Images and Generated Images
    for t in target_image:
        container.append(t)

        # Normalize the target and source images
        t = normalize_imagenet_chw(t)
        # source_image = normalize_imagenet_chw(source_image)
        source_image = normalize_arcface_chw(source_image)

        # Generate the image
        generated_row = generate_image(simswap, t.unsqueeze(0).to(device), source_image.to(device))
        
        # Unnormalize the generated image
        generated_row = generated_row.detach().cpu()
        generated_row = unnormalize_imagenet_chw(generated_row)
       
        container.extend(generated_row)

    grid = make_grid(container, nrow=nrow)
    return grid

def log_training_step(
    writer,
    iteration: int,
    simswap,
    dataset,
    scalars,
    device,
):

    # Log Scalars
    for name, value in scalars.items():
        writer.add_scalar(name, value, iteration)

    # Evaluate on LFW (Training) Dataset
    source = [image_to_tensor(dataset.image_paths[i]) for i in CUMULATIVE_SAME_IDENTITY_IMAGES]
    target = source
    lfw_cumulative_same_grid = get_grid(simswap=simswap, source_image=source, target_image=target, device=device)
    writer.add_image(
        "LFW_Eval/Cumulative_Same_Identity",
        lfw_cumulative_same_grid,
        iteration,
    )

    source = [image_to_tensor(dataset.image_paths[i]) for i in CUMULATIVE_DIFFERENT_IDENTITY_SOURCE]
    target = [image_to_tensor(dataset.image_paths[i]) for i in CUMULATIVE_DIFFERENT_IDENTITY_TARGET]
    lfw_cumulative_diff_grid = get_grid(simswap=simswap, source_image=source, target_image=target, device=device)
    writer.add_image(
        "LFW_Eval/Cumulative_Different_Identity",
        lfw_cumulative_diff_grid,
        iteration,
    )

    same_ident = random.choice(dataset.positive_identities)
    idx_a, idx_b = random.sample(dataset.identity_to_indices[same_ident], k=2)
    source = [
        image_to_tensor(dataset.image_paths[idx_a]),
        image_to_tensor(dataset.image_paths[idx_b]),
    ]
    target = source
    lfw_current_same_grid = get_grid(simswap=simswap, source_image=source, target_image=target, device=device)
    writer.add_image(
        "LFW_Eval/Current_Same_Identity",
        lfw_current_same_grid,
        iteration,
    )

    ids = random.sample(list(dataset.all_identities), k=4)
    source = [
        image_to_tensor(
            dataset.image_paths[random.choice(dataset.identity_to_indices[ids[0]])]
        ),
        image_to_tensor(
            dataset.image_paths[random.choice(dataset.identity_to_indices[ids[1]])]
        ),
    ]
    target = [
        image_to_tensor(
            dataset.image_paths[random.choice(dataset.identity_to_indices[ids[2]])]
        ),
        image_to_tensor(
            dataset.image_paths[random.choice(dataset.identity_to_indices[ids[3]])]
        ),
    ]
    lfw_current_diff_grid = get_grid(simswap=simswap, source_image=source, target_image=target, device=device)
    writer.add_image(
        "LFW_Eval/Current_Different_Identity",
        lfw_current_diff_grid,
        iteration,
    )

    # Evaluation on CelebVHQ Dataset
    cumulative_lfw_to_celeb_grid = get_grid(simswap=simswap, source_image=EVAL_LFW_IMAGES, target_image=EVAL_CELEBVHQ_IMAGES, device=device)    
    writer.add_image(
        "CelebVHQ_Eval/LFW_to_CelebVHQ",
        cumulative_lfw_to_celeb_grid,
        iteration,
    )
    cumulative_celeb_to_lfw_grid = get_grid(simswap=simswap, source_image=EVAL_CELEBVHQ_IMAGES, target_image=EVAL_LFW_IMAGES, device=device)
    writer.add_image(
        "CelebVHQ_Eval/CelebVHQ_to_LFW",
        cumulative_celeb_to_lfw_grid,
        iteration,
    )
    cumulative_celeb_to_celeb_grid = get_grid(simswap=simswap, source_image=EVAL_CELEBVHQ_IMAGES, target_image=EVAL_CELEBVHQ_IMAGES, device=device)
    writer.add_image(
        "CelebVHQ_Eval/CelebVHQ_to_CelebVHQ",
        cumulative_celeb_to_celeb_grid, 
        iteration,
    )


    # Evaluation on SNL Dataset
    cumulative_celeb_to_snl_grid = get_grid(simswap=simswap, source_image=EVAL_CELEBVHQ_IMAGES, target_image=EVAL_SNL_IMAGES, device=device)
    writer.add_image(
        "SNL_Eval/CelebVHQ_to_SNL",
        cumulative_celeb_to_snl_grid,
        iteration,
    )
    cumulative_snl_to_snl_grid = get_grid(simswap=simswap, source_image=EVAL_SNL_IMAGES, target_image=EVAL_SNL_IMAGES, device=device)
    writer.add_image(
        "SNL_Eval/SNL_to_SNL",
        cumulative_snl_to_snl_grid,
        iteration,
    )

