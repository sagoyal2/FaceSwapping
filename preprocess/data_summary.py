from collections import Counter
from pathlib import Path

import cv2

RAW_DATA = Path(
    "/home/ubuntu/DemoFaceSwappingData/lfw_funneled_cropped_aligned_224"
)
# EXPECTED_SHAPE = (250, 250, 3)

def count_images_in_folder(folder: Path) -> int:
    count = 0
    for file in folder.iterdir():
        if file.is_file():
            count += 1
            img = cv2.imread(file)
    
            # assert img.shape == EXPECTED_SHAPE, f"Image {file} has shape {img.shape}"
    return count

def main() -> None:
    root = RAW_DATA

    # Find out total number of folders
    folders = sorted(d for d in root.iterdir() if d.is_dir())
    total_folders = len(folders)
    print(f"Total folders: {total_folders}")

    # Find out total number of images
    images_per_folder = [count_images_in_folder(name) for name in folders]
    total_images = sum(images_per_folder)
    print(f"Total images: {total_images}")

    # Find out distribution of images per folder
    tally = Counter(images_per_folder)
    line = ", ".join(f"{n}: {tally[n]}" for n in sorted(tally))
    print(line)

if __name__ == "__main__":
    main()
