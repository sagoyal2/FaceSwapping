import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
from face_parsing import FaceParsingONNX

def blur_pass(image: np.ndarray) -> bool:
    """
    Source: https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/

    Only allow images with a Laplacian score greater (sharper) than the threshold.
    """
    LAPLACIAN_THRESHOLD = 13.4

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)  
    score = np.var(laplacian) 

    return score > LAPLACIAN_THRESHOLD

def crop_pass(image: np.ndarray) -> bool:
    """
    Only allow images with less than 40% of the frame black.
    """
    BLACK = np.array([0, 0, 0])
    BLACK_THRESHOLD = 0.09

    black_fraction = (image == BLACK).mean()
    return black_fraction < BLACK_THRESHOLD

def accessory_occlusion_pass(segmentation_mask: np.ndarray) -> bool:
    """
    Accessory Definition: https://github.com/yakhyo/face-parsing/blob/main/utils/common.py 
    
    Only allow images with less than 65% of the frame occluded by accessories.
    """
    accessories_classes = [6, 7, 8, 9, 16, 17, 18]
    accessories_mask = np.isin(segmentation_mask, accessories_classes)
    accessories_fraction = float(accessories_mask.mean())

    return accessories_fraction < 0.65


# Initialize Models
# Face Detection Model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Face Parsing Model
SEGMENTATION_MODEL = Path("/home/ubuntu/DemoFaceSwappingData/segmentation_model/resnet18.onnx")
engine = FaceParsingONNX(SEGMENTATION_MODEL)


# Define Input and Output Directories
RAW_DATA = Path(
    "/home/ubuntu/DemoFaceSwappingData/lfw_funneled"
)


OUTPUT_DIR = Path(
    f"/home/ubuntu/DemoFaceSwappingData/lfw_funneled_cropped_aligned_224"
)


def main() -> None:

    total_folders = len(list(RAW_DATA.iterdir()))
    for index, folder in tqdm(enumerate(RAW_DATA.iterdir()), desc="Cropping and aligning images", total=total_folders):

        # if index > 10:
        #     break
        
        if folder.is_dir():
            output_folder = OUTPUT_DIR / folder.name
            output_folder.mkdir(parents=True, exist_ok=True)

            for file in folder.iterdir():
                if file.is_file():
                    img = cv2.imread(file)

                    # Check if original image is sharp
                    if not blur_pass(img):
                        continue

                    # Detect highest confidence face and align
                    faces = app.get(img)
                    if len(faces) == 0:
                        continue

                    face = faces[0]

                    kps = face.kps.astype("float32")
                    img_cropped_aligned_224 = face_align.norm_crop(
                        img,
                        kps,
                        image_size=224,
                        mode="arcface",
                    )

                    # Check if cropped and aligned image is not wrong face (too black) 
                    if not crop_pass(img_cropped_aligned_224):
                        continue

                    # Check if cropped and aligned image is not occluded by accessories
                    segmentation_mask = engine.predict(img_cropped_aligned_224)
                    if not accessory_occlusion_pass(segmentation_mask):
                        continue

                    cv2.imwrite(output_folder / file.name, img_cropped_aligned_224)

            
            if not any(output_folder.iterdir()):
                output_folder.rmdir()
                print(f"Removed empty folder: {output_folder}")


if __name__ == "__main__":
    main()