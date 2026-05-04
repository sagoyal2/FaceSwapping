import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple

"""
Copied from https://github.com/yakhyo/face-parsing/blob/main/onnx_inference.py
"""
class FaceParsingONNX:
    """Face parsing inference using ONNXRuntime."""

    def __init__(self, model_path: str, session: ort.InferenceSession = None) -> None:
        """Initializes the FaceParsingONNX class.

        Args:
            model_path (str): Path to the ONNX model file.
            session (ort.InferenceSession, optional): ONNX Session. Defaults to None.

        Raises:
            AssertionError: If model_path is None and session is not provided.
            FileNotFoundError: If model_path does not exist.
        """
        self.session = session
        if self.session is None:
            assert model_path is not None, 'Model path is required for the first time initialization.'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'ONNX model not found at path: {model_path}')

            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if ort.get_device() == 'GPU'
                else ['CPUExecutionProvider']
            )
            self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_size = (512, 512)
        self.input_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.input_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name

        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            np.ndarray: Preprocessed image tensor (1, C, H, W).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        image = image.astype(np.float32) / 255.0
        image = (image - self.input_mean) / self.input_std

        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        image_batch = np.expand_dims(image, axis=0).astype(np.float32)  # CHW → BCHW

        return image_batch

    def postprocess(self, output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output to segmentation mask.

        Args:
            output (np.ndarray): Raw model output.
            original_size (Tuple[int, int]): Original image size (width, height).

        Returns:
            np.ndarray: Segmentation mask resized to original dimensions.
        """
        predicted_mask = output.squeeze(0).argmax(0).astype(np.uint8)
        restored_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        return restored_mask

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run face parsing inference on an image.

        Args:
            image (np.ndarray): Input image in BGR format (H, W, C).

        Returns:
            np.ndarray: Segmentation mask with the same size as input image.
        """
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], original_size)

