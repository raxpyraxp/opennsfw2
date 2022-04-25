"""
Inference utilities.
"""
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

from ._download import get_default_weights_path
from ._image import preprocess_image, Preprocessing
from ._inspection import make_and_save_nsfw_grad_cam
from ._model import make_open_nsfw_model
from ._typing import NDFloat32Array


def predict_image(
        image_path: str,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path(),
        grad_cam_path: Optional[str] = None,
        alpha: float = 0.8
) -> float:
    """
    Pipeline from single image path to predicted NSFW probability.
    Optionally generate and save the Grad-CAM plot.
    """
    pil_image = Image.open(image_path)
    image = preprocess_image(pil_image, preprocessing)
    model = make_open_nsfw_model(weights_path=weights_path)
    nsfw_probability = float(model(np.expand_dims(image, 0)).numpy()[0][1])

    if grad_cam_path is not None:
        make_and_save_nsfw_grad_cam(
            pil_image, preprocessing, model, grad_cam_path, alpha
        )

    return nsfw_probability


def predict_images(
        image_paths: Sequence[str],
        batch_size: int = 8,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path(),
        grad_cam_paths: Optional[Sequence[str]] = None,
        alpha: float = 0.8
) -> List[float]:
    """
    Pipeline from image paths to predicted NSFW probabilities.
    Optionally generate and save the Grad-CAM plots.
    """
    images = tf.convert_to_tensor([
        preprocess_image(Image.open(image_path), preprocessing)
        for image_path in image_paths
    ])
    model = make_open_nsfw_model(weights_path=weights_path)
    predictions = model.predict(images, batch_size=batch_size, verbose=0)
    nsfw_probabilities: List[float] = predictions[:, 1].tolist()

    if grad_cam_paths is not None:
        for image_path, grad_cam_path in zip(image_paths, grad_cam_paths):
            make_and_save_nsfw_grad_cam(
                Image.open(image_path), preprocessing, model,
                grad_cam_path, alpha
            )

    return nsfw_probabilities


class Aggregation(str, Enum):
    MEAN = auto()
    MEDIAN = auto()
    MAX = auto()
    MIN = auto()


def _get_aggregation_fn(
        aggregation: Aggregation
) -> Callable[[NDFloat32Array], float]:

    def fn(x: NDFloat32Array) -> float:
        agg: Any = {
            Aggregation.MEAN: np.mean,
            Aggregation.MEDIAN: np.median,
            Aggregation.MAX: np.max,
            Aggregation.MIN: np.min,
        }[aggregation]
        return float(agg(x))

    return fn
