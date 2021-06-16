import os

import cv2
import numpy as np
from typing import Iterator, Dict, Any

import mri_project.muscle_detector as md
from mri_project.utility import dfe, scale_img


def compute_lever_arms_for_trace_file(file: str, angles: Iterator[float],
                                      cache: Dict[str, Any] = None) -> Dict[int, np.ndarray]:
    print(file)
    # plt.imshow(md.read_image(file))
    # plt.show()
    t = md.MuscleDetector(file, file, traced_image=file)
    t.get_traced_binary_mask()
    for deg in angles:
        t.get_traced_contours(int(deg))

    if cache is not None:
        cache['t'] = t
    return t.traced_lever_arm_images


def save_img_dict(img_dict: Dict[str, np.ndarray], input_image_file: str, ):
    d, f, e = dfe(input_image_file)
    for degree, img in img_dict.items():
        filename = os.path.join(d, f"{f}_{degree}{e}")
        os.makedirs(d, exist_ok=True)
        print(filename)
        cv2.imwrite(filename, scale_img(img))
