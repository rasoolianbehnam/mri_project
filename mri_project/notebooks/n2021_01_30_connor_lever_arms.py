import numpy as np
from typing import Iterator, Dict, Any
from varname import nameof
import mri_project.muscle_detector as md


def compute_lever_arms_for_trace_file(file: str, angles: Iterator[float],
                                      cache: Dict[str, Any]=None) -> Dict[float, np.ndarray]:
    print(file)
    # plt.imshow(md.read_image(file))
    # plt.show()
    t = md.MuscleDetector(file, file, traced_image=file)
    t.get_traced_binary_mask()
    for deg in angles:
        t.get_traced_contours(deg)

    if cache is not None:
        cache[nameof(t)] = t
    return t.traced_lever_arm_images