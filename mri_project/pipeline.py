import logging

import cv2
import numpy as np

import matplotlib.pyplot as plt

from mri_project.contour_ops import get_muscle_contours, sort_muscle_contours_by_dist_from_center, \
    get_muscle_contours_dict
from mri_project.utility import draw_lever_arms

logger = logging.getLogger(__name__)


def get_largest_contour_of_each_color(img):
    out = np.zeros_like(img)
    for c in np.unique(img):
        _, cnts, hierarchy = cv2.findContours(np.uint8(img == c), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        good_cnts = sorted(cnts, key=lambda x: -cv2.contourArea(x))
        imt = np.zeros_like(img)
        cv2.drawContours(imt, good_cnts, 0, int(c), -1)
        out += imt
    return out


def resize_muscle_image(img, shape):
    imgs = []
    for c in np.unique(img):
        this_muscle = np.uint8(img == c)
        resized = cv2.resize(this_muscle, shape)
        res = np.uint((resized > 0) * c)
        imgs.append(res)
    out = np.zeros(shape[::-1])
    for image in imgs:
        out += image
    return out


def predict_image(model, x):
    gray = x
    if len(x.shape) == 3 and x.shape[-1] == 3:
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, (256, 256))
    normalized_image = (resized_image - resized_image.mean()) / resized_image.std()
    inp = normalized_image.reshape(1, *normalized_image.shape, 1)
    prediction = model.predict(inp)
    res = prediction.argmax(axis=3)[0].astype('uint8')
    resized_res = resize_muscle_image(res, x.shape[:2][::-1])
    cleaned_res = get_largest_contour_of_each_color(resized_res)
    return cleaned_res


def show_lever_arms(img, angle, numbered=False, scale=1,
                    ax=None, plot=True, img_color_coefficient=1):
    if angle > np.pi:
        angle = np.pi / 180 * angle
    if numbered:
        good_cnts = get_muscle_contours_dict(img)
        if good_cnts.get(0) is not None:
            del good_cnts[0]
        sorted_cnts = [good_cnts[i][0] for i in sorted(good_cnts.keys())]
    else:
        good_cnts = get_muscle_contours(img)
        sorted_cnts = sort_muscle_contours_by_dist_from_center(good_cnts)
    if len(good_cnts) not in {9, 11}:
        logger.warning("muscles not of size 9 or 11")
        if len(good_cnts) > 11:
            sorted_cnts = sorted_cnts[:11]
    logger.info(f"Number of muscles = {len(sorted_cnts)}")
    center_point = np.int32(np.mean(sorted_cnts[0], axis=(0, 1))).reshape(-1)
    out, lever_arms = draw_lever_arms(img, sorted_cnts, angle, center_point, scale)
    out = img_color_coefficient * img + out
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(out)
    return sorted_cnts, lever_arms, out
