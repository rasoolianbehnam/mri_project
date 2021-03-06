import glob
import logging
import os
from collections import namedtuple
from functools import reduce
from numbers import Number
from operator import add
from os.path import dirname, splitext, basename
from typing import Iterable, Dict, Tuple

import numpy as np
import pandas as pd

from mri_project.contour_ops import *
from mri_project.contour_ops import get_muscle_contours

logger = logging.getLogger(__name__)

Line = namedtuple("Line", 'a, b')

MriImage = namedtuple("MriImage",
                      "id raw_image traced_image muscle_contours predicted_muscles predicted_muscle_contours")


def match_names_by_dir(a, b):
    a_dirs = {dirname(x): x for x in a}
    b_dirs = {dirname(x): x for x in b}
    common_dirs = a_dirs.keys() & b_dirs.keys()
    return [a_dirs[x] for x in common_dirs], [b_dirs[x] for x in common_dirs]


def draw_sorted_contours(img, sorted_cnts):
    """
    assigns distinct numbers in sorted order to the contour colors
    """
    out = np.zeros_like(img)
    for i, cnt in enumerate(sorted_cnts, 1):
        out += (draw_contours(img, [cnt]) * int(i)).astype('uint8')
    return out


def elongation(cnt):
    """
    contour elongation
    """
    *_, rw, rh = cv2.boundingRect(cnt)
    return rw / rh


def get_muscles(img):
    """
    detects muscles from raw image and returns thresholded image with the muscles
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    traces = (hsv_img[..., 1] > 60).astype('uint8')
    im_floodfill = traces.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = traces.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 1)
    im_floodfill = 1 - im_floodfill  # , np.ones((3, 3))
    return im_floodfill


def line_passing_from_point_at_angle(p, angle):
    p = p.reshape(-1)
    x, y = p
    a = np.tan(angle)
    b = y - x * a
    return Line(a, b)


def y_on_line(x, line):
    return x * line.a + line.b


def x_on_line(y, line):
    return (y - line.b) / line.a


def vertical_line_passing_from_point(line, point):
    x0, y0 = point
    return Line(-1 / line.a, y0 + x0 / line.a)


def line_intersection(l1, l2):
    a = np.array([[l1.a, -1], [l2.a, -1]])
    b = -np.array([l1.b, l2.b])
    return np.linalg.solve(a, b)


def get_contour_center_points(cnts):
    return [np.mean(cnt, axis=(0, 1)) for cnt in cnts]


def get_lever_arms(center_points, angle, center_point):
    line = line_passing_from_point_at_angle(center_point, angle)
    intersections = []
    for cnt_center in center_points:
        l2 = vertical_line_passing_from_point(line, cnt_center)
        intersections.append(line_intersection(line, l2))
    return intersections


def get_center_muscle_index(center_points):
    if center_points.shape[1] != 2:
        raise ValueError(f"center_points should be 2-dimensionals but is of shape {center_points.shape}")
    diff = center_points - center_points.mean(axis=0)
    idx = np.argmin((diff ** 2).sum(axis=1))
    return idx


def get_center_muscle_index_v02(cnts: List[np.ndarray]) -> int:
    for cnt in cnts:
        if tuple(cnt.shape[1:]) != (1, 2):
            raise ValueError("The shape of contours must be [?, 1, 2]")
    center = np.concatenate(cnts).mean(axis=(0, 1))
    centers = np.array([cnt.mean((0, 1)) for cnt in cnts])
    return int(np.argmin(((centers - center) ** 2).sum(axis=1)))


def draw_lever_arms(img: np.ndarray, sorted_cnts: Iterable[np.ndarray], angle: float, center_point=None, scale=1):
    if angle == 90 or angle == np.pi / 2:
        angle = 89.9
    if angle > np.pi:
        angle = angle * np.pi / 180
    center_points = np.array([v.mean(axis=(0, 1)) for v in sorted_cnts])
    # print([len(v) for v in sorted_cnts])
    if center_point is None:
        center_point = center_points[get_center_muscle_index_v02(list(sorted_cnts))]
    # print(center_point)
    w, h = img.shape
    line = line_passing_from_point_at_angle(center_point, angle)
    # print(line)
    out = np.zeros_like(img)
    y0 = np.int32(y_on_line(0, line))
    y1 = np.int32(y_on_line(1 * w, line))
    cv2.line(out, (0, y0), (1 * w, y1), 1, 3)
    intersections = get_lever_arms(center_points, angle, center_point)

    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_font_scale = 1
    text_color = 1
    text_thickness = 2
    text_displacement = np.int32(np.array([0.01098097, 0.01072961]) * np.array(img.shape))

    lever_arms = []
    for cnt_center, intersection in zip(center_points, intersections):
        cv2.line(out, tuple(np.int32(cnt_center)), tuple(np.int32(intersection)), 1, 3)
        distance = np.sqrt(np.sum((cnt_center - intersection) ** 2)) * scale
        text_ord = tuple(np.int32(cnt_center - text_displacement))

        cv2.putText(out, '%4.2f' % distance, text_ord, text_font,
                    text_font_scale, text_color, text_thickness, cv2.LINE_AA)
        lever_arms.append(
            dict(
                zip(
                    ['center_x', 'center_y', 'scale', 'lever_arm'],
                    [*cnt_center, scale, distance]
                )
            )
        )
    return out, lever_arms


def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def dfe(file):
    d = dirname(file)
    b = basename(file)
    f, e = splitext(b)
    return d, f, e


def convert_list_col_to_multiple_cols(df: pd.DataFrame, col, col_names, drop_col=True):
    df[col_names] = pd.DataFrame(df[col].tolist(), index=df.index)
    if drop_col:
        df.drop(col, axis=1, inplace=True)


def imsshow(*images, n_cols=None, single_size=(10, 10), **ax_args):
    n_images = len(images)
    n_cols = n_cols or n_images
    n_rows = n_images // n_cols
    n_rows = n_rows + 1 if n_images % n_cols else n_rows
    w, h = single_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_rows, h * n_cols))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img, **ax_args)
    return fig, axes


def get_outliers(x, r=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    cond = (x > q3 + r * iqr) | (x < q1 - r * iqr)
    return np.where(~cond), np.where(cond)


def multi_label_image_to_dict(img: np.ndarray, transform_fun=lambda x: x) -> Dict[Number, np.ndarray]:
    return {i: transform_fun(img == i) for i in np.unique(img)}


def multi_label_image_from_dict(d):
    x = [k * v for k, v in d.items()]
    return np.array(list(reduce(add, x)))


def show_lever_arms(img, angle, binary=False, scale=1,
                    ax=None, plot=True, img_color_coefficient=1.):
    if angle > np.pi:
        angle = np.pi / 180 * angle
    cnts = get_muscle_contours_dict(img, binary)
    if 0 in cnts:
        del cnts[0]
    cnts = {k: v[0] for k, v in cnts.items() if len(v)}
    logger.info(f"Number of muscles = {len(cnts)}")
    out, lever_arms = draw_lever_arms(img, cnts.values(), angle, scale=scale)
    lever_arms = dict(zip(cnts.keys(), lever_arms))
    out = img_color_coefficient * img + out
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(out)
    return cnts, lever_arms, out


txt = """Background ,0
Left banana  ,250
Right banana  ,240
Left central  ,230
Right central  ,220
Bottom left  ,210
Bottom right  ,200
Top left  ,190
Top right  ,180
Btw bottom and center left  ,170
Btw bottom and center right  ,160
Center  ,150
Left half banana  ,140
Right half banana  ,130"""
muscle_colors_map = dict([x.split(',') for x in txt.split('\n')])
reverse_muscle_colors_map = {int(v): k for k, v in muscle_colors_map.items()}


def get_muscle_contours_dict(img: np.ndarray, binary: bool) -> Dict[Number, List[np.ndarray]]:
    """
    :param img: a binary or multiclass image mask
    :param binary: whether the mask is binary or not
    :return:
    """
    res = multi_label_image_to_dict(img)
    cnts = {k: sorted(get_muscle_contours(v, min_area_threshold=0.02), key=lambda x: -cv2.contourArea(x))
            for k, v in res.items()}
    if binary:
        cnts = dict(enumerate([[cnt] for v in cnts.values() for cnt in v]))
    return cnts


def write_areas(img, centers, areas, color):
    out = np.zeros_like(img)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_font_scale = 1
    text_color = color
    text_thickness = 2
    text_displacement = np.int32(np.array([0.01098097, 0.01072961]) * np.array(img.shape))
    for cnt, area in zip(centers, areas):
        text_ord = tuple(np.int32(cnt - text_displacement))
        cv2.putText(out, '%4.2f' % area, text_ord, text_font,
                    text_font_scale, text_color, text_thickness, cv2.LINE_AA)
    return img + out


def scale_img(img, max=255, dtype='uint8'):
    mn, mx = np.min(img), np.max(img)
    out = (img - mn) / (mx - mn)
    return (out * max).astype(dtype)


def get_all_images(input_path: str, extension=''):
    if extension:
        extension = '.' + extension
    files = glob.glob(f"{input_path}/**/*{extension}", recursive=True)
    return files


def shape_matches(x: np.ndarray, shape: Tuple[int]) -> bool:
    assert len(shape) == len(x.shape)
    for x_d, true_d in zip(x.shape, shape):
        if true_d is not None:
            if x_d == true_d or x_d in true_d:
                continue
            return False
    return True


def replace_path(file_path, in_root, out_root):
    abspath = os.path.abspath
    file_path, in_root, out_root = abspath(file_path), abspath(in_root), abspath(out_root)
    assert in_root in file_path
    return file_path.replace(in_root, out_root)