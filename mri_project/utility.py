import numpy as np
import pandas as pd
from os.path import dirname, splitext, basename
import cv2
from collections import namedtuple
from mri_project.contour_ops import *
import logging

logger = logging.getLogger(__name__)

Line = namedtuple("Line", 'a, b')

MriImage = namedtuple("MriImage", "id raw_image traced_image muscle_contours predicted_muscles predicted_muscle_contours")

def match_names_by_dir(a, b):
    a_dirs = {dirname(x): x for x in a}
    b_dirs = {dirname(x): x for x in b}
    common_dirs = a_dirs.keys() & b_dirs.keys()
    return ([a_dirs[x] for x in common_dirs], [b_dirs[x] for x in common_dirs])


def draw_sorted_contours(img, sorted_cnts):
    """
    assigns distinct numbers in sorted order to the contour colors
    """
    out = np.zeros_like(img)
    for i, cnt in enumerate(sorted_cnts, 1):
        out += (draw_contours(img, [cnt])*int(i)).astype('uint8')
    return out


def elongation(cnt):
    """
    contour elongation
    """
    *_, rw, rh = cv2.boundingRect(cnt)
    return rw/rh


def get_muscles(img):
    """
    detects muscles from raw image and returns thresholded image with the muscles
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    traces = (hsv_img[...,1]>60).astype('uint8')
    im_floodfill = traces.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = traces.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 1);
    im_floodfill = 1-im_floodfill#, np.ones((3, 3))
    return im_floodfill


def line_passing_from_point_at_angle(p, angle):
    x, y = p
    a = np.tan(angle)
    b = y - x * a
    return Line(a, b)


def y_on_line(x, line):
    return x * line.a + line.b


def x_on_line(y, line):
    return (y - b) / line.a


def vertical_line_passing_from_point(line, point):
    x0, y0 = point
    return Line(-1/line.a, y0 + x0/line.a)


def line_intersection(l1, l2):
    a = np.array([[l1.a, -1], [l2.a, -1]])
    b = -np.array([l1.b, l2.b])
    return np.linalg.solve(a, b)


def get_contour_center_points(cnts):
    return  [np.mean(cnt, axis=(0, 1)) for cnt in cnts]



def get_lever_arms(center_points, angle, center_point):
    line = line_passing_from_point_at_angle(center_point, angle)
    intersections = []
    for cnt_center in center_points:
        l2 = vertical_line_passing_from_point(line, cnt_center)
        intersections.append(line_intersection(line, l2))
    return intersections


def draw_lever_arms(img, sorted_cnts, angle, center_point, scale=1):
    if np.allclose(angle, np.pi/2):
        angle += .001
    w, h = img.shape
    line = line_passing_from_point_at_angle(center_point, angle)
    out = np.zeros_like(img)
    y0 = np.int32(y_on_line(0, line))
    y1 = np.int32(y_on_line(5*w, line))
    cv2.line(out, (0, y0), (5*w, y1), 1, 3)
    center_points = get_contour_center_points(sorted_cnts)
    intersections = get_lever_arms(center_points, angle, center_point)
    
    text_font = cv2.FONT_HERSHEY_SIMPLEX 
    text_fontScale = 1
    text_color = 1
    text_thickness = 2
    text_displacement = np.int32(np.array([0.01098097, 0.01072961]) * np.array(img.shape))
   
    lever_arms = []
    for cnt_center, intersection in zip(center_points, intersections):
        cv2.line(out, tuple(np.int32(cnt_center)), tuple(np.int32(intersection)), 1, 3)
        distance = np.sqrt(np.sum((cnt_center - intersection)**2)) * scale
        text_ord = tuple(np.int32(cnt_center - text_displacement))
 
        cv2.putText(out, '%4.2f'%distance, text_ord, text_font,   
                    text_fontScale, text_color, text_thickness, cv2.LINE_AA) 
        lever_arms.append(
            dict(
                zip(
                    ['center_x', 'center_y', 'scale', 'lever_arm'], 
                    [*cnt_center, scale, distance]
                )
            )
        )
    return out, lever_arms


def show_lever_arms(img, angle, scale=1, ax=None, plot=True):
    if angle > np.pi:
        angle = np.pi / 360 * angle
    good_cnts = get_muscle_contours(img)
    sorted_cnts = sort_muscle_contours_by_dist_from_center(good_cnts)
    if len(good_cnts) not in {9, 11}:
        logger.warning("muscles not of size 9 or 11")
        if len(good_cnts) > 11:
            sorted_cnts = sorted_cnts[:11]
    logger.info(f"Number of muscles = {len(sorted_cnts)}")
    center_point = np.int32(np.mean(sorted_cnts[0], axis=(0, 1)))
    out, lever_arms = draw_lever_arms(img, sorted_cnts, angle, center_point, scale)
    out = img+out
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.imshow(out)
    return lever_arms, out

def dfe(file):
    d = dirname(file)
    b = basename(file)
    f, e = splitext(b)
    return d, f, e

muscle_colors_map = dict([x.split(',') for x in """Background ,0
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
Right half banana  ,130""".split('\n')])
reverse_muscle_colors_map = {int(v): k for k, v in muscle_colors_map.items()}