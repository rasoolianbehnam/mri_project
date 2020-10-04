import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_uint8(img):
    if img.dtype != np.uint8:
        raise ValueError(f"Image data type should be uint8 but is {img.dtype}")


def get_contours(img):
    img, cnts, hierarchy = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def draw_contours(img, cnts, thickness=-1):
    check_uint8(img)
    w, h = img.shape[:2]
    imt = np.zeros((w, h))
    cv2.drawContours(imt, cnts, -1, 1, thickness)
    return imt


def elongation(cnt):
    *_, rw, rh = cv2.boundingRect(cnt)
    return rw / rh


def get_muscle_contours(img):
    img, cnts, hierarchy = cv2.findContours(np.uint8(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_areas = [cv2.contourArea(x) for x in cnts]
    mx_area = max(cnts_areas) or 1e-9
    good_cnts = [x for x, a in zip(cnts, cnts_areas) if a / mx_area > 0.046 and elongation(x) < 9]
    return good_cnts


def get_center_diffs(cnts):
    centers = np.array([x[:, 0, :].mean(axis=0) for x in cnts])
    center = np.mean(centers, axis=0, keepdims=True)
    cent_diff = centers - center
    r = np.sqrt(np.sum(cent_diff ** 2, axis=1))
    return cent_diff, r


def sort_muscle_contours_by_angle(cnts):
    cent_diff, r = get_center_diffs(cnts)
    angles = np.arcsin(cent_diff[:, 0] / r) * 180 / np.pi
    sorted_cnts = [x for x, y in sorted(zip(cnts, angles), key=lambda x: x[1])]
    return sorted_cnts


def sort_muscle_contours_by_dist_from_center(cnts):
    cent_diff, r = get_center_diffs(cnts)
    indices = np.argsort(r)
    return [cnts[i] for i in indices]


def draw_contours_dict(img, contours_dict, debug=0):
    """
    assigns distinct numbers in sorted order to the contour colors
    """
    out = np.zeros_like(img)
    for i, cnt in contours_dict.items():
        cv2.drawContours(out, cnt, -1, int(i), -1)
        if debug == 1:
            plt.imshow(out)
            plt.title((i, len(cnt[0])))
            plt.show()
        # out += np.uint8(draw_contours(img, [cnt])*i)
    return out


def erode_by_muscle_class(img, kernel_size_dict):
    check_uint8(img)
    out = np.zeros_like(img)
    for c in np.unique(img):
        k = kernel_size_dict.get(c)
        if not k:
            continue
        out += cv2.erode(np.uint8(img == c), np.ones((k, k))) * np.uint8(c)
    return out
