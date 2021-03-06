import glob

import cv2
from sklearn.cluster import KMeans
import numpy as np
import joblib
from mri_project.pipeline import predict_image
from mri_project.resources import models_dir
from mri_project.utility import get_muscles, show_lever_arms
import mri_project.contour_ops as cntops

from mri_project.contour_ops import get_muscle_contours
import logging

logger = logging.getLogger(__name__)
nine_to_11_dict = {0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11}


def read_image(x):
    if isinstance(x, str):
        x = cv2.imread(x)
    elif not isinstance(x, np.ndarray):
        raise ValueError("x should be either a path to image or an image")
    return x


def read_generic(x):
    if isinstance(x, str):
        x = joblib.load(x)
    return x


def train_clustering_model_on_dir(img_dir_regex, n_clusters=2, rgb=False):
    channels = 1
    read_flag = 0
    if rgb:
        channels = 3
        read_flag = 1
    train_files = glob.glob(img_dir_regex)
    x = np.concatenate([cv2.imread(file, read_flag).reshape(-1, channels) for file in train_files])
    model = KMeans(n_clusters=n_clusters)
    model.fit(x)
    return model


def get_thresh(img, cluster_model, rgb=False):
    channels = 3 if rgb else 1
    thresh = cluster_model.predict(img.reshape(-1, channels))
    if thresh.sum() / len(thresh) > .5:
        thresh = 1 - thresh
    return thresh.reshape(*img.shape[:2])


class MuscleDetector(object):
    def __init__(self, id_, raw_image, scale=None, traced_image=None):
        self.id = id_
        self.raw_image = read_image(raw_image)
        self.traced_image = None
        self.scale = scale
        if traced_image is not None:
            self.traced_image = read_image(traced_image)
        if scale is None:
            clustering_model = joblib.load(models_dir+"scale_bar_clustering_model.pkl")
            self.get_scale(clustering_model)

        self.traced_lever_arm_images = {}
        self.predicted_lever_arm_images = {}
        self.traced_features = {}
        self.predicted_features = {}
        self.traced_binary_mask = None
        self.traced_multilabel_mask = None
        self.predicted = None
        self.traced_contours = None
        self.predicted_contours = None

    def get_scale(self, clustering_model):
        thresh = get_thresh(self.raw_image.mean(axis=2), clustering_model)
        _, cnts, _ = cntops.find_contours(thresh)
        cnts = [cnt for cnt in cnts if len(cnt) > 50]
        idx = np.argsort([cntops.elongation(cnt) for cnt in cnts])[::-1]
        picked_cnts = [cnts[i] for i in idx[:1]]
        mx = picked_cnts[0].max(axis=(0, 1))
        mn = picked_cnts[0].min(axis=(0, 1))
        length = np.max(mx - mn)
        scale = 10 / length
        self.scale = scale
        return scale

    def get_traced_binary_mask(self, img=None):
        if img is None:
            img = self.traced_image
        out = get_muscles(img)
        self.traced_binary_mask = out
        return out

    def get_traced_multilabel_mask(self, unmatched_pixel_value=0):
        rszd = cv2.resize(self.traced_binary_mask, self.predicted.shape[::-1])
        cnts = get_muscle_contours(rszd)
        cnt_map = {}
        predicted_unique = np.unique(self.predicted)
        result = np.zeros_like(rszd)
        for i, cnt in enumerate(cnts, 1):
            max_overlap = .1
            chosen_j = unmatched_pixel_value
            for j in predicted_unique:
                if j == 0:
                    continue
                imt = np.zeros_like(rszd)
                cv2.drawContours(imt, [cnt], -1, 1, -1)
                common = (imt > 0) & (self.predicted > 0) & ((imt > 0) == (self.predicted == j))
                overlap = np.sum(common)
                if overlap > max_overlap:
                    max_overlap = overlap
                    chosen_j = j
                # plt.imshow(common)
                # plt.show()
            cnt_map[i] = chosen_j
            cv2.drawContours(result, [cnt], -1, int(chosen_j), -1)
        if len(cnt_map.keys()) != len(set(cnt_map.values())):
            logger.warning(f"Matches not 100%. The map is {cnt_map}")
        self.traced_multilabel_mask = result
        return result

    def has_good_prediction(self, refresh=False):
        if refresh:
            self.get_traced_multilabel_mask()
        if self.predicted is None or self.traced_multilabel_mask is None:
            return False
        return len(np.unique(self.predicted)) == len(np.unique(self.traced_multilabel_mask))

    def get_contour_areas(self):
        self.traced_features['area'] = {
            k: cv2.contourArea(x) * (self.scale ** 2) for k, x in self.traced_contours.items()
        }
        self.predicted_features['area'] = {
            k: cv2.contourArea(x) * (self.scale ** 2) for k, x in self.predicted_contours.items()
        }

    def get_contour_centers(self):
        self.traced_features['center'] = {
            k: x.mean(axis=(0, 1)) for k, x in self.traced_contours.items()
        }
        self.predicted_features['center'] = {
            k: x.mean(axis=(0, 1)) for k, x in self.predicted_contours.items()
        }

    def predict(self, model):
        self.predicted = np.uint8(predict_image(model, self.raw_image))
        return self.predicted

    def get_traced_contours(self, angle, img_color_coefficient=1 / 11, binary=False):
        if self.traced_image is None:
            return
        if self.has_good_prediction():
            cnts, cnt_features, lever_image = show_lever_arms(self.traced_multilabel_mask, angle, scale=self.scale,
                                                              plot=False, img_color_coefficient=img_color_coefficient)
        else:
            cnts, cnt_features, lever_image = show_lever_arms(self.traced_binary_mask, angle, binary=True,
                                                              scale=self.scale, plot=False)
        self.traced_contours = cnts
        self.traced_lever_arm_images[angle] = lever_image
        self.traced_features[f'lever_arm_{angle}'] = {k: x['lever_arm'] for k, x in cnt_features.items()}

    def get_predicted_contours(self, angle, img_color_coefficient=1 / 11):
        cnts, cnt_features, lever_image = show_lever_arms(self.predicted, angle, scale=self.scale,
                                                          plot=False, img_color_coefficient=img_color_coefficient)
        self.predicted_contours = cnts
        self.predicted_lever_arm_images[angle] = lever_image
        self.predicted_features[f'lever_arm_{angle}'] = {k: x['lever_arm'] for k, x in cnt_features.items()}

    def get_attributes(self):
        return [x for x in dir(self) if not x.startswith('_') and not callable(getattr(self, x))]

    @classmethod
    def load_from_dict(cls, d):
        x = cls(d['id'], d['raw_image'], d['scale'])
        attrs = set(x.get_attributes())
        assert len(d.keys() - attrs) == 0
        for k, v in d.items():
            setattr(x, k, v)
        return x

    def save_to_dict(self):
        out = {}
        attrs = self.get_attributes()
        for attr in attrs:
            out[attr] = getattr(self, attr)
        return out
