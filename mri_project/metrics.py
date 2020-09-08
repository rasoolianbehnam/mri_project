from collections import defaultdict

import joblib
import numpy as np

all_stats = []


def stat(fun):
    all_stats.append(fun.__name__)
    return fun


def get_confusion_matrix(x, y):
    class_numbers = np.unique(x)  # np.uint8([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])//10
    print(class_numbers, len(class_numbers))
    matches = np.zeros([np.max(class_numbers) + 1] * 2, dtype='float')
    all_positives = matches.copy() + 1e-9
    print(matches.shape)
    for cls1 in class_numbers:
        for cls2 in class_numbers:
            a = ((x == cls1) & (y == cls2)).sum()
            b = (x == cls2).sum()
            matches[int(cls1), int(cls2)] = a
            all_positives[int(cls1), int(cls2)] = b + 1e-9
    return matches / all_positives


@stat
def total_accuracy(pred, orig):
    w, h = orig.shape
    total_pixels = w * h
    return np.sum(pred == orig) / total_pixels


def true_positive(pred, orig):
    return np.sum((pred > 0) & (orig > 0))


def true_negative(pred, orig):
    return np.sum((pred == 0) & (orig == 0))


def false_positive(pred, orig):
    return np.sum((pred > 0) & (orig == 0))


def false_negative(pred, orig):
    return np.sum((pred == 0) & (orig > 0))


@stat
def accuracy(pred, orig):
    uniques = set(np.unique(pred)) | set(np.unique(orig))
    out = {}
    for u in uniques:
        om = orig == u
        pm = pred == u
        tp = true_positive(pm, om)
        tn = true_negative(pm, om)
        fp = false_positive(pm, om)
        fn = false_negative(pm, om)
        out[u] = (tp + tn) / (tp + tn + fp + fn)
    return out


@stat
def precision(pred, orig):
    uniques = set(np.unique(pred)) | set(np.unique(orig))
    out = {}
    for u in uniques:
        pm = pred == u
        om = orig == u
        tp = true_positive(pm, om)
        fp = false_positive(pm, om)
        out[u] = tp / (tp + fp)
    return out


@stat
def recall(pred, orig):
    uniques = set(np.unique(pred))
    out = {}
    for u in uniques:
        pm = pred == u
        om = orig == u
        tp = true_positive(pm, om)
        fn = false_negative(pm, om)
        out[u] = tp / (tp + fn)
    return out


def get_stats(files, model=None, transform_fun=None):
    if model is not None and not isinstance(model, dict):
        model = defaultdict(lambda x: model)
    pred_stats = defaultdict(lambda: defaultdict(dict))
    bad_files = []
    for index, file_ in enumerate(files):
        # if index < 676: continue
        print(f"{index}: {file_}")
        data_ = joblib.load(file_)
        if transform_fun:
            data_ = transform_fun(data_)
        data_.get_traced_contours(89.8)
        n_muscles = len(data_.traced_contours)
        if model is not None:
            try:
                data_.predict(model[n_muscles])
            except KeyError:
                bad_files.append(file_)
                continue
        data_.get_predicted_contours(89.8)

        if len(data_.traced_contours) != len(data_.predicted_contours):
            bad_files.append(file_)
            continue
        data_.get_contour_areas()
        data_.get_contour_centers()
        if n_muscles in {9, 11}:
            pred_stats[n_muscles][data_.id] = get_stats_(data_)
        else:
            bad_files.append(file_)
    return pred_stats, bad_files


def get_stats_(data_):
    stats_ = {}
    # area
    a_ = np.array(data_.predicted_features['area'])
    b_ = np.array(data_.traced_features['area'])
    stats_['area_ratio'] = dict(enumerate(a_ / b_))
    stats_['area_difference'] = dict(enumerate(np.abs(a_ - b_)))
    # center
    a_ = np.array(data_.predicted_features['center'])
    b_ = np.array(data_.traced_features['center'])
    stats_['center_distance'] = dict(enumerate(np.sqrt(np.sum((a_ - b_) ** 2, axis=1))
                                                                        * data_.scale))
    # lever arm
    a_ = np.array(data_.predicted_features['lever_arm_89.8'])
    b_ = np.array(data_.traced_features['lever_arm_89.8'])
    stats_['lever_arm_89.8_distance'] = dict(enumerate(np.abs((a_ - b_))))

    # other
    a_ = data_.predicted
    b_ = data_.traced_multilabel_mask
    stats_['total_accuracy'] = {-1: total_accuracy(a_, b_)}
    stats_['accuracy'] = accuracy(a_, b_)
    stats_['precision'] = precision(a_, b_)
    stats_['recall'] = recall(a_, b_)
    stats_['has_good_prediction'] = {-1, data_.has_good_prediction()}

    return stats_
