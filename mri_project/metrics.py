from collections import defaultdict
import pandas as pd

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
    pred_stats = [] # defaultdict(lambda: defaultdict(dict))
    bad_files = []
    for index, file_ in enumerate(files):
        print(f"{index}: {file_}")
        data_, bad_files_ = prepare_file(file_, model, transform_fun)
        bad_files.extend(bad_files_)
        n_muscles = len(data_.traced_contours)
        if n_muscles in {9, 11}:
            pred_stats.append(get_stats_(data_, file_))
        else:
            bad_files.append(file_)
    return pd.concat(pred_stats), bad_files


def prepare_file(file, model, transform_fun):
    bad_files = []
    data = joblib.load(file)
    if transform_fun:
        data = transform_fun(data)
    data.get_traced_contours(89.8)
    n_muscles = len(data.traced_contours)
    if model is not None:
        try:
            data.predict(model[n_muscles])
        except KeyError:
            bad_files.append(file)
            return data, bad_files
    data.get_predicted_contours(89.8)

    if len(data.traced_contours) != len(data.predicted_contours):
        bad_files.append(file)
        return data, bad_files
    data.get_contour_areas()
    data.get_contour_centers()
    return data, bad_files


def get_stats_(data, file=''):
    out = pd.concat([
        *get_various_metrics(data, 'area'),
        *get_various_metrics(data, 'lever_arm_89.8'),
        get_centroid_distance(data)
    ]).reset_index()

    precicted_labels = data.predicted
    traced_labels = data.traced_multilabel_mask
    accuracy_df = pd.DataFrame(accuracy(precicted_labels, traced_labels).items(),
                               columns=['muscle', 'value']).assign(measure='pixel', metric='accuracy')
    precision_df = pd.DataFrame(precision(precicted_labels, traced_labels).items(),
                                columns=['muscle', 'value']).assign(measure='pixel', metric='precision')
    recall_df = pd.DataFrame(recall(precicted_labels, traced_labels).items(),
                             columns=['muscle', 'value']).assign(measure='pixel', metric='recall')
    out = pd.concat([out, accuracy_df, precision_df, recall_df])
    out = out.append({'muscle': -1, 'value': total_accuracy(precicted_labels, traced_labels),
                      'measure': '-', 'metric': 'accuracy'},
                     ignore_index=True)
    out = out.append({'muscle': -1, 'value': total_accuracy(precicted_labels, traced_labels),
                      'measure': '-', 'metric': 'has_good_prediction'},
                     ignore_index=True)
    return out.assign(file=file)


def get_centroid_distance(data):
    a = pd.DataFrame(data.predicted_features['center'].items(), columns=['muscle', 'value']) \
        .set_index('muscle')
    b = pd.DataFrame(data.traced_features['center'].items(), columns=['muscle', 'value']) \
        .set_index('muscle')
    diff = (a - b) ** 2
    return (
        pd.DataFrame(np.sqrt(diff['value'].map(np.sum)), index=diff.index).assign(measure='centroid', metric='distance')
    )


def get_various_metrics(data, measure):
    a = pd.DataFrame(data.predicted_features[measure].items(), columns=['muscle', 'value']).set_index('muscle')
    b = pd.DataFrame(data.traced_features[measure].items(), columns=['muscle', 'value']).set_index('muscle')
    return [
        a.assign(measure=measure, metric='predicted'),
        b.assign(measure=measure, metric='traced'),
        (a / b).assign(measure=measure, metric='ratio'),
        (a - b).abs().assign(measure=measure, metric='diff'),
        ((a - b).abs() / b).assign(measure=measure, metric='relative_diff')
    ]
