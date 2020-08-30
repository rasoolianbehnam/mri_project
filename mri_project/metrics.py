import numpy as np

all_stats = []
def stat(fun):
    all_stats.append(fun.__name__)
    return fun
    
def get_confusion_matrix(x, y):
    class_numbers = np.unique(x)# np.uint8([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])//10
    print(class_numbers, len(class_numbers))
    matches = np.zeros([class_numbers.max()+1]*2, dtype='float')
    all_positives = matches.copy()+1e-9
    print(matches.shape)
    for cls1 in class_numbers:
        for cls2 in class_numbers:
            a = ((x==cls1) & (y==cls2)).sum()
            b = ((x==cls2)).sum()
            matches[int(cls1), int(cls2)] = a 
            all_positives[int(cls1), int(cls2)] = b+1e-9
    return matches/all_positives


@stat
def total_accuracy(pred, orig):
    w, h = orig.shape
    total_pixels = w * h
    return (pred == orig).sum() / total_pixels


@stat
def accuracy(pred, orig):
    uniques = set(np.unique(pred)) | set(np.unique(orig))
    out = {}
    for u in uniques:
        orig_mask = orig == u
        pred_mask = pred == u
        out[u] = np.sum(pred_mask & orig_mask) / np.sum(orig_mask)
    return out

@stat
def precision(pred, orig):
    uniques = set(np.unique(pred)) | set(np.unique(orig))
    out = {}
    for u in uniques:
        pred_mask = pred == u
        orig_mask = orig != u
        out[u] = np.sum(pred_mask & orig_mask) / np.sum(orig_mask)
    return out

@stat
def recall(pred, orig):
    uniques = set(np.unique(pred))
    out = {}
    for u in uniques:
        pred_mask = pred != u
        orig_mask = orig == u
        out[u] = np.sum(pred_mask & orig_mask) / np.sum(orig_mask)
    return out