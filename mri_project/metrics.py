import numpy as np

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
