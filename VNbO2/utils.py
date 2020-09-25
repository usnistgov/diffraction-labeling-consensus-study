import numpy as np

def order_labels(labels, C):
    """ order labels consistently with ground truth labels -- start from pure VO2 """
    C_avg = [np.mean(C[labels == label]) for label in np.unique(labels)]
    label_order = np.argsort(C_avg)

    reordered = np.zeros_like(labels)

    for new_label, label in enumerate(label_order):
        reordered[labels == label] = new_label

    return reordered
