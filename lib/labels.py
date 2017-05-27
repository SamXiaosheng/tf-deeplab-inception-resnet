from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

_LABELS_AS_DICT = dict( zip(_LABELS, range(len(_LABELS))) )

def index_of_label(label):
    return _LABELS_AS_DICT.get(label)

def label_of_index(ind):
    return _LABELS[ind]

def color_of_label(label):
    pass

def color_of_index(ind):
    pass
