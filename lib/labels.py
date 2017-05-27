from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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

def _bitiget(num, pos):
    return ((num & (1 << pos)) != 0)

def _colormap():
    cmap = np.zeros((255, 3))
    for i in range(1, 256):
        id = i-1
        r = 0; g = 0; b = 0
        for j in range(8):
            r = r | (_bitiget(id, 0) << 7-j)
            g = g | (_bitiget(id, 1) << 7-j)
            b = b | (_bitiget(id, 2) << 7-j)
            id = id >> 3

        cmap[i-1, :] = np.array([ r, g, b ])

    return cmap

_COLORMAP = _colormap()

def index_of_label(label):
    return _LABELS_AS_DICT.get(label)

def label_of_index(ind):
    return _LABELS[ind]

def color_of_label(label):
    return color_of_index(index_of_label(label))

def color_of_index(ind):
    return _COLORMAP[ind, :]
