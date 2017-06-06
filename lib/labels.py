from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

IgnoreLabel = -1

Labels = [
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

LabelsToIndices = dict( zip(Labels, range(len(Labels))) )

NumClasses = len(Labels)

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

Colormap = _colormap()

def index_of_label(label):
    return LabelsToIndices.get(label)

def label_of_index(ind):
    return Labels[ind]

def color_of_label(label):
    return color_of_index(index_of_label(label))

def color_of_index(ind):
    return Colormap[ind, :]

def to_one_hot(tensor, scope="ToOneHot"):
    with tf.name_scope(scope):
        return tf.one_hot(tensor, NumClasses)

def to_labels(tensor, scope="ToLabels"):
    with tf.name_scope(scope):
        labeled_tensor = IgnoreLabel * tf.ones_like(tensor, dtype=tf.int32)[:, :, :, 0]

        for label in Labels:
            opname = "Mask_%s" % (label)
            color = color_of_label(label)
            index = index_of_label(label)
            mask = tf.reduce_all(tf.equal(tensor, color, name=opname), axis=3, name=opname)

            labeled_tensor = tf.where(mask, index * tf.ones_like(labeled_tensor), labeled_tensor)

        return labeled_tensor

def _image_layers(shape):
    layers = []
    for _ in range(3):
        layers.append(255.0 * tf.ones([ shape[0], shape[1], shape[2] ], dtype=tf.float32))

    return layers

def to_images(tensor, scope="ToImage"):
    with tf.name_scope(scope):
        r_layer, g_layer, b_layer = _image_layers(tensor.shape)

        for label in Labels:
            index = index_of_label(label)
            r, g, b = color_of_label(label)
            mask = tf.equal(tensor, index)

            r_layer = tf.where(mask, r * tf.ones_like(r_layer), r_layer)
            g_layer = tf.where(mask, g * tf.ones_like(g_layer), g_layer)
            b_layer = tf.where(mask, b * tf.ones_like(b_layer), b_layer)

        return tf.stack([ r_layer, g_layer, b_layer ], axis=3)
