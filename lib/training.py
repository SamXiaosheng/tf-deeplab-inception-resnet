from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from labels import IgnoreLabel, Labels, index_of_label, to_one_hot, NumClasses

def _sum(tensor):
    return tf.reduce_sum(tf.cast(tensor, dtype=tf.int64), axis=[1, 2])

def _avg_label_accuracy(label, gt, preds):
    with tf.name_scope("LabelAverageAccuracy_%s" % (label)):
        index = index_of_label(label)

        preds_true = tf.equal(preds, index)
        gt_true = tf.equal(gt, index)
        ignore_pixels = tf.equal(gt, IgnoreLabel)

        true_positives = tf.logical_and(preds_true, gt_true)
        false_negatives = tf.logical_and(gt_true, tf.logical_not(preds_true))

        # Note, we need to account for pixels that have the special IgnoreLabel.
        # These pixels should not be counted as false positives
        false_positives = tf.logical_and(
            tf.logical_not(ignore_pixels),
            tf.logical_and(preds_true, tf.logical_not(gt_true)))

        denoms = tf.add_n([ _sum(true_positives), _sum(false_positives), _sum(false_negatives) ])

        accuracies = _sum(true_positives) / denoms
        accuracies = tf.where(tf.is_nan(accuracies), tf.ones_like(accuracies), accuracies)

        return tf.reduce_mean(accuracies)

def average_accuracy(gt, preds):
    with tf.name_scope("AverageAccuracy"):
        predictions = tf.argmax(preds, axis=3)
        avg_acc = tf.constant(0.0, dtype=tf.float64, name="avg_accuracy")

        for label in Labels:
            avg_acc += _avg_label_accuracy(label, gt, predictions)

        return avg_acc  / len(Labels)

def cross_entropy(gt, logits):
    with tf.name_scope("CrossEntropy"):
        raw_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=logits)
        not_nan = tf.logical_not(tf.is_nan(raw_xentropy))

        mask_ones = tf.ones_like(raw_xentropy)
        mask_zeros = tf.zeros_like(raw_xentropy)

        xentropy = tf.reduce_sum(tf.where(not_nan, raw_xentropy, mask_zeros))
        xentropy /= tf.reduce_sum(tf.where(not_nan, mask_ones, mask_zeros))

        return xentropy
