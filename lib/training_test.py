from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from labels import IgnoreLabel, NumClasses
from training import average_accuracy, cross_entropy

# GT
# 0  1
# 1 20

# PRED
# 0 1
# 1 1

# Expected Accuracy (0)  = 1 / (1 + 0 + 0) = 100%
# Expected Accuracy (1)  = 2 / (2 + 0 + 1) = 66%
# Expected Accuracy (20) = 0 / (0 + 0 + 1) = 0%
# Avg = 55.33%

def create_gt():
    gt = np.zeros((1, 2, 2))
    gt[0, 0, 0] = 0
    gt[0, 0, 1] = 1
    gt[0, 1, 0] = 1
    gt[0, 1, 1] = 20

    return tf.constant(gt)

def create_preds():
    preds = np.zeros((1, 2, 2, NumClasses))
    preds[0, 0, 0, 0] = 1.0
    preds[0, 0, 1, 1] = 1.0
    preds[0, 1, 0, 1] = 1.0
    preds[0, 1, 1, 1] = 1.0

    return tf.constant(preds)

class TrainingTest(tf.test.TestCase):
    # Next couple of test cases just test some random scenarios that ought to be satisfied
    def test_average_accuracy_true_positive(self):
        gt = create_gt()
        preds = create_preds()
        expected_accuracy = (19 + (2/3) + 0) / NumClasses

        with self.test_session() as sess:
            computed_accuracy = sess.run(average_accuracy(gt, preds))
            self.assertAlmostEqual(computed_accuracy, expected_accuracy)

    def test_average_accuracy_false_negative(self):
        gt = tf.constant(np.array([1]).reshape((1, 1, 1)))
        p = np.zeros((1, 1, 1, NumClasses))
        p[0, 0, 0, 0] = 1
        preds = tf.constant(p)
        expected_accuracy = 19 / 21

        with self.test_session() as sess:
            computed_accuracy = sess.run(average_accuracy(gt, preds))

            self.assertAlmostEqual(expected_accuracy, computed_accuracy)

    def test_average_accuracy_with_multiple_samples(self):
        gt = np.zeros((2, 1, 21))
        gt[0, 0, :] = np.arange(21)
        gt[1, 0, :] = (np.arange(21) + 1) % 21
        gt = tf.constant(gt)

        preds = np.zeros((2, 1, 21, 21))
        for i in range(21):
            preds[:, 0, i, i] = 1.0
        preds = tf.constant(preds)

        with self.test_session() as sess:
            computed_accuracy = sess.run(average_accuracy(gt, preds))

            self.assertAlmostEqual(0.5, computed_accuracy)

    def test_average_accuracy_with_ignore_pixels(self):
        gt = np.zeros((1, 1, 2))
        gt[0, 0, 1] = IgnoreLabel
        gt = tf.constant(gt)

        preds = np.zeros((1, 1, 2, NumClasses))
        preds[0, 0, :, 0] = 1.0
        preds = tf.constant(preds)

        with self.test_session() as sess:
            computed_accuracy = sess.run(average_accuracy(gt, preds))

            self.assertAlmostEqual(1.0, computed_accuracy)

    def test_cross_entropy(self):
        gt = np.zeros((1, 1, 1))
        gt[0, 0, 0] = 1
        gt = tf.constant(gt, dtype=tf.int32)

        logits = -1000.0 * np.ones((1, 1, 1, 21))
        logits[0, 0, 0, 0] = 0.0
        logits[0, 0, 0, 1] = 0.0
        logits = tf.constant(logits)

        with self.test_session() as sess:
            computed_xentropy = sess.run(cross_entropy(gt, logits))

            self.assertAlmostEqual(-np.log(0.5), computed_xentropy)

if __name__ == '__main__':
    tf.test.main()
