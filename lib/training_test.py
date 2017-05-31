from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from labels import NumClasses
from training import average_accuracy

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
    def test_average_accuracy_true_positive(self):
        gt = create_gt()
        preds = create_preds()
        expected_accuracy = (19 + (2/3) + 0) / NumClasses

        with self.test_session() as sess:
            computed_accuracy = sess.run(average_accuracy(gt, preds))
            self.assertAlmostEqual(computed_accuracy, expected_accuracy)

if __name__ == '__main__':
    tf.test.main()
