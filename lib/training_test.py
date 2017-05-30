from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from labels import NumClasses
from training import cross_entropy_loss

def create_gt():
    gt = np.zeros((1, 2, 2))
    gt[0, 0, 0] = 0
    gt[0, 0, 1] = 1
    gt[0, 1, 0] = 1
    gt[0, 1, 1] = 20

    return tf.constant(gt)

def create_preds():
    preds = np.zeros((1, 2, 2, NumClasses))

    return tf.constant(preds)

class TrainingTest(tf.test.TestCase):
    def test_cross_entropy_loss(self):
        gt = create_gt()
        preds = create_preds()
        expected_xentropy = 123.0

        with self.test_session() as sess:
            computed_xentropy = cross_entropy_loss(gt, preds)
            self.assertAlmostEqual(expected_xentropy, computed_xentropy)



if __name__ == '__main__':
    tf.test.main()
