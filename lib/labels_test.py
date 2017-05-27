from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import labels

class LabelsTest(tf.test.TestCase):
    def test_index_of_label(self):
        self.assertEqual(labels.index_of_label("background"), 0)
        self.assertEqual(labels.index_of_label("bus"), 6)
        self.assertIsNone(labels.index_of_label("fizzypoof"))

    def test_label_of_index(self):
        self.assertEqual(labels.label_of_index(0), "background")
        self.assertEqual(labels.label_of_index(20), "tvmonitor")

    def test_color_of_label(self):
        self.assertAllEqual(labels.color_of_label("background"), np.array([0, 0, 0]))
        self.assertAllEqual(labels.color_of_label("bus"), np.array([0, 128, 128]))
        self.assertAllEqual(labels.color_of_label("person"), np.array([192, 128, 128]))

    def test_equality_of_colors(self):
        for label in labels.Labels:
            self.assertAllEqual(labels.color_of_label(label),
                labels.color_of_index(labels.index_of_label(label)))

if __name__ == '__main__':
    tf.test.main()
