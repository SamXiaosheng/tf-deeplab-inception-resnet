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

    def test_tensor_conversion(self):
        with self.test_session() as sess:
            for label in labels.Labels:
                gt_image = np.array(labels.color_of_label(label)).reshape((1, 1, 1, 3))
                expected = np.array(labels.index_of_label(label)).reshape((1, 1, 1))

                gt = tf.constant(gt_image)
                labeled = sess.run(labels.to_labels(gt))

                self.assertAllEqual(labeled, expected)

    def test_tensor_reverse_conversion(self):
        with self.test_session() as sess:
            for label in labels.Labels:
                labeled = np.array(labels.index_of_label(label)).reshape((1, 1, 1))
                expected = np.array(labels.color_of_label(label)).reshape((1, 1, 1, 3))
                img = sess.run(labels.to_images(tf.constant(labeled)))

                self.assertAllEqual(img, expected)

if __name__ == '__main__':
    tf.test.main()
