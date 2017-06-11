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

    def test_to_labels_returns_appropriate_dtype(self):
        gt_image = np.array(labels.color_of_label("bus")).reshape((1, 1, 1, 3))
        labeled_image = labels.to_labels(tf.constant(gt_image))

        self.assertTrue(labeled_image.dtype == tf.int32)

    def test_tensor_reverse_conversion(self):
        with self.test_session() as sess:
            for label in labels.Labels:
                labeled = np.array(labels.index_of_label(label)).reshape((1, 1, 1))
                expected = np.array(labels.color_of_label(label)).reshape((1, 1, 1, 3))
                img = sess.run(labels.to_images(tf.constant(labeled)))

                self.assertAllEqual(img, expected)

    def test_bijective_behavior(self):
        imgs = 255.0 * np.ones((3, 2, 2, 3))
        for i in range(3):
            imgs[i, :, :, :] = labels.color_of_index(i)

        with self.test_session() as sess:
            labeled = labels.to_labels(tf.constant(imgs))
            converted = labels.to_images(labeled)
            converted_imgs = sess.run(converted)

            self.assertAllClose(converted_imgs, imgs)

    def test_ignore_pixels_handled(self):
        imgs = 255.0 * np.ones((1, 2, 2, 3))
        expected = labels.IgnoreLabel * np.ones((1, 2, 2))

        with self.test_session() as sess:
            labeled = labels.to_labels(tf.constant(imgs))
            labeled = sess.run(labeled)

            self.assertAllClose(expected, labeled)

    def test_to_one_hot(self):
        with self.test_session() as sess:
            for label in labels.Labels:
                idx = labels.index_of_label(label)
                labeled = np.array(idx).reshape((1, 1, 1))
                expected = np.zeros((1, 1, 1, 21))
                expected[0, 0, 0, idx] = 1.0

                one_hot = sess.run(labels.to_one_hot(tf.constant(labeled)))
                self.assertAllEqual(one_hot, expected)

if __name__ == '__main__':
    tf.test.main()
