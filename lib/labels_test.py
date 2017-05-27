from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

if __name__ == '__main__':
    tf.test.main()
