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

if __name__ == '__main__':
    tf.test.main()
