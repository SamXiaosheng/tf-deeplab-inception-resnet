from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2

class InceptionResnetV2Test(tf.test.TestCase):
    def testNetworkSetup(self):
        images = tf.placeholder(tf.float32, shape=[10, 299, 299, 3])
        net, _ = inception_resnet_v2(images)
        self.assertListEqual(net.get_shape().as_list(), [10, 35, 35, 1204])

if __name__ == "__main__":
    tf.test.main()
