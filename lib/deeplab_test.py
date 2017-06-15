from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from deeplab import _atrous_conv, network

NUM_BATCH = 10
RESNET_OUTPUT_SHAPE = [NUM_BATCH, 35, 35, 1204]
NUM_CLASSES=15

class DeeplabTest(tf.test.TestCase):
    def testAtrous(self):
        input_net = tf.placeholder(tf.float32, shape=RESNET_OUTPUT_SHAPE)

        atrous = _atrous_conv(input_net, rate=6, num_classes=NUM_CLASSES)
        self.assertListEqual(atrous.get_shape().as_list(), [NUM_BATCH, 35, 35, NUM_CLASSES])

    def testNetwork(self):
        images = tf.placeholder(tf.float32, [NUM_BATCH, 299, 299, 3])
        deeplab_net, _ = network(images, num_classes=NUM_CLASSES)
        self.assertListEqual(deeplab_net.get_shape().as_list(), [10, 35, 35, NUM_CLASSES])

    def testNetworkWithLargeImages(self):
        images = tf.placeholder(tf.float32, [NUM_BATCH, 350, 500, 3])
        deeplab_net, _ = network(images, num_classes=NUM_CLASSES)
        self.assertListEqual(deeplab_net.get_shape().as_list(), [10, 41, 60, NUM_CLASSES])

    def testNetworkWithResize(self):
        images = tf.placeholder(tf.float32, [NUM_BATCH, 350, 500, 3])
        deeplab_net, _ = network(images, num_classes=NUM_CLASSES, resize=[350, 500])
        self.assertListEqual(deeplab_net.get_shape().as_list(), [10, 350, 500, NUM_CLASSES])

if __name__ == "__main__":
    tf.test.main()
