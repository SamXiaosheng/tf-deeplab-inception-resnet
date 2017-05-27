from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2

def _atrous_conv(net, rate, num_classes, scope="AtrousConv2d"):
    with tf.variable_scope(scope):
        num_filters = net.get_shape()[-1]

        filters = tf.get_variable("weights", shape=[ 3, 3, num_filters, num_classes],
            trainable=True)

        biases = tf.get_variable("biases", shape=[ num_classes ], trainable=True)

        atrous = tf.nn.atrous_conv2d(net, filters, rate, padding="SAME")
        atrous = tf.nn.bias_add(atrous, biases)

        return atrous

def _atrous_spatial_pyramind_pooling(net, num_classes):
    with tf.variable_scope("ASPP"):
        aspp_r6 = _atrous_conv(net, 6, num_classes, scope="branch_6")
        aspp_r12 = _atrous_conv(net, 6, num_classes, scope="branch_12")
        aspp_r18 = _atrous_conv(net, 6, num_classes, scope="branch_18")
        aspp_r24 = _atrous_conv(net, 6, num_classes, scope="branch_24")

        return tf.add_n([ aspp_r6, aspp_r12, aspp_r18, aspp_r24 ])

def network(imgs, num_classes=21, is_training=True,
    dropout_keep_prob=0.8, reuse=None):

    resnet = inception_resnet_v2(imgs, is_training=is_training,
        dropout_keep_prob=dropout_keep_prob, reuse=reuse)

    aspp = _atrous_spatial_pyramind_pooling(resnet, num_classes)

    return aspp
