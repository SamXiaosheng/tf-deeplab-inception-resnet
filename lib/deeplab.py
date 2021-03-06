from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

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

def _create_summaries(resnet_endpoints):
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    for end_point in resnet_endpoints:
        x = resnet_endpoints[end_point]
        summaries.add(tf.summary.histogram('activations/' + end_point, x))
        summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))

    for variable in slim.get_model_variables():
        summaries.add(tf.summary.histogram(variable.op.name, variable))

    return list(summaries)

def network(imgs, num_classes=21, is_training=True,
    dropout_keep_prob=0.8, reuse=None, resize=None):

    arg_scope = inception_resnet_v2_arg_scope(weight_decay=0.0005)
    with slim.arg_scope(arg_scope) as scope:
        resnet, endpoints = inception_resnet_v2(imgs, is_training=is_training,
            dropout_keep_prob=dropout_keep_prob, reuse=reuse)

    aspp = _atrous_spatial_pyramind_pooling(resnet, num_classes)

    summaries = _create_summaries(endpoints)

    if (resize is not None):
        aspp = tf.image.resize_images(aspp, resize, method=tf.image.ResizeMethod.BILINEAR)

    return aspp, summaries
