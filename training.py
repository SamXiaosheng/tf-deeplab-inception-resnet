from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

OUT_DIR = "/tmp/deeplab"
LIB_DIR = os.path.abspath("./lib")
sys.path.extend([ LIB_DIR ])

import tensorflow as tf
import deeplab

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

with tf.Session() as sess:
    imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    net = deeplab.network(imgs)

    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)
    summary_writer.flush()
