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

from pipeline import PipelineManager

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

with tf.Session() as sess:
    manager = PipelineManager("/root/tf-deeplab-inception-resnet/DATA", "dev.txt")
    q = manager.create_queues()

    # deq = q.dequeue_up_to(1, name="FizzyPoof")
    deq = q.dequeue(name="FizzyPoof")

    manager.start_queues(sess)

    for i in range(10):
        img, gt = sess.run(deq)
        print(i, img.shape, gt.shape)

#     imgs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
#     net = deeplab.network(imgs)


    manager.stop_queues()

    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)
    summary_writer.flush()
