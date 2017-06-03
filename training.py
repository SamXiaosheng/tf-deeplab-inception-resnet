from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

OUT_DIR = "/tmp/deeplab"
LIB_DIR = os.path.abspath("./lib")
sys.path.extend([ LIB_DIR ])

import numpy as np
import tensorflow as tf
import deeplab

from pipeline import PipelineManager

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

with tf.Session() as sess:


    manager = PipelineManager("/root/tf-deeplab-inception-resnet/DATA", "dev2.txt")
    img_queue = manager.create_queues()
    manager.start_queues(sess)

    image_batch, ground_truth_batch = img_queue.dequeue_up_to(2, name="ImageBatchDequeue")
    net = deeplab.network(image_batch)

    sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])

    for i in range(1):
        # img, gt = sess.run([ image_batch, ground_truth_batch])
        # print(i, img.shape, gt.shape)
        print(">>", sess.run(net))

    manager.stop_queues()

    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)
    summary_writer.flush()
