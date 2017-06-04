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
    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)
    manager = PipelineManager("/root/tf-deeplab-inception-resnet/DATA", "dev2.txt")
    img_queue = manager.create_queues()
    manager.start_queues(sess)

    image_batch, ground_truth_batch = img_queue.dequeue_up_to(2, name="ImageBatchDequeue")
    net = deeplab.network(image_batch)
    print("NN", net.get_shape())
    net_resized = tf.image.resize_images(net, [350, 500], method=tf.image.ResizeMethod.BILINEAR)

    sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])

    for i in range(1):
        n_summ, nr_summ = sess.run([ net, net_resized ])
        print("NSUMM", n_summ.shape)
        print("NSUMM", nr_summ.shape)


        # n, nr = sess.run([ net, net_resized ])
        # img, gt = sess.run([ image_batch, ground_truth_batch])
        # print(i, img.shape, gt.shape)
        # xx = sess.run(ground_truth_batch)
        # print(">>", xx.shape)
        # print(np.unique(xx))

        # img_summ = sess.run(tf.summary.image("GroundTruth", ground_truth_batch))
        # summary_writer.add_summary(n_summ, i)
        # summary_writer.add_summary(nr_summ, i)

    manager.stop_queues()

    summary_writer.flush()
