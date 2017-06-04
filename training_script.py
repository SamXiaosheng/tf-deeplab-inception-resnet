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
from training import average_accuracy
from labels import to_labels

TARGET_SIZE = [350, 500]

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)
    manager = PipelineManager("/mnt/hdd0/datasets/pascal/VOCdevkit/VOC2012", "dev.txt",
        target_size=TARGET_SIZE)

    img_queue = manager.create_queues()
    manager.start_queues(sess)

    image_batch, ground_truth_batch = img_queue.dequeue_up_to(2, name="ImageBatchDequeue")
    preds = deeplab.network(image_batch)
    resized_preds = tf.image.resize_images(preds, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    avg_accuracy = average_accuracy(to_labels(ground_truth_batch), resized_preds)

    sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])

    for i in range(1):
        foo = sess.run(avg_accuracy)
        print("FOO", foo)



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
