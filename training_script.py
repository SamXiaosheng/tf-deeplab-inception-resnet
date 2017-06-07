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
from training import average_accuracy, cross_entropy
from labels import to_labels, to_images

TARGET_SIZE = [350, 500]
BATCH_SIZE = 10

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

def create_and_start_queues():
    manager = PipelineManager("/mnt/hdd0/datasets/pascal/VOCdevkit/VOC2012", "train.txt",
        target_size=TARGET_SIZE, device="/cpu:0", threads=20)

    img_queue = manager.create_queues()
    manager.start_queues(sess)

    image_batch, ground_truth_batch = img_queue.dequeue_up_to(BATCH_SIZE,  name="ImageBatchDequeue")

    return manager, image_batch, ground_truth_batch

def create_image_summaries(imgs, gt, predicted):
    with tf.name_scope("ImageSummaries"):
        im_summ = tf.summary.image("Image", imgs[0:2, :, :, :])
        gt_summ = tf.summary.image("GroundTruth", gt[0:2, :, :, :])

        pred_imgs = to_images(tf.argmax(predicted, axis=3))
        pred_summ = tf.summary.image("Prediction", pred_imgs[0:2, :, :, :])

        return [ im_summ, gt_summ, pred_summ ]

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=sess.graph)

    manager, image_batch, ground_truth_batch = create_and_start_queues()

    preds = deeplab.network(image_batch)

    labeled_ground_truth = to_labels(ground_truth_batch)
    resized_preds = tf.image.resize_images(preds, TARGET_SIZE,
        method=tf.image.ResizeMethod.BILINEAR)

    avg_accuracy = average_accuracy(labeled_ground_truth, resized_preds)
    xentropy = cross_entropy(labeled_ground_truth, resized_preds)
    train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(xentropy)

    img_summaries = create_image_summaries(image_batch, ground_truth_batch, resized_preds)

    sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])

    for i in range(100000):
        acc, _ = sess.run([ avg_accuracy, train_step ])

        if (i % 100 == 0):
            print("Accuracy = %7.3f" % (100 * acc))
            for summary in sess.run(img_summaries):
                summary_writer.add_summary(summary, i)

    manager.stop_queues()
    summary_writer.flush()
