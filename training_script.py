from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

LIB_DIR = os.path.abspath("./lib")
sys.path.extend([ LIB_DIR ])

import numpy as np
import tensorflow as tf
import deeplab

from pipeline import PipelineManager
from training import average_accuracy, cross_entropy
from labels import to_labels, to_images

OUT_DIR = "/tmp/deeplab"
TARGET_SIZE = [350, 500]
BATCH_SIZE = 2
STEPS = 100000
SAVE_EVERY = 1000

def create_and_start_queues(sess):
    manager = PipelineManager("/mnt/hdd0/datasets/pascal/VOCdevkit/VOC2012", "dev.txt",
        target_size=TARGET_SIZE, device="/cpu:0", threads=(2*BATCH_SIZE))

    img_queue = manager.create_queues()
    manager.start_queues(sess)

    with tf.device("/cpu:0"):
        image_batch, ground_truth_batch = img_queue.dequeue_up_to(BATCH_SIZE, name="ImageBatchDequeue")

    return manager, image_batch, ground_truth_batch

def create_image_summaries(imgs, gt, predicted):
    with tf.device("/cpu:0"):
        with tf.name_scope("ImageSummaries"):
            im_summ = tf.summary.image("Image", imgs[0:2, :, :, :])
            gt_summ = tf.summary.image("GroundTruth", gt[0:2, :, :, :])

            pred_imgs = to_images(tf.argmax(predicted, axis=3))
            pred_summ = tf.summary.image("Prediction", pred_imgs[0:2, :, :, :])

            return [ im_summ, gt_summ, pred_summ ]

def create_savers(graph):
    summary_writer = tf.summary.FileWriter(OUT_DIR, graph=graph)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

    return summary_writer, saver

def save_checkpoint(step, sess, saver, summary_writer, avg_accuracy, xentropy, img_summaries):
    _avg_accuracy, _xentropy, _img_summaries = sess.run([ avg_accuracy, xentropy, img_summaries ])
    print("%7d: Accuracy = %7.3f, Xentropy = %7.3f" % (step, 100 * _avg_accuracy, _xentropy))

    model_path = os.path.join(OUT_DIR, "model.ckpt")
    saver.save(sess, model_path, global_step=step)
    for summary in _img_summaries:
        summary_writer.add_summary(summary, step)

def main(_):
    os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

    with tf.Session() as sess:
        manager, image_batch, ground_truth_batch = create_and_start_queues(sess)
        preds = deeplab.network(image_batch)
        labeled_ground_truth = to_labels(ground_truth_batch, device="/cpu:0")
        resized_preds = tf.image.resize_images(preds, TARGET_SIZE,
            method=tf.image.ResizeMethod.BILINEAR)
        img_summaries = create_image_summaries(image_batch, ground_truth_batch, resized_preds)

        avg_accuracy = average_accuracy(labeled_ground_truth, resized_preds, device="/cpu:0")
        xentropy = cross_entropy(labeled_ground_truth, resized_preds, device="/cpu:0")
        reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_losses = tf.add_n(reg_vars)
        total_loss = xentropy + reg_losses

        train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(total_loss)

        sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])
        summary_writer, saver = create_savers(sess.graph)

        for i in range(STEPS):
            sess.run(train_step)

            if (i % SAVE_EVERY == 0):
                save_checkpoint(i, sess, saver, summary_writer, avg_accuracy, xentropy,
                    img_summaries)

        manager.stop_queues()
        summary_writer.flush()

if __name__ == '__main__':
    tf.app.run()
