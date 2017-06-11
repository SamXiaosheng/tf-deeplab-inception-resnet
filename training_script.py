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
from training import average_accuracy, cross_entropy
from labels import to_labels, to_images

TARGET_SIZE = [350, 500]
BATCH_SIZE = 2

os.system("rm %s" % (os.path.join(OUT_DIR, "*")))

def create_and_start_queues():
    manager = PipelineManager("/mnt/hdd0/datasets/pascal/VOCdevkit/VOC2012", "dev.txt",
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
    reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses = tf.add_n(reg_vars)
    total_loss = xentropy + reg_losses

    train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(total_loss)

    img_summaries = create_image_summaries(image_batch, ground_truth_batch, resized_preds)
    sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])

    for i in range(100):
        acc, xent, _ = sess.run([ avg_accuracy, xentropy, train_step ])
        if (i % 10 == 0):
            print("%7d: Accuracy = %7.3f, Xentropy = %7.3f" % (i, 100 * acc, xent))

            for summary in sess.run(img_summaries):
                summary_writer.add_summary(summary, i)

    manager.stop_queues()
    summary_writer.flush()

    # avg_accuracy = average_accuracy(labeled_ground_truth, resized_preds)
    # xentropy, denom = cross_entropy(labeled_ground_truth, resized_preds)
    # reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_losses = tf.add_n(reg_vars)
    #
    # loss_fn = xentropy + reg_losses
    #
    # print("REG", reg_losses)
    #
    # test_loss = reg_losses
    #
    # grads_and_vars = tf.train.MomentumOptimizer(0.001, 0.9).compute_gradients(test_loss)
    # grads_and_vars = filter(lambda gv: gv[0] is not None, grads_and_vars)
    #
    # for g, v in grads_and_vars:
    #     if (g == None):
    #         print("NONE", g, v)
    #
    #
    # grads = [ gv[0] for gv in grads_and_vars ]
    # variables = [ gv[1] for gv in grads_and_vars ]
    # #
    # # img_summaries = create_image_summaries(image_batch, ground_truth_batch, resized_preds)
    #
    # # train_step = tf.train.MomentumOptimizer(0.00001, 0.9).minimize(test_loss)
    # sess.run([ tf.local_variables_initializer(), tf.global_variables_initializer() ])
    #
    # # train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # # [ print(v) for v in train_vars ]
    #
    # print("GRADS", grads)
    # evaled_grads = sess.run(grads)
    # evaled_vars = sess.run(variables)
    # foo = zip(variables, evaled_grads)
    #
    # for g1, g2 in foo:
    #     print("*********")
    #     print("grad", g1.name)
    #     print(g2)
    #
    #
    #
    # # for i in range(100000):
    # #     print("TRAIN", sess.run(train_step))
    # #     print("TEST LOSS", sess.run(test_loss))
    # #     # acc, xent, _, _denom = sess.run([ avg_accuracy, xentropy, train_step, denom ])
    # #     #
    # #     # if (i % 1 == 0):
    # #     #     print("Accuracy = %7.3f, Xentropy = %7.3f, denom = %f" % (100 * acc, xent, _denom))
    # #
    # #         for summary in sess.run(img_summaries):
    # #             summary_writer.add_summary(summary, i)
