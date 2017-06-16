from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

ExcludedScopes = [
    "InceptionResnetV2/Mixed_6a",
    "InceptionResnetV2/Mixed_7a",
    "InceptionResnetV2/Repeat_2/block8",
    "InceptionResnetV2/Block8",
    "InceptionResnetV2/Repeat_1/block17"
]

def _excluded(name):
    excluded = False
    for excluded_scope in ExcludedScopes:
        if name.startswith(excluded_scope):
            excluded = True
            break

    return excluded

def load_checkpoint(checkpoint_dir, train_dir, sess):
    load_dir = None

    if tf.train.latest_checkpoint(train_dir):
        print("Training checkpoint exists at %s" % (train_dir))
        load_dir = tf.train.latest_checkpoint(train_dir)
    elif (checkpoint_dir is not None):
        print("Found checkpoint at %s" % (checkpoint_dir))
        load_dir = checkpoint_dir

    if (load_dir):
        variables_to_restore = []
        for var in slim.get_model_variables():
            if (not _excluded(var.op.name)):
                variables_to_restore.append(var)

        assign_fn = slim.assign_from_checkpoint_fn(
            checkpoint_dir,
            variables_to_restore,
            ignore_missing_vars=True)

        assign_fn(sess)
