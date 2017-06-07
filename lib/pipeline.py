from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

def assert_dir_exists(dirname):
    assert os.path.exists(dirname), "Dir: %s does not exist!" % (dirname)

def create_paths(name, img_path, seg_path):
    with tf.name_scope("PathCreator"):
        with tf.name_scope("ImagePath"):
            img_filename = tf.constant(img_path + "/") + name + tf.constant(".jpg")

        with tf.name_scope("GroundTruthPath"):
            seg_filename = tf.constant(seg_path + "/") + name + tf.constant(".png")

    return [ img_filename, seg_filename ]

class PipelineManager(object):
    DirImageSets = "ImageSets/Segmentation"
    DirJPEGImages = "JPEGImages"
    DirSegmentationClass = "SegmentationClass"
    Capacity = 1000

    def __init__(self, root, data_set, target_size=[350, 500], device="/cpu:0", threads=10):
        self.root = root
        self.target_size = target_size
        self.device = device
        self.threads = threads

        self.dir_image_sets = os.path.join(root, PipelineManager.DirImageSets)
        self.dir_jpeg_images = os.path.join(root, PipelineManager.DirJPEGImages)
        self.dir_segmentations = os.path.join(root, PipelineManager.DirSegmentationClass)
        self.data_set = os.path.join(self.dir_image_sets, data_set)

        [ assert_dir_exists(d) for d in [self.dir_image_sets, self.dir_jpeg_images, self.dir_segmentations, self.data_set] ]

    def create_queues(self):
        with tf.device(self.device):
            with tf.name_scope("Pipeline"):
                path_name_queue = self._path_name_queue()
                image_queue = self._image_tensors_queue(path_name_queue)

                return image_queue

    def start_queues(self, sess):
        with tf.device(self.device):
            self.coordinator = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coordinator)

    def stop_queues(self):
        self.coordinator.request_stop()
        self.coordinator.join(self.threads)

    def _path_name_queue(self):
        with open(self.data_set) as f:
            cases = [ line.strip() for line in f ]

        case_name_queue = tf.train.string_input_producer(tf.constant(cases, name="case_names"),
            shuffle=True, capacity=len(cases), name="CaseNameQueue")

        path_name_queue = tf.FIFOQueue(len(cases), dtypes=[tf.string, tf.string],
            shapes=[[], []], name="ImagePathQueue")

        enqueue_op = path_name_queue.enqueue(
            create_paths(case_name_queue.dequeue(), self.dir_jpeg_images, self.dir_segmentations))

        qr = tf.train.QueueRunner(queue=path_name_queue, enqueue_ops=[enqueue_op])
        tf.train.add_queue_runner(qr)

        return path_name_queue

    def _read_and_decode_image(self, path, decoder, size, scope="ImageFile"):
        with tf.name_scope(scope):
            img = decoder(tf.read_file(path), channels=3)
            resized = tf.image.resize_images(img, size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return tf.cast(resized, dtype=tf.float32)

    def _image_tensors_queue(self, path_queue):
        paths = path_queue.dequeue()

        img_shape = self.target_size + [3]
        img_queue = tf.FIFOQueue(1000, [tf.float32, tf.float32],
            shapes=[img_shape, img_shape], name="ImageQueue")

        img_enqueue_op = img_queue.enqueue([
            self._read_and_decode_image(paths[0], tf.image.decode_jpeg, self.target_size,
                scope="ReadImageFile"),
            self._read_and_decode_image(paths[1], tf.image.decode_png, self.target_size,
                scope="ReadGroundTruthFile")
        ] * self.threads)

        qr = tf.train.QueueRunner(queue=img_queue, enqueue_ops=[ img_enqueue_op ] * 2)
        tf.train.add_queue_runner(qr)

        return img_queue
