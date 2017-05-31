from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

def assert_dir_exists(dirname):
    assert os.path.exists(dirname), "Dir: %s does not exist!" % (dirname)


class PipelineManager(object):
    DirImageSets = "ImageSets/Segmentation"
    DirJPEGImages = "JPEGImages"
    DirSegmentationClass = "SegmentationClass"

    def __init__(self, root):
        self.root = root
        self.dir_image_sets = os.path.join(root, PipelineManager.DirImageSets)
        self.dir_jpeg_images = os.path.join(root, PipelineManager.DirJPEGImages)
        self.dir_segmentations = os.path.join(root, PipelineManager.DirSegmentationClass)

        [ assert_dir_exists(d) for d in [self.dir_image_sets, self.dir_jpeg_images, self.dir_segmentations] ]
