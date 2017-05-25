from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.inception import inception_resnet_v2

def network(imgs):
    return inception_resnet_v2(imgs)
