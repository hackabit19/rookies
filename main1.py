
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops

# What model to download.
MODEL_PATH = 'object_detection/ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_PATH + '/frozen_inference_graph.pb'

PATH_TO_LABELS = MODEL_PATH + '/mscoco_label_map.pbtxt'



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



def main():
    print("1. Outdoor")
    print("2. Indoor")
    print("3. Panic")



    x=int(input("Enter the option:"))

    if x==1:
        print("outdoor")
        from real_time import outdoor
        outdoor()
    elif x==2:
        print("indoor")
        from indoor1 import indoor
        indoor()

        #outdoor()

    elif x==3:
        print("panic mode")
        from panic1 import alert

main()




