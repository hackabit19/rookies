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

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#Object Detection Imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



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


#######################################################################################################################

import pyttsx3 as pyttsx
engine = pyttsx.init()
engine.setProperty('rate', 170)

def run_inference_for_single_image(image, graph,tensor_dict,sess):
    if 'detection_masks' in tensor_dict:
        tensor_dict.seek(0)
               
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})
##    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict




#############################################################

import cv2


#cap= cv2.VideoCapture('C:/Users/Akanksha/Desktop/detect/video1.mp4')

def outdoor():
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


  ##  m()
  cap=cv2.VideoCapture(0)
  count=0

  try:

    with detection_graph.as_default():
      with tf.Session() as sess:
        print("gh",sess)
          
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        print(tensor_dict)

        old_dic={}

        while True:
          ret,image_np = cap.read()
          count=count+1

          if count%10!=0:
            continue
                  #print(ret,image_np)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
          output_dict = run_inference_for_single_image(image_np_expanded, detection_graph,tensor_dict,sess)
  ##                print(output_dict)
    # Visualization of the results of a detection.
          image_np,labels = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
  ##                print(output_dict.get('detection_classes'))
##          print(labels)
          di= {}
          for i in labels:
            l=i.split(':')[0]
            p =i.split(':')[1].split('%')[0]
##            print(p)
            if int(p)>60:
              if i.split(':')[0] not in di:
                di[l]=1
              else:
                di[l] = di[l]+1
          print(di)

          cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
          if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            from main1 import main
            main()
            break

          if old_dic==di:
            continue

          

          if 'Traffic sign' in di:
            engine.say("there are"+str(di['Traffic sign'])+"Traffic sign")
            engine.runAndWait()
            del di['Traffic sign']
          if 'Stop sign' in di:
            engine.say("there are"+str(di['Stop sign'])+"Stop sign")
            engine.runAndWait()
            del di['Stop sign']
          if 'stop sign' in di:
            engine.say("there are"+str(di['Stop sign'])+"Stop sign")
            engine.runAndWait()
            del di['Stop sign']
          if 'Traffic sign' in di:
            engine.say("there are"+str(di['Traffic sign'])+"Traffic sign")
            engine.runAndWait()
            del di['Traffic sign']
          if 'train' in di:
            engine.say("there are"+str(di['train'])+"trains")
            engine.runAndWait()
            del di['train']
          if 'car' in di:
            engine.say("there are"+str(di['car'])+"car")
            engine.runAndWait()
            del di['car']
          if 'bus' in di:
            engine.say("there are"+str(di['bus'])+"bus")
            engine.runAndWait()
            del di['bus']
          if 'motorbike' in di:
            engine.say("there are"+str(di['motorbike'])+"motorbike")
            engine.runAndWait()
            del di['motorbike']
          if 'bicycle' in di:
            engine.say("there are"+str(di['bicycle'])+"bicycle")
            engine.runAndWait()
            del di['bicycle']
##
          for i in di.keys():
            engine.say("There are "+str(di[i])+str(i))
            engine.runAndWait()
          
               
          
          old_dic=di
          
          
          

  except Exception as e:
    print(e)
##    from main import main
##    main()
    cap.release()



outdoor()










