import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

  def __init__(self, frozen_inference_graph_path):

    self.graph = tf.Graph()

    with tf.gfile.FastGFile(frozen_inference_graph_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')#
      self.sess = tf.Session(graph=self.graph)


  def run(self, image):

    target_size = image.size
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    
    return resized_image, seg_map

  def label_to_color_image(self, label):

    #cityscapes 
    colormap = np.asarray([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [70, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32],
    ])

    return colormap[label]


### Step_1
# Setting path & list
###

# current working path
work_path = '/home/csk/Documents/01_self_driving_lab/00_inference_cityscapes/'

# frozen graph of pretrained model
#frozen_inference_graph_path = work_path + 'model/01_deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb'
#frozen_inference_graph_path = work_path + 'model/02_deeplabv3_cityscapes_train/frozen_inference_graph.pb'
frozen_inference_graph_path = work_path + 'model/03_train_fine/frozen_inference_graph.pb'
#frozen_inference_graph_path = work_path + 'model/04_trainval_fine/frozen_inference_graph.pb'




# file list
file_list = work_path + 'file_list.txt'

	
### Step_2
# Loading deep learing model
###
MODEL = DeepLabModel(frozen_inference_graph_path)


### Step_3
# inferencing images & saving result labels
###

file = open(file_list, 'r') 
file_list = file.readlines()

for n in file_list:
  file_name = n.strip()
  input_image_name = work_path + 'input_images/' + file_name + '.jpg'
  output_label_name = work_path + 'output_labels/' + file_name + '.png'

  # loading image
  original_im = Image.open(input_image_name)

  # inferencing
  resized_im, seg_map = MODEL.run(original_im)

  # convering raw label into color label
  seg_image = MODEL.label_to_color_image(seg_map).astype(np.uint8)

  # saving result color label
  result = Image.fromarray(seg_image)
  result.save(output_label_name)

file.close()























