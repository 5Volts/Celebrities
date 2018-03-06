# Original Code can be found in Tensorflow For Poet 2 website
# -5Volts
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import glob
from face_detection import check
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  names=[]
  file = str(input("State your Image File Name:"))
  imgg = check(cv2.imread(file))
  cv2.imwrite("Cropped_Image.png",imgg)
  file_name = "Cropped_Image.png"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  with tf.Session(graph=graph) as sess:
    start = time.time()
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end=time.time()
  results = np.squeeze(results)
  top_k = results.argsort()[-15:][::-1]
  labels = load_labels(label_file)
  print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
  for i in top_k:
    print(labels[i],":", results[i])
    names.append(labels[i])
  cel= labels[np.argmax(results)]
  font = {'family': 'normal','size': 8}
  matplotlib.rc('font', **font)
  y_pos = np.arange(len(labels))
  fig = plt.figure(figsize=(9, 4))
  plt.barh(y_pos, np.sort(results), align='edge', alpha=1)
  plt.yticks(y_pos, names[::-1])
  plt.xlabel('Similarities')
  plt.title('Convolutional Neural Network for Celebrities')
  plt.savefig("image.png")
  img = np.array(cv2.imread("image.png"))
  img2 = cv2.imread(file_name)
  img2 = np.array(cv2.resize(img2,(450,450)))
  for i in glob.glob("tf_files/Celebrity/{}/*.jpeg".format(cel)):
    img3 = cv2.imread(i)
    break
  img3 = np.array(cv2.resize(img3,(450,450)))
  img4 = np.concatenate((img2,img3),axis=1)
  img5 = np.concatenate((img4,img),axis=0)
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img5, 'You look like {}!'.format(cel), (10, 430), font, 1,(226, 244, 66), 2, cv2.LINE_AA)
  img5=cv2.resize(img5,(800,600))
  cv2.imshow("Image",img5)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

