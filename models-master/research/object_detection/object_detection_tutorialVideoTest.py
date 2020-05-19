# coding: utf-8


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

video = cv2.VideoCapture("G:\\PycharmProjects\\models-master\\dataset\\Video\\input\\bandicam.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# opencv支持不同的编码格式
video_writer = cv2.VideoWriter('G:\\PycharmProjects\\models-master\\dataset\\Video\\output\\outputVideo.avi',
                               cv2.VideoWriter_fourcc(*'XVID'), fps, size)
success, frame = video.read()


sys.path.append("..")

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'G:\\PycharmProjects\\models-master\\dataset\\exModel\\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'G:\\PycharmProjects\\models-master\\dataset\\poker_label_map.pbtxt'

NUM_CLASSES = 54

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

gpu_options = tf.GPUOptions(allow_growth=True)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        while success and cv2.waitKey(1) & 0xFF != ord('q'):
            # 扩展维度，应为模型期待: [1, None, None, 3]
            image_np_expanded = np.expand_dims(frame, axis=0)
            print(image_np_expanded)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # 每个框代表一个物体被侦测到
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # 每个分值代表侦测到物体的可信度.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # 执行侦测任务.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # 检测结果的可视化
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            video_writer.write(frame)
            success, frame = video.read()
video_writer.release()