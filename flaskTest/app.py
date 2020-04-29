import os
import tensorflow as tf
from flask import Flask, Blueprint, send_file, request, jsonify
from datetime import timedelta
import matplotlib.image as mpimg
import uuid
from templates import indexPython, tool
from PIL import Image
import cv2 as cv
import os, io
import PIL.Image
import numpy as np
from object_detection.utils import label_map_util
import pandas as pd
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('host', '10.9.2.241', '')
tf.app.flags.DEFINE_integer('port', '3221', 'server with port,if no port, use deault port 80')
tf.app.flags.DEFINE_boolean('debug', True, '')
tf.app.flags.DEFINE_string('upload_folder', 'tmp/', '')

UPLOAD_FOLDER = FLAGS.upload_folder + "upload/"
RESULT_FOLDER = FLAGS.upload_folder + "result/"
ALLOWED_EXTENSIONS = set(['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG'])  # 图片格式

app = Flask(__name__, static_folder='', static_url_path='')  # 创建1个Flask实例
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


labArr = [{"id": 1, "name": '3~1'},
          {"id": 2, "name": '3~2'},
          {"id": 3, "name": '3~3'},
          {"id": 4, "name": '3~4'},
          {"id": 5, "name": '4~1'},
          {"id": 6, "name": '4~2'},
          {"id": 7, "name": '4~3'},
          {"id": 8, "name": '4~4'},
          {"id": 9, "name": '5~1'},
          {"id": 10, "name": '5~2'},
          {"id": 11, "name": '5~3'},
          {"id": 12, "name": '5~4'},
          {"id": 13, "name": '6~1'},
          {"id": 14, "name": '6~2'},
          {"id": 15, "name": '6~3'},
          {"id": 16, "name": '6~4'},
          {"id": 17, "name": '7~1'},
          {"id": 18, "name": '7~2'},
          {"id": 19, "name": '7~3'},
          {"id": 20, "name": '7~4'},
          {"id": 21, "name": '8~1'},
          {"id": 22, "name": '8~2'},
          {"id": 23, "name": '8~3'},
          {"id": 24, "name": '8~4'},
          {"id": 25, "name": '9~1'},
          {"id": 26, "name": '9~2'},
          {"id": 27, "name": '9~3'},
          {"id": 28, "name": '9~4'},
          {"id": 29, "name": '10~1'},
          {"id": 30, "name": '10~2'},
          {"id": 31, "name": '10~3'},
          {"id": 32, "name": '10~4'},
          {"id": 33, "name": '11~1'},
          {"id": 34, "name": '11~2'},
          {"id": 35, "name": '11~3'},
          {"id": 36, "name": '11~4'},
          {"id": 37, "name": '12~1'},
          {"id": 38, "name": '12~2'},
          {"id": 39, "name": '12~3'},
          {"id": 40, "name": '12~4'},
          {"id": 41, "name": '13~1'},
          {"id": 42, "name": '13~2'},
          {"id": 43, "name": '13~3'},
          {"id": 44, "name": '13~4'},
          {"id": 45, "name": '14~1'},
          {"id": 46, "name": '14~2'},
          {"id": 47, "name": '14~3'},
          {"id": 48, "name": '14~4'},
          {"id": 49, "name": '15~1'},
          {"id": 50, "name": '15~2'},
          {"id": 51, "name": '15~3'},
          {"id": 52, "name": '15~4'},
          {"id": 53, "name": '17~3'},
          {"id": 54, "name": '18~4'}
          ]

PATH_TO_CKPT = os.path.join(os.getcwd(), 'static/dataset/exModel/frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(os.getcwd(), 'static/dataset/poker_label_map.pbtxt')

NUM_CLASSES = 54

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

output_image_path = os.path.join(os.getcwd(), 'tmp/result/')
output_csv_path = os.path.join(os.getcwd(), 'tmp/csv/')

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()

detection_sess = tf.Session(graph=detection_graph)

with detection_sess.as_default():
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def PNG_JPG(file_path):
    with tf.gfile.GFile(file_path, 'rb') as fid:
        encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        outfile = os.path.splitext(file_path)[0] + ".jpg"
        if image.format != 'JPEG':
            img = cv.imread(file_path, 0)
            w, h = img.shape[::-1]
            img = Image.open(file_path)
            img = img.resize((int(w), int(h)), Image.ANTIALIAS)
            try:
                if len(img.split()) == 4:
                    # prevent IOError: cannot write mode RGBA as BMP
                    r, g, b, a = img.split()
                    img = Image.merge("RGB", (r, g, b))
                    img.convert('RGB').save(outfile, quality=100)
                    # os.remove(PngPath)
                else:
                    img.convert('RGB').save(outfile, quality=100)
                    # os.remove(PngPath)
                return outfile
            except Exception as e:
                print("PNG转换JPG 错误", e)
        else:
            return outfile


def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name

def isE(path):
    return os.path.exists(path)


@app.route('/')
def first_flask():  # 视图函数
    return send_file('templates/index.html')  # response


@app.route('/capIndex')
def capIndex():  # 视图函数
    return send_file('templates/capIndex.html')  # response


@app.route('/api/saveImg', methods=['POST', 'GET'])
def saveImg():
    if request.method == 'POST':
        file = request.files['uploadFile']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            file_path = os.getcwd() + '/' + PNG_JPG(file_path)
            print(file_path)
            if os.path.exists(file_path):
                return jsonify({'ok': 1, 'res': filename})
            else:
                return jsonify({'ok': -2, 'res': "文件上传失败"})
        return jsonify({'ok': 0, 'res': "文件错误"})
    return jsonify({'ok': -1, 'res': "请求错误"})


@app.route('/api/detection1', methods=['POST', 'GET'])
def detection1():
    if request.method == 'POST':
        data = request.form
        file1 = data["file1"]
        file2 = data["file2"]
        print(file1)
        print(file2)
        if file2 == "file2":
            getRes = indexPython.getRes(file1)
            return jsonify({'ok': 1, 'res': str(file1).split('.')[0] + '.png', "data": getRes})
        else:
            if tool.phash_img_similarity(file1, file2) < 1:
                getRes = indexPython.getRes(file2)
                return jsonify({'ok': 2, 'res': str(file2).split('.')[0] + '.png', "data": getRes})
            else:
                return jsonify({'ok': 0, 'res': "图片没改变"})
    else:
        return jsonify({'ok': -1, 'res': "请求错误"})

@app.route('/api/detection', methods=['POST', 'GET'])
def detection():
    if request.method == 'POST':
        data = request.form
        fileName = data["fileName"]
        if isE(os.path.join(UPLOAD_FOLDER, fileName)):
            getRes = indexPython.getRes(fileName)
            return jsonify({'ok': 1, 'res': str(fileName).split('.')[0] + '.png', "data": getRes})
        return jsonify({'ok': 0, 'res': "文件不存在"})
    else:
        return jsonify({'ok': -1, 'res': "请求错误"})



@app.route('/api/uploadImg', methods=['POST', 'GET'])
def uploadImg():
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            # print('file saved to %s' % file_path)
            conv_filename = PNG_JPG(file_path)  # 图片转换
            while True:
                if os.path.exists(os.getcwd() + '/' + conv_filename):
                    start = time.time()
                    data = pd.DataFrame()

                    PATH_TO_TEST_IMAGES_DIR = os.getcwd() + '/' + conv_filename

                    image = Image.open(PATH_TO_TEST_IMAGES_DIR)
                    width, height = image.size
                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = load_image_into_numpy_array(image)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes, scores, classes, num) = detection_sess.run(
                        [detection_boxes,
                         detection_scores,
                         detection_classes,
                         num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                    # min_score_thresh 新增
                    end = time.time()
                    # write images
                    # 保存识别结果图片
                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(image_np)
                    plt.savefig(os.path.join(os.getcwd(), output_image_path) + filename.split('.')[0])
                    # cv2.imwrite(os.path.join(os.getcwd(), output_image_path) + filename, image_np)

                    # s_boxes = boxes[scores > 0.5]
                    # s_classes = classes[scores > 0.5]
                    # s_scores = scores[scores > 0.5]
                    s_boxes = boxes[scores > 0.2]
                    s_classes = classes[scores > 0.2]
                    s_scores = scores[scores > 0.2]
                    print(classes)

                    # write table
                    # 保存位置坐标结果到 .csv表格
                    for i in range(len(s_classes)):
                        newdata = pd.DataFrame(0, index=range(1), columns=range(8))
                        newdata.iloc[0, 0] = filename.split('.')[0]
                        newdata.iloc[0, 1] = s_boxes[i][0] * height  # ymin
                        newdata.iloc[0, 2] = s_boxes[i][1] * width  # xmin
                        newdata.iloc[0, 3] = s_boxes[i][2] * height  # ymax
                        newdata.iloc[0, 4] = s_boxes[i][3] * width  # xmax
                        newdata.iloc[0, 5] = s_scores[i]
                        newdata.iloc[0, 6] = s_classes[i]

                        tmp=int(s_classes[i])-1
                        newdata.iloc[0, 7] = labArr[tmp]["name"]


                        data = data.append(newdata)

                    # 行数>0
                    if data.shape[0] > 0:
                        data = data.sort_values([1, 2], ascending=[True, True])  # 排序

                    data.to_csv(output_csv_path + filename.split('.')[0] + '.csv', index=False)
                    end = time.time()
                    return jsonify({'ok': 1,
                                    'num':len(s_classes),
                                    'res': conv_filename.split('/')[2].split('.')[0] + '.png',
                                    "data": {"state": 1, "resStr": end - start, "data": data.to_json(orient='split')}
                                    })

                    # getRes = indexPython.getRes(conv_filename.split('/')[2])
                    # if getRes["state"] == -2:
                    #     return jsonify({'ok': -2, 'res': "图片错误"})
                    # else:
                    #     return jsonify({'ok': 1, 'res': filename.split('.')[0] + '.png', "data": getRes})
                else:
                    return jsonify({'ok': -3, 'res': "图片上传失败"})

        return jsonify({'ok': 0, 'res': "图片格式错误"})
    return jsonify({'ok': -1, 'res': "请求错误"})


@app.route('/api/wxdetection', methods=['POST', 'GET'])
def wxdetection():
    if request.method == 'GET':
        fileName = request.args.get("fileName")
        print(fileName)
        getRes = indexPython.getRes(fileName)
        return jsonify({'ok': 1, 'res': str(fileName).split('.')[0] + '.png', "data": getRes})
    else:
        return jsonify({'ok': -1, 'res': "请求错误"})


if __name__ == '__main__':
    app.run(host=FLAGS.host, port=FLAGS.port, debug=FLAGS.debug, threaded=True)  # 启动socket
