from django.shortcuts import render
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import base64


def convert_jpg_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return base64_string


def reconstruct(pb_path):
    if not os.path.isfile(pb_path):
        print("Error: %s not found" % pb_path)

    print("Reconstructing Tensorflow model")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Success!")
    return detection_graph


def image2np(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)


def image2tensor(image):
    npim = image2np(image)
    return np.expand_dims(npim, axis=0)


def detect(detection_graph, test_image_path):
    with open('annotations.json') as json_file:
        data = json.load(json_file)

    categories = data['categories']
    label_map = label_map_util.load_labelmap('labelmap.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=60, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.01)
        with tf.compat.v1.Session(graph=detection_graph, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            image = Image.open(test_image_path)
            (boxes, scores, classes, num) = sess.run(  # type: ignore
                [detection_boxes, detection_scores,
                    detection_classes, num_detections],
                feed_dict={image_tensor: image2tensor(image)}
            )

            npim = image2np(image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                npim,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=15)
            plt.figure(figsize=(12, 8), dpi=150)
            plt.imshow(npim)
            plt.axis("off")
            plt.savefig('output.png',format='png', bbox_inches='tight', pad_inches=0.1)
            plt.close()


def index(request):
    print(request.method)
    if request.method == 'POST':
        print(request.FILES)
        detection_graph = reconstruct("ViTTrashClass.pb")
        image = request.FILES['image']
        # image.save('image.jpg', image)
        with open('image.jpg', 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)
        detect(detection_graph, 'image.jpg')
        base64_data = convert_jpg_to_base64("output.png")
        data_uri = f"data:image/jpeg;base64,{base64_data}"
        return render(request, 'base/index.html', {'image': data_uri})

    return render(request, 'base/index.html')
