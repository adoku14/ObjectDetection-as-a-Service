#!/usr/bin/env python3
import cv2
import numpy as np
import os
import json


def getCurrentPath():
    return os.getcwd()

def loadFiles():
    with open('application.properties') as file:
        file_info = dict()
        lines = file.readlines()
        for line in lines:
            infos = line.split('=')
            file_info[infos[0].strip()] = infos[1].strip()

        return file_info

def loadNetwork():
    file_info = loadFiles()

    net = cv2.dnn.readNet(file_info['weights'], file_info['cfg_file'])
    with open(file_info['classes_file'], 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def predict(received_img, net, classes):
    image = cv2.imdecode(np.fromstring(received_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)  # imread(received_img)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    # image_np = load_image_into_numpy_array(image)

    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    json_list = {}
    cnt = 1
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        data = {}
        data['class'] = classes[class_ids[i]]
        data['confidence'] = str(round(confidences[i], 4))
        data['top'] = str(x)
        data['left'] = str(y)
        data['bottom'] = str(x + w)
        data['right'] = str(y + h)

        json_list[cnt] = data
        cnt = cnt + 1
    return json_list
