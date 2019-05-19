import cv2
import argparse
import numpy as np
import threading
import queue
import time
import copy
WEIGHTS_PATH="config/yolov3.weights"
CONFIG_PATH="config/yolo.cfg"
CLASS_NAME_PATH="config/coco.names"


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, x, y, x_plus_w, y_plus_h):
    # label = str(classes[class_id])
    #
    # color = COLORS[class_id]
    # print(str(classes[class_id]))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), COLORS[0], 2)

    # cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# image = cv2.imread(args.image)



classes = None

with open(CLASS_NAME_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)


def get_yolo_box(image1):
    image=copy.deepcopy(image1)
    return_box=[]
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]#得到不同类别的置信度分数
            class_id = np.argmax(scores)#找出最大的置信度分数的类别
            confidence = scores[class_id]#得到最大的置信度分数的类别的置信度
            if confidence > 0.5:   #做一个简单的过滤
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)  #添加到类别的列表
                confidences.append(float(confidence))#并且添加列表对应的置信度分数列表
                boxes.append([x, y, w, h])   #添加bbox
    # print(boxes)
    print(len(confidences))
    # print(conf_threshold)
    # print(nms_threshold)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)#进行非最大化抑制后的参数

   #遍历所以bbox，画出person的框
    for i in indices:
        i = i[0]
        if(class_ids[i]==0):#检测是否为person
            return_box = boxes[i]
            break
            #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
    return return_box


class Detect_thread(threading.Thread):
    def __init__(self,q):
        threading.Thread.__init__(self)
        self.q =q

    def getframe(self,frame):
        self.frame=frame

    def run(self):
        while True:
            print('开始检测')
            initBB = get_yolo_box(self.frame)
            print('检测完毕')
            print(initBB)
            if initBB :
                self.q.put(initBB)  # 加入队列
                print('加入队列')
                # time.sleep(1)



