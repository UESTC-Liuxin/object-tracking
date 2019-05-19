from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from yolo import *
import threading
import queue


class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # 开启摄像头
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(1)
        self.frame = self.vs.read()
    def run(self):
        while True:
            self.frame = self.vs.read()
            bbox=get_yolo_box(self.frame)
            draw_prediction(self.frame,10,100,100,100)
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF
            print(bbox)


thread =myThread()
thread.start()



while True:
    print('主线程')

# # 开启摄像头
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(1)
# frame = vs.read()
# while True:
#     frame = vs.read()
#     # 检查是否读取图像
#     if frame is None:
#         break
#     # resize图像
#     frame = imutils.resize(frame, width=500)
#     (H, W) = frame.shape[:2]
#     bbox=get_yolo_box(frame)
#     draw_prediction(frame,10,100,100,100)
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF


