from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from yolo import *
import threading
import queue

#跟踪器字典初始化
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
#初始化跟踪器
traker_name='csrt'

tracker = OPENCV_OBJECT_TRACKERS[traker_name]()

#开启摄像头
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#睡眠一秒等待开启
time.sleep(1.0)
# initialize the FPS throughput estimator
fps = None
#初始化BBox
initBB = None





def tracking():
    initBBqueue =queue.Queue(1)
    threading.Timer(frame)
    global initBB,fps,tracker
    #开始循环读取图像
    while True:
        frame = vs.read()
        #检查是否读取图像
        if frame is None:
            break
        #resize图像
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        #检查队列是否为空，不为空重新初始化跟踪器
        # if initBBqueue.empty()!=True:
        #     initBB=get_yolo_box(frame)
        #     tracker = OPENCV_OBJECT_TRACKERS[traker_name]()
        #     tracker.init(frame, tuple(initBB))
        #     fps = FPS().start()

        # 检查是否已经有跟踪对象
        if initBB is not None:
            # 获取新的边框
            (success, box) = tracker.update(frame)
            # 检查是否成功获取
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 更新fps计数器
            fps.update()
            fps.stop()
            info = [("Tracker", traker_name), ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())), ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            time.sleep(0.05)

        # else:
        #     detect_thread=Detect_thread(initBBqueue)
        #     detect_thread.getframe(frame)
        #     detect_thread.start()
        #     initBB =initBBqueue.get()
        #     print('队列已添加')
            # fps = FPS().start()


        # key = cv2.waitKey(1) & 0xFF


        # if (key == ord("s")):
        #     # select the bounding box of the object we want to track (make
        #     # sure you press ENTER or SPACE after selecting the ROI)
        #     # initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        #     initBB=get_yolo_box(frame)
        #     tracker = OPENCV_OBJECT_TRACKERS[traker_name]()
        #     tracker.init(frame, tuple(initBB))
        #     fps = FPS().start()
        #
        # # if the `q` key was pressed, break from the loop
        # elif key == ord("q"):
        #     break






tracking()
vs.stop()
cv2.destroyAllWindows()