import cv2

#img = cv2.imread('tejo.PNG')

#0,1 based on your webcam number
cap = cv2.VideoCapture(0)
cap.set(3, 640)  #
cap.set(4, 480)  #

classNames = []

#coco.names are the classes we can detect
#https://opencv.org/introduction-to-the-coco-dataset/
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#good balance between speed and accuracy
#https://towardsdatascience.com/everything-you-need-to-know-about-mobilenetv3-and-its-comparison-with-previous-versions-a5d5e5a6eeaa
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#frozen_inference_graph.pb, is a frozen graph that cannot be trained anymore
weightPath = 'frozen_inference_graph.pb'

#https://docs.opencv.org/4.x/d3/df1/classcv_1_1dnn_1_1DetectionModel.html
#initialize detection model
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    #threshold to detect objects - confidence
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow('output', img)
    cv2.waitKey(1)