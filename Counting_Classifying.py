import cv2
import numpy as np
from time import sleep
import cv2
import matplotlib.pyplot as plt
# Non maximum supressions
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

min_width = 80
min_height = 80
# Allowed error between pixels
offset = 20
# Count line position
line_pos = 500
# FPS do vídeo
delay = 60
detect = []
vehicle = 0



def center_value(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# Capturing the video
cap = cv2.VideoCapture('vid_3.mp4')
# Foregroundmask using MOG2
# It is also a Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
# It provides better adaptability to varying scenes due illumination changes etc.
sub = cv2.createBackgroundSubtractorMOG2()
#Setting the font
font = cv2.FONT_HERSHEY_PLAIN
# Using yolo tiny weigths and configurations
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')
classes = []

#using coco dataset to classify objects in the video
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# From line 51 to 62 we are pre-processing the frames
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = sub.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# We are drawing a line on tht identifies the vehicles crossing it.
# We are enumerating(taking one by one) each contour.

    cv2.line(frame, (25, line_pos), (1500, line_pos), (255, 127, 0), 3)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contours = (w >= min_width) and (h >= min_height)
        if not validate_contours:
            continue
# We are assigning each contour a rectangle and a circle in between to identify.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center = center_value(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
# If the contour gets on the line between points 580 to 520 then we will be incrementing hte vehicle count
        for (x, y) in detect:
            if y < (line_pos + offset) and y > (line_pos - offset):
                vehicle += 1
                cv2.line(frame, (25, line_pos), (1200, line_pos), (0, 127, 255), 3)
                detect.remove((x, y))


    _, frame1 = cap.read()
    height, width, _ = frame1.shape

#We are creating a blob here to connect pixels in a binary image.
    blob = cv2.dnn.blobFromImage(frame1, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
# Put everything in rectangle box to track the confidence or the scores with their class's
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            #Argmax is used to find the largest number among the scores
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
#  apply non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            cv2.rectangle(frame1, (x, y), (x + w, y + h),(0, 255, 255), 2)
            cv2.putText(frame1, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)


    cv2.putText(frame, "VEHICLE COUNT : " + str(vehicle), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("VehicleCount", frame)
    cv2.imshow("YOLO-Tiny", frame1)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()