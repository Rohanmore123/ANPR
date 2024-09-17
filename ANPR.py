#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import os
import shutil
import ultralytics
from ultralytics import YOLO

model = YOLO('best.pt')
detection_model = YOLO('char-best.pt')

class_names = {}
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
for i,c in enumerate(characters):
    class_names[i] = c

def detect_characters(plate):
    cls_list = []
    x_list = []             
    results = detection_model(plate)
    for result in results:
        plate_tensor = result.boxes.xyxy
        if plate_tensor is not None:
            for dims in plate_tensor.tolist():
                px1,py1,px2,py2 = dims
                pp1 = (int(px1), int(py1))
                pp2 = (int(px2), int(py2))
                x_list.append(px1)
            cls_tensor = result.boxes.cls
            for i in cls_tensor.tolist():
                cls_list.append(i)
            conf = sum(result.boxes.conf.tolist())/10
    return cls_list, x_list, conf

def concat_plateNum(cls_list, x_list):
    sorted_list = []
    plate_number=''
    indices = sorted(range(len(x_list)), key=lambda k: x_list[k])
    for idx in indices:
        sorted_list.append(cls_list[idx])
    for i in sorted_list:
        plate_number += str(class_names[i])
    return plate_number

video_path = 'D:\ANPR_final\VID20231006133434.mp4'
cap = cv.VideoCapture(video_path)

vehicles_list = []
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model(frame)
        
        for result in results:
            tensor = result.boxes.xyxy
            if tensor is not None:
                for dims in tensor.tolist():
                    x1,y1,x2,y2 = dims
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
                    cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                    cls_list, x_list, conf = detect_characters(cropped_frame)
                    plateNum = concat_plateNum(cls_list, x_list)                    
                    if len(plateNum)==10 :
                        text = plateNum
                        font = cv.FONT_HERSHEY_SIMPLEX
                        cv.rectangle(frame, (int(x1-30),int(y1-50)), (int(x2+30),int(y1)), (0,255,255), -1)
                        cv.putText(frame, plateNum, (int(x1 - 10), int(y1) - 20),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                
        frame = cv.resize(frame, (720, 480))
        cv.imshow('Detection_video', frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()


# In[ ]:




