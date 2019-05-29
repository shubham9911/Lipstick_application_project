# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:42:29 2019

@author: bunty
"""

import cv2
import numpy as np
import dlib


video = cv2.VideoCapture(0)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    ret,frame = video.read()
    
        
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face = detector(frame)
    for faces in face:
        
        x1 = faces.left()
        y1 = faces.top()
        x2 = faces.right()
        y2 = faces.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        
        landmarks = predictor(frame, faces)
        landmarks_points = []
        for n in range(48, 67):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #cv2.circle(frame, (x, y), 2,(0,0,255), -1)
            landmarks_points.append((x,y))
    
        points = np.array(landmarks_points, np.int32)
        #convexhull = cv2.convexHull(points)
         
        cv2.fillPoly(frame, [points], (0,0,255))
        
   
    
    
    
   
    cv2.imshow("frame",frame)
   # cv2.imshow("lipstick",frame)
    
    key = cv2.waitKey(1)
    
    if key == ord('b'):
        break
        
        

        
        
        
        
video.release()

cv2.destroyAllWindows()
