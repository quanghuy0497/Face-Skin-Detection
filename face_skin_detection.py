import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

face_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,78,74,79,73,72,80,71,70,69,68,76,75,77]
img=cv2.imread("Image.jpg")
height, width, channels = img.shape
size = width * height

#=====================================================FACE====================================================
def Face_detection(img):
    predictor_path = 'face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    face_cal = 0
    face_count = 0
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        face_count += 1
        face = 0
        coordinate = np.array([[shape.parts()[num].x , shape.parts()[num].y] for num in face_index])
        for num in range(30):
            if (num == 29): nextt = 0
            else: nextt = num + 1
            face += (coordinate[num][0]* coordinate[nextt][1] - coordinate[num][1]* coordinate[nextt][0])
            cv2.circle(img, (coordinate[num][0], coordinate[num][1]), 3, (0,255,0), -1)
        face_cal += abs(face) / 2
        cv2.imshow("Face.jpg",img)
    return face_cal
#=====================================================SKIN=====================================================
def Skin_detection(img):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    final_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    final_mask=cv2.medianBlur(final_mask,3)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    cv2.imshow("Skin.jpg",final_mask)

    skin_cal = cv2.countNonZero(final_mask)
    
    return skin_cal

ratio_face_size = (Face_detection(img) / size) * 100
ratio_skin_size = (Skin_detection(img) / size) * 100
ratio_face_skin = ratio_face_size / ratio_skin_size
print(size)
print("Ratio face:",ratio_face_size,"%")
print("Ratio skin:",ratio_skin_size,"%")
print("Ratio face and skin:", ratio_face_skin * 100,"%")
cv2.waitKey(0)