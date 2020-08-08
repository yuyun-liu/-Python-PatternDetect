import numpy as np
import cv2
import os
import datetime
import matplotlib.pyplot as plt

from Lib.GetPattern import ImageProcessing, AdvanceImageProcessing

FOLDER_PATH = os.getcwd()
IMG_PATH = FOLDER_PATH + "\\images\\PatternCam1.png"
PATTERNS_PATH = np.array([FOLDER_PATH + "\\images\\Pattern0.png",
FOLDER_PATH + "\\images\\Pattern1.png",
FOLDER_PATH + "\\images\\Pattern2.png",
FOLDER_PATH + "\\images\\Pattern3.png",
FOLDER_PATH + "\\images\\Pattern4.png",
FOLDER_PATH + "\\images\\Pattern5.png",
FOLDER_PATH + "\\images\\Pattern6.png",
FOLDER_PATH + "\\images\\Pattern7.png"])

PATTERNS_LABEL_PATH = np.array([FOLDER_PATH + "\\images\\Pattern0Label.png",
FOLDER_PATH + "\\images\\Pattern1Label.png",
FOLDER_PATH + "\\images\\Pattern2Label.png",
FOLDER_PATH + "\\images\\Pattern3Label.png",
FOLDER_PATH + "\\images\\Pattern4Label.png",
FOLDER_PATH + "\\images\\Pattern5Label.png",
FOLDER_PATH + "\\images\\Pattern6Label.png",
FOLDER_PATH + "\\images\\Pattern7Label.png"])

ALPHA = 0.9
BETA = 30
CANNY_THRESHOLD1 = 30
CANNY_THRESHOLD2 = 100
CANNY_KERNAL = (5, 5)
MATRIX_SIZE = 9
THRESH_MIN_VALVE = 150
THRESH_MAX_VALVE = 255
IMG = cv2.imread(IMG_PATH)
ENABLE_CAM = True
CAM_INDEX = 0
PATTERN_IMAGE_HIGHTH = 150
PATTERN_IMAGE_WIDTH = 130

cam =  cv2.VideoCapture(CAM_INDEX)
im = AdvanceImageProcessing()
patterns_matrix = np.copy(im.get_patterns_bool_matrix(PATTERNS_PATH, 8, MATRIX_SIZE))
patterns_matrix = np.bitwise_not(patterns_matrix)
ret = True
last_loop_time = datetime.datetime.now()
this_loop_time = datetime.datetime.now()
while(ret):
    if(ENABLE_CAM):
        ret, img = cam.read()
        img = im.bcg_lookup(img, ALPHA, BETA)
        #cv2.imshow("camera_capture", img)
    else:
        img = IMG
        #cv2.imshow("IMG_File_Read", img)
    cnts_array = im.get_image_pattern_contours(img, THRESH_MIN_VALVE, THRESH_MAX_VALVE, CANNY_KERNAL, CANNY_THRESHOLD1, CANNY_THRESHOLD2, False)      
    i = 0
    imgs = np.zeros((PATTERN_IMAGE_HIGHTH, PATTERN_IMAGE_WIDTH * np.size(cnts_array, 0), 3))
    for cnts in cnts_array:
        pattern_return = im.get_pattern(im.get_subROI(img, cnts), patterns_matrix)
        print("Pattern = ", pattern_return)
        img_pattern = cv2.imread(PATTERNS_LABEL_PATH[pattern_return])
        img_pattern = cv2.resize(img_pattern, (PATTERN_IMAGE_WIDTH, PATTERN_IMAGE_HIGHTH), interpolation=cv2.INTER_LINEAR)
        #img_pattern = cv2.cvtColor(img_pattern, cv2.COLOR_RGB2GRAY)
        imgs[0:PATTERN_IMAGE_HIGHTH, i*PATTERN_IMAGE_WIDTH:(i+1)*PATTERN_IMAGE_WIDTH, 0:3] = img_pattern
        cv2.imshow("Patterns Finded", imgs)
        i += 1
    if i == 0:
        cv2.destroyWindow("Patterns Finded")
    this_loop_time = datetime.datetime.now()
    #print(this_loop_time.second - last_loop_time.second)
    last_loop_time = this_loop_time
    if cv2.waitKey(1) & 0xff == ord('q') :
            break
    