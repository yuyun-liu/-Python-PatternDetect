import os
import cv2
import numpy as np

from Lib.GetPattern import ImageProcessing

FOLDER_PATH = os.getcwd()
IMG_PATH = FOLDER_PATH + "\\images\\Patterns.png"
IMG_WRITE_PATH = FOLDER_PATH + "\\images"
IMG = cv2.imread(IMG_PATH)
CANNY_THRESHOLD1 = 30
CANNY_THRESHOLD2 = 100
CANNY_KERNAL = (5, 5)
ERODE_DILATE_KERNAL = (5, 5)

im = ImageProcessing()
IMG = im.gray_scale(IMG)
bin_img = im.binarize(IMG, 200, 255, True)
bin_img = im.erode_dilate(bin_img, ERODE_DILATE_KERNAL, True, 1)
bin_img = im.morphology_fill(bin_img)
edged_img = im.canny(bin_img,  CANNY_KERNAL, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
_, cnts_array = im.draw_contours(edged_img, cv2.cvtColor(IMG, cv2.COLOR_GRAY2RGB))
i = 0
for img_cnt in cnts_array:
    im.write_image_file(IMG_WRITE_PATH + "\\Pattern" +  str(i) + ".png", im.get_subROI(IMG, img_cnt))
    i += 1
cv2.waitKey(1000)