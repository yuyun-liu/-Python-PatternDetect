import cv2
import numpy as np
import os

class ImageProcessing:
    
    def __init__(self):
        super().__init__()

    def write_image_file(self, path, img):
        cv2.imwrite(path, img)

    def bcg_lookup(self, img, alpha, beta):
        img = np.uint8( np.clip ( ( alpha * img + beta) , 0, 255))
        return img

    def gray_scale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def binarize(self, gray_img, min_valve = 150 , max_valve = 255 , inverse = True):
        if(inverse):
            _, img = cv2.threshold(gray_img, min_valve, max_valve, cv2.THRESH_BINARY_INV)
        else:
            _, img = cv2.threshold(gray_img, min_valve, max_valve, cv2.THRESH_BINARY)    
        return img

    def morphology_fill(self, binary_img):
        img = binary_img.copy()
        img_inv = img.copy()
        highth, width = img.shape
        #print("highth = ", highth, "width = ", width)
        mask = np.zeros((highth + 2, width + 2), np.uint8)
        cv2.floodFill(img_inv, mask, (0, 0), 255)
        img_inv = cv2.bitwise_not(img_inv)
        img = cv2.bitwise_or(img, img_inv)
        #cv2.imshow("img_inv", img_inv) // fill
        #cv2.imshow("img_", img)
        return img

    def erode_dilate(self, img, erode_or_dialate, iteration, kernal = (5, 5)):
        kernal = np.ones((kernal), np.uint8)
        if erode_or_dialate:
            img_out = cv2.erode(img, kernal, iterations = iteration)
        else:
            img_out = cv2.dilate(img, kernal, iterations = iteration)
        return img_out

    def remove_small_object(self, img, min_size):
        nb_obj, output, status, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = status[1 : -1]
        nb_obj = nb_obj - 1
        img_out = np.zeros((output.shape), dtype = np.uint8)
        for i in range(0, nb_obj):
            if sizes[i] >= min_size:
                img_out[output == i + 1] = 255
        nb_obj , _ , _ , _ = cv2.connectedComponentsWithStats(img_out, connectivity=8)
        return img_out, nb_obj
    
    def canny(self, img, kernal = (5, 5), threshold1 = 30, threshold2 = 100):
        blur = cv2.GaussianBlur(img, kernal, 0)
        edged = cv2.Canny(blur, threshold2, threshold1)
        return edged

    def draw_contours(self, edged, img, area_size_max = 50000, area_size_min = 4000):
        cnts_array = np.zeros((8,4))

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = img.copy()
        cv2.drawContours(contours, cnts, -1, (0, 255, 0), 1)
        #cv2.imshow("contours", contours)
        i = 0
        for(_, c) in enumerate(cnts):
            if (cv2.contourArea(c) < area_size_min) or (cv2.contourArea(c) > area_size_max):
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            if((x + y + w + h) == 0):
                break
            else:
                cnts_array[i][0] = x
                cnts_array[i][1] = y
                cnts_array[i][2] = x + w
                cnts_array[i][3] = y + h
                cv2.rectangle(img , (x , y) , (x + w , y + h) , (0 , 255 ,0) , 2)
                i += 1
                if(i >= np.size(cnts_array, 0)):
                    break

        cnts_array = cnts_array[0: i, 0:4]
        cv2.imshow(" rect_finded " , img)
        return cnts, cnts_array

    def get_subROI(self, img, rect_tuple):
        sub_img = img.copy()
        (x1, y1, x2, y2) = rect_tuple
        if(x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0):
            sub_img = img
        else:
            sub_img = sub_img[int(y1): int(y2), int(x1) : int(x2)]
        return sub_img

    def histogram(self, img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        sum_hist = 0
        c = 0
        for value in hist:
            sum_hist += (c * value)
            c += 1

        if(c == 0):
            sum_hist = 0
        else:
            sum_hist /= (img.shape[0] * img.shape[1])

        return sum_hist

    def get_image_bool_matrix(self, img, matrix_size = 9, valve = 128):
        img_bool_matrix = np.full((matrix_size, matrix_size), False, dtype=bool)
        pixel_width = img.shape[1] / matrix_size
        pixel_highth = img.shape[0] / matrix_size
        for iy in range (0, matrix_size):
            for ix in range(0, matrix_size):
                sub_image_rect = (pixel_width * ix, pixel_highth * iy, 
                pixel_width + pixel_width * ix, 
                pixel_highth + pixel_highth * iy)
                if(self.histogram(self.get_subROI(img, sub_image_rect)) >= valve):
                    img_bool_matrix[iy][ix] = True
                else:
                    img_bool_matrix[iy][ix] = False
        return img_bool_matrix

class AdvanceImageProcessing(ImageProcessing):
    def get_patterns_bool_matrix(self, patterns_path, number_of_matrix = 8, matrix_size = 9):
        tmp_patterns_bool_matrix = np.full((number_of_matrix, matrix_size, matrix_size), False, dtype = bool)
        i = 0
        for path in patterns_path:
            img = cv2.imread(path)
            img = self.gray_scale(img)
            img = self.binarize(img, 150, 255, True)
            tmp_patterns_bool_matrix[i] = self.get_image_bool_matrix(img, matrix_size, 128)
            i += 1
        i = 0
        patterns_bool_matrix = np.full((number_of_matrix, matrix_size - 4 , matrix_size - 4), False, dtype = bool)
        for bool_matrix in tmp_patterns_bool_matrix:
            bool_matrix = np.delete(bool_matrix, [0, 1, 7, 8], 0)
            bool_matrix = np.delete(bool_matrix, [0, 1, 7, 8], 1)
            patterns_bool_matrix[i] = bool_matrix
            i += 1
        return patterns_bool_matrix

    def get_image_pattern_contours(self, source_img, threshold_min_valve = 150, threshold_max_valve = 255, canny_kernal = (5, 5),  canny_threshold1 = 30, canny_threshold2 = 100, show = False):
        source_img = self.gray_scale(source_img)
        bin_img = self.binarize(source_img, threshold_min_valve, threshold_max_valve, True)
        #morphology_img = self.morphology_fill(bin_img)
        edged_img = self.canny(bin_img,  canny_kernal, canny_threshold1, canny_threshold2)
        if show:
            cv2.imshow("gray_scale", source_img)
            cv2.imshow("binary_img", bin_img)
            cv2.imshow("edged_img", edged_img)

        _ , cnts_array = self.draw_contours(edged_img, cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB))
        cnts_array = np.flip(cnts_array, 0)
        return cnts_array

    
    def get_pattern(self, source_img, patterns_bool_matrix, matrix_size = 9, valve = 128):
        source_bool_matrix = np.full((matrix_size, matrix_size), False, dtype = bool)
        pattern_score_matrix = np.zeros((5,5)) #Create a zeros array to compute the score of pattern compare.
        source_bool_matrix = self.get_image_bool_matrix(source_img, matrix_size, valve) #Get bool matrix from image
        global_score = 0
        pattern_index = 0
        def matrix_convolotion(source_matrix, pattern_matrix):
            score = 0
            tmp_source_matrix = np.full((matrix_size - 4, matrix_size - 4), False, dtype = bool)
            score_source_matrix = np.copy(tmp_source_matrix)
            for ix in range(1, 4):
                for iy in range(1, 4):
                    tmp_source_matrix = source_matrix[ix:ix+5, iy:iy+5]
                    score_source_matrix = np.logical_or(np.logical_and(tmp_source_matrix, pattern_matrix), np.logical_and(np.bitwise_not(tmp_source_matrix), np.bitwise_not(pattern_matrix)))
                    local_score = np.sum(score_source_matrix)
                    if(local_score > score):
                        score = local_score
                    else:
                        score = score
            return score
        loop_index = 0
        for pattern in patterns_bool_matrix:
            score_return = matrix_convolotion(source_bool_matrix, pattern)
            if(score_return > global_score):
                global_score = score_return
                pattern_index = loop_index
            else:
                global_score = global_score
            loop_index += 1

        return pattern_index
