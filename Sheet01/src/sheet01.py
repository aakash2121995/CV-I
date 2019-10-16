import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import time
import sys

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cal_integral(img):
    int_ = cv.copyMakeBorder(img.copy().astype(np.int64),top=1,left=1,bottom=0,right=0,borderType=cv.BORDER_CONSTANT,value=0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_[i+1, j+1] = int_[i, j+1] + int_[i+1,j] - int_[i, j] + img[i, j]
    return int_

if __name__ == '__main__':
    # img_path = sys.argv[1]
    img_path = "../src/bonn.png"
    img = cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE)


#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1:');
    # int = cv.integral(img)
    integral = cal_integral(img)
    normalise_int = (255.0 * (integral - integral.min() / (integral.max() - integral.min()))).astype(np.uint8)
    display_image("Integral Img", normalise_int)




#    =========================================================================
#    ==================== Task 2 =================================
#    =========================================================================
    print('Task 2:');





#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');





#    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');





#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');





#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');










