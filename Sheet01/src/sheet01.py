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
    img_path = "../images/bonn.png"
    img = cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE)


#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1:');
    # int = cv.integral(img)
    # a)
    integral = cal_integral(img)
    normalise_int = (255.0 * (integral /integral.max() )).astype(np.uint8)
    display_image("Integral Img", normalise_int)

    # b)
    print("Mean pixel value by summing up each pixel value in the image: " ,img.sum()/(img.shape[0]*img.shape[1]))
    print("Mean pixel value by computing an integral image using the function cv.integral: ", cv.integral(img)[-1,-1]/ (img.shape[0] * img.shape[1]))
    print("Mean pixel value by computing an integral image with own function,: ", cal_integral(img)[-1,-1] / (img.shape[0] * img.shape[1]))

    # c)
    PATCH_SIZE = 100
    time_pixel_wise, time_cv_func, time_own_func = 0, 0, 0
    for i in range(10):
        rand_coord = random.randint(0, img.shape[0] - PATCH_SIZE), random.randint(0, img.shape[1] - PATCH_SIZE)
        patch = img[rand_coord[0]:rand_coord[0]+PATCH_SIZE, rand_coord[1]:rand_coord[1]+PATCH_SIZE]

        start = time.time()
        val = patch.sum()/(PATCH_SIZE*PATCH_SIZE)
        end = time.time()
        time_pixel_wise += end-start

        start = time.time()
        val = cv.integral(patch)[-1,-1] / (PATCH_SIZE * PATCH_SIZE)
        end = time.time()
        time_cv_func += end - start

        start = time.time()
        val = cal_integral(patch)[-1,-1]/ (PATCH_SIZE * PATCH_SIZE)
        end = time.time()
        time_own_func += end - start

    print("\nTime taken by summing up each pixel value in the image: ", time_pixel_wise)
    print("Time taken by computing an integral image using the function cv.integral: ",
          time_cv_func)
    print("Time taken by computing an integral image with own function,: ",
          time_own_func)
    print()
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










