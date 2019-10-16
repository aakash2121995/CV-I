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


def gaussian_kernel2D(size=15, sigma=5):
    ax = np.arange(-(size // 2 ), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

def gaussian_kernel1D(size=15, sigma=5):
    allnum = np.linspace(-(size // 2), size // 2, size)
    gaus = np.exp(-0.5 * ((allnum / sigma) ** 2))
    return (gaus / gaus.sum())

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
    img_cpy = np.copy(img)
    cv.equalizeHist(img_cpy)
    display_image('2 - a - Histogram Equalization', img_cpy)
    



#    =========================================================================
#    ==================== Task 4 =================================
#    =========================================================================
    print('Task 4:');

    #get a copy of the image
    img_cpy = np.copy(img)
    #set sigma
    sigma = 2*np.sqrt(2)
    
    #########################     Gaussian Blur    ############################
    #apply gaussian blur on the image and display the result
    #ksize set to (0,0) to get them calculated from sigma
    img_gaussian_blurred = cv.GaussianBlur(img_cpy, (0, 0), sigma, 0)       
    display_image('4 - a - Gaussian Blur', img_gaussian_blurred)
    #calculate absolute pixel wise differnce
    pixel_error = np.absolute(img - img_gaussian_blurred)
    #print the maximum pixel error
    print('Maximum pixel error for Gaussian Blur: {}'.format(np.max(pixel_error)))
    
    
    #########################     Filter2D     ################################
    #calculate the kernel of size 15 x 15
    kernel = gaussian_kernel2D(size=15, sigma=sigma)
    #container for output image
    img_filtered_2d = img_cpy
    #apply filter2D on the image and display the result
    cv.filter2D(src=img, dst=img_filtered_2d, ddepth=-1, kernel=kernel)
    display_image('4 - b - Filter2D', img_filtered_2d)
    #calculate absolute pixel wise differnce
    pixel_error = np.absolute(img - img_filtered_2d)
    #print the maximum pixel error
    print('Maximum pixel error for Filter2D: {}'
          .format(np.max(pixel_error)))
    
    
    #########################     Seperate Filter2D    ########################
    #calculate the kernel of size 15 x 1
    kernel = gaussian_kernel1D(size=15, sigma=2)
    #container for output image
    img_sepFiltered_2d = img_cpy
    #apply seperate filter2D on the image and display the result
    cv.sepFilter2D(src=img, dst=img_sepFiltered_2d, ddepth=-1, kernelX=kernel, 
                   kernelY=kernel)
    display_image('4 - c - Seperate Filter2D', img_sepFiltered_2d)
    #calculate absolute pixel wise differnce
    pixel_error = np.absolute(img - img_sepFiltered_2d)
    #print the maximum pixel error
    print('Maximum pixel error for Seperate Filter2D: {}'
          .format(np.max(pixel_error)))



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










