import cv2 as cv
import numpy as np


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def integral(img):
    int_ = cv.copyMakeBorder(img.copy().astype(np.int64),top=1,left=1,bottom=0,right=0,borderType=cv.BORDER_CONSTANT,value=0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            int_[i+1, j+1] = int_[i, j+1] + int_[i+1,j] - int_[i, j] + img[i, j]

    return int_


img = cv.imread('../images/bonn.png', flags=cv.IMREAD_GRAYSCALE)
# int = cv.integral(img)
int = integral(img)
normalise_int = (255.0 * int / int.max()).astype(np.uint8)
display_image("Integral Img", normalise_int)
