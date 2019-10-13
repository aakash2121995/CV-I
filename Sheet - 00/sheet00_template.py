import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # set image path
    img_path = 'bonn.png'

    # 2a: read and display the image 
    img = cv.imread(img_path)
    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image
    img_gray = cv.cvtColor(img,code=cv.COLOR_RGB2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation
    img_cpy = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_cpy[i,j,0] = max(img[i,j,0]-img_gray[i,j]*0.5,0)
            img_cpy[i,j,1] = max(img[i,j,1]-img_gray[i,j]*0.5,0)
            img_cpy[i,j,2] = max(img[i,j,2]-img_gray[i,j]*0.5,0)

    display_image('2 - c - Reduced Intensity Image', img_cpy)

    # 2d: one-line statement to perfom the operation above
    img_cpy = np.maximum(img-np.expand_dims((img_gray*0.5).astype(int),axis=2),0).astype(np.uint8)
    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)

    # 2e: Extract the center patch and place randomly in the image
    PATCH_SIZE = 16
    img_center = int(img.shape[0]/2),int(img.shape[1]/2)
    top_left_corner = img_center[0] - int(PATCH_SIZE/2),img_center[1] - int(PATCH_SIZE/2)
    bottom_right_corner = img_center[0] + int(PATCH_SIZE/2),img_center[1] + int(PATCH_SIZE/2)
    img_patch = img[top_left_corner[0]:bottom_right_corner[0],top_left_corner[1]:bottom_right_corner[1],:]

    display_image('2 - e - Center Patch', img_patch)

    # Random location of the patch for placement
    rand_coord = random.randint(0, img.shape[0] - PATCH_SIZE), random.randint(0, img.shape[1] - PATCH_SIZE)
    img_cpy = np.copy(img)
    img_cpy[rand_coord[0]:rand_coord[0]+PATCH_SIZE,rand_coord[1]:rand_coord[1]+PATCH_SIZE, :] = img_patch[:,:,:]
    display_image('2 - e - Center Patch Placed Random %d, %d' % (rand_coord[0], rand_coord[1]), img_cpy)

    # 2f: Draw random rectangles and ellipses
    display_image('2 - f - Rectangles and Ellipses', img_cpy)

    # destroy all windows
    cv.destroyAllWindows()
