import cv2
import numpy as np
import matplotlib.pylab as plt

def show_corners(val):
    pass

def get_derivatives(I):
    I_x = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    I_y = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return I_x, I_y

def main():
    # Load the image
    filepath = 'data/exercise2/building.jpeg'
    image = cv2.imread(filepath, 0)
    # Compute Structural Tensor
    I_x, I_y = get_derivatives(image)
    w = (1. / 9) * np.ones((3, 3))
    #M = np.block([[np.square(I_x), I_xy], [I_xy, np.square(I_y)]])
    #M = cv2.filter2D(M, -1, w)
    IxIx = cv2.filter2D(np.square(I_x), -1, w)
    IyIy = cv2.filter2D(np.square(I_y), -1, w)
    IxIy = cv2.filter2D(np.multiply(I_x, I_y), -1, w)
    det_M = np.multiply(IxIx, IyIy) - np.square(IxIy)
    trace_M = np.add(IxIx, IyIy)
    # Harris Corner Detection
    k = 0.04
    corner_response = det_M - (k * np.square(trace_M))
    print(corner_response)

    # Forstner Corner Detection



if __name__ == '__main__':
    main()
