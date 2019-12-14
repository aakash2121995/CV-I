import cv2
import numpy as np
import matplotlib.pylab as plt

def display_corners(image, corners, title):
    dilated_corners = cv2.dilate(corners, None)
    display_image(dilated_corners, title + ': Dilated Corners')
    image[dilated_corners > 0.1 * dilated_corners.max()]=[0,0,255]
    display_image(image, title)

def threshold_corners(corners, threshold):
    corners_normed = cv2.normalize(corners, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return (corners_normed > threshold).astype('float')


def display_image(image, title):
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()


def get_derivatives(I):
    I_x = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    I_y = cv2.Sobel(I, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    return I_x, I_y

def main():
    # Load the image
    filepath = 'data/exercise2/building.jpeg'
    image_color = cv2.imread(filepath)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    # Compute Structural Tensor
    I_x, I_y = get_derivatives(image)
    box_kernel = (1. / 9) * np.ones((3, 3))
    #M = np.block([[np.square(I_x), I_xy], [I_xy, np.square(I_y)]])
    #M = cv2.filter2D(M, -1, box_kernel)
    IxIx = cv2.filter2D(np.square(I_x), -1, box_kernel)
    IyIy = cv2.filter2D(np.square(I_y), -1, box_kernel)
    IxIy = cv2.filter2D(np.multiply(I_x, I_y), -1, box_kernel)
    det_M = np.multiply(IxIx, IyIy) - np.square(IxIy)
    trace_M = np.add(IxIx, IyIy)

    # Harris Corner Detection
    k = 0.04
    hcorner_thresh = 0.3
    corner_response = det_M - (k * np.square(trace_M))
    R = threshold_corners(corner_response, hcorner_thresh)
    display_image(R, 'Harris Corner Response: Thresholded')
    display_corners(image_color, corner_response, 'Harris Corners')

    # Forstner Corner Detection
    epsilon = 1e-12                         #term to avoid division by zero
    threshold_w = 0.1
    threshold_q = 0.95
    w = np.divide(det_M, trace_M + epsilon)
    w_thresholded = threshold_corners(w, threshold_w)
    q = np.divide(4 * det_M, np.square(trace_M) + epsilon)
    q_thresholded = threshold_corners(q, threshold_q)
    display_image(q_thresholded, 'Förstner Corner Response - Q: Thresholded')
    display_image(w_thresholded, 'Förstner Corner Response - W: Thresholded')
    forstners_corners = w_thresholded  * q_thresholded
    display_corners(image_color, forstners_corners, 'Förstner Corners')

if __name__ == '__main__':
    main()
