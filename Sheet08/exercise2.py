import cv2
import numpy as np
import matplotlib.pylab as plt

def display_corners(image, corners, title):
    '''

    :param image: Original Image
    :param corners: Corners
    :param title: Title of the plot
    :return:
    '''
    #find local maxima using dilate for non-maximum supression
    dilated_corners = cv2.dilate(corners, None)
    #display the dilated corners
    display_image(dilated_corners, title + ': Dilated Corners')
    #mark the corners
    image[dilated_corners > 0.1 * dilated_corners.max()]=[0, 0, 255]
    #display the image with marked corners
    display_image(image, title)

def threshold_corners(corners, threshold):
    '''
    Thresholding corners
    :param corners: Corners
    :param threshold: Threshold value
    :return:
    '''
    #normalize the corners
    corners_normed = cv2.normalize(corners, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #return the corners with intensity above threshold
    return (corners_normed > threshold).astype('float')


def display_image(image, title):
    '''
        display the image
        :param image: Image
        :param title: Title of the plot
        :return:
        '''
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.show()

def main():
    # Load the image
    filepath = 'data/exercise2/building.jpeg'
    image_color = cv2.imread(filepath)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # Compute Structural Tensor
    I_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    I_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
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
    epsilon = np.finfo(np.float).eps                        #additive term to avoid division by zero
    threshold_w = 0.1
    threshold_q = 0.95

    w = np.divide(det_M, trace_M + epsilon)
    w_thresholded = threshold_corners(w, threshold_w)
    display_image(w_thresholded, 'Förstner Corner Response - W: Thresholded')

    q = np.divide(4 * det_M, np.square(trace_M) + epsilon)
    q_thresholded = threshold_corners(q, threshold_q)
    display_image(q_thresholded, 'Förstner Corner Response - Q: Thresholded')

    forstners_corners = w_thresholded  * q_thresholded
    display_corners(image_color, forstners_corners, 'Förstner Corners')

if __name__ == '__main__':
    main()
