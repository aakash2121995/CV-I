import cv2
import pprint
import numpy as np
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)

def decompose(M):
    rank_M = np.linalg.matrix_rank(M)
    if rank_M == 1:
        print('Separable as rank of the matrix is {}'.format(rank_M))
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        print('A can be factorized as: u * s * vh where,')
        print('u = {}'.format(np.around(u, decimals=5)))
        print('s = {}'.format(np.diag(np.around(s, decimals=5))))
        print('vh = {}'.format(np.around(vh, decimals=5)))
    else:
        print('Not Separable as rank of the matrix is {}; greater than 1.'.format(rank_M))

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task_1_b():
    image = cv2.imread("img/fourier1.png", 0)
    original_image = np.real(np.fft.ifftn(image))
    display = np.hstack((image, np.fft.ifftshift(original_image)))
    display_image('Fourier - Original', display.astype(np.uint8))

def task_1_d():
    A = np.array([[3, 1, -9, -2, 0],
                  [5, 2, 2, 3, -1],
                  [9, 4, -9, -8, 1],
                  [2, 10, -20, -20, 0],
                  [4, 8, 4, -6, 0]])
    decompose(A)

def task_1_e():
    A = np.array([[-21, 6 ,  3, 12,  9],
                  [7  ,-2 , -1, -4, -3],
                  [0  , 0 ,  0,  0,  0],
                  [35 ,-10, -5,-20,-15],
                  [-14, 4 ,  2,  8,  6]])
    decompose(A)


if __name__ == "__main__":
    #task_1_b()
    task_1_d()
    task_1_e()