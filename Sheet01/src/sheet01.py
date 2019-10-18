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

def get_pixel_error(source, target):
    """
    Computes pixel wise absolute difference between source and target image
    :param source: original image
    :param target: resulting image after processing
    :return: pixel-wise difference and maximum pixel error
    """
    pixel_error = np.absolute(source - target)
    return pixel_error, np.max(pixel_error)
    
def get_cdf(img):
    """
    Computes cumulative distribution function of the image
    :param img: original image
    :return: cumulative distribution function of the image
    """
    hist, _ = np.histogram(img.flatten(),256,[0,256])
    return hist.cumsum()

def histogram_equalization(img):
    """
    Computes histogram equalization of the image
    :param img: original image
    :return: histogram equalization of the image
    """
    cdf = get_cdf(img)
    cdf_masked = np.ma.masked_equal(cdf, 0)
    equalizedHist = (cdf_masked - cdf_masked.min())*255/(cdf_masked.max()-cdf_masked.min())
    equalizedHist = np.ma.filled(equalizedHist, 0).astype('uint8')
    return equalizedHist[img]

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
    img_path = "../images/bonn.png"
    img = cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE)


#    =========================================================================
#    ==================== Task 1 =================================
#    =========================================================================
    print('Task 1:');
    # int = cv.integral(img)
    # a)
    integral = cal_integral(img)
    normalise_int = (255.0 * (integral /integral.max())).astype(np.uint8)
    display_image("1 - a Integral Image", normalise_int)

    # b)
    print("Mean pixel value by summing up each pixel value in the image: ", img.sum()/(img.shape[0]*img.shape[1]))
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
    #get a copy of the image
    img_cpy = np.copy(img)
    #histogram equalization of the image using function provided by OpenCV
    img_equalizedHist_cv = cv.equalizeHist(img_cpy)
    display_image('2 - a - Histogram Equalization by OpenCV', img_equalizedHist_cv)
    #histogram equalization of the image by our implementation
    img_equalizedHist_custom = histogram_equalization(img_cpy)
    display_image('2 - a - Histogram Equalization by Custom Implementation', img_equalizedHist_custom)
    
    #calculate absolute pixel wise differnce
    pixel_error, max_pixel_error = get_pixel_error(img_equalizedHist_cv, img_equalizedHist_custom)
    #print the maximum pixel error
    print('Maximum pixel error for Histogram Equalization: {}'.format(max_pixel_error))
    
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
    #    ==================== Task 5 =================================
    #    =========================================================================
    print('Task 5:');

    img_gaussian_twice = cv.GaussianBlur(cv.GaussianBlur(img, (0, 0), 2, 0) , (0, 0), 2, 0)
    img_gaussian_once = cv.GaussianBlur(img, (0, 0), 2*np.sqrt(2), 0)
    display_image("5 - a Twice with a Gaussian kernel with sigma = 2",img_gaussian_twice)
    display_image("5 - b once with a Gaussian kernel with sigma = 2root2",img_gaussian_once)
    abs_diff = np.abs(img_gaussian_twice - img_gaussian_once)

    print("Maximum pixel error: ", abs_diff.max())
    #    =========================================================================
#    ==================== Task 6 =================================
#    =========================================================================
    print('Task 6:');





#    =========================================================================
#    ==================== Task 7 =================================
#    =========================================================================
    print('Task 7:');
    random_probs = np.random.rand(img.shape[0],img.shape[1])
    bool_pepper = random_probs <= 0.15
    bool_salt = np.logical_and(random_probs > 0.15, random_probs <= 0.3)
    img_cpy = img.copy()
    img_cpy[bool_pepper] = 255
    img_cpy[bool_salt] = 0
    display_image("7 - Noisy image",img_cpy)

    best_diff = 256
    best_denoised_img = []
    for filter_size in [1,3,5,7,9]:
        denoised_img = cv.GaussianBlur(img_cpy, (filter_size, filter_size), 3, 0)
        if best_diff > np.abs(denoised_img.mean() - img.mean()):
            best_diff = np.abs(denoised_img.mean() - img.mean())
            best_denoised_img = denoised_img

    display_image("Denoised Image with gaussian blur",best_denoised_img)

    best_diff = 256
    best_denoised_img = []
    for filter_size in [1, 3, 5, 7, 9]:
        denoised_img = cv.medianBlur(img_cpy,filter_size)
        if best_diff > np.abs(denoised_img.mean() - img.mean()):
            best_diff = np.abs(denoised_img.mean() - img.mean())
            best_denoised_img = denoised_img

    display_image("Denoised Image with Median blur", best_denoised_img)

    best_diff = 256
    best_denoised_img = []
    for filter_size in [1, 3, 5, 7, 9]:
        denoised_img = cv.bilateralFilter(img_cpy, filter_size, 3, 3)
        if best_diff > np.abs(denoised_img.mean() - img.mean()):
            best_diff = np.abs(denoised_img.mean() - img.mean())
            best_denoised_img = denoised_img

    display_image("Denoised Image with Bilateral blur", best_denoised_img)

#    =========================================================================
#    ==================== Task 8 =================================
#    =========================================================================
    print('Task 8:');
    kernel_1 = np.array([[0.0113, 0.0838, 0.0113], [0.0838, 0.6193, 0.0838], [0.0113, 0.0838, 0.0113]])
    kernel_2 = np.array([[-0.8984, 0.1472, 1.1410], [-1.9075, 0.1566, 2.1359], [-0.8659, 0.0573, 1.0337]])
    # get a copy of the image
    img_cpy = np.copy(img)
    image_filtered = cv.filter2D(img_cpy, -1, kernel_1)
    display_image('8 - a - Filtered using Kernel 1', image_filtered)
    image_filtered = cv.filter2D(img_cpy, -1, kernel_2)
    display_image('8 - a - Filtered using Kernel 2', image_filtered)
    w1, u1, v1_t = cv.SVDecomp(kernel_1)
    sigma1 = np.max(w1)
    # u1 = u1[0, :]
    # v1_t = v1_t.T[:,]
    #container for output image
    img_sepFiltered_2d = img_cpy
    #apply seperate filter2D on the image and display the result
    # cv.sepFilter2D(src=img, dst=img_sepFiltered_2d, ddepth=-1, kernelX=np.sqrt(sigma1)*v1_t,
    #                kernelY=np.sqrt(sigma1)*u1)
    cv.sepFilter2D(src=img, dst=img_sepFiltered_2d, ddepth=-1, kernelX=np.sqrt(sigma1) * v1_t.T[:, 0],
                   kernelY=np.sqrt(sigma1) * u1[:, 0])
    display_image('8 - b - Seperate Filter2D with kernel 1', img_sepFiltered_2d)
    #calculate absolute pixel wise differnce
    pixel_error, max_pixel_error = get_pixel_error(img, img_sepFiltered_2d)
    #print the maximum pixel error
    print('Maximum pixel error for Seperate Filter2D with kernel 1: {}'
          .format(max_pixel_error))
    
    w2, u2, v2_t = cv.SVDecomp(kernel_2)
    sigma2 = np.max(w2)
    # u2 = u2[0 : 1][0]
    # v2_t = v2_t[0 : 1][0]
    
    #container for output image
    img_sepFiltered_2d = img_cpy
    #apply seperate filter2D on the image and display the result
    cv.sepFilter2D(src=img, dst=img_sepFiltered_2d, ddepth=-1, kernelX=np.sqrt(sigma2)*v2_t.T[:, 0],
                   kernelY=np.sqrt(sigma2)*u2[:, 0])
    display_image('8 - b - Seperate Filter2D with kernel 2', img_sepFiltered_2d)
    #calculate absolute pixel wise differnce
    pixel_error, max_pixel_error = get_pixel_error(img, img_sepFiltered_2d)
    #print the maximum pixel error
    print('Maximum pixel error for Seperate Filter2D with kernel 2: {}'
          .format(max_pixel_error))
    









