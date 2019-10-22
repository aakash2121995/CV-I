import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def gaussian_kernel2D(size = 15, sigma=5):
    ax = np.arange(-(size // 2 ), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_convolution_using_fourier_transform(image, kernel):
    '''fft_kernel = np.fft.fft2(kernel, s=image.shape[:2], axes=(0, 1))

    # convolve
    fft_image = np.fft.fft2(image, axes=(0, 1))
    fft_image_new = fft_kernel * fft_image
    filtered_image = np.fft.ifft2(fft_image_new, axes=(0, 1)).real

    # clip values to range
    filtered_image = np.clip(filtered_image, 0, 1)
    sz = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1])  # total amount of padding
    kernel = np.pad(kernel, (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)), 'constant')
    kernel = np.fft.ifftshift(kernel)

    filtered_image = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(kernel)))
    return filtered_image'''
    l, w = image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1
    fft_image = np.fft.fft2(image, [l, w])
    fft_kernel = np.fft.fft2(kernel, [l, w])
    result = np.fft.ifft2(np.multiply(fft_image.real, fft_kernel.real))
    return result.real

def task1():
    image = cv2.imread("data/einstein.jpeg", 0)
    display_image('Task - 1: Original Image', image)
    # calculate kernel
    #kernel = cv2.getGaussianKernel(ksize=7, sigma=1, ktype=cv2.CV_64F)
    kernel = gaussian_kernel2D(7, 1)
    print(kernel.shape)
    # calculate convolution of image and kernel
    # container for output image
    conv_result = image
    # apply filter2D on the image and display the result
    cv2.filter2D(src=image, dst=conv_result, ddepth=-1, kernel=kernel)
    display_image('1 - a - Gaussian Blur with Filter2D', conv_result)
    fft_result = get_convolution_using_fourier_transform(image, kernel)
    display_image('1 - a - Gaussian Blur with Fourier Transform', fft_result)

    # compare results


def sum_square_difference(image, template):
    output_image = np.empty((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))

    for i in range(image.shape[0] - template.shape[0] + 1):
        for j in range(image.shape[1] - template.shape[1] + 1):
            patch = image[i:i + template.shape[0], j:j + template.shape[1]]
            squared_diff = np.square(patch - template)
            output_image[i, j] = squared_diff.sum()

    # min_ = output_image.min()
    # max_ = output_image.max()
    # range_ = max_ - min_
    # output_image = 255 * ((output_image - min_) / range_)
    # output_image = output_image.astype(np.uint8)
    return output_image


def normalized_cross_correlation(image, template):
    mean_template = template.mean()
    mean_norm_template = template - mean_template
    squared_sum_template = np.square(mean_norm_template).sum()

    output_image = np.empty((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))

    for i in range(image.shape[0] - template.shape[0] + 1):
        for j in range(image.shape[1] - template.shape[1] + 1):
            patch = image[i:i + template.shape[0], j:j + template.shape[1]]
            mean_norm_patch = patch - patch.mean()
            numerator = (mean_norm_template * mean_norm_patch).sum()
            deno = np.sqrt(squared_sum_template * np.square(mean_norm_patch).sum())
            output_image[i, j] = numerator / deno

    # min_ = output_image.min()
    # max_ = output_image.max()
    # range_ = max_ - min_
    # output_image = 255 * ((output_image - min_) / range_)
    # output_image = output_image.astype(np.uint8)
    return output_image


def task2():
    image = cv2.imread("data/lena.png", 0)
    template = cv2.imread("data/eye.png", 0)

    result_ssd = sum_square_difference(image, template)
    img_cpy = image.copy()
    indexes = np.unravel_index(result_ssd.argmax(),result_ssd.shape)
    cv2.rectangle(img_cpy,indexes,(indexes[0]+template.shape[0], indexes[1]+template.shape[1]),color=(200,0,0,),thickness=1)
    display_image("sum", img_cpy)
    result_ncc = normalized_cross_correlation(image, template)
    img_cpy = image.copy()
    indexes = np.unravel_index(result_ncc.argmax(), result_ncc.shape)
    cv2.rectangle(img_cpy, indexes, (indexes[0] + template.shape[0], indexes[1] + template.shape[1]),
                  color=(200, 0, 0,), thickness=1)
    display_image("sum", img_cpy)
    # display_image("sum", result_ncc)
    # result_cv_sqdiff = cv2.  # calculate using opencv
    result_cv_ncc = None  # calculate using opencv

    # draw rectangle around found location in all four results
    # show the results


def build_gaussian_pyramid_opencv(image, num_levels):
    return None


def build_gaussian_pyramid(image, num_levels, sigma):
    return None


def template_matching_multiple_scales(pyramid, template):
    return None


def task3():
    image = cv2.imread("../data/traffic.jpg", 0)
    template = cv2.imread("../data/template.jpg", 0)

    cv_pyramid = build_gaussian_pyramid_opencv(image, 8)
    mine_pyramid = build_gaussian_pyramid(image, 8)

    # compare and print mean absolute difference at each level
    result = template_matching_multiple_scales(pyramid, template)

    # show result


def get_derivative_of_gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size,sigma,)
    kernel_x = kernel*kernel.T
    kernel_y = kernel_x.copy()
    kernel_x_padded = cv2.copyMakeBorder(kernel_x[:, :-1].copy(),top=0,bottom=0,left=1,right=0,borderType=cv2.BORDER_REFLECT)
    kernel_y_padded = cv2.copyMakeBorder(kernel_y[:-1, :].copy(),top=1,bottom=0,left=0,right=0,borderType=cv2.BORDER_REFLECT)

    return kernel_x_padded-kernel_x, kernel_y_padded - kernel_y


def task4():
    image = cv2.imread("data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = cv2.filter2D(image, -1, kernel_x)  # convolve with kernel_x
    edges_y = cv2.filter2D(image, -1, kernel_y)  # convolve with kernel_y

    magnitude = np.sqrt(np.square(edges_x) + np.square(edges_y))  # compute edge magnitude
    direction = np.arctan2(edges_y,edges_x)  # compute edge direction

    magnitude = (255*magnitude/magnitude.max()).astype(np.uint8)
    direction = (255*direction/direction.max()).astype(np.uint8)

    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    return None


def task5():
    image = cv2.imread("data/traffic.jpg", 0)

    edges = cv2.Canny(image,100,200)  # compute edges
    edge_function = None  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, positive_inf, negative_inf
    )
    dist_transfom_cv = None  # compute using opencv

    # compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    #task2()
    # task3()
    #task4()
    #task5()
