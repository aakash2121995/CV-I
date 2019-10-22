import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def gaussian_kernel2D(size=15, sigma=5):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)


def get_pixel_error(source, target):
    """
    Computes pixel wise absolute difference between source and target image
    :param source: original image
    :param target: resulting image after processing
    :return: pixel-wise difference and mean pixel error
    """
    pixel_error = np.absolute(source - target)
    return pixel_error, np.mean(pixel_error, dtype=np.float64)


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_rect(label, result, image, template):
    indexes = np.unravel_index(result.argmax(), result.shape)
    cv2.rectangle(image, indexes, (indexes[0] + template.shape[0], indexes[1] + template.shape[1]),
                  color=(200, 0, 0,), thickness=1)
    display_image(label, image)


def get_convolution_using_fourier_transform(image, kernel):
    l, w = image.shape[0] + kernel.shape[0] - 1, image.shape[1] + kernel.shape[1] - 1
    fft_image = np.fft.fft2(image, [l, w])
    fft_kernel = np.fft.fft2(kernel, [l, w])
    filtered_image = np.real(np.fft.ifft2(np.multiply(fft_image, fft_kernel)))
    return filtered_image.real


def task1():
    image = cv2.imread("data/einstein.jpeg", 0)
    display_image('Task - 1: Original Image', image)
    # calculate kernel
    # kernel = cv2.getGaussianKernel(ksize=7, sigma=1, ktype=cv2.CV_64F)
    kernel = gaussian_kernel2D(7, 1)
    # calculate convolution of image and kernel using OpenCV
    # container for output image
    conv_result = image
    # apply filter2D on the image and display the result
    cv2.filter2D(src=image, dst=conv_result, ddepth=-1, kernel=kernel)
    display_image('1 - a - Gaussian Blur with Filter2D', conv_result)
    # calculate convolution of image and kernel using FFT
    fft_result = get_convolution_using_fourier_transform(image, kernel).astype(np.uint8)
    display_image('1 - a - Gaussian Blur with Fourier Transform', fft_result)
    # compare results
    # pad the convolved image with zero to resolve the size difference
    conv_result_padded = np.pad(conv_result, [(fft_result.shape[0]-conv_result.shape[0])//2,
                                              (fft_result.shape[1]-conv_result.shape[1])//2], mode='constant')
    _, mean_abs_diff = get_pixel_error(conv_result_padded, fft_result)
    print('Mean absolute difference between two blurred images: {}'.format(mean_abs_diff))


def sum_square_difference(image, template):
    output_image = np.empty((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1))

    for i in range(image.shape[0] - template.shape[0] + 1):
        for j in range(image.shape[1] - template.shape[1] + 1):
            patch = image[i:i + template.shape[0], j:j + template.shape[1]]
            squared_diff = np.square(patch - template)
            output_image[i, j] = squared_diff.sum()

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

    return output_image


def task2():
    image = cv2.imread("data/lena.png", 0)
    template = cv2.imread("data/eye.png", 0)

    img_cpy = image.copy()
    result_ssd = sum_square_difference(image, template)
    display_rect("2 - Own Sum Square Differece", result_ssd, img_cpy, template)

    img_cpy = image.copy()
    result_ncc = normalized_cross_correlation(image, template)
    display_rect("2 - Own NCC", result_ncc, img_cpy, template)

    img_cpy = image.copy()
    result_cv_sqdiff = cv2.matchTemplate(image, template, method=cv2.TM_SQDIFF)  # calculate using opencv
    display_rect("2 - CV Sum Squared Differences", result_cv_sqdiff, img_cpy, template)

    img_cpy = image.copy()
    result_cv_ncc = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)  # calculate using opencv
    display_rect("2 - CV NCC", result_cv_ncc, img_cpy, template)

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
    kernel = cv2.getGaussianKernel(size, sigma, )
    kernel_x = kernel * kernel.T
    kernel_y = kernel_x.copy()
    kernel_x_padded = cv2.copyMakeBorder(kernel_x[:, :-1].copy(), top=0, bottom=0, left=1, right=0,
                                         borderType=cv2.BORDER_REFLECT)
    kernel_y_padded = cv2.copyMakeBorder(kernel_y[:-1, :].copy(), top=1, bottom=0, left=0, right=0,
                                         borderType=cv2.BORDER_REFLECT)

    return kernel_x_padded - kernel_x, kernel_y_padded - kernel_y


def task4():
    image = cv2.imread("data/einstein.jpeg", 0)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = cv2.filter2D(image, -1, kernel_x)  # convolve with kernel_x
    edges_y = cv2.filter2D(image, -1, kernel_y)  # convolve with kernel_y

    magnitude = np.sqrt(np.square(edges_x) + np.square(edges_y))  # compute edge magnitude
    direction = np.arctan2(edges_y, edges_x)  # compute edge direction

    magnitude = (255 * magnitude / magnitude.max()).astype(np.uint8)
    direction = (255 * direction / direction.max()).astype(np.uint8)

    display_image("Magnitude", magnitude)
    display_image("Direction", direction)


def l2_distance_transform_1D(column, positive_inf, negative_inf):
    k = 0
    v = np.empty(column.shape[0]).astype(int)
    v[0] = 0
    z = np.empty(column.shape[0]).astype(int)
    z[0] = negative_inf
    z[1] = positive_inf

    output = np.empty_like(column)

    for q in range(1, column.shape[0]):
        while True:
            s = (column[q] + q * q - (column[v[k]] + v[k] ** 2)) / (2 * q - 2 * v[k])
            if s > z[k]:
                break
            k = k - 1
        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = positive_inf

    k = 0

    for q in range(column.shape[0]):
        while z[k + 1] < q:
            k = k + 1
        output[q] = (q - v[k]) ** 2 + column[v[k]]

    return output


def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
    for row_ind in range(edge_function.shape[0]):
        edge_function[row_ind, :] = l2_distance_transform_1D(edge_function[row_ind, :], positive_inf, negative_inf)

    for col_ind in range(edge_function.shape[1]):
        edge_function[:, col_ind] = l2_distance_transform_1D(edge_function[:, col_ind], positive_inf, negative_inf)

    return edge_function


def task5():
    image = cv2.imread("data/traffic.jpg", 0)

    edges = cv2.Canny(image, 100, 200)  # compute edges
    display_image("5 - a Edges", edges)
    edges = edges.astype(np.int64)
    edges[(edges == 0)] = 10 ** 10
    edges[(edges == 255)] = 0
    edges[(edges == 10 ** 10)] = 1
    edges = edges.astype(np.uint8)
    edge_function = edges  # prepare edges for distance transform

    dist_transfom_mine = l2_distance_transform_2D(
        edge_function, 999999, -999999
    )
    dist_transfom_cv = cv2.distanceTransform(edge_function, distanceType=cv2.DIST_L2,
                                             maskSize=3)  # compute using opencv
    display_image("Display Transform", dist_transfom_cv)
    # compare and print mean absolute difference


if __name__ == "__main__":
    task1()
    # task2()
    # task3()
    # task4()
    # task5()
