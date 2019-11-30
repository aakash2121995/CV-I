#!/usr/bin/python3.5

import numpy as np
import cv2 as cv

'''
    load the image and foreground/background parts
    image: the original image
    background/foreground: numpy array of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''


def display_image(img, window_name=''):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def read_image(filename):
    image = cv.imread(filename) / 255.0
    height, width = image.shape[:2]
    bounding_box = np.zeros(image.shape)
    bounding_box[90:350, 110:250, :] = 1
    bb_width, bb_height = 140, 260
    background = image[bounding_box == 0].reshape((height * width - bb_width * bb_height, 3))
    foreground = image[bounding_box == 1].reshape((bb_width * bb_height, 3))

    return image, foreground, background


class GMM(object):

    def gaussian_scores(self, data):
        # TODO
        pass

    def fit_single_gaussian(self, data):
        mean = data.mean(axis=0)
        diagonal = np.var(data, axis=0)
        covariance = np.diag(diagonal)
        self.gaussians = [(mean, covariance, 1)]

    def estimate_r(self, mean, covariance, lmbda, data):
        inverse = np.linalg.inv(covariance)
        deno = np.sqrt(np.linalg.det(2 * np.pi * covariance))
        d = data - mean
        num = lmbda * np.exp(-0.5 * np.sum((d @ inverse) * d, axis=1))
        return num / deno

    def estep(self, data):
        n, m = data.shape
        k = len(self.gaussians)
        r = np.zeros((n, k))
        for idx, (mean, covariance, lmbda) in enumerate(self.gaussians):
            r[:, idx] = self.estimate_r(mean, covariance, lmbda, data)
        r /= r.sum(axis=1).reshape(n, 1)
        return r

    def mstep(self, data, r):
        new_gaussians = []
        for idx in range(len(self.gaussians)):
            lmbda_new = r[:, idx].sum() / r.sum()
            mean_new = r[:, idx] @ data / r[:, idx].sum()
            d = data - mean_new
            covariance_new = (r[:, idx] * d.T) @ d / r[:, idx].sum()
            new_gaussians.append((mean_new.copy(), covariance_new.copy(), lmbda_new.copy()))
        self.gaussians = new_gaussians

    def em_algorithm(self, data, n_iterations=10):
        for i in range(n_iterations):
            r = self.estep(data)
            self.mstep(data, r)

    def split(self, epsilon=0.1):
        new_gaussians = []
        for mean, covariance, lmbda in self.gaussians:
            lmbda_new = lmbda / 2
            mean1, mean2 = mean + epsilon * covariance.diagonal(), mean - epsilon * covariance.diagonal()
            covariance1, covariance2 = covariance.copy(), covariance.copy()
            new_gaussians.extend([(mean1, covariance1, lmbda_new), (mean2, covariance2, lmbda_new)])
        self.gaussians = new_gaussians

    def probability(self, data):
        prob = np.zeros(data.shape[0])
        for idx, (mean, covariance, lmbda) in enumerate(self.gaussians):
            prob += self.estimate_r(mean, covariance, lmbda, data)
        return prob

    def sample(self):
        # TODO
        pass

    def train(self, data, n_splits):
        self.fit_single_gaussian(data)
        for i in range(n_splits):
            self.split()
        self.em_algorithm(data)


image, foreground, background = read_image('person.jpg')

'''
TODO: compute p(x|w=background) for each image pixel and manipulate the image such that everything below the threshold is black, display the resulting image
Hint: Slide 64
'''
gmm_background = GMM()
gmm_background.train(background, 3)
image_flat = image.reshape(image.shape[0] * image.shape[1], 3)
vals = gmm_background.probability(image_flat).reshape(image.shape[0], image.shape[1])
image[np.where(vals > 50)] = 0
display_image(image)
