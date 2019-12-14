import cv2
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import euclidean_distances

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

def match_points(descriptors_1, descriptors_2, threshold=0.7):
    '''
    match points given two descriptor, from 1 to 2
    :param descriptors_1: Query Descriptor
    :param descriptors_2: Train Descriptor
    :param threshold: max value for Lowe's ratio test, Default 0.7
    :return:
    '''
    epsilon = 1e-12
    infinity = 1e20
    accepted_matches = []
    pairwise_distances = euclidean_distances(descriptors_1, descriptors_2)
    queryIdx = np.arange(descriptors_1.shape[0]).astype('int')
    trainIdx = pairwise_distances.argmin(axis=1).astype('int')
    best_distances = pairwise_distances[queryIdx, trainIdx]
    pairwise_distances[queryIdx, trainIdx] = infinity
    second_best_distances = pairwise_distances[queryIdx, pairwise_distances.min(axis=1).astype('int')]
    ratio = np.divide(best_distances, second_best_distances + epsilon)
    queryIdx_accpeted = queryIdx[ratio < threshold]
    trainIdx_accepted = trainIdx[ratio < threshold]
    for i in range(queryIdx_accpeted.shape[0]):
        match = cv2.DMatch(queryIdx_accpeted[i], trainIdx_accepted[i],
                           pairwise_distances[queryIdx_accpeted[i], trainIdx_accepted[i]])
        accepted_matches.append([match])
    return accepted_matches


def display_keypoints(image, keypoints, title=''):
    '''

    :param image: Original Image
    :param keypoints: Keypoints
    :param title: Title of the plot
    :return:
    '''
    img = cv2.drawKeypoints(image, keypoints, None)
    display_image(img, title + ' Keypoints')

def main():
    # Load the images
    mountain_1 = cv2.imread('data/exercise3/mountain1.png', 0)
    mountain_2 = cv2.imread('data/exercise3/mountain2.png', 0)

    # extract sift keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(mountain_1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(mountain_2, None)
    display_keypoints(mountain_1, keypoints_1, 'Mountain - 1')
    display_keypoints(mountain_2, keypoints_2, 'Mountain - 2')

    # implementation of matching
    matched_keypoints = match_points(descriptors_1, descriptors_2, threshold=0.4)

    # display the matches
    image_matched = cv2.drawMatchesKnn(mountain_1, keypoints_1, mountain_2, keypoints_2, matched_keypoints,
                                       None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    display_image(image_matched, 'Keypoints Matched Using Ratio Test')

if __name__ == '__main__':
    main()
