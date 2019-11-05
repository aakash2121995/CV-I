import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

##############################################
#     Task 1        ##########################
##############################################


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    edges = cv.Canny(img, 50, 150, apertureSize=3)
    # display_image("1 - Edges",edges)
    lines = cv.HoughLines(edges, 0.5, np.pi / 180,50)
    lines = np.squeeze(lines,axis=1)
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    display_image("1 - Hough transform CV", img)


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
    edge_indexes = np.where(img_edges == 255)
    edge_indexes = list(map(lambda x, y: (x, y), edge_indexes[0], edge_indexes[1]))

    for x,y in edge_indexes:
        for theta in range(accumulator.shape[0]):
            d = abs(y*np.cos(np.pi*theta*theta_step_sz/180) - x*np.sin(np.pi*theta*theta_step_sz/180))
            if d <=0:
                print("x = {0}, y = {1}, theta = {2}".format(x,y,theta*theta_step_sz))
                print(d)
            accumulator[theta, int(d/d_resolution)] += 1

    detected_lines = np.where(accumulator >= threshold)
    thetas = detected_lines[0]*theta_step_sz*np.pi/180
    d_ = detected_lines[1]*d_resolution
    detected_lines = list(map(lambda x, y: (x, y), thetas, d_))
    return detected_lines, accumulator


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,50,150) # detect the edges
    detected_lines, accumulator = myHoughLines(edges, 1, 2, 50)
    for theta, rho in detected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    display_image("1 - Hough transform Own implementation", img)


##############################################
#     Task 2        ##########################
##############################################

def getNeighbors(data, centroid, radius = 10):
    neighbors = []
    intensity_threshold = 0.5*np.amax(data)
    for datapoint in data:
        #distance_between = np.sqrt(np.sum((datapoint - centroid)**2))
        distance_between = np.linalg.norm(datapoint - centroid, axis=None)
        if distance_between <= radius:
            neighbors.append(datapoint)
    return neighbors

class Center:
    def __init__(self, position, intensity):
        self.position = position
        self.instensity = 0.
        self.neighbors = []

    def calculateNeighbors(self, data, radius):
        intensity_threshold = 0.5 * np.amax(data)
        x_left = int(np.maximum(np.ceil(self.position[0])- radius, 0))
        x_right = int(np.minimum(np.floor(self.position[0]) + radius, data.shape[0]))
        y_left = int(np.maximum(np.ceil(self.position[1]) - radius, 0))
        y_right = int(np.minimum(np.floor(self.position[1]) + radius, data.shape[1]))
        neighbors = []
        #print(x_left, x_right, y_left, y_right)
        dataSet = data[x_left:x_right, y_left:y_right]

        '''for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if x_left <= i <= x_right and y_left <= j <= y_right and self.instensity - intensity_threshold < data[i, j] < self.instensity + intensity_threshold:
                    neighbors.append([i, j, data[i, j]])'''
        for i in range(dataSet.shape[0]):
            for j in range(dataSet.shape[1]):
                if self.instensity - intensity_threshold < dataSet[i, j] < self.instensity + intensity_threshold:
                    neighbors.append([i, j, dataSet[i, j]])
        self.neighbors = np.array(neighbors)

    def shiftCenter(self):
        self.position[0], self.position[1], self.instensity = np.average(self.neighbors, axis=0)



def mean_shift(data, window_size=5):
    centers = [Center([i, j], data[i, j]) for i in range(data.shape[0]) for j in range(data.shape[1])]
    converged = []
    i = 0
    while True:
        current_converged = []
        for center in centers:
            center.calculateNeighbors(data, window_size)
            prev_center = [center.position[0], center.position[1], center.instensity]
            center.shiftCenter()

            if center.position[0] == prev_center[0] and center.position[1] == prev_center[1] and center.instensity == prev_center[2]:
                current_converged.append(center)
        converged.extend(current_converged)
        centers = list(set(centers).difference(set(current_converged)))
        if len(centers) == 0:
            break
        i+=1
        print("iteration {}".format(i))

    return converged


'''def meanShift(data, window_size=5, max_iteration=5):
    centers = data.copy()
    convergence = False
    iter = 0
    while not convergence:
        newCenters = []
        prev_centers = centers.copy()
        for center in centers:
            neighbors = getNeighbors(data, center)
            newCenter = np.mean(neighbors, axis=0)
            newCenters.append(tuple(newCenter))
        uniqueCenters = sorted(list(set(newCenters)))
        centers = np.array(uniqueCenters)
        convergence = np.array_equal(prev_centers, centers)
        iter += 1
    print("iterations {}".format(iter))
    return centers'''


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image into grayscale
    edges = cv.Canny(img_gray, 50, 150)  # detect the edges
    theta_res = 2 # set the resolution of theta
    d_res = 1 # set the distance resolution
    detected_lines, accumulator = myHoughLines(edges, d_res, theta_res, 50)
    centers = mean_shift(accumulator)
    centers = np.array([[c.position[0], c.position[1]] for c in centers])
    detected_lines_ = centers.max()
    for theta, rho in detected_lines:
        a = np.cos(theta / 2)
        b = np.sin(theta / 2)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    display_image("2 - Peaks using mean shift algorithm", img)
    accumulator = (255 * (accumulator - accumulator.min()) / (accumulator.max() - accumulator.min())).astype(np.uint8)
    display_image("Accumulator with detected lines", accumulator)


##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    data = data.astype(np.float32)
    centers = np.zeros((k, data.shape[1]))
    index = np.zeros(data.shape[0], dtype=int)
    clusters = [[] for i in range(k)]

    # initialize centers using some random points from data
    centers = data[np.random.choice(data.shape[0],k),:].copy()
    # ....

    convergence = False
    iterationNo = 0
    while not convergence:
        # assign each point to the cluster of closest center
        # ...
        distances = np.empty((data.shape[0],k))
        for cluster_ind in range(k):
            print(centers[cluster_ind])
            distances[:,cluster_ind] =  np.linalg.norm(data - centers[cluster_ind],axis=1)

        # update clusters' centers and check for convergence
        old_centers = centers.copy()
        index = distances.argmin(axis=1)
        for cluster_ind in range(k):
            centers[cluster_ind] = data[np.where(index == cluster_ind)].mean(axis=0)
        if np.linalg.norm(old_centers-centers) < 1:
            convergence = True
        # ...

        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def display_clustered_img(img, index, centers,k,label):
    for cluster_ind in range(k):
        indices = np.unravel_index(np.where(index==cluster_ind), (img.shape[0],img.shape[1]))
        img[indices] = centers[cluster_ind]

    display_image("{0} Image  k = {1}".format(label,k),img)

def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    data = img.reshape(img.shape[0]*img.shape[1],1)

    k = 2
    img_cpy = img.copy()
    index,centers = myKmeans(data,k)
    display_clustered_img(img_cpy,index,centers,k,"Intensity")

    k = 4
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    display_clustered_img(img_cpy, index, centers, k, "Intensity")

    k = 6
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    display_clustered_img(img_cpy, index, centers, k, "Intensity")



def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    data = img.reshape(img.shape[0]*img.shape[1],3)

    k = 2
    img_cpy = img.copy()
    index,centers = myKmeans(data,k)
    display_clustered_img(img_cpy,index,centers,k,"Color")

    k = 4
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    display_clustered_img(img_cpy, index, centers, k, "Color")

    k = 6
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    display_clustered_img(img_cpy, index, centers, k, "Color")


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = np.linspace(0,255,img.shape[0])
    cols = np.linspace(0,255,img.shape[1])
    xx, yy = np.meshgrid(cols,rows)
    data = np.vstack((img.flatten(),xx.flatten(),yy.flatten())).T

    k = 2
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    centers = centers[:, 0]
    display_clustered_img(img_cpy, index, centers, k, "Intensity with position")

    k = 4
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    centers = centers[:, 0]
    display_clustered_img(img_cpy, index, centers, k, "Intensity with position")

    k = 6
    img_cpy = img.copy()
    index, centers = myKmeans(data, k)
    centers = centers[:,0]
    display_clustered_img(img_cpy, index, centers, k, "Intensity with position")


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    start_vertex = 'A'
    W = np.array([[0., 1., 0.2, 1., 0., 0., 0., 0.],
                  [1., 0., 0.1, 0., 1., 0., 0., 0.],
                  [0.2, 0.1, 0., 1., 0., 1., 0.3, 0.],
                  [1., 0., 1., 0., 0., 1., 0., 0.],
                  [0., 1., 0., 0., 0., 0., 1., 1.],
                  [0., 0., 1., 1., 0., 0., 1., 0.],
                  [0., 0., 0.3, 0., 1., 1., 0., 1.],
                  [0., 0., 0., 0., 1., 0., 1., 0.]])  # construct the W matrix
    d = W.sum(axis=1)
    D =  np.diag(d)
    L = D - W
    D_sqrt = np.diag(np.power(d, -1./2.))
    eigenEquation = np.dot(D_sqrt, np.dot(L, D_sqrt))
    lambdas, Z = cv.eigen(eigenEquation)[1:]
    eigen_value = np.amin(lambdas[lambdas != np.amin(lambdas)])
    print('Second smallest eigen value: {}'.format(eigen_value))
    y = Z[np.where(lambdas==eigen_value)[0]].T
    y = np.dot(D_sqrt, y)
    print('Corresponding eigen vector: {}'.format(y))
    min_N_cut = np.dot(y.T, np.dot(L, y))/np.dot(y.T, np.dot(D, y))
    print('Minimum NCut: {}'.format(min_N_cut[0][0]))
    c1 = np.where(y > 0)[0]
    c2 = np.where(y < 0)[0]
    cluster1, cluster2 = [], []
    for i in range(len(c1)):
        cluster1.append(chr(ord(start_vertex) + c1[i]))
        cluster2.append(chr(ord(start_vertex) + c2[i]))
    print('Vertices in cluster - 1: {}'.format(cluster1))
    print('Vertices in cluster - 2: {}'.format(cluster2))


##############################################
##############################################
##############################################

if __name__ == "__main__":
    #task_1_a()
    #task_1_b()
    task_2()
    #task_3_a()
    #task_3_b()
    #task_3_c()
    #task_4_a()

