import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('data/hand.jpg', 0)


def get_pixel_locations(landmarks):
    W = landmarks.T
    return W[0], W[1]

def get_distance_transformed():
    '''
    Calculate the distance transformation of the image
    :return: distance transformation of the image
    '''
    edges = cv2.Canny(image, 40, 80)
    edges[np.where(edges == 255)] = 1.0
    edges[np.where(edges == 0)] = 255.0
    edges[np.where(edges == 1)] = 0.0
    distance_transformed = cv2.distanceTransform(edges, cv2.DIST_L2, 3).astype(np.uint8)
    return distance_transformed.astype(np.float32)

def get_transformation(landmarks, image_points):
    '''
    Try to find an affine transformation matrix that maps landmarks to the image_points
    :param landmarks: initial guess of the shape
    :param image_points:
    :return:
    '''
    dim = landmarks.shape[0]
    I = np.vstack([np.diag([1., 1.])]*dim)
    xn = np.array([image_points.T[1], image_points.T[0]]).T.flatten()
    first = np.zeros((landmarks.shape[0]*2, landmarks.shape[1]))
    first[::2] = np.fliplr(landmarks)
    second = np.vstack([[0., 0.], first])[:-1]
    points = np.hstack([first, second, I])
    psi = np.dot(np.linalg.pinv(points), xn)
    correspondence = np.reshape(points @ psi, (int(xn.shape[0] / 2), 2))
    new_landmarks = np.column_stack([correspondence.T[1], correspondence.T[0]]).astype(int)
    return psi, new_landmarks

def plot_landmarks(landmarks, title, x=None):
    '''
    Plot the landmarks on the image
    :param landmarks: approximation of the points on the shape
    :param title: Title of the plot
    :param x: Image points if available
    :return:
    '''
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    if x is not None:
        plt.scatter(x[:, 1], x[:, 0], c='r', s=10)
    plt.scatter(landmarks[:, 1], landmarks[:, 0], c='y', s=10)
    plt.grid()
    plt.title(title)
    plt.show()

def ICP(W, epsilon=0.001):
    '''
    Try to map the W to the closest point on the edge of the shape
    :param D: distance Transformed image
    :param W: initial landmarks points
    :param epsilon: covergence threshold
    :return: the final landmark points closest to the shape
    '''
    D = get_distance_transformed()
    G_y, G_x = np.gradient(D)
    delta = False
    wn_old = W
    count = 0
    psi_old = None
    while not delta:
        X, Y = get_pixel_locations(wn_old)
        Gx = G_x[X, Y]
        Gy = G_y[X, Y]
        gradient = np.column_stack([Gy, Gx])
        Dn = D[X, Y]
        denominator = np.hypot(Gy, Gx).reshape(-1, 1)
        numerator = Dn.reshape(-1, 1) * gradient
        xn_new = wn_old - np.divide(numerator, denominator, where=denominator != 0)
        psi, wn_new = get_transformation(wn_old, xn_new.astype('int'))
        plot_landmarks(wn_new.astype('int'), 'Iteration: {}'.format(count+1), xn_new.astype('int'))
        count += 1
        wn_old = wn_new.astype('int')
        delta = np.array_equal(psi, psi_old)
        psi_old = psi
    return wn_old


with open("data/hand_landmarks.txt", "r") as file:
    raw_data = [tuple(map(int, line[1:-2].split(','))) for line in file]
landmarks_data = np.array(raw_data).T
landmarks = np.column_stack([landmarks_data[1], landmarks_data[0]])
new_landmarks = ICP(landmarks)

