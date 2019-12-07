import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

image = cv2.imread('data/hand.jpg', 0)

def get_transformation(landmarks, image_points):
    '''
    Try to find an affine transformation matrix that maps landmarks to the image_points
    :param landmarks: initial guess of the shape
    :param image_points:
    :return:
    '''
    dim = landmarks.shape[0]
    I = np.vstack([np.diag([1., 1.])]*dim)
    xn = np.expand_dims(image_points, axis=-1).reshape(image_points.shape[0]*2, 1)
    first = np.zeros((landmarks.shape[0]*2, landmarks.shape[1]))
    first[::2] = landmarks
    second = np.vstack([[0., 0.], first])[:-1]
    points = np.hstack([first, second, I])
    inv_points = np.dot(np.linalg.inv(points.T @ points), points.T)
    psi = inv_points @ xn
    return psi, points @ psi

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
        plt.scatter(x[:, 0], x[:, 1], c='r', s=10)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='y', s=10)
    plt.grid()
    plt.title(title)
    plt.show()

def ICP(D, W, epsilon=0.1):
    '''
    Try to map the W to the closest point on the edge of the shape
    :param D: distance Transformed image
    :param W: initial landmarks points
    :param epsilon: covergence threshold
    :return: the final landmark points closest to the shape
    '''
    G = np.gradient(D)
    delta = 1
    wn_old = W
    count = 0
    while delta > epsilon:
        print(count)
        G_x = G[1][wn_old[:, 1], wn_old[:, 0]]
        G_y = G[0][wn_old[:, 1], wn_old[:, 0]]
        gradient = np.column_stack([G_y, G_x])
        Dn = D[wn_old[:, 1], wn_old[:, 0]]
        denominator = np.hypot(G_y, G_x)[:, None]
        multipliers = Dn[:, None] / denominator
        xn_new = wn_old - np.multiply(multipliers, gradient)
        psi, wn_new = get_transformation(wn_old, xn_new)
        wn_new = np.reshape(wn_new, (int(wn_new.shape[0] / 2), wn_new.shape[1] * 2))
        plot_landmarks(wn_new.astype('int'), 'Iteration: {}'.format(count+1), xn_new.astype('int'))
        count += 1
        delta = np.mean(Dn)
        wn_old = wn_new.astype('int')
    return wn_old

rgx = re.compile('[%s]' % "(,)")
with open("data/hand_landmarks.txt", "r") as f:
    lines = f.readlines()
X, Y = [], []
for line in lines:
    line = rgx.sub(' ', line).lstrip().rstrip().split()
    X.append(int(line[0]))
    Y.append(int(line[1]))
landmarks = np.column_stack([X, Y])
plot_landmarks(landmarks, 'Given Landmarks')


image_blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.0)
edges = cv2.Canny(image_blurred, 60, 150, L2gradient=True)
edges = np.float64(edges)
edges = 255.0 - edges  # this is because of the input of the cv2.distanceTransform.
distance_transformed = cv2.distanceTransform(
    src=np.uint8(edges),
    distanceType=cv2.DIST_L2,
    maskSize=cv2.DIST_MASK_PRECISE,
)
plt.imshow(distance_transformed, cmap="gray")
plt.show()
new_landmarks = ICP(distance_transformed, landmarks)

