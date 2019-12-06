import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def to_homogeneous(A):
    dim = A.shape[1]
    new_A = np.ones((dim+1,A.shape[0]))
    new_A[:dim, :] = np.copy(A.T)
    return new_A

def from_homogeneous(A):
    dim = A.shape[0] - 1
    return A[:dim,:].T

def translate_point(point, t):
    translation = np.array([[1., 0., t[0]], [0., 1., t[1]], [0., 0., 1]])
    new_point = translation @ point
    return new_point[:point.shape[0]]

def get_transformation(wn, xn):
    '''d = wn.shape[1]
    center_w = np.mean(wn, axis=0)
    center_x = np.mean(xn, axis=0)
    w = wn - center_w
    x = xn - center_x
    U, L2, VT = np.linalg.svd(w.T @ x)
    rotation = U @ VT
    translation = center_x.T - (rotation @ center_w.T)
    psi = np.identity(d+1)
    psi[:d, :d] = rotation
    psi[:d, d] = translation'''
      dim = landmarks.shape[0]/2
      I = np.vstack([np.diag([1., 1.])]*dim)
      first = np.zeros((landmarks.shape[0]*2, landmarks.shape[1]))
      first[::2] = landmarks
      second = np.vstack([[0., 0.], first])[:-1]
      points = np.hstack([first, second, I])
      psi = image_points / np.linalg.pinv(points)
      return psi



def plot_landmarks(image, landmarks):
    plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='y', s=10)
    plt.grid()
    plt.show()

rgx = re.compile('[%s]' % "(,)")
with open("data/hand_landmarks.txt", "r") as f:
    lines = f.readlines()
X, Y = [], []
for line in lines:
    line = rgx.sub(' ', line).lstrip().rstrip().split()
    X.append(int(line[0]))
    Y.append(int(line[-1]))
landmarks = np.column_stack([X, Y])
image = plt.imread('data/hand.jpg')
#plot_landmarks(image, landmarks)
image_blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.0)
edges = cv2.Canny(image_blurred, 100, 250, 3)
edges = np.float32(edges) / 255.0
edges[np.where(edges <= 0.7)] = 0.0
edges[np.where(edges > 0.7)] = 1.0
#print(edges)
edges = 1 - edges  # this is because of the input of the cv2.distanceTransform.
#print(edges)
distance_transformed = cv2.distanceTransform(
    src=np.uint8(edges * 255.0),
    distanceType=cv2.DIST_L2,
    maskSize=cv2.DIST_MASK_PRECISE,
)
#print(distance_transformed)
epsilon = 0.1
G = np.gradient(distance_transformed)
G_x = G[1][Y, X]
G_y = G[0][Y, X]
delta = 1
denominator = np.hypot(G_x, G_y)[:, None]
Dn = distance_transformed[Y, X][:, None]
wn_old = landmarks.astype('float64')
gradient = np.column_stack([G_x, G_y])
count = 0
while delta > epsilon:
    numerator = Dn * gradient
    xn_new = wn_old - numerator/denominator
    delta = np.amax(wn_old - xn_new)
    psi = get_transformation(wn_old, xn_new)
    wn_new = from_homogeneous(psi @ to_homogeneous(wn_old))
    new_landmark = wn_new.astype('int')
    Dn = distance_transformed[new_landmark[:, 1], new_landmark[:, 0]][:, None]
    wn_old = wn_new
    plot_landmarks(image, wn_new.astype(int))
    count += 1
    delta = np.mean(Dn)

