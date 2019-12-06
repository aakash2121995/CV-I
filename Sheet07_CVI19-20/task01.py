import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

dx = np.array([[-1 / 2, 0, 1 / 2]], 'float64')

def get_exponent(x, y, sigma):
    return -1 * (x * x + y * y) / (2 * sigma)

def get_derivative_of_gaussian_kernel(size, sigma):
    assert size > 0 and size % 2 == 1 and sigma > 0

    kernel_x = np.zeros((size, size))
    kernel_y = np.zeros((size, size))

    size_half = size // 2

    for i in range(size):
        y = i - size_half
        for j in range(size):
            x = j - size_half
            kernel_x[i, j] = (
                -1
                * (x / (2 * np.pi * sigma * sigma))
                * np.exp(get_exponent(x, y, sigma))
            )
            kernel_y[i, j] = (
                -1
                * (y / (2 * np.pi * sigma * sigma))
                * np.exp(get_exponent(x, y, sigma))
            )

    return kernel_x, kernel_y

def get_transformation(landmarks, image_points):
    dim = landmarks.shape[0]
    I = np.vstack([np.diag([1., 1.])]*dim)
    xn = np.expand_dims(image_points, axis=-1).reshape(image_points.shape[0]*2, 1)
    #print(xn)
    first = np.zeros((landmarks.shape[0]*2, landmarks.shape[1]))
    first[::2] = landmarks
    second = np.vstack([[0., 0.], first])[:-1]
    points = np.hstack([first, second, I])
    #print(points.shape)
    psi = np.linalg.pinv(points) @ xn
    return psi, points @ psi

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
print(image.shape)
#plot_landmarks(image, landmarks)
image_blurred = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=0.0)
edges = cv2.Canny(image_blurred, 100, 250, 3)
plt.imshow(edges, cmap="gray")
plt.show()
edges = np.float64(edges) / 255.0
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
G_x = cv2.filter2D(distance_transformed, -1, dx)[Y, X]
G_y = cv2.filter2D(distance_transformed, -1, dx.T)[Y, X]

delta = 1
denominator = np.hypot(G_x, G_y)[:, None]
Dn = distance_transformed[Y, X][:, None]
wn_old = landmarks.astype('float64')
gradient = np.column_stack([G_y, G_x])
count = 0
while delta > epsilon:
    print(count)
    numerator = Dn * gradient
    xn_new = wn_old - numerator/denominator
    print(xn_new)
    plot_landmarks(image, xn_new.astype('int'))
    psi, wn_new = get_transformation(wn_old, xn_new)
    wn_new = np.reshape(wn_new, (int(wn_new.shape[0]/2), wn_new.shape[1]*2))
    new_landmark = wn_new.astype('int')
    print(new_landmark)
    Dn = distance_transformed[new_landmark[:, 0], new_landmark[:, 1]][:, None]
    #plot_landmarks(image, new_landmark)
    count += 1
    delta = np.mean(wn_new - wn_old)
    wn_old = wn_new
