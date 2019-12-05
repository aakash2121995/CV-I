import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

def pca(centered_data, threshold=0.1):
    wwt = centered_data @ centered_data.T
    U, L2, Vt = np.linalg.svd(wwt)
    L2_ratio = np.cumsum(L2) / np.sum(L2)
    compared = L2_ratio[L2_ratio > 1 - threshold]
    return U, L2, 5

training_data = np.loadtxt('data/hands_aligned_train.txt.new', skiprows=1)
image = plt.imread('data/hand.jpg')
weights = np.array([(-0.4, -0.2, 0.0, 0.2, 0.4)])
mu = np.mean(training_data, axis=0)
centered_data = training_data - mu
#print(np.mean(centered_data, axis=0))
'''points = training_data
x = points[:int(points.shape[0] / 2)]
y = points[int(points.shape[0] / 2):]
plt.figure()
plt.imshow(image)
plt.scatter(x, y, c='r', s=40)
plt.show()'''


U, L2, K = pca(centered_data)
LK = np.diag(L2[:K])
UK = U[:, :K]
variance_hat = np.sum(L2[K:])/(L2.shape[0] - K)
I = np.eye(LK.shape[0])
phi_hat = UK @ np.sqrt(np.subtract(LK, variance_hat*I))
print(phi_hat[:K, :K].shape)
w = mu + phi_hat[:K, :K] @ weights.T
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
count = 0
for i in range(w.shape[1]):
    points = training_data[:, i]
    x = points[:int(points.shape[0]/2)]
    print(x.shape)
    y = points[int(points.shape[0]/2):]
    print(y.shape)
    ax1.clear()
    ax1.imshow(image, cmap='gray')
    ax1.scatter(x, y, color='red', s=3)
    plt.pause(0.01)
plt.show()
