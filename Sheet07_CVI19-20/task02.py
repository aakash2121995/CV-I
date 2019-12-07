import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from task01 import ICP, plot_landmarks

def pca(centered_data, threshold=0.1):
    wwt = centered_data @ centered_data.T
    U, L2, Vt = np.linalg.svd(wwt)
    L2_ratio = np.cumsum(L2) / np.sum(L2)
    compared = L2_ratio[L2_ratio <= 1 - threshold]
    return U, L2, compared.size

def plot_shape(points, title):
    fig = plt.figure(figsize=(16, 8))
    x = points[:int(points.shape[0] / 2)]
    y = points[int(points.shape[0] / 2):]
    plt.plot(x, y, '-o', color='red', zorder=2)
    plt.title(title)
    plt.show()

training_data = np.loadtxt('data/hands_aligned_train.txt.new', skiprows=1)
weights = np.array([(-0.4, -0.2, 0.0, 0.2, 0.4)])
mu = np.mean(training_data, axis=1)
plot_shape(mu, 'Mean Shape')
centered_data = training_data - mu.reshape(112, 1)
U, L2, K = pca(centered_data)
LK = np.diag(L2[:K])
UK = U[:K, :K]
variance_hat = np.sum(L2[K:])/(L2.shape[0] - K)
I = np.eye(LK.shape[0])
phi_hat = UK @ np.sqrt(np.subtract(LK, variance_hat*I))
prev_weights = weights
for i in range(K):
    weights_str = ",".join(map(str, prev_weights.tolist()))
    hphi = phi_hat * prev_weights
    w = mu + np.sum(hphi)
    plot_shape(w, 'K = {}, weights = {}'.format(i+1, weights_str))
    prev_weights = np.roll(prev_weights, 1)

