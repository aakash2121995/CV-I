import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75

def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32')/255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi.astype(np.float64)


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================

def get_derivative_of_gaussian_kernel(size=5, sigma=3):
    kernel = cv2.getGaussianKernel(size, sigma, )
    kernel_x = kernel * kernel.T
    kernel_y = kernel_x.copy()
    kernel_x_padded = cv2.copyMakeBorder(kernel_x[:, :-1].copy(), top=0, bottom=0, left=1, right=0,
                                         borderType=cv2.BORDER_REFLECT)
    kernel_y_padded = cv2.copyMakeBorder(kernel_y[:-1, :].copy(), top=1, bottom=0, left=0, right=0,
                                         borderType=cv2.BORDER_REFLECT)

    return kernel_x_padded - kernel_x, kernel_y_padded - kernel_y


def get_gradient_magnitude(image):
    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

    edges_x = cv2.filter2D(image, -1, kernel_x).astype(np.int32) # convolve with kernel_x
    edges_y = cv2.filter2D(image, -1, kernel_y).astype(np.int32)  # convolve with kernel_y

    magnitude = np.square(edges_x) + np.square(edges_y)
    return magnitude

def resolve_index(i, j, height, width):
    x, y = i, j
    if j == 0:
        y = j + 1
    if j >= width - 1:
        y = j - 1
    if i == 0:
        x = i + 1
    if i >= height - 1:
        x = i - 1
    return x, y

# FUNCTIONS
# ------------------------
# your implementation here
def meanCurvatureMotion(phi):
    phi_x = np.array(np.gradient(phi, axis=0), dtype=np.float64)
    phi_y = np.array(np.gradient(phi, axis=1), dtype=np.float64)
    epsilon = 1e-4
    phi_xx = np.array(np.gradient(phi_x, axis=0), dtype=np.float64)
    phi_yy = np.array(np.gradient(phi_y, axis=1), dtype=np.float64)
    phi_xy = np.array(np.gradient(phi_x, axis=1), dtype=np.float64)
    phi_x_sqr = np.square(phi_x)
    phi_y_sqr = np.square(phi_y)
    result = phi_xx*phi_y_sqr.astype(np.float64) + phi_yy*phi_x_sqr.astype(np.float64) - (2*phi_x*phi_y*phi_xy)
    return result / (phi_x_sqr.astype(np.float64) + phi_y_sqr.astype(np.float64) + epsilon)


def forwardDifference(phi, phi_y=None):
    diff_x = np.diff(phi, axis=0)
    phi_x_forward = phi.copy()
    phi_x_forward[1:] = diff_x
    phi_x_forward[:-1] -= phi[1:2]
    # print(phi_x_forward.shape)
    # phi_y_forward = phi - phi_y_forward
    if phi_y is None:
        phi_y_forward = phi.copy()
    else:
        phi_y_forward = phi_y.copy()
    diff_y = np.diff(phi, axis=1)
    phi_y_forward[:, 1:] = diff_y
    phi_y_forward[:, :1] -= phi[:, 1:2]
    return phi_x_forward, phi_y_forward

def backwardDifference(phi):
    phi_x = np.flipud(phi)
    phi_y = np.fliplr(phi)
    phi_x_backward, phi_y_backward = forwardDifference(phi_x, phi_y)
    return np.flipud(phi_x_backward), np.fliplr(phi_y_backward)

def frontPropagation(phi, w_x, w_y):
    phi_x_forward, phi_y_forward = forwardDifference(phi)
    phi_x_backward, phi_y_backward = backwardDifference(phi)
    dphi = np.maximum(w_x, 0)*phi_x_forward + np.minimum(w_x, 0)*phi_x_backward + np.maximum(w_y, 0)*phi_y_forward + np.minimum(w_y, 0)*phi_y_backward
    return dphi
# ------------------------



if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 10
    Im, phi = load_data()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    image_gradient = get_gradient_magnitude(Im).astype(np.float64)
    w = 1. / (np.sqrt(image_gradient) +  1.)
    w_x = np.gradient(w, axis=0)
    w_y = np.gradient(w, axis=1)
    tau = 1. / 4. * np.amax(w)
    # ------------------------

    for t in range(n_steps):

        # ------------------------
        # your implementation here
        dphi =  w * meanCurvatureMotion(phi) + frontPropagation(phi, w_x, w_y)
        phi +=  tau * dphi
        # ------------------------

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)
            #phi = contour
            #phi = contour
    #plt.show()


