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

# FUNCTIONS
# ------------------------
# your implementation here
def get_phi_xx(phi):
    filter_xx = np.array([1,-2,1]).reshape(1,-1)
    return cv2.filter2D(phi,-1,filter_xx)
def get_phi_yy(phi):
    filter_yy = np.array([1, -2, 1])
    return cv2.filter2D(phi, -1, filter_yy)
def get_phi_xy(phi):
    filter_xy = np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    return 0.25*cv2.filter2D(phi,-1,filter_xy)

def meanCurvatureMotion(phi):
    phi_x = np.gradient(phi)[1]
    phi_y = np.gradient(phi)[0]
    epsilon = 1e-4
    phi_xx = get_phi_xx(phi)
    phi_yy = get_phi_yy(phi)
    phi_xy = get_phi_xy(phi)
    result = phi_xx*phi_y**2 + phi_yy*phi_x**2 - (2*phi_x*phi_y*phi_xy)
    return result / (phi_x**2 + phi_y**2 + epsilon)


def forwardDifference(phi, phi_y=None):
    phi_x_forward = phi.copy()
    if phi_y is None:
        phi_y_forward = phi.copy()
    else:
        phi_y_forward = phi_y.copy()
    diff_x = np.diff(phi, axis=1)
    diff_y = np.diff(phi_y_forward, axis=0)
    phi_x_forward[:, :-1] = diff_x
    phi_x_forward[:, -1] = 0
    phi_y_forward[1:] = diff_y
    phi_y_forward[:1] = 0
    return phi_x_forward, phi_y_forward

def backwardDifference(phi):
    phi_x = np.flipud(phi)
    phi_y = np.fliplr(phi)
    phi_x_backward, phi_y_backward = forwardDifference(phi_x, phi_y)
    return np.flipud(phi_x_backward), np.fliplr(phi_y_backward)

def frontPropagation(phi, w_x, w_y):
    phi_x_forward, phi_y_forward = forwardDifference(phi)
    phi_x_backward, phi_y_backward = backwardDifference(phi)
    dphi = np.max((w_x,np.zeros_like(w_x)),axis=0)*phi_x_forward + np.min((w_x,np.zeros_like(w_x)),axis=0)*phi_x_backward \
           + np.max((w_y,np.zeros_like(w_y)),axis=0)*phi_y_forward + np.max((w_y,np.zeros_like(w_y)),axis=0)*phi_y_backward
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
    image_gradient = np.hypot(np.gradient(Im)[1], np.gradient(Im)[0])
    w = 1. / (image_gradient +  1.)
    w_x = np.gradient(w)[1]
    w_y = np.gradient(w)[0]
    tau = 1. / 4. * np.max(w)
    # ------------------------

    for t in range(n_steps):

        # ------------------------
        # your implementation here
        dphi = w * meanCurvatureMotion(phi) + frontPropagation(phi, w_x, w_y)
        phi = phi + tau * dphi
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
    plt.show()
