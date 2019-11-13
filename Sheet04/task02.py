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


def getSeondDerivative(phi):
    height, width = phi.shape
    phi_xx = phi.copy()
    phi_yy = phi.copy()
    phi_xy = phi.copy()
    for j in range(width):
        y = j
        if j == 0:
            y = j + 1
        if j >= width - 1:
            y = j - 1
        for i in range(height):
            x = i
            if i == 0:
                x = i + 1
            if i >= height - 1:
                x = i - 1
            phi_xx[x, y] = phi_xx[x + 1, y] - 2 * phi_xx[x, y] + phi_xx[x - 1, y]
            phi_yy[x, y] = phi_yy[x, y + 1] - 2 * phi_yy[x, y] + phi_yy[x, y - 1]
            phi_xy[x, y] = (phi_xy[x + 1, y + 1] - phi_xy[x + 1, y - 1] - phi_xy[x - 1, y + 1] + phi_xy[
                x - 1, y - 1]) * 1. / 4.
    return phi_xx.astype(np.float64), phi_yy.astype(np.float64), phi_xy.astype(np.float64)


# FUNCTIONS
# ------------------------
# your implementation here
def meanCurvatureMotion(phi):
    phi_x = np.array(np.gradient(phi, axis=0), dtype=np.float64)
    phi_y = np.array(np.gradient(phi, axis=1), dtype=np.float64)
    epsilon = 1e-4
    phi_xx, phi_yy, phi_xy = getSeondDerivative(phi)
    phi_x_sqr = np.square(phi_x)
    phi_y_sqr = np.square(phi_y)
    result = phi_xx*phi_y_sqr.astype(np.float64) - 2*phi_x*phi_y*phi_xy + phi_yy*phi_x_sqr.astype(np.float64)
    return result / (phi_x_sqr.astype(np.float64) + phi_y_sqr.astype(np.float64) + epsilon)


def frontPropagation(phi, w_x, w_y):
    height, width = phi.shape
    dphi = np.zeros(phi.shape)
    for j in range(width):
        y = j
        if j == 0:
            y = j + 1
        if j >= width - 1:
            y = j - 1
        for i in range(height):
            x = i
            if i == 0:
                x = i + 1
            if i >= height - 1:
                x = i - 1
            dphi[x, y] = max(w_x, 0)*(phi[x+1, y] - phi[x, y]) + min(w_x, 0)*(phi[x, y] - phi[x - 1, y])\
                     + max(w_y, 0)*(phi[x, y+1] - phi[x, y]) + min(w_x, 0)*(phi[x, y] - phi[x, y - 1])
    return dphi
# ------------------------



if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 1
    #np.seterr(all='ignore')
    Im, phi = load_data()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here
    image_gradient = cv2.Laplacian(Im.astype(np.uint8), cv2.CV_64F)
    w = 1. / (np.abs(image_gradient) +  1.)
    w_x = np.gradient(w, axis=0)

    w_y = np.gradient(w, axis=1)
    eta = 1. / 4. * np.amax(w)
    # ------------------------

    for t in range(n_steps):

        # ------------------------
        # your implementation here

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
        dphi = eta * w * meanCurvatureMotion(phi) + frontPropagation(phi, np.amax(w_x), np.amax(w_y))
        phi += dphi
        plt.show()
