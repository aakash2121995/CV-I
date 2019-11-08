import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0, :]).reshape((-1, 2))
    ax.plot(V_plt[:, 0], V_plt[:, 1], color=line, alpha=alpha)
    ax.scatter(V[:, 0], V[:, 1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    Im = cv2.imread(fpath, 0)
    h, w = Im.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return Im, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here
def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def get_all_states(node,max_row,max_col):
    one_d_states_x = np.linspace(node[0] - 1, node[0] + 1, 3, dtype=np.int32)
    one_d_states_y = np.linspace(node[1] - 1, node[1] + 1, 3, dtype=np.int32)
    k = np.meshgrid( one_d_states_x,one_d_states_y)
    k[0] = k[0].flatten()
    k[1] = k[1].flatten()

    return k


def snake_optimisation(V_trunc, gradient, alpha):
    all_transitions = []
    all_transition_indexes = []
    node = V_trunc[0]
    k = get_all_states(node,gradient.shape[0],gradient.shape[1])
    S_n = -gradient[k[1],k[0]].copy()

    for index in range(V_trunc.shape[0] - 1):
        next_node = V_trunc[index + 1]
        k_next = get_all_states(next_node, gradient.shape[0],gradient.shape[1])
        S_n_1 = -gradient[k_next[1],k_next[0]].copy()
        new_states_n = []
        state_indexes = []
        for next_state_index in range(k_next[0].shape[0]):
            diff_0 = k[0] - k_next[0][next_state_index]
            diff_1 = k[1] - k_next[1][next_state_index]
            P = alpha * (diff_0 ** 2 + diff_1 ** 2)
            S_n_plus_P = S_n + P
            edge_index = S_n_plus_P.argmin()
            S_n_1[next_state_index] += S_n_plus_P[edge_index]
            new_states_n.append((k[0][edge_index], k[1][edge_index]))
            state_indexes.append(edge_index)

        all_transitions.append(new_states_n)
        all_transition_indexes.append(state_indexes)
        S_n = S_n_1
        k = k_next
    edge_index = S_n.argmin()
    minimum_transitions = []
    last_state = (k[0][edge_index], k[1][edge_index])
    minimum_transitions.append(last_state)
    all_transition_indexes = all_transition_indexes[::-1]
    for index, transitions in enumerate(all_transitions[::-1]):
        edge_index = all_transition_indexes[index][edge_index]
        minimum_transitions.append(transitions[edge_index])

    for idx, node in enumerate(V_trunc[::-1]):
        node[0] = minimum_transitions[idx][0]
        node[1] = minimum_transitions[idx][1]


# ------------------------


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    Im, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200

    # ------------------------
    # your implementation here

    ALPHA = 0.1
    gradient = get_gradient_magnitude(Im).astype(np.float32)
    # display_image("gradient",gradient)
    V_trunc = V
    snake_optimisation(V_trunc,gradient,ALPHA)
    # ------------------------

    for t in range(n_steps):
        # print("V = ",V)
        # ------------------------
        # your implementation here
        random_pos = np.random.randint(V.shape[0])
        rolled_V = np.roll(V,random_pos,axis=0)
        V_trunc = rolled_V
        snake_optimisation(V_trunc,gradient,ALPHA)
        V = np.roll(rolled_V,-random_pos,axis=0)
        # ------------------------

        ax.clear()
        ax.imshow(Im, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
