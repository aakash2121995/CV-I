import numpy as np
from scipy.linalg import block_diag
import matplotlib.pylab as plt

observations = np.load('observations.npy')

dt = 0.5
I = np.diag([1., 1., 1., 1.])
O = np.diag([0., 0., 0., 0.])

def get_observation(t):
    return observations[t]

def get_lagged_psi(tau, psi, phi):
    _I = []
    psi_tau = np.hstack([psi, np.hstack([O]*tau)])
    phi_state = np.hstack([I, np.hstack([O]*tau)])
    O2 = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]])
    phi_tau = np.hstack([phi, np.hstack([O2] * tau)])
    for i in range(tau+1):
        _I.append(I)
    _I = np.array(_I).reshape(tau+1, 4, 4)
    I_tau = block_diag(*_I)
    return  np.vstack([psi_tau, I_tau])[:-4], phi_tau, phi_state

class KalmanFilter(object):
    def __init__(self, psi, sigma_p, phi, sigma_m, tau):
        self.psi = psi          #F
        self.sigma_p = sigma_p  #Q
        self.phi = phi          #H
        self.sigma_m = sigma_m  #R
        self.state = None
        self.covariance = None
        self.tau = tau
        if self.tau != 0:
            self.xSmooth = []
            self.lagged_psi, self.lagged_phi_x, self.lagged_phi_state = get_lagged_psi(self.tau, self.psi, self.phi)

    def init(self, init_state):
        self.state = init_state
        self.covariance = np.diag([100., 100., 100., 100.])
        self.time_step = 0
        self.iter_1 = 0
        self.iter_2 = 0

    def track(self, xt):
        self.state = np.dot(self.psi, self.state)
        self.covariance = np.dot(self.psi, np.dot(self.covariance, self.psi.T)) + self.sigma_p
        self.v = np.subtract(xt, np.dot(self.phi, self.state))
        self.S = np.dot(self.phi, np.dot(self.covariance, self.phi.T)) + self.sigma_m
        self.K = np.dot(self.covariance, np.dot(self.phi.T, np.linalg.inv(self.S)))
        self.state = np.add(self.state, np.dot(self.K, self.v))
        self.covariance = np.dot((I - np.dot(self.K, self.phi)), self.covariance)
        if self.tau != 0:
            self.xSmooth.append(self.state[:, None])

    def get_current_location(self):
        location = np.dot(self.phi, self.state)
        if self.time_step >= self.tau and self.tau!=0:
            states = self.xSmooth[-(self.tau+1):]
            if len(states) == self.tau+1:
                self.iter_1 += 1
                states = np.array(states[::-1]).reshape(24, 1)
                smoothed_state = np.dot(self.lagged_psi, states.reshape(24, 1))
                location = np.dot(self.lagged_phi_x, smoothed_state)
                state = np.dot(self.lagged_phi_state, smoothed_state)
                self.state = state.reshape(4, )
        self.time_step += 1
        return location

def perform_tracking(tracker):
    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())

    return track

def main():
    init_state = np.array([0, 1, 0, 0])

    psi = np.array([[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])

    phi = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])
    sm = 0.05       #sm is too small; smaller covariance => relaiable measurement
    sigma_m = np.array([[sm, 0],
                        [0, sm]])


    tracker = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=0)
    tracker.init(init_state)

    fixed_lag_smoother = KalmanFilter(psi, sigma_p, phi, sigma_m, tau=5)
    fixed_lag_smoother.init(init_state)

    track = perform_tracking(tracker)
    track_smoothed = perform_tracking(fixed_lag_smoother)
    plt.figure(figsize=(16, 8))
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.plot([x[0] for x in track_smoothed], [x[1] for x in track_smoothed])
    plt.legend(['Observations', 'tau = 0', 'tau = 5'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    main()
