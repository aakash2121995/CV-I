import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow

class OpticalFlow:
    def __init__(self):
        # Parameters for Lucas_Kanade_flow()
        self.EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade
        self.WINDOW_SIZE = [25, 25] # the number of points taken in the neighborhood of each pixel

        # Parameters for Horn_Schunck_flow()
        self.EPSILON= 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
        self.MAX_ITERS = 1000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
        self.ALPHA = 1.0 # smoothness term

        # Parameter for flow_map_to_bgr()
        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        return True

    #***********************************************************************************
    # function for converting flow map to to BGR image for visualisation
    # return bgr image
    def flow_map_to_bgr(self, flow):
        flow_img = np.zeros((self.next.shape[0], self.next.shape[1], 3), dtype=np.uint8)
        flow_img[..., 1] = 255
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        flow_img[..., 0] = angle /2
        flow_img[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        flow_bgr = cv.cvtColor(flow_img, cv.COLOR_HSV2BGR)
        return flow_bgr

    #***********************************************************************************
    # implement Lucas-Kanade Optical Flow 
    # returns the Optical flow based on the Lucas-Kanade algorithm and visualisation result
    def Lucas_Kanade_flow(self):
        kernel = np.ones((self.WINDOW_SIZE))
        rows, cols = self.next.shape
        flow = np.zeros((rows, cols, 2), dtype=np.float32)
        IxIx = cv.filter2D(np.square(self.Ix), -1, kernel)
        IyIy = cv.filter2D(np.square(self.Iy), -1, kernel)
        IxIy = cv.filter2D(np.multiply(self.Ix, self.Iy), -1, kernel)
        IxIt = cv.filter2D(np.multiply(self.Ix, self.It), -1, kernel)
        IyIt = cv.filter2D(np.multiply(self.Iy, self.It), -1, kernel)
        A = np.dstack((IxIx, IxIy, IxIy, IyIy))
        B = np.dstack((-IxIt, -IyIt))
        for row in range(rows):
            for col in range(cols):
                a = A[row, col].reshape((2, 2))
                b = B[row, col]
                if np.min(np.linalg.eigvals(a)) < self.EIGEN_THRESHOLD:
                    print('Invalid flow encountered at pixel ({}, {})'.format(row, col))
                    continue
                #flow[row, col, :] = np.dot(np.linalg.pinv(a), b)[0]
                flow[row, col, :] = np.linalg.lstsq(a, b, rcond=None)[0]
        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    # implement Horn-Schunck Optical Flow 
    # returns the Optical flow based on the Horn-Schunck algorithm and visualisation result
    def Horn_Schunck_flow(self):
        rows, cols = self.next.shape
        flow = np.zeros((rows, cols, 2), dtype=np.float32)
        u = np.zeros((rows, cols))
        v = np.zeros((rows, cols))
        kernel = np.array([[0., 1./4, 0.], [1./4, -1., 1./4], [0., 1./4, 0.]])
        iterator = 0
        delta = 1
        denominator = self.ALPHA**2 + np.square(self.Ix) + np.square(self.Iy)
        while delta > self.EPSILON and iterator < self.MAX_ITERS:
            delta_u = cv.filter2D(u, -1, kernel)
            delta_v = cv.filter2D(v, -1, kernel)
            u_hat = u + delta_u
            v_hat = v + delta_v
            numarator = np.multiply(self.Ix, u_hat) + np.multiply(self.Iy, v_hat) + self.It
            update = np.divide(numarator, denominator)
            u_old = u.copy()
            v_old = v.copy()
            u = u_hat - np.multiply(self.Ix, update)
            v = v_hat - np.multiply(self.Iy, update)
            delta = np.linalg.norm(u - u_old) + np.linalg.norm(v - v_old)
            iterator += 1
        flow[:, :, 0] = u
        flow[:, :, 1] = v
        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    #***********************************************************************************
    #calculate the angular error here
    # return average angular error and per point error map
    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        uc = groundtruth_flow[:, :, 0]
        vc = groundtruth_flow[:, :, 1]
        u = estimated_flow[:, :, 0]
        v = estimated_flow[:, :, 1]
        numarator = np.multiply(uc, u) + np.multiply(vc, v) + 1
        denominator = np.sqrt(np.multiply((uc**2 + vc**2 + 1), (u**2 + v**2 + 1)))
        aae_per_point = (180./np.pi)*np.arccos(np.divide(numarator, denominator))
        aae = np.sum(aae_per_point) / aae_per_point.size
        return aae, aae_per_point


if __name__ == "__main__":

    data_list = [
        'data/frame_0001.png',
        'data/frame_0002.png',
        'data/frame_0007.png',
    ]

    gt_list = [
        './data/frame_0001.flo',
        './data/frame_0002.flo',
        './data/frame_0007.flo',
    ]

    Op = OpticalFlow()
    
    for (i, (frame_filename, gt_filemane)) in enumerate(zip(data_list, gt_list)):
        groundtruth_flow = load_FLO_file(gt_filemane)
        img = cv.cvtColor(cv.imread(frame_filename), cv.COLOR_BGR2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        print('Average Angular error for Luacas-Kanade Optical Flow: %.4f' %(aae_lucas_kanade))
        #
        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow)
        print('Average Angular error for Horn-Schunck Optical Flow: %.4f' %(aae_horn_schunk))

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)
        fig = plt.figure()

        # Display
        fig.add_subplot(2, 3, 1)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 2)
        plt.imshow(flow_lucas_kanade_bgr)
        fig.add_subplot(2, 3, 3)
        plt.imshow(aae_lucas_kanade_per_point)
        fig.add_subplot(2, 3, 4)
        plt.imshow(flow_bgr_gt)
        fig.add_subplot(2, 3, 5)
        plt.imshow(flow_horn_schunck_bgr)
        fig.add_subplot(2, 3, 6)
        plt.imshow(aae_horn_schunk_per_point)
        plt.show()

        print("*"*20)
