import cv2
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

NUM_IMAGES=14
NUM_Boards = NUM_IMAGES
image_prefix = "../images/"
image_suffix = ".png"
images_files_list = [osp.join(image_prefix, f) for f in os.listdir(image_prefix)
                     if osp.isfile(osp.join(image_prefix, f)) and f.endswith(image_suffix)]
board_w = 10
board_h = 7
board_size = (board_w, board_h)
board_n = board_w * board_h
img_shape = (0,0)
obj = []
for ptIdx in range(0, board_n):
    obj.append(np.array([[ptIdx/board_w, ptIdx%board_w, 0.0]],np.float32))
obj = np.vstack(obj)

def display_images(title, images):
    cv2.imshow(title, images)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def show_images(images, cols=7, rows=2):
    fig = plt.figure(figsize=(50, 50))
    for i in range(NUM_IMAGES):
        a = fig.add_subplot(cols, np.ceil(NUM_IMAGES / float(cols)), i + 1)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    #fig.set_size_inches(np.array(fig.get_size_inches()) * NUM_IMAGES)
    plt.show()

def task1(images):
    #implement your solution
    # termination criteria
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objectPoints, imagePoints = [], []
    final_images = []
    for img in images:
        image = img.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(image_gray, board_size, None)
        # If found, add object points, image points (after refining them)
        if retval:
            objectPoints.append(obj)
            # Refining corners position with sub-pixels based algorithm
            subpix_corners = cv2.cornerSubPix(image_gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imagePoints.append(subpix_corners)
            final_image = cv2.drawChessboardCorners(image, board_size, subpix_corners, retval)
            final_images.append(final_image)
            display = np.hstack((img, final_image))
            display_images('original --> With_corners', display)
        else:
            print('Chessboard not detected in image ')
    #show_images(final_images)
    return imagePoints, objectPoints

def task2(imagePoints, objectPoints, image_shape):
    _, cameraMatrix, distortions, rotation, translation \
        = cv2.calibrateCamera(objectPoints, imagePoints, image_shape[::-1], None, None)
    return cameraMatrix, distortions, rotation, translation

def task3(images, imagePoints, objectPoints, CM, D, rvecs, tvecs):
    projection_error = 0
    red = [0, 0, 255]
    green = [0, 255, 0]
    for i in range(len(images)):
        image = images[i].copy()
        newImgpoints, _ = cv2.projectPoints(objectPoints[i], rvecs[i], tvecs[i], CM, D)
        projection_error += cv2.norm(imagePoints[i], newImgpoints, cv2.NORM_L1) / len(newImgpoints)
        for j in range(newImgpoints.shape[0]):
            cv2.circle(image, tuple(imagePoints[i][j].reshape(-1)), 5, green)
            cv2.circle(image, tuple(newImgpoints[j].reshape(-1)), 5, red)
        display_images('With reprojected points', image)
    projection_error /= len(objectPoints)
    print('Re-projection error: {}'.format(projection_error))

def task4(images, CM, D):
    undistorted_images = []
    for image in images:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        undistorted = cv2.undistort(image_gray, CM, D, None, None)
        undistorted_images.append(undistorted)
        display = np.hstack((image_gray, undistorted))
        display_images('Distorted --> Undistorted', display)

def task5(CM, rvecs, tvecs):
    #implement your solution
    pass

def main():
    images = []
    #Showing images
    for img_file in images_files_list:
        print(img_file)
        img = cv2.imread(img_file)
        image_shape = img.shape[:2]
        images.append(img)
        #cv2.imshow("Task1", img)
        #cv2.waitKey(10)
    #show_images(images)
    imagePoints, objectPoints = task1(images) #Calling Task 1
    
    CM, D, rvecs, tvecs = task2(imagePoints, objectPoints, image_shape) #Calling Task 2

    print('Camera matrix:\n {}'.format(CM))
    print('distortion matrix:\n {}'.format(D))
    for i in range(len(rvecs)):
        print('for image - {}:'.format(i + 1))
        print('rotation vector:\n {}'.format(rvecs[i]))
        print('translation vector:\n {}'.format(tvecs[i]))

    task3(images, imagePoints, objectPoints, CM, D, rvecs, tvecs)  # Calling Task 3

    task4(images, CM, D) # Calling Task 4

    #task5(CM, rvecs, tvecs) # Calling Task 5
    
    print("FINISH!")

main()