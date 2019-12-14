import os
import cv2
import random
import numpy as np
from sklearn import neighbors
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def match_labels(labels, y_pred, y_test):
    '''

    :param labels: all labels
    :param y_pred: index of all predicted labels
    :param y_test: index of all original labels
    :return: list of [original labels   predicted labels
    '''
    def get_label(i):
        predicted = labels[y_pred[i]].rsplit(' ', 1)[-1]
        original = labels[y_test[i]].rsplit(' ', 1)[-1]
        return '{} \t {}'.format(original, predicted)

    matched = [get_label(i) for i in range(y_pred.shape[0])]
    return matched


def calculate_error(images, reconstructed):
    '''
    calculates recondtruction error
    :param images:
    :param reconstructed:
    :return: recondtruction error
    '''
    error = mean_squared_error(images.T, reconstructed.T, multioutput='raw_values')
    return np.sqrt(error)

def faceDetection(images, reconstructed_images, filenames, directory_name='faces', threshold=0.1):
    for i in range(images.shape[0]):
        isFace = False
        if isFaceDetected(images[i:i+1], reconstructed_images[i:i+1], threshold):
            print('Face detected in {} in {}'.format(filenames[i], directory_name))
        else:
            print('Face not detected in {} in {}'.format(filenames[i], directory_name))

def isFaceDetected(face, reconstructed_face, threshold=0.1):
    isFace = False
    loss = np.sqrt(mean_squared_error(face, reconstructed_face))
    if  loss < threshold:
        isFace = True
    return isFace

def reconstruct_image(images, pca):
    '''

    :param images:
    :param pca:
    :return: reconstructed image
    '''
    return pca.inverse_transform(pca.transform(images))

def plot_images(faces, h, w, rows=5, columns=2, title='Faces'):
    '''

    :param faces: collection of images
    :param h: height of an image
    :param w: width of an image
    :param rows: no of rows in the figure
    :param columns: no of columns in the figure
    :return:
    '''
    fig = plt.figure(figsize=(2 * columns, 2.5 * rows))
    plt.subplots_adjust(hspace=.35)
    fig.suptitle(title)
    for i in range(rows * columns):
        subplot = plt.subplot(rows, columns, i + 1)
        subplot.imshow(faces[i].reshape((h, w)), cmap='gray')
        subplot.set_title('Image - {}'.format(i+1))
    plt.xticks(())
    plt.yticks(())
    plt.show()

def readImageData(directory, h, w):
    images = []
    filenames = os.listdir(directory)
    for filename in filenames:
        img = cv2.resize(cv2.imread(os.path.join(directory, filename), 0), (w, h), interpolation=cv2.INTER_CUBIC)
        if img is not None:
            images.append(np.ravel(img))
    return filenames, np.vstack(images)

def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.data
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)


    # Compute the PCA
    k = 100                #no of principal components
    #calculating PCA
    pca = PCA(n_components=k, svd_solver='randomized').fit(X_train)

    # extracting eigen faces
    eigen_faces = pca.components_.reshape((k, h, w))
    # Visualize Eigen Faces (first 10)
    plot_images(eigen_faces[:10], h, w, title='Eigen Faces')

    # Compute reconstruction error for images in face and other
    face_names, faces = readImageData('data/exercise1/detect/face', h, w)
    plot_images(faces, h, w, 5, 1)
    other_names, others = readImageData('data/exercise1/detect/other', h, w)
    plot_images(others, h, w, 5, 1, title='Others')

    reconstructed_faces = reconstruct_image(faces, pca)
    plot_images(reconstructed_faces, h, w, 5, 1, title='Reconstructed Faces')
    face_reconstruction_error = calculate_error(faces, reconstructed_faces)
    print('Reconstruction error for face images: {}'.format(face_reconstruction_error))

    reconstructed_others = reconstruct_image(others, pca)
    plot_images(reconstructed_others, h, w, 5, 1, title='Reconstructed Others')
    other_reconstruction_error = calculate_error(others, reconstructed_others)
    print('Reconstruction error for other images: {}'.format(other_reconstruction_error))

    # Perform face detection
    threshold = np.mean([face_reconstruction_error.max(), other_reconstruction_error.min()])
    faceDetection(faces, reconstructed_faces, face_names, threshold=threshold)
    faceDetection(others, reconstructed_others, other_names, 'other', threshold=threshold)
    #
    #
    # # Perform face recognition
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    knn = knn.fit(X_train_reduced, y_train)
    y_pred = knn.predict(X_test_reduced)
    matched_labels = match_labels(lfw.target_names, y_pred, y_test)
    print('[Original Names \t Predicted Names]')
    for i in range(len(matched_labels)):
        print(matched_labels[i])
    print(classification_report(y_test, y_pred, target_names=lfw.target_names))

if __name__ == '__main__':
    main()
