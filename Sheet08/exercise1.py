import os
import cv2
import random
import sklearn
import numpy as np
from sklearn import neighbors
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#change mean square error to root_mean_square

def match_labels(labels, y_pred, y_test):
    def get_label(i):
        predicted = labels[y_pred[i]].rsplit(' ', 1)[-1]
        original = labels[y_test[i]].rsplit(' ', 1)[-1]
        return 'P: {} \n T: {}'.format(predicted, original)

    matched = [get_label(i) for i in range(y_pred.shape[0])]
    return matched


def faceDetection(images, reconstructed_images, filenames, directory_name='faces', threshold=0.1):
    for i in range(images.shape[0]):
        isFace = False
        if isFaceDetected(images[i:i+1], reconstructed_images[i:i+1], threshold):
            print('Face detected in {} in {}'.format(filenames[i], directory_name))

def isFaceDetected(face, reconstructed_face, threshold=0.1):
    isFace = False
    loss = ((face - reconstructed_face) ** 2).mean()
    if  np.sqrt(loss) < threshold:
        print(threshold, loss)
        isFace = True
    return isFace

def reconstruct_image(images, pca):
    mean_image = images.mean(axis=0)
    eigen_images = pca.components_
    print(images.shape)
    centered_images = images - mean_image
    #print(eigen_images.shape, images.shape)
    coefficients =  pca.score_samples(images)
    print(coefficients)
    reconstructed_images = mean_image + pca.inverse_transform(pca.transform(centered_images))
    #reconstructed_images = mean_image + pca.transform(images) @ eigen_images
    return reconstructed_images

def plot_faces(faces, h, w, rows=5, columns=2, titles='Faces'):
    '''

    :param faces: collection of face images
    :param h: height of an image
    :param w: width of an image
    :param rows: no of rows in the figure
    :param columns: no of columns in the figure
    :return:
    '''
    plt.figure(figsize=(1.8 * columns, 2.4 * rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * columns):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(faces[i].reshape((h, w)), cmap=plt.cm.gray)
        #plt.title('eigen face - {}'.format(i+1))
        plt.title(titles)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def readImageData(directory, h, w):
    images = []
    filenames = os.listdir(directory)
    for filename in filenames:
        img = cv2.resize(cv2.imread(os.path.join(directory, filename), 0), (w, h), interpolation=cv2.INTER_CUBIC)
        #img = cv2.imread(os.path.join(directory, filename))
        print(img.shape)
        if img is not None:
            images.append(np.ravel(img))
    return filenames, np.vstack(images)

def main():
    random.seed(0)
    np.random.seed(0)

    # Loading the LFW dataset
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    print(n_samples, h, w)
    X = lfw.data
    n_pixels = X.shape[1]
    y = lfw.target  # y is the id of the person in the image

    # splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    print(X_train.shape)

    #plot_faces(X_train, h, w)
    # Compute the PCA
    k = 100                #no of principal components
    #calculating PCA
    pca = PCA(n_components=k, svd_solver='randomized').fit(X_train)
    #plot_faces(pca.inverse_transform(pca.fit_transform(X_train - X_train.mean(axis=0))), h, w)
    plot_faces(X_test, h, w)
    plot_faces(pca.inverse_transform(pca.transform(X_test)), h, w)
    # extracting eigen faces
    eigen_faces = pca.components_.reshape((k, h, w))
    # Visualize Eigen Faces
    #plot_faces(eigen_faces, h, w)
    # Compute reconstruction error
    face_names, faces = readImageData('data/exercise1/detect/face', h, w)
    plot_faces(faces, h, w, 5, 1)
    other_names, others = readImageData('data/exercise1/detect/other', h, w)
    #plot_faces(others, h, w, 5, 1)
    reconstructed_test = pca.inverse_transform(pca.transform(X_train))
    plot_faces(reconstructed_test, h, w, 5, 1)
    error = loss = ((X_train - reconstructed_test) ** 2).mean()
    print(error)
    reconstructed_faces = pca.inverse_transform(pca.transform(faces))
    plot_faces(reconstructed_faces, h, w, 5, 1)
    #error = mean_squared_error(X_train.mean(axis=0) + faces, reconstructed_faces)
    face_reconstruction_error = mean_squared_error(faces, reconstructed_faces)
    print(face_reconstruction_error)

    reconstructed_others = pca.inverse_transform(pca.transform(others))
    plot_faces(reconstructed_others, h, w, 5, 1)
    other_reconstruction_error = mean_squared_error(others, reconstructed_others)
    print(other_reconstruction_error)

    threshold = np.mean([face_reconstruction_error, other_reconstruction_error])*2./3

    # Perform face detection
    faceDetection(faces, reconstructed_faces, face_names, threshold=threshold)
    faceDetection(others, reconstructed_others, other_names, 'other', threshold=threshold)


    # Perform face recognition
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    knn = knn.fit(X_train_pca, y_train)
    print("Predicting people's names on the test set")
    y_pred = knn.predict(X_test_pca)
    matched_labels = match_labels(lfw.target_names, y_pred, y_test)
    #print(matched_labels)
    #print(classification_report(y_test, y_pred, target_names=lfw.target_names))
    plot_faces(X_test, h, w)

if __name__ == '__main__':
    main()
