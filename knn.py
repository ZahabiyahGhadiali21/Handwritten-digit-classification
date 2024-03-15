import numpy as np
import pickle
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from MNIST_Dataset_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def load_mnist_data(data_path):
    data = MNIST(data_path)
    img_train, labels_train = data.load_training()
    img_test, labels_test = data.load_testing()
    return np.array(img_train), np.array(labels_train), np.array(img_test), np.array(labels_test)

def train_knn_classifier(train_img, train_labels, test_size=0.2, n_neighbors=5, n_jobs=10):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(train_img, train_labels, test_size=test_size)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto', n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    return clf, X_val, y_val

def evaluate_classifier(clf, X_val, y_val):
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    conf_mat = confusion_matrix(y_val, y_pred)
    return accuracy, conf_mat

def save_classifier(clf, filename):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

def load_classifier(filename):
    with open(filename, 'rb') as f:
        clf = pickle.load(f)
    return clf

def plot_confusion_matrix(conf_mat, title):
    plt.matshow(conf_mat)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def predict_and_evaluate_test_data(clf, test_img, test_labels):
    test_labels_pred = clf.predict(test_img)
    acc = accuracy_score(test_labels, test_labels_pred)
    conf_mat_test = confusion_matrix(test_labels, test_labels_pred)
    return acc, conf_mat_test

def visualize_test_images(test_img, test_labels, test_labels_pred, num_images=20):
    sample_indices = np.random.randint(0, len(test_img), num_images)
    for idx in sample_indices:
        plt.figure()
        plt.title('Original Label: {}  Predicted Label: {}'.format(test_labels[idx], test_labels_pred[idx]))
        plt.imshow(test_img[idx].reshape(28, 28), cmap='gray')
        plt.show()

if __name__ == "__main__":
    data_path = './MNIST_Dataset_Loader/dataset/'
    train_img, train_labels, test_img, test_labels = load_mnist_data(data_path)
    clf, X_val, y_val = train_knn_classifier(train_img, train_labels)
    accuracy, conf_mat = evaluate_classifier(clf, X_val, y_val)
    print('Accuracy of trained classifier:', accuracy)
    print('Confusion matrix for validation data:\n', conf_mat)
    plot_confusion_matrix(conf_mat, 'Confusion Matrix for Validation Data')

    save_classifier(clf, 'MNIST_KNN.pickle')
    clf_loaded = load_classifier('MNIST_KNN.pickle')

    acc_test, conf_mat_test = predict_and_evaluate_test_data(clf_loaded, test_img, test_labels)
    print('Accuracy of classifier on test data:', acc_test)
    print('Confusion matrix for test data:\n', conf_mat_test)
    plot_confusion_matrix(conf_mat_test, 'Confusion Matrix for Test Data')

    visualize_test_images(test_img, test_labels, clf_loaded.predict(test_img))
