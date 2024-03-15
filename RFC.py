import sys
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
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

def train_random_forest_classifier(train_img, train_labels, n_estimators=100, n_jobs=10, test_size=0.1):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(train_img, train_labels, test_size=test_size)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
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

def visualize_confusion_matrix(conf_mat, title):
    plt.matshow(conf_mat)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def visualize_test_images(test_img, test_labels, test_labels_pred, num_images=10):
    sample_indices = np.random.randint(0, len(test_img), num_images)
    for idx in sample_indices:
        two_d = (np.reshape(test_img[idx], (28, 28)) * 255).astype(np.uint8)
        plt.title('Original Label: {}  Predicted Label: {}'.format(test_labels[idx], test_labels_pred[idx]))
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        plt.show()

if __name__ == "__main__":
    data_path = './MNIST_Dataset_Loader/dataset/'
    train_img, train_labels, test_img, test_labels = load_mnist_data(data_path)
    clf, X_val, y_val = train_random_forest_classifier(train_img, train_labels)
    accuracy, conf_mat = evaluate_classifier(clf, X_val, y_val)
    print('Accuracy of trained classifier:', accuracy)
    print('Confusion matrix for validation data:\n', conf_mat)
    visualize_confusion_matrix(conf_mat, 'Confusion Matrix for Validation Data')

    save_classifier(clf, 'MNIST_RFC.pickle')
    clf_loaded = load_classifier('MNIST_RFC.pickle')

    acc_test, conf_mat_test = evaluate_classifier(clf_loaded, test_img, test_labels)
    print('Accuracy of classifier on test data:', acc_test)
    print('Confusion matrix for test data:\n', conf_mat_test)
    visualize_confusion_matrix(conf_mat_test, 'Confusion Matrix for Test Data')

    visualize_test_images(test_img, test_labels, clf_loaded.predict(test_img))
