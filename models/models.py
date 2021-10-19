import os
import math
import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from src.constants import log
from matplotlib import pyplot as plt

dirname = os.path.dirname(__file__)

path = r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\results"
logger = log(path=path, file="ml_metrics.logs")

def knn_model(dataset):
    features, labels = dataset.iloc[:,0:63].values, dataset.iloc[:,63].values
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(feature_train, label_train)
    # model.kneighbors_graph(features, n_neighbors=3,mode='connectivity')
    model_location = dirname + r"\knn.pkl"
    joblib.dump(knn, model_location)
    trained_model = joblib.load(model_location, mmap_mode='r')
    predictions = trained_model.predict(feature_test)
    logger.info("------------ KNN Model ---------------")
    metrics(label_test, predictions, "KNN")
    return

def rf_model(dataset):
    features, labels = dataset.iloc[:,0:63].values, dataset.iloc[:,63].values
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(feature_train, label_train)
    model_location = dirname + r"\rf.pkl"
    # joblib.dump(rf, model_location)
    trained_model = joblib.load(model_location, mmap_mode='r')
    predictions = trained_model.predict(feature_test)
    logger.info("------------ Random Forest Model ---------------")
    metrics(label_test, predictions, "RF")
    return

def cap_curve(labels, predictions, model_title):
    # CAP curve

    total = len(labels)
    class_1_count = np.sum(labels)
    plt.figure(figsize=(20, 12))
    plt.plot([0, total], [0, class_1_count], c='r', linestyle='--', label='Random Model')
    plt.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c='grey', linewidth=2, label='Perfect Model')

    model_y = [y for _, y in sorted(zip(predictions, labels), reverse=True)]
    y_values = np.append([0], np.cumsum(model_y))
    x_values = np.arange(0, total + 1)
    plt.plot(x_values,
             y_values,
             c='b',
             label=model_title,
             linewidth=4)
    # Plot information
    plt.xlabel('Total observations', fontsize=16)
    plt.ylabel('Class 1 observations', fontsize=16)
    plt.title('Cumulative Accuracy Profile', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.show()
    return

def plot_roc(fpr, tpr, model_title):
    #  ROC curve
    plt.figure(figsize=(20, 12))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, c='g', label=model_title, linewidth=4)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.show()
    return

def metrics(labels, predictions, model_title):
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    rmse = math.sqrt(mean_squared_error(labels, predictions))
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, model_title)
    cap_curve(predictions, labels, model_title)

    logger.info("Accuracy: {:.3f}".format(accuracy))
    logger.info("F Score: {:.3f}".format(f1))
    logger.info("RMSE: {:.3f}".format(rmse))
    logger.info("AUC: {:.3f}".format(roc_auc))
    return

def k_selector(dataset):
    features, labels = dataset.iloc[:, 0:63].values, dataset.iloc[:, 63].values
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.3)
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(feature_train, label_train)
        pred_i = knn.predict(feature_test)
        error_rate.append(np.mean(pred_i != label_test))

    fig = plt.figure(figsize=(10, 4))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
             markersize=10)
    plt.title('Error Rate vs. K-Values')
    plt.xlabel('K-Values')
    plt.ylabel('Error Rate')
    plt.show()
    fig.savefig(r"C:\Users\srika\Desktop\Flosonics Backup\URA\Arithmetic\results\best_k_value.png")
    return