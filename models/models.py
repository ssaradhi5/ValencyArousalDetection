import os
import math
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

dirname = os.path.dirname(__file__)


def knn_model(dataset):

    # features, labels = construct_dataframe(dataset).iloc[:,0:63].values, construct_dataframe(dataset).iloc[:,63].values
    features, labels = dataset.iloc[:,0:63].values, dataset.iloc[:,63].values

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(feature_train, label_train)
    # model.kneighbors_graph(features, n_neighbors=3,mode='connectivity')
    model_location = dirname + r"\knn.pkl"
    joblib.dump(knn, model_location)
    trained_model = joblib.load(model_location, mmap_mode='r')
    predictions = trained_model.predict(feature_test)
    mse = mean_squared_error(label_test, predictions)
    rmse = math.sqrt(mse)
    log_loss(label_test, predictions)
    print("KNN accuracy: ", accuracy_score(label_test, predictions))
    print('KNN rmse', rmse)
    return

def rf_model(dataset):
    # features = dataset[:, 0:63]
    # labels = dataset[:, 63]

    features, labels = dataset.iloc[:,0:63].values, dataset.iloc[:,63].values

    feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(feature_train, label_train)
    model_location = dirname + r"\rf.pkl"
    # joblib.dump(rf, model_location)
    trained_model = joblib.load(model_location, mmap_mode='r')
    predictions = trained_model.predict(feature_test)
    mse = mean_squared_error(label_test, predictions)
    rmse = math.sqrt(mse)
    print("RF accuracy: ", accuracy_score(label_test, predictions))
    print('RF rmse', rmse)
    return