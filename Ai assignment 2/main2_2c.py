import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def testAndTrainVariables():
    iris = load_iris()
    splitData = train_test_split(iris.data, iris.target, test_size=0.25, train_size=0.75, random_state=0, shuffle=True,
                                 stratify=iris.target)
    xTrain = splitData[0]
    xTest = splitData[1]
    yTrain = splitData[2]
    yTest = splitData[3]
    return xTrain, xTest, yTrain, yTest


def get_logistic_regression_matrix(xTrain, xTest, yTrain, yTest):
    LR = LogisticRegression()
    LR.fit(xTrain, yTrain)

    yPrediction = LR.predict(xTest)

    logisticRegressionMatrix = confusion_matrix(yTest, yPrediction)

    return logisticRegressionMatrix


def get_knn_uniform_matrix(xTrain, xTest, yTrain, yTest):
    knnUniform = KNeighborsClassifier(n_neighbors=60, weights='uniform')
    knnUniform.fit(xTrain, yTrain)
    yPrediction = knnUniform.predict(xTest)
    knnUniformMatrix = confusion_matrix(yTest, yPrediction)
    return knnUniformMatrix


def get_knn_distance_matrix(xTrain, xTest, yTrain, yTest):
    knnDistance = KNeighborsClassifier(n_neighbors=60, weights='distance')
    knnDistance.fit(xTrain, yTrain)
    yPrediction = knnDistance.predict(xTest)
    knnDistanceMatrix = confusion_matrix(yTest, yPrediction)
    return knnDistanceMatrix


if __name__ == '__main__':
    xTrain, xTest, yTrain, yTest = testAndTrainVariables()

    print(get_logistic_regression_matrix(xTrain, xTest, yTrain, yTest))

    print(get_knn_uniform_matrix(xTrain, xTest, yTrain, yTest))

    print(get_knn_distance_matrix(xTrain, xTest, yTrain, yTest))


