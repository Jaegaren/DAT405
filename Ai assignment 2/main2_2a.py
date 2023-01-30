import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier



if __name__ == '__main__':
    iris = load_iris()
    xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, random_state=0)
    #xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target)

    logisticRegression = LogisticRegression()
    logisticRegression.fit(xTrain, yTrain)


    yPrediction = logisticRegression.predict(xTest)

    confusionMatrix = confusion_matrix(yTest, yPrediction)

    print(confusionMatrix)
    #conf_mat.plot()
    #plt.show()


    #actual = np.random.binomial(1, 0.9, size=1000)
    #predicted = np.random.binomial(1, 0.9, size=1000)
    #confusion_matrix = metrics.confusion_matrix(actual, predicted)

    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    #cm_display.plot()
    #plt.show()
