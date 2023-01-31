import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier



if __name__ == '__main__':
    iris = load_iris()

    splitData = train_test_split(iris.data, iris.target, test_size=0.25, train_size=0.75, random_state=0, shuffle=True,
                                 stratify=iris.target)
    xTrain = splitData[0]
    xTest = splitData[1]
    yTrain = splitData[2]
    yTest = splitData[3]

    logisticRegression = LogisticRegression()
    logisticRegression.fit(xTrain, yTrain)

    yPrediction = logisticRegression.predict(xTest)

    confusionMatrix = confusion_matrix(yTest, yPrediction)

    print(confusionMatrix)

    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=[0, 1, 2])

    cm_display.plot()
    plt.show()
