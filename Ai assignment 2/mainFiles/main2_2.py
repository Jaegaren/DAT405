from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def firstExercise():
    x_train, x_test, y_train, y_test = returnTrainingData()
    flowerTypes = load_iris().target_names

    logisticRegress = LogisticRegression(max_iter=1000, random_state=0)
    logisticRegress.fit(x_train, y_train)

    predictYValues = logisticRegress.predict(x_test)
    cm = confusion_matrix(y_test, predictYValues, labels=logisticRegress.classes_)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=flowerTypes)
    cm_disp.plot()
    plt.show()


def kNearestNeighbour():
    x_train, x_test, y_train, y_test = returnTrainingData()
    logisticRegress = LogisticRegression(max_iter=1000, random_state=0)
    logisticRegress.fit(x_train, y_train)
    clf = KNeighborsRegressor(11)
    clf.fit(x_train, y_train)
    yPredict = clf.predict(x_test)
    print(1 - mean_squared_error(y_test, yPredict))



def returnTrainingData():
    X = load_iris().data
    y = load_iris().target
    print(X)
    print(y)
    return train_test_split(X, y, random_state=0)



if __name__ == '__main__':
    returnTrainingData()
