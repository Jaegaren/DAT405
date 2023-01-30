from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay





if __name__ == '__main__':
    x = load_iris().data
    y = load_iris().target
    flowerTypes = load_iris().target_names

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    linearRegress = LogisticRegression(max_iter=1000, random_state=0)
    linearRegress.fit(x_train, y_train)

    predictYValues = linearRegress.predict(x_test)
    print(predictYValues)

    cm = confusion_matrix(y_test, predictYValues, labels=linearRegress.classes_)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=flowerTypes)
    cm_disp.plot()
    plt.show()

