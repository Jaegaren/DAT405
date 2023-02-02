import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

def readHouseTable():
    df = pd.read_csv('../data2/data_assignment2.csv')
    return df

if __name__ == '__main__':
    soldResidents = readHouseTable()
    residents = dropOutliers(soldResidents, [40, 45, 9, 24])