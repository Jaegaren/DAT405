import pandas as pd
def readData():
    df = pd.read_csv('data3/data_assignment3.csv')
    return df


if __name__ == '__main__':
    print(readData())