# import necessary packages
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


def get_files(directory_name):
    """
    This function takes a directory name as an argument and returns the list of files in that directory.

    Parameters:
        directory_name (string): The name of the directory

    Returns:
        file_list (list): The list of files in the directory
    """
    file_list = [f for f in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, f))]
    return file_list


def read_files_to_dataframe(file_list, directory_name):
    """
    This function takes a list of files and a directory name as arguments and returns a pandas dataframe
    with the text from the files and the target (0 for ham and 1 for spam)

    Parameters:
        file_list (list): The list of files
        directory_name (string): The name of the directory

    Returns:
        data (pandas dataframe): The pandas dataframe with the text from the files and the target
    """
    data = []
    for f in file_list:
        with open(directory_name + '/' + f, encoding='latin1') as file:
            data.append(file.read())
    data = pd.DataFrame(data, columns=['text'])
    if directory_name == 'easy_ham' or directory_name == 'hard_ham':
        data['target'] = 0
    else:
        data['target'] = 1
    return data


def combine_dataframes(data_list):
    """
    This function takes a list of pandas dataframes as an argument and returns a single combined dataframe.

    Parameters:
        data_list (list): The list of pandas dataframes

    Returns:
        data (pandas dataframe): The combined dataframe
    """
    data = pd.concat(data_list, ignore_index=True)
    return data


def transform_data(data):
    """
    This function takes a pandas dataframe as an argument and returns a transformed pandas dataframe
    using CountVectorizer.

    Parameters:
        data (pandas dataframe): The dataframe

    Returns:
        X (array): The transformed array
        y (array): The target
    """
    cv = CountVectorizer()
    X = cv.fit_transform(data['text'])
    y = data['target']
    return X, y


def split_data(X, y):
    """
    This function takes two arrays as arguments and returns four arrays after splitting.

    Parameters:
        X (array): The transformed array
        y (array): The target

    Returns:
        X_train (array): The training array
        X_test (array): The testing array
        y_train (array): The training target
        y_test (array): The testing target
    """
    X_train = X[0:len(data_hamtrain) + len(data_spamtrain)]
    X_test = X[len(data_hamtrain) + len(data_spamtrain):]
    y_train = y[0:len(data_hamtrain) + len(data_spamtrain)]
    y_test = y[len(data_hamtrain) + len(data_spamtrain):]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    This function takes two arrays as arguments and returns two trained models.

    Parameters:
        X_train (array): The training array
        y_train (array): The training target

    Returns:
        mnb (model): The trained multinomial naive bayes model
        bnb (model): The trained bernoulli naive bayes model
    """
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    return mnb, bnb


def predict(X_test, mnb, bnb):
    """
    This function takes three arguments and returns two predictions.

    Parameters:
        X_test (array): The testing array
        mnb (model): The trained multinomial naive bayes model
        bnb (model): The trained bernoulli naive bayes model

    Returns:
        mnb_pred (array): The prediction from the multinomial naive bayes model
        bnb_pred (array): The prediction from the bernoulli naive bayes model
    """
    mnb_pred = mnb.predict(X_test)
    bnb_pred = bnb.predict(X_test)
    return mnb_pred, bnb_pred


def evaluate(mnb_pred, y_test, bnb_pred):
    """
    This function takes three arguments and returns four evaluation metrics.

    Parameters:
        mnb_pred (array): The prediction from the multinomial naive bayes model
        y_test (array): The testing target
        bnb_pred (array): The prediction from the bernoulli naive bayes model

    Returns:
        mnb_true_pos (float): The true positive rate for the multinomial naive bayes model
        mnb_false_neg (float): The false negative rate for the multinomial naive bayes model
        bnb_true_pos (float): The true positive rate for the bernoulli naive bayes model
        bnb_false_neg (float): The false negative rate for the bernoulli naive bayes model
    """
    mnb_true_pos = (mnb_pred == y_test).sum() / len(y_test)
    mnb_false_neg = (mnb_pred != y_test).sum() / len(y_test)
    bnb_true_pos = (bnb_pred == y_test).sum() / len(y_test)
    bnb_false_neg = (bnb_pred != y_test).sum() / len(y_test)
    return mnb_true_pos, mnb_false_neg, bnb_true_pos, bnb_false_neg


def print_results(mnb_true_pos, mnb_false_neg, bnb_true_pos, bnb_false_neg):
    """
    This function takes four arguments and prints the results.

    Parameters:
        mnb_true_pos (float): The true positive rate for the multinomial naive bayes model
        mnb_false_neg (float): The false negative rate for the multinomial naive bayes model
        bnb_true_pos (float): The true positive rate for the bernoulli naive bayes model
        bnb_false_neg (float): The false negative rate for the bernoulli naive bayes model
    """
    print('Multinomial Naive Bayes True Positive Rate: ', mnb_true_pos)
    print('Multinomial Naive Bayes False Negative Rate: ', mnb_false_neg)
    print('Bernoulli Naive Bayes True Positive Rate: ', bnb_true_pos)
    print('Bernoulli Naive Bayes False Negative Rate: ', bnb_false_neg)


if __name__ == '__main__':
    # initialize data
    hamtrain_files = get_files('easy_ham')
    spamtrain_files = get_files('spam')
    hamtest_files = get_files('hard_ham')
    spamtest_files = get_files('spam')

    # read data
    data_hamtrain = read_files_to_dataframe(hamtrain_files, 'easy_ham')
    data_spamtrain = read_files_to_dataframe(spamtrain_files, 'spam')
    data_hamtest = read_files_to_dataframe(hamtest_files, 'hard_ham')
    data_spamtest = read_files_to_dataframe(spamtest_files, 'spam')

    # combine dataframes
    data = combine_dataframes([data_hamtrain, data_spamtrain, data_hamtest, data_spamtest])

    # transform data using countvectorizer
    X, y = transform_data(data)

    # split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # train model
    mnb, bnb = train_model(X_train, y_train)

    # predict
    mnb_pred, bnb_pred = predict(X_test, mnb, bnb)

    # evaluate
    mnb_true_pos, mnb_false_neg, bnb_true_pos, bnb_false_neg = evaluate(mnb_pred, y_test, bnb_pred)

    # print results
    print_results(mnb_true_pos, mnb_false_neg, bnb_true_pos, bnb_false_neg)