#import necessary packages
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    # initialize data
    hamtrain_files = [f for f in os.listdir('easy_ham') if os.path.isfile(os.path.join('easy_ham', f))]
    spamtrain_files = [f for f in os.listdir('spam') if os.path.isfile(os.path.join('spam', f))]
    hamtest_files = [f for f in os.listdir('hard_ham') if os.path.isfile(os.path.join('hard_ham', f))]
    spamtest_files = [f for f in os.listdir('spam') if os.path.isfile(os.path.join('spam', f))]

    # read data
    data_hamtrain = []
    for f in hamtrain_files:
        with open('easy_ham/' + f, encoding='latin1') as file:
            data_hamtrain.append(file.read())
    data_spamtrain = []
    for f in spamtrain_files:
        with open('spam/' + f, encoding='latin1') as file:
            data_spamtrain.append(file.read())
    data_hamtest = []
    for f in hamtest_files:
        with open('hard_ham/' + f, encoding='latin1') as file:
            data_hamtest.append(file.read())
    data_spamtest = []
    for f in spamtest_files:
        with open('spam/' + f, encoding='latin1') as file:
            data_spamtest.append(file.read())

    # create dataframe
    data_hamtrain = pd.DataFrame(data_hamtrain, columns=['text'])
    data_hamtrain['target'] = 0
    data_spamtrain = pd.DataFrame(data_spamtrain, columns=['text'])
    data_spamtrain['target'] = 1
    data_hamtest = pd.DataFrame(data_hamtest, columns=['text'])
    data_hamtest['target'] = 0
    data_spamtest = pd.DataFrame(data_spamtest, columns=['text'])
    data_spamtest['target'] = 1

    # combine dataframes
    data = pd.concat([data_hamtrain, data_spamtrain, data_hamtest, data_spamtest], ignore_index=True)

    # transform data using countvectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(data['text'])
    y = data['target']

    # split data
    X_train = X[0:len(data_hamtrain) + len(data_spamtrain)]
    X_test = X[len(data_hamtrain) + len(data_spamtrain):]
    y_train = y[0:len(data_hamtrain) + len(data_spamtrain)]
    y_test = y[len(data_hamtrain) + len(data_spamtrain):]

    # train model
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    # predict
    mnb_pred = mnb.predict(X_test)
    bnb_pred = bnb.predict(X_test)

    # evaluate
    mnb_true_pos = (mnb_pred == y_test).sum() / len(y_test)
    mnb_false_neg = (mnb_pred != y_test).sum() / len(y_test)
    bnb_true_pos = (bnb_pred == y_test).sum() / len(y_test)
    bnb_false_neg = (bnb_pred != y_test).sum() / len(y_test)

    # print results
    print('Multinomial Naive Bayes True Positive Rate: ', mnb_true_pos)
    print('Multinomial Naive Bayes False Negative Rate: ', mnb_false_neg)
    print('Bernoulli Naive Bayes True Positive Rate: ', bnb_true_pos)
    print('Bernoulli Naive Bayes False Negative Rate: ', bnb_false_neg)