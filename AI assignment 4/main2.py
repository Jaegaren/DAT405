import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # order the data into 4 files; hamtrain, spamtrain, hamtest, spamtest
    hamtrain = [f for f in os.listdir('easy_ham') if os.path.isfile(os.path.join('easy_ham', f))]
    spamtrain = [f for f in os.listdir('spam') if os.path.isfile(os.path.join('spam', f))]
    hamtest = [f for f in os.listdir('hard_ham') if os.path.isfile(os.path.join('hard_ham', f))]
    spamtest = [f for f in os.listdir('spam') if os.path.isfile(os.path.join('spam', f))]

    ham_data = [file for file in os.listdir("/content/easy_ham")]
    spam_data = [file for file in os.listdir("/content/spam")]

    # read all the data in the whole directory
    data_hamtrain = []
    for f in hamtrain:
        with open('easy_ham/' + f, encoding='utf-8', errors="replace") as file:
            data_hamtrain.append(file.read())
    data_spamtrain = []
    for f in spamtrain:
        with open('spam/' + f, encoding='utf-8', errors="replace") as file:
            data_spamtrain.append(file.read())
    data_hamtest = []
    for f in hamtest:
        with open('hard_ham/' + f, encoding='utf-8', errors="replace") as file:
            data_hamtest.append(file.read())
    data_spamtest = []
    for f in spamtest:
        with open('spam/' + f, encoding='utf-8', errors="replace") as file:
            data_spamtest.append(file.read())


    def createDataFrame(data, dir, label):
        text_data = []
        for elem in data:
            with open(dir+elem, errors="replace") as f:
                text_data.append((f.read(), label))
        df = pd.DataFrame(text_data, columns=["mail_contents", "label"])
        return df


    # create dataframe
    data_hamtrain = pd.DataFrame(data_hamtrain, columns=['text'])
    data_hamtrain['target'] = 0
    data_spamtrain = pd.DataFrame(data_spamtrain, columns=['text'])
    data_spamtrain['target'] = 1
    data_hamtest = pd.DataFrame(data_hamtest, columns=['text'])
    data_hamtest['target'] = 0
    data_spamtest = pd.DataFrame(data_spamtest, columns=['text'])
    data_spamtest['target'] = 1


    ham_dataframe = createDataFrame(ham_data, "/content/easy_ham", 0)
    spam_dataframe = createDataFrame(spam_data, "/content/spam", 1)


    # put together dataframes
    data = pd.concat([data_hamtrain, data_spamtrain, data_hamtest, data_spamtest], ignore_index=True)

    # transform data with countvectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(data['text'])
    y = data['target']

    # split data
    X_train = X[0:len(data_hamtrain) + len(data_spamtrain)]
    X_test = X[len(data_hamtrain) + len(data_spamtrain):]
    y_train = y[0:len(data_hamtrain) + len(data_spamtrain)]
    y_test = y[len(data_hamtrain) + len(data_spamtrain):]

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)

    # predict
    mnb_pred = mnb.predict(X_test)
    bnb_pred = bnb.predict(X_test)

    # evaluate the accuracy
    mnb_true_pos = (mnb_pred == y_test).sum() / len(y_test)
    mnb_false_neg = (mnb_pred != y_test).sum() / len(y_test)
    bnb_true_pos = (bnb_pred == y_test).sum() / len(y_test)
    bnb_false_neg = (bnb_pred != y_test).sum() / len(y_test)

    # print results
    print('Multinomial Naive Bayes True Positive Rate: ', mnb_true_pos)
    print('Multinomial Naive Bayes False Negative Rate: ', mnb_false_neg)
    print('Bernoulli Naive Bayes True Positive Rate: ', bnb_true_pos)
    print('Bernoulli Naive Bayes False Negative Rate: ', bnb_false_neg)