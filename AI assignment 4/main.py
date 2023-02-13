import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

if __name__ == '__main__':
    # Defining path to the data
    data_path = os.path.join('data', 'easy_ham', 'easy_ham.csv')

    # Loading the data
    df_ham = pd.read_csv('easy_ham/easy_ham.csv')
    df_spam = pd.read_csv('spam/spam.csv')

    # Split ham and spam data into train and test datasets
    ham_train, ham_test = np.split(df_ham, [int(.75 * len(df_ham))])
    spam_train, spam_test = np.split(df_spam, [int(.75 * len(df_spam))])

    vectorizer = CountVectorizer()
    X_train_ham = vectorizer.fit_transform(ham_train['message'])
    X_train_spam = vectorizer.fit_transform(spam_train['message'])
    X_test_ham = vectorizer.transform(ham_test['message'])
    X_test_spam = vectorizer.transform(spam_test['message'])

    mnb = MultinomialNB()
    bnb = BernoulliNB()
    mnb.fit(X_train_ham, ham_train['target'])
    bnb.fit(X_train_spam, spam_train['target'])

    mnb_pred_ham = mnb.predict(X_test_ham)
    bnb_pred_spam = bnb.predict(X_test_spam)

    mnb_true_positive = np.sum(mnb_pred_ham == ham_test['target'])
    mnb_false_negative = np.sum(mnb_pred_ham != ham_test['target'])
    bnb_true_positive = np.sum(bnb_pred_spam == spam_test['target'])
    bnb_false_negative = np.sum(bnb_pred_spam != spam_test['target'])

    print("Multinomial Naive Bayes - True Positive: {0}, False Negative: {1}".format(mnb_true_positive,
                                                                                     mnb_false_negative))
    print(
        "Bernoulli Naive Bayes - True Positive: {0}, False Negative: {1}".format(bnb_true_positive, bnb_false_negative))