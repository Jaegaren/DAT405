import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ham_data = [f for f in os.listdir('easy_ham') if os.path.isfile(os.path.join('easy_ham', f))]
    spam_data = [f for f in os.listdir('spam') if os.path.isfile(os.path.join('spam', f))]
    hard_ham_data = [f for f in os.listdir('hard_ham') if os.path.isfile(os.path.join('hard_ham', f))]


    def createDataFrame(data, dir, label):
        text_data = []
        for elem in data:
            with open(os.path.join(dir, elem), errors="replace") as f:
                text_data.append((f.read(), label))
        df = pd.DataFrame(text_data, columns=["text", "label"])
        return df


    ham_dataframe = createDataFrame(ham_data, 'easy_ham/', 0)
    spam_dataframe = createDataFrame(spam_data, 'spam/', 1)
    hard_ham_dataframe = createDataFrame(hard_ham_data, "hard_ham/", 2)

    data = pd.concat([ham_dataframe, spam_dataframe], ignore_index=True)
    data2 = pd.concat([hard_ham_dataframe, spam_dataframe], ignore_index=True)

    cv = CountVectorizer()
    vectors = cv.fit_transform(data["text"])

    xTrain, xTest, yTrain, yTest = train_test_split(vectors, data["label"], test_size=0.25)
    # test
    mnb = MultinomialNB()
    mnb.fit(xTrain, yTrain)
    bnb = BernoulliNB()
    bnb.fit(xTrain, yTrain)

    # predict
    mnb_pred = mnb.predict(xTest)
    bnb_pred = bnb.predict(xTest)

    # evaluate the accuracy
    mnb_true_pos = (mnb_pred == yTest).sum() / len(yTest)
    mnb_false_neg = (mnb_pred != yTest).sum() / len(yTest)
    bnb_true_pos = (bnb_pred == yTest).sum() / len(yTest)
    bnb_false_neg = (bnb_pred != yTest).sum() / len(yTest)

    # print results
    print('Multinomial Naive Bayes True Positive Rate: ', mnb_true_pos)
    print('Multinomial Naive Bayes False Negative Rate: ', mnb_false_neg)
    print('Bernoulli Naive Bayes True Positive Rate: ', bnb_true_pos)
    print('Bernoulli Naive Bayes False Negative Rate: ', bnb_false_neg)
    print("----------------------------------------------------------")

    vectors2 = cv.fit_transform(data2["text"])

    xTrain, xTest, yTrain, yTest = train_test_split(vectors2, data2["label"], test_size=0.25)
    # test
    mnb = MultinomialNB()
    mnb.fit(xTrain, yTrain)
    bnb = BernoulliNB()
    bnb.fit(xTrain, yTrain)

    # predict
    mnb_pred = mnb.predict(xTest)
    bnb_pred = bnb.predict(xTest)

    # evaluate the accuracy
    mnb_true_pos = (mnb_pred == yTest).sum() / len(yTest)
    mnb_false_neg = (mnb_pred != yTest).sum() / len(yTest)
    bnb_true_pos = (bnb_pred == yTest).sum() / len(yTest)
    bnb_false_neg = (bnb_pred != yTest).sum() / len(yTest)

    # print results
    print('Multinomial Naive Bayes True Positive Rate: ', mnb_true_pos)
    print('Multinomial Naive Bayes False Negative Rate: ', mnb_false_neg)
    print('Bernoulli Naive Bayes True Positive Rate: ', bnb_true_pos)
    print('Bernoulli Naive Bayes False Negative Rate: ', bnb_false_neg)