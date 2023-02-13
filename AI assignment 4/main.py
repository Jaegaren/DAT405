import os

def setUp():
    # Create train and test directories
    os.mkdir("train")
    os.mkdir("test")

    # Create ham and spam directories within train and test
    os.mkdir("train/ham")
    os.mkdir("train/spam")
    os.mkdir("test/ham")
    os.mkdir("test/spam")

if __name__ == '__main__':

    # Get file list from easy_ham
    filenames_ham = os.listdir("easy_ham")

    # Separate files into train and test
    for i in range(len(filenames_ham)):
        if i % 4 == 0:
            os.rename("easy_ham/" + filenames_ham[i],
                      "test/ham/" + filenames_ham[i])
        else:
            os.rename("easy_ham/" + filenames_ham[i],
                      "train/ham/" + filenames_ham[i])

    # Get file list from easy_spam
    filenames_spam = os.listdir("easy_spam")

    # Separate files into train and test
    for i in range(len(filenames_spam)):
        if i % 4 == 0:
            os.rename("easy_spam/" + filenames_spam[i],
                      "test/spam/" + filenames_spam[i])
        else:
            os.rename("easy_spam/" + filenames_spam[i],
                      "train/spam/" + filenames_spam[i])
