import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import time


class SpamClassifierTraining:

    def train(self):
        df= pd.read_csv("spam.csv", encoding="latin-1")
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        # Features and Labels
        df['label'] = df['class'].map({'ham': 0, 'spam': 1})
        X = df['message']
        y = df['label']

        # Extract Feature With CountVectorizer
        cv = CountVectorizer()
        X = cv.fit_transform(X) # Fit the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        #Naive Bayes Classifier
        clf = MultinomialNB()
        clf.fit(X_train,y_train)
        clf.score(X_test,y_test)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Alternative Usage of Saved Model
        joblib.dump(cv, 'vectorizer.pkl')
        joblib.dump(clf, 'NB_spam_model.pkl')


if __name__=="__main__":

    # model training
    start_time = time.time()
    sp = SpamClassifierTraining()
    sp.train()
    end_time = time.time()
    exe_time = end_time - start_time
    print("model training time: %f seconds" % exe_time)

    # start_time = time.time()
    # sp = SpamClassifier()
    # end_time = time.time()
    # exe_time = end_time - start_time
    # print("total Model loading time: %f seconds" % exe_time)

