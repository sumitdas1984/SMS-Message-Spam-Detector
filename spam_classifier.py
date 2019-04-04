from sklearn.externals import joblib
import time


class SpamClassifier:

    def __init__(self):
        vectorizer = open('vectorizer.pkl','rb')
        self.cv = joblib.load(vectorizer)
        NB_spam_model = open('NB_spam_model.pkl','rb')
        self.clf = joblib.load(NB_spam_model)

    def classify(self, text):
        data = [text]
        vect = self.cv.transform(data).toarray()
        my_prediction = self.clf.predict(vect)
        return my_prediction


if __name__=="__main__":

    start_time = time.time()
    sp = SpamClassifier()
    end_time = time.time()
    exe_time = end_time - start_time
    print("model loading time: %f seconds" % exe_time)

    text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    start_time = time.time()
    prediction = sp.classify(text)
    exe_time = end_time - start_time
    print(prediction)
    print("prediction time: %f seconds" % exe_time)
