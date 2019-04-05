# SMS-Message-Spam-Detector
A simple Flask API to detect spam or ham using Python and sklearn

## My implementation of a Multinomial Naive Bayes Classifier
===========================================================

### Uses the Naive Bayes Classifier to predict whether a given text/document is either spam or not spam (ham).
1. Set training folder with spam/ham subfolders.
2. Set test folder with spam/ham subfolders.
3. In case you want to classify a separate file as spam/ham,
pass the file through the read_file function first, which returns
terms in the file as a list. (probably will change this soon)

### A short explanation:
given a message M, find P(C | M) where C = (spam OR ham)
Bayes Rule states that :
P(C | M)   =   (P(M | C) * P(C)) / (P(M))
(posterior)      (likelihood) (prior)   (evidence)
where P(M | C) can be stated as the product of P(m | C) for every term m in M.
The predicted class returned is the one that has maximum value for P(C | M).

### More explanation on creating the classification as service using Flask:
https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
