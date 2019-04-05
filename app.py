from flask import Flask,render_template,url_for,request
from flask import jsonify
import spam_classifier
import json

# load the model from disk
sp = spam_classifier.SpamClassifier()

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
	sms_text = str(request.args.get('sms'))
	print('sms: '+sms_text)
	pred = sp.classify(str(sms_text))[0]
	output = {}
	if pred == 0:
		output['class'] = 'ham'
	else:
		output['class'] = 'spam'
	return jsonify(output)


if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=5001)
