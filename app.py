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
    output = {}
    output['class'] = sp.classify(str(sms_text))
    d1 = json.dumps(output)
    output = json.loads(output)
    return jsonify(output)


if __name__ == '__main__':
    app.debug = False
    app.run(host='0.0.0.0', port=5001)
