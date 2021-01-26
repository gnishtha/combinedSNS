from flask import Flask, request, Response
from flask_api import status

app = Flask(__name__)

from combinedSNS import preprocessing

@app.route('/api/predict/',  methods=['POST', 'GET'])
def predict():
    prediction = ''
    if request.method == 'POST':
        data = request.get_json(force = True)
        comment = data.get('comment','')
        if comment=='':
            return "Empty Comment", status.HTTP_400_BAD_REQUEST
        prediction = preprocessing(comment)
    return prediction


if __name__ == '__main__':
    app.run(debug=True)
