from flask import render_template, Flask, request

app = Flask(__name__)

from combinedSNS import preprocessing

@app.route('/predict',  methods=['POST', 'GET'])
def predict():
    prediction = ''
    if request.method == 'POST':
        comment = request.form['comment']
        prediction = preprocessing(comment)
    return render_template('end.html', prediction=prediction)


@app.route('/')
def index():
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True)
