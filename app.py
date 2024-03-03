import flask
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PredictInventory
import TotalAnalyize

app = Flask(__name__)
CORS(app)
global uploadedFile


@app.route('/uploadData', methods=['POST'])
def get_uploaded_data():
    f = request.files['file']
    f.save("./uploadFiles/" + secure_filename(f.filename))
    global uploadedFile
    uploadedFile = f.filename
    return 'Uplaod Success!'

@app.route('/getAnalysisList', methods=["POST", "GET"])
def get_analysis_List():
    data = TotalAnalyize.getTotalAnalyizeList(uploadedFile)
    return data

@app.route('/getLSTMData', methods=["POST", "GET"])
def get_LSTM_data():
    inputValues = request.json
    data = PredictInventory.getLSTMData(inputValues, uploadedFile)
    return data


if __name__ == "__main__":
    app.run()