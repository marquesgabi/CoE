# class_code.py
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
from class_code import model_init, model_pipeline
from keras.models import Sequential
#import xmltodict

app = Flask(__name__)
#model_weights_file = 'model/model.h5'
#model_arquitecture_file = 'model/model.json'
model_file = 'model/saved_model.h5'
model = model_init(model_file) #model_weights_file, model_arquitecture_file, model_file)

@app.route('/')
def test():
    return 'API running!'


@app.route('/json/', methods=['POST'])
def json_process():
    print(request.is_json)
    if not request.is_json:
        return jsonify('Error: Input must be a json format.')
    data = request.get_json()
    #prediction = np.array2string(np.array(model_pipeline(data, model, 'json')))
    prediction = model_pipeline(data, model, 'json')
    return jsonify(prediction)

@app.route('/xml/', methods=['POST'])
def xml_process():
    xml_data = request.form['NFe']
    content_dict = xmltodict.parse(xml_data)
    return jsonify(content_dict) #jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
