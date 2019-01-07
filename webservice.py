
'''
curl --header "Content-Type: application/json"   --request POST   --data @test2.json   http://127.0.0.1:5000/cuisine/api/json

# output
{"predictions": [{"id": 18009, "cuisine": "italian", "probability": 0.17846838753494407}, {"id": 28583, "cuisine": "italian", "probability": 0.1918703125538735}]}
'''

from flask import Flask, jsonify
from flask import make_response
from flask import abort
from flask import request
import os, io
import sys
import json, urllib
from pprint import pprint
import requests
import pickle
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from pandas.io.json import json_normalize
from collections import OrderedDict
from collections import defaultdict


app = Flask(__name__)
categoricalTransformer = {9: 'italian', 0: 'italian', 13: 'mexican', 16: 'mexican', 18: 'cajun_creole', 4: 'southern_us', 3: 'mexican', 17: 'mexican', 7: 'filipino', 2: 'mexican', 14: 'brazilian', 10: 'spanish', 1: 'brazilian', 19: 'mexican', 11: 'indian', 5: 'thai', 6: 'italian', 12: 'indian', 8: 'chinese', 15: 'mexican'}

def flatten_json(input_file):
    corpus_file = open(input_file,"r")
    corpus = corpus_file.read()
    entries =  json.loads(corpus)
    df =  json_normalize(entries)
    df['flat_ingredients'] = df.apply(lambda row: ' '.join(ingredient for ingredient in row['ingredients']), axis=1)
    df['word_count'] = df.apply(lambda row: len(row['flat_ingredients'].split(' ')), axis=1)
    df.drop('ingredients', axis=1, inplace=True)
    df.sort_values(['word_count'], ascending=False, inplace=True)
    return df



# read POST JSON
@app.route('/')
def index():
    return "Cuisine Prediction Service"

@app.route('/cuisine/api/json',methods=['POST'])
def getCuisine():
    data = request.json
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)

    test = flatten_json('./data.json')

    tfidf_vect = pickle.load(open("vectorizer.pickle", "rb"))

    test_transform = tfidf_vect.transform(test['flat_ingredients'])

    le = preprocessing.LabelEncoder()
    X_test = test_transform

    # load the model
    filename = 'finalized_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    #predict
    predicted_labels = loaded_model.predict(X_test)

    predicted_probability = loaded_model.predict_proba(X_test)

    json_data = []
    for i in range (0, len(predicted_labels)):
        data = {}
        data['id'] = int(test['id'][i])
        data['cuisine'] = categoricalTransformer[predicted_labels[i]]
        item = predicted_probability[i]
        data['probability'] = item[predicted_labels[i]]
        json_data.append(data)
    json_return = {"predictions":json_data}

    # ordering to make it in expected output format
    key_order = ["id", "cuisine", "probability"]
    result = defaultdict(list)
    for dic in json_return["predictions"]:
        ordered = OrderedDict((key, dic.get(key)) for key in key_order)
        result["predictions"].append(ordered)
    return (json.dumps(result))

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
