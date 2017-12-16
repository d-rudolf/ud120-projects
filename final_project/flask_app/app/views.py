from collections import OrderedDict

import numpy as np
import pickle
from flask import jsonify, render_template
from . import myapp
import sys
sys.path.append("../")
print('views.py')


@myapp.route('/')
def home():
    with open("../final_project_dataset.pkl", "br") as data_file:
        data_dict = pickle.load(data_file)
    params = OrderedDict(sorted(data_dict["ALLEN PHILLIP K"].items(), key=lambda t: t[0])).keys()
    return render_template('home.html', params=params)

@myapp.route('/data/<feature_x>&<feature_y>')
def view_data(feature_x, feature_y):
    with open("../final_project_dataset.pkl", "br") as data_file:
        data_dict = pickle.load(data_file)
    data_dict = _remove_outlier(data_dict)
    data_dict_ordered = OrderedDict(sorted(data_dict.items(), key=lambda t: t[0]))
    data = []
    for key, value in data_dict_ordered.items():
        if value[feature_x] != 'NaN' and value[feature_y] != 'NaN':
            print(value[feature_x], value[feature_y])
            x = value[feature_x]
            y = value[feature_y]
            data.append((x, y, key))
    data.sort(key=lambda x: x[0])
    return jsonify(data)

def _scale_data(mylist):
    """
    - scale (x,y) before plotting
    :param data_dict:
    :return:
    """
    myarray = np.array(mylist)
    array_min = myarray.min()
    array_max = myarray.max()
    myarray = (myarray-array_min)/(array_max-array_min)
    return myarray

@myapp.route('/test_data')
def view_test_data():
    with open("app/templates/data.tsv", "r") as data_file:
        data = data_file.read()
        return data

def _remove_outlier(data_dict):
    outlier_list = ['TOTAL', 'LAY KENNETH L', 'SKILLING JEFFREY K']
    for name in list(data_dict.keys()):
        if name in outlier_list:
            data_dict.pop(name)
    return data_dict