import os
import pickle
from flask import jsonify
from . import myapp

print('views.py')


@myapp.route('/')
def home():
    with open("../final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    return jsonify(data_dict)