from pprint import pprint
from time import time

import numpy as np
from sklearn import svm, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.model_selection import cross_validate


def _get_features():
    features_list = ['poi',
                     'bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees',
                     'email_address',
                     'exercised_stock_options',
                     'expenses',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'loan_advances',
                     'long_term_incentive',
                     'other',
                     'restricted_stock',
                     'restricted_stock_deferred',
                     'salary',
                     'shared_receipt_with_poi',
                     'to_messages',
                     'total_payments',
                     'total_stock_value']

    features_list.remove('email_address')
    myfeatures = ['poi',
                  'salary',
                  'bonus',
                  'deferral_payments',
                  'deferred_income',
                  'director_fees',
                  'expenses',
                  'loan_advances',
                  'long_term_incentive',
                  'total_payments']
    return myfeatures

def _remove_outlier(data_dict):
    outlier_list = ['TOTAL', 'LAY KENNETH L']
    for name in data_dict.keys():
        if name in outlier_list:
            data_dict.pop(name)
    return data_dict

def _scale_data(features):
    myarray = np.array(features)
    array_min = myarray.min()
    array_max = myarray.max()
    myarray = (myarray-array_min)/(array_max-array_min)
    return myarray

def _select_features(key):
    feat_select = {'pca': PCA(n_components=5),
                   'var_threshold': VarianceThreshold(threshold = 0.1),
                   'k_best': SelectKBest(chi2, k=5)}
    return feat_select[key]

def _get_classifier(key):
    clf_dict ={'svm': svm.SVC(kernel='rbf'),
               'lin_reg': linear_model.LinearRegression()}
    return clf_dict[key]

def _get_train_test_data(features, labels):
    feature_train, feature_test, label_train, label_test = train_test_split(
    features, labels, test_size = 0.4, random_state = 0)
    return feature_train, feature_test, label_train, label_test


def _cross_validate(pipeline, features, labels):
    """
     precision = Tp/(Tp + Fp)
     recall = Tp/(Tp + Fn)
    """
    scoring = ['precision', 'recall']
    scores = cross_validate(pipeline, features, labels, scoring=scoring,
                             cv=5, return_train_score=False)
    print(scores.keys())
    recall = _get_mean_and_std(scores['test_recall'])
    precision = _get_mean_and_std(scores['test_precision'])
    print('recall: {0:0.3f} +/- {1:0.3f}'.format(recall[0], recall[1]))
    print('precision: {0:0.3f} +/- {1:0.3f}'.format(precision[0], precision[1]))

def _get_mean_and_std(array):
    mean = array.mean()
    std = array.std()
    return mean, std

def _get_parameters():
    parameters = {'feat_select__n_components': (1, 3, 5, 7, 9),
                  'clf__C': (1e-3, 1, 1e3),
                  'clf__gamma': (0.1, 1, 10)}
    return parameters

def _evaluate_grid_search(grid_search, mypipeline, parameters, feature, label):
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in mypipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(feature, label)
    print("done in %0.3fs" % (time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))