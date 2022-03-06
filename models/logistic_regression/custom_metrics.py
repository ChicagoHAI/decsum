#!/usr/bin/env python3

import random

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


def train_val_compute(train_notes, val_notes, train_y, val_y, training_pipeline, parameters, verbose = True, func = roc_auc_score):

    """Using a validation set instead of cross validation"""
    parameter_grid = ParameterGrid(parameters)
    params, scores, best_score, best_params = [], [], 10000000000, None
    for C in parameter_grid:
        if verbose == True:
            print ("Currently using parameter :")
            print (C)
        training_pipeline.set_params(**C)
        training_pipeline.fit(train_notes, train_y)
        predicted_val_y = training_pipeline.predict(val_notes)
        score = func(val_y, predicted_val_y)
        print(score)
        params.append(C)
        scores.append(score)
        if (score < best_score):
            best_score = score
            best_params = C
    training_pipeline.set_params(**best_params)
    training_pipeline.fit(train_notes, train_y)
    return training_pipeline, best_score, best_params, params, scores

def mortality_rate_at_k(y_true, y_pred, K=100):

    results = [(v, random.random(), t) for t, v in zip(y_true, y_pred)]
    results.sort(reverse=True)
    print("average rate: ", np.mean(y_true))
    return np.mean([t for _, _, t in results[:K]])
