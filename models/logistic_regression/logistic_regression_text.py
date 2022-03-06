#!/usr/bin/env python3
import argparse
import os
import pickle
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion

from models.logistic_regression.custom_metrics import train_val_compute
from models.logistic_regression.feature_definitions import BOWFeatures, IdentityFeatures
from models.utils import load_jsonl_gz


def flatReviews(df):
    for i in range(len(df)):
        df['reviews'].iloc[i] = " ".join(df['reviews'].iloc[i])

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data/out/50reviews/",
                        type=str, help='Path to the data of in-hospital mortality task')
    parser.add_argument('--feature_used', default="all",
                        type=str, help='feature used', choices=["all", "notes", "all_but_notes"])
    parser.add_argument('--balanced', dest="balanced", action="store_true", help = 'whether to use balanced class weights')
    parser.add_argument('--metric', default="mse", type=str, choices=["mse"], help = 'metrics')
    args = parser.parse_args()
    print (args)

    print("Loading data")
    train = pd.DataFrame(load_jsonl_gz(os.path.join(args.data_dir,"train.jsonl.gz")))
    valid = pd.DataFrame(load_jsonl_gz(os.path.join(args.data_dir,"dev.jsonl.gz")))
    test = pd.DataFrame(load_jsonl_gz(os.path.join(args.data_dir,"test.jsonl.gz")))
    # data = pd.read_json(open(args.data))
    # train, test = train_test_split(data, test_size=0.2, random_state=19)
    # train, valid = train_test_split(train, test_size=0.2, random_state=19)

    train = flatReviews(train)
    valid = flatReviews(valid)
    test = flatReviews(test)

    metric = mean_squared_error

    union_list = []
    if args.feature_used in ['all', 'notes']:
        print ("add Bag of Words features .....")
        union_list.append(("tfidf_pipe",
                            Pipeline([
                            ("tfidf", BOWFeatures()),
                            ])))
    if args.feature_used in ['all','all_but_notes']:
        print ("add structured variable features ..... ")
        union_list.append(("structured",
                           Pipeline([
                               ("identity", IdentityFeatures()),
                               ("imputer", SimpleImputer())
                           ])))

    print("Total number of training data:", len(train))
    print("Total number of validation data:", len(valid))
    print("Total number of test data:", len(test))

    pipeline = Pipeline([
        ('union', FeatureUnion(union_list)),
        ('ridge', Ridge(solver="auto", max_iter = 500)),
    ])

    parameters = {
        "ridge__alpha": np.logspace(-5, 3, 9, base = 2)
    }

    # Display of parameters

    print("Now doing training on training set and hyperparameter tuning using the validation set...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    # Training on training data and hyperparameter tuning on validation data

    t0 = time()
    pipeline, best_score, best_parameters, params, scores = train_val_compute(train, valid, train['avg_score'], valid['avg_score'], pipeline, parameters, func=metric)
    print("done in %0.3fs" % (time() - t0))
    print()

    # Displaying training results

    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Best models")
    print(pipeline)
    print ("Mean test score:")
    print(scores)
    print("Best score: \n%0.3f" % best_score)


    # Displaying test results

    test_predicted = pipeline.predict(test)
    test['predicted'] = test_predicted
    test.to_csv('test_with_predicted.csv')
    print ("MSE on Test Set:")
    print(mean_squared_error(test['avg_score'], test_predicted))
    test_score = mean_squared_error(test['avg_score'], test_predicted)

    print("save model")

    model_name = '.chkpt'
    if args.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif args.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name
    path_dir = f'/data/joe/yelp/logistic_regression/models/'
    import pathlib
    pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(pipeline, open(os.path.join(path_dir, model_name), 'wb'))

    # save result
    result_dir = "./models/logistic_regression/results/"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    outname = f'.csv'
    if args.feature_used == "all":
        outname = "feature_text_" + outname
    elif args.feature_used == "notes":
        outname = "text_" + outname
    else:
        outname = "feature_" + outname

    print("Write Result to ", outname)
    with open(os.path.join(result_dir, outname), 'w') as f:
        f.write("TYPE,MSE\n")
        f.write(f"valid,{best_score}\n")
        f.write(f"test,{test_score}")

    if args.feature_used == "notes":
        tfidf_words = dict(pipeline.named_steps['union']
                        .transformer_list).get('tfidf_pipe').named_steps['tfidf'].get_feature_names()
        lr_coefs_pos = pipeline.named_steps['ridge'].coef_.argsort()[::-1][:10]
        print(lr_coefs_pos)
        lr_coefs_neg = pipeline.named_steps['ridge'].coef_.argsort()[:10]
        print("important pos words")
        for i in lr_coefs_pos:
            print(tfidf_words[i], pipeline.named_steps['ridge'].coef_[i])
        print("important neg words")
        for i in lr_coefs_neg:
            print(tfidf_words[i], pipeline.named_steps['ridge'].coef_[i])