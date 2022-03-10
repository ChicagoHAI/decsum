#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 
# =============================================================================

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

test = pd.read_csv('ridge_prediction.csv')
test = test[test.business != "#NAME?"]
longformer_pred = pd.read_csv('longformer_prediction.csv')
longformer_pred = longformer_pred[longformer_pred.business != "#NAME?"]

combined = pd.merge(test, longformer_pred, on="business", how="inner")
combined['first_10_avg'] = combined['scores'].apply(lambda x: np.mean(ast.literal_eval(x)))

print(f'Longformer MSE: {mean_squared_error(combined.pred, combined.avg_score)}')
print(f'Ridge Regression MSE: {mean_squared_error(combined.predicted, combined.avg_score)}')
def group(x):
    if x < 1.5:
        return 0
    elif x < 2.5:
        return 2
    elif x < 3.5:
        return 3
    elif x < 4.5:
        return 4
    else:
        return 5

combined['group'] = combined['first_10_avg'].apply(lambda x: group(x))

combined_show = combined[combined['group'] > 0]

sns.displot(data=combined_show, x='pred', hue="group", kind='kde')
plt.show()
sns.displot(data=combined_show, x='predicted', hue="group", kind='kde')
plt.show()

