#!/usr/bin/env python3

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

class BOWFeatures(TfidfVectorizer):
    def __init__(self, tokenizer=None, norm="l2",
                 use_idf=True, max_df=1.0, min_df=5,
                 lowercase=True, ngram_range=(1, 1),
                 token_pattern=r"\w+", smooth_idf=True):
        super().__init__(tokenizer=tokenizer,
                                          norm=norm,
                                          max_df=max_df,
                                          min_df=min_df,
                                          lowercase=lowercase,
                                          use_idf=use_idf,
                                          ngram_range=ngram_range,
                                          token_pattern=token_pattern,
                                          smooth_idf=smooth_idf)

    def fit(self, data, y=None):
        return super().fit(data['reviews'], y)


    def transform(self, data, y=None):
        X = super().transform(data['reviews'])
        return X

    def fit_transform(self, X, y=None):
        # make sure that the base class does not do "clever" things
        return self.fit(X, y).transform(X, y)

    def get_feature_names(self):
        feature_names = super(BOWFeatures, self).get_feature_names()
        return feature_names

class IdentityFeatures(TransformerMixin):
    def __init__(self):
        self.hi = 0

    def transform(self, X, *_):
        return np.stack(X['scores'].values)

    def fit(self, *_):
        return self

    def fit_transform(self, X, y=None):
        # make sure that the base class does not do "clever" things
        return self.fit(X, y).transform(X, y)
