#-*- coding:utf-8 -*-
#==========================================================
#      https://www.nltk.org/book/ch06.html
#==========================================================

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
#import pandas_datareader as pdr
from pandas_datareader.data import DataReader
from datetime import datetime, timedelta

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pydot
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

def data_preprocessing():
    import pandas as pd

    X  = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
                       'title': ["His Last Bow", "How Watson Learned the Trick","A Moveable Feast", "The Grapes of Wrath"],
                       'expert_rating': [5, 3, 4, 5],
                       'user_rating': [4, 5, 4, 3]})

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

    column_trans = ColumnTransformer ([('citi_label',OneHotEncoder(),['city']),('title_feature', CountVectorizer(), 'title'),('nExport_Rating', MinMaxScaler(), ['expert_rating','user_rating'])],remainder='passthrough')


    result = column_trans.fit_transform(X)
    #print column_trans.get_feature_names()
    print result

def moving_average (source, period):
    def fun (data):
        return data + "_MA" + str(int(period))

    data = source.copy()
    data.columns = list(map(fun, data.columns))
    return data.rolling (period).mean()

def macd (source, period1, period2):
    def fun (data):
        return data + "_MACD(%s,%s)" %(str(int(period1)),str(int(period2)))

    data = source.copy()
    data.columns = list(map(fun, data.columns))
    return data.rolling (period1).mean() - data.rolling (period2).mean()

def signal_noise_ratio (source, period):
    import numpy as np
    def fun (data):
        return data + "_SN"

    data = source.copy()
    data.columns = list(map(fun, data.columns))
    return data.pct_change(1).rolling(period).mean() / data.pct_change(1).rolling(period).std() * np.sqrt(252.0/period)

#-----------------------------------------------------------------------------------------------------------------------
# INPUT
TARGET = "sp500"

pDATA = DataReader(TARGET,'fred',datetime(1985,1,1), datetime.today()).dropna(how='any')

# STEP 1.2 Build Target Value
INVEST_HORIZON = 20
pDATA ['TARGET'] = pDATA[TARGET].pct_change(INVEST_HORIZON).shift(-INVEST_HORIZON)

#-----------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTOR
#pDATA = pd.concat([pDATA, moving_average(pDATA, 20), moving_average(pDATA, 60), macd(pDATA,20, 60), signal_noise_ratio(pDATA,60)], axis=1)
pDATA = pd.concat([pDATA, macd(pDATA[TARGET].to_frame(),20, 60), signal_noise_ratio(pDATA[TARGET].to_frame(),60)], axis=1)

pDATA2 = pDATA.dropna(how='any')

#Y = pDATA2['TARGET'].apply(lambda x: 1 if x > 0.0 else 0 ).values
Y = pDATA2['TARGET'].apply(lambda x: -1 if x < 0.0 else 1 if x > 0.035 else 0).values
#Y = pDATA2['TARGET'].values
X = pDATA2[pDATA2.columns[2:]].values

#-----------------------------------------------------------------------------------------------------------------------
# STEP 2 - MACHINE LEARNING
DCTree_classifier = DecisionTreeClassifier(max_depth=3, random_state=0)
DCTree_classifier.fit(X, Y)
Y_pred = DCTree_classifier.predict(X)

print('Decision Tree Accuracy: %.2f' % accuracy_score(Y, Y_pred))

export_graphviz(DCTree_classifier, out_file="decision_tree1.dot",
                class_names=['DOWN','NEUTRAL','UP'],
                feature_names=[u"MACD", u"SN"],
                impurity=False, filled=True)

(graph,) = pydot.graph_from_dot_file("decision_tree1.dot", encoding='utf8')
graph.write_png("decision_tree1.png")

Classifiers = {'Logistic Regression': LogisticRegression(), 'Gaussian Naive Bayes': GaussianNB (), "SGD Classifier": SGDClassifier(), "BNB Bayes": BernoulliNB(), "SVC": SVC()}

# Training -------------------------------------------------------------------------------------------------------------
Trained = {"Realized": Y}

for key in Classifiers.keys():
    classifier = Classifiers[key]
    classifier.fit(X, Y)
    Y_pred = classifier.predict(X)
    Trained [key] = Y_pred
    print(key + ' Accuracy: %.2f' % accuracy_score(Y, Y_pred))

# Prediction -----------------------------------------------------------------------------------------------------------
LASTEST_FEATURE =  pDATA[pDATA.columns[2:]].tail(1).values

for key in Classifiers.keys():
    classifier = Classifiers[key]
    #print(key + "expect current status as %i" % classifier.predict_proba(LASTEST_FEATURE))
    if hasattr (classifier, "predict_proba"):
        print key, classifier.predict_proba(LASTEST_FEATURE)

#pTrained = pd.DataFrame (Trained)
#pTrained.plot()
#plt.show()