#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:43:15 2020

@author: gutianjiao
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import urllib
from urllib import request

import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



def Trial():

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv"

    df = pd.read_csv(url)

    #print(df.head)

    #df without age column
    #df = df.drop('Age', axis=1)

    #df with age column and rounded values
    df['Age'] = df['Age'].div(10)

    df['Age'] = df['Age'].astype(int)


    df = df.drop('Age', axis=1)
    df = df.drop('Itching', axis=1)
    df = df.drop('Irritability', axis=1)
    df = df.drop('Obesity', axis=1)
    df = df.drop('sudden weight loss', axis=1)
    df = df.drop('Gender', axis=1)
    df = df.drop('muscle stiffness', axis=1)
    df = df.drop('visual blurring', axis=1)
    df = df.drop('Genital thrush', axis=1)
    df = df.drop('weakness', axis=1)

    #dropping these two causes 10% drop
    #df = df.drop('Polyuria', axis=1)
    #df = df.drop('Polydipsia', axis=1)

    df = df.drop('Polyphagia', axis=1)
    df = df.drop('delayed healing', axis=1)
    df = df.drop('partial paresis', axis=1)
    df = df.drop('Alopecia', axis=1)
   

    #print(df.head)


    df_encode = df.apply(LabelEncoder().fit_transform)

    testing_set, training_set = train_test_split(df_encode, test_size = 0.50)

    nb = CategoricalNB()

    nb.fit(training_set.drop('class', axis=1).values.tolist(), training_set['class'].tolist())

    y_predict = nb.predict(testing_set.drop('class', axis=1))

    #print(df_encode)
    print(f"Accuracy score is: {accuracy_score(testing_set['class'].tolist(), y_predict)}") 

    #df_encode.hist(column='Itching')
    
    #plt.show()




def main():
    for i in range(10):
        Trial()

main()