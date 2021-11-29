#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 00:42:16 2020

@author: gutianjiao
"""
import numpy as np
import pandas as pd

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

from matplotlib import pyplot as plt

accuracy = 0

def rand_jitter(arr):
    stdev = .2*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def Trial(url, trials):

   

    df = pd.read_csv(url)

    #print(df.head)

    #df without age column

    #df with age column and rounded values
    #df['Age'] = df['Age'].div(10)

    #df['Age'] = df['Age'].astype(int)


    #print(df.head)

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
    df_encode = df.apply(LabelEncoder().fit_transform)

    xplot = df_encode['Polyuria'].values.tolist()
    yplot = df_encode['Polydipsia'].values.tolist()

    plt.scatter(rand_jitter(xplot), rand_jitter(yplot), s=7, c=df_encode['class'].values.tolist())

    plt.show()

    nb = CategoricalNB()


    print(df_encode)
  

    global accuracy
    nb = CategoricalNB()
    total = 0

    for i in range(trials):

        total+=1
        
        testing_set, training_set = train_test_split(df_encode, test_size = 0.50)

        nb.fit(training_set.drop('class', axis=1).values.tolist(), training_set['class'].tolist())

        y_predict = nb.predict(testing_set.drop('class', axis=1))

        current_acc = accuracy_score(testing_set['class'].tolist(), y_predict)
        #print(df_encode)
        print("Accuracy score for trial {} is: {}".format(total, current_acc))
        accuracy+= current_acc

    print('Average accuracy for {} trials is: {}'.format(total, accuracy / total))


        



def observeplot(url):
    
    df = pd.read_csv(url)
    df = df.drop('Age', axis=1)

    df_encode = df.apply(LabelEncoder().fit_transform)

    #df_encode.hist()

    xplot = df_encode['weakness'].values.tolist()
    yplot = df_encode['Polyphagia'].values.tolist()

    #plt.scatter(rand_jitter(xplot), rand_jitter(yplot), s=2, c=df_encode['class'].values.tolist())
    """ import seaborn as sns
    plt.figure(figsize=(16,6))
   
    ax = sns.countplot(x='Polyphagia', data=df_encode)
    ax.set_xticklabels(labels=['Yes', 'No'])
    plt.show() """

    import seaborn as sns

    grouped = df_encode.groupby(['class', 'Polyphagia'])['Polyphagia'].count()

    grouped = df.groupby(['class', 'Polyphagia'])['Polyphagia'].count()
    

    print(grouped.head)

    fig, ax = plt.subplots(figsize=(15,15))
    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:green']

    #grouped.plot(ax=ax, kind='bar', stacked=True, color=colors[:])
    grouped.plot.bar(color=colors)
    #ax=sns.countplot(x=grouped)
    
    ax.set_xticklabels(labels=['No', 'Yes', 'No', 'Yes'])
    
    plt.show()


def main():
    
    url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv'

    Trial(url1, 10)
    #observeplot(url1)
main()