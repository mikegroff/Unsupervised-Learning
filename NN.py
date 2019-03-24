#Michael Groff

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import random
import time
from sklearn.decomposition import KernelPCA as KP, FactorAnalysis as FA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics
from scipy.stats import kurtosis,skew
from sklearn.random_projection import GaussianRandomProjection as RP
from sklearn.decomposition import FastICA as ICA, PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def split(array,size=1000,sizet=100):
    m,n = array.shape
    ind = np.random.choice(m,size, replace = False)
    all = set(range(0,m))
    left = all - set(ind)
    left = list(left)
    m = len(left)
    indt = np.random.choice(m,sizet,replace = False)
    left = np.take(left,indt,axis=0)
    test = np.take(array,ind,axis=0)
    trial = np.take(array,left,axis=0)
    return test,trial

def readin():
    #returns two dataframes with data to be used, last row being the result
    df1 = pd.read_csv('winequality-white.csv', sep=";")
    df2 = pd.read_csv('adult.txt', sep=",", header=None)
    df2.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
    stacked = df2[["1","3","5","6","7","8","9","13","14"]].stack()
    df2[["1","3","5","6","7","8","9","13","14"]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return df1,df2

def learn(array):
    X = array[:,:-1]
    Y = array[:,-1]
    clf = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=1000)
    clf = clf.fit(X,Y)
    return clf

def crossval(X,Y,size,p=10,k=10):
    score = []
    for i in range(0,k-1):
        rem = range(size*i,size*i+size)
        rem = set(rem)
        m = X.shape[0]
        left = set(range(0,m)) - rem
        left = list(left)
        train = np.take(X,left,axis=0)
        tree = learn(train)
        a,b = dt.test(tree,Y)
        c = dt.accr(a,b)
        score.append(c)
    return score

def learningcurve(df,p=10,n=100):
    m,z = df.shape
    size = int(0.7*m/n)
    sizes =[]
    traina = []
    testa = []
    times = []
    for i in range(1,n):
        train,trial = dt.split(df,size*i,int(0.3*m))
        s= time.clock()
        tree = learn(train)
        a,b = dt.test(tree, trial)
        score = dt.accr(a,b)
        c,d = dt.test(tree, train)
        scoret = dt.accr(c,d)
        e = time.clock()
        sizes.append(size*i)
        traina.append(scoret)
        testa.append(score)
        times.append(e-s)
    print("Trial Times")
    print(times)
    return sizes,testa,traina

def test(tree, array):
    X = array[:,:-1]
    Y = array[:,-1]
    Z = tree.predict(X)
    return Y,Z

def accr(a,b):
    c = np.where(a==b)
    c = np.asarray(c)
    k,tot = c.shape
    per = tot/a.size
    return per

if __name__=="__main__":
    print ("Nueral Network")
    df1,df2 = readin()
    array = df2.values
    m,z = array.shape
    X = array[:,:-1]
    Y = array[:,-1:]
    pca = FA(n_components = 7)
    Z = pca.fit_transform(X)
    clust = GaussianMixture(n_components = 1).fit(Z)
    Z = clust.predict(Z)
    b = np.zeros((m,1))
    b[:,0] = Z
    Z = b
    print(Z.shape,Y.shape)
    array = np.hstack((Z,Y))
    train1, trial1 = split(array,int(0.7*m),int(0.3*m))
    s = time.clock()
    tree1 = learn(train1)
    #dt.prune(tree1,10)
    a,b = test(tree1, trial1)
    score = accr(a,b)
    c,d = test(tree1, train1)
    scoret = accr(c,d)
    e = time.clock()
    print("Testing Set Score for Collection 1")
    print(score)
    print("Training Set Score for Collection 1")
    print(scoret)
    print("Runtime")
    print(e-s)
