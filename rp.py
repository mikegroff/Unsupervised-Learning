#Michael Groff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.random_projection import GaussianRandomProjection as RP
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics
from scipy.stats import kurtosis,skew

def readin():
    #returns two dataframes with data to be used, last row being the result
    df1 = pd.read_csv('winequality-white.csv', sep=";")
    df2 = pd.read_csv('adult.txt', sep=",", header=None)
    df2.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
    stacked = df2[["1","3","5","6","7","8","9","13","14"]].stack()
    df2[["1","3","5","6","7","8","9","13","14"]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return df1,df2

if __name__=="__main__":
    print ("RP")
    df1,df2 = readin()
    array = df2.values
    X = array[:,:-1]
    Y = array[:,-1]
    labels = set(Y)
    maxc = X.shape[1]
    l = 25
    ks = range(1,l)
    kur = np.zeros(maxc)
    scr = np.zeros(maxc)
    ss = np.zeros(l-1)
    ll = np.zeros(l-1)

    """

    for i in ks:
        rp = RP(n_components = i)
        Z = rp.fit_transform(X)
        W = Z@np.linalg.pinv(rp.components_).T-X
        error = np.sum(np.multiply(W,W))
        scr[i-1] = error/(maxc*X.shape[0])



    plt.plot(ks,scr)
    plt.title("Adult Income - RP")
    plt.xlabel("# of components")
    plt.ylabel("Distance")
    plt.legend(['Avg error of inverse_transform'])
    plt.show()

    for k in ks:
        clust = KMeans(n_clusters = k).fit(Z)
        W = clust.predict(Z)
        ss[k-1] = clust.inertia_

    plt.plot(ks,ss)
    plt.title("Adult Income - KM")
    plt.xlabel("# of clusters")
    plt.ylabel("Sum of Squares")
    plt.legend(["kmeans"])
    plt.show()
    """

    pca = RP(n_components = 11)
    Z = pca.fit_transform(X)

    for k in ks:
        clust= GaussianMixture(n_components = k,reg_covar = 1e-5).fit(Z)
        W = clust.predict(Z)
        ll[k-1] = clust.score(Z)

    plt.plot(ks,ll)
    plt.title("Adult Income - EM")
    plt.xlabel("# of clusters")
    plt.ylabel("log of likelihood")
    plt.legend(["EM"])
    plt.show()
