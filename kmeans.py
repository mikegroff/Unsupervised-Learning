#Michael Groff

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import metrics
from scipy.spatial.distance import cdist

def readin():
    #returns two dataframes with data to be used, last row being the result
    df1 = pd.read_csv('winequality-white.csv', sep=";")
    df2 = pd.read_csv('adult.txt', sep=",", header=None)
    df2.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14"]
    stacked = df2[["1","3","5","6","7","8","9","13","14"]].stack()
    df2[["1","3","5","6","7","8","9","13","14"]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return df1,df2

if __name__=="__main__":
    print ("Kmeans")
    df1,df2 = readin()

    array = df1.values
    X = array[:,:-1]
    Y = array[:,-1]
    labels = set(Y)
    l = 50
    hom = np.zeros(l-1)
    com = np.zeros(l-1)
    vme = np.zeros(l-1)
    ran = np.zeros(l-1)
    ks = range(2,l+1)
    min = np.zeros(l-1)
    avg = np.zeros(l-1)
    cmin = np.zeros(l-1)
    cavg = np.zeros(l-1)
    ss = np.zeros(l-1)


    for k in ks:
        clust = KMeans(n_clusters = k).fit(X)
        Z = clust.predict(X)
        ss[k-2] = clust.inertia_
        hom[k-2] = metrics.homogeneity_score(Y,Z)
        com[k-2] = metrics.completeness_score(Y,Z)
        vme[k-2] = metrics.completeness_score(Y,Z)
        ran[k-2] = metrics.adjusted_rand_score(Y,Z)

        props = np.zeros((k,len(labels)))
        csiz = np.zeros(k)
        for i in range(0,k):
            a = Y[np.where(Z == i)]
            csiz[i] = a.size
            p = 0
            for j in labels:
                b = a[np.where(a==j)]
                props[i,p] = b.size/a.size
                p += 1
        maxes = np.amax(props,axis = 1)
        min[k-2] = np.amin(maxes)
        avg[k-2] = np.mean(maxes)
        cmin[k-2] = np.amin(csiz)
        cavg[k-2] = np.mean(csiz)
    cs = np.divide(cmin,cavg)
    sa = np.divide(min,avg)


    plt.plot(ks,hom,ks,com,ks,vme,ks,ran)
    plt.title("Wine Quality Data - KM")
    plt.xlabel("# of clusters")
    plt.ylabel("Score")
    plt.legend(["Homoegenity", 'Completeness','V_measure', 'Adjusted Rand index' ])
    plt.show()

    plt.plot(ks,cs)
    plt.title("Wine Quality Data - KM")
    plt.xlabel("# of clusters")
    plt.ylabel("Proportional Cluster size")
    plt.legend(["Minimal Cluster",])
    plt.show()

    plt.plot(ks,ss)
    plt.title("Wine Quality Data - KM")
    plt.xlabel("# of clusters")
    plt.ylabel("Sum of Squares")
    plt.legend(["kmeans",])
    plt.show()


    plt.plot(ks,min,ks,avg)
    plt.title("Wine Quality Data - KM")
    plt.xlabel("# of clusters")
    plt.ylabel("Highest label proportion agreement")
    plt.legend(["Minimal scoring cluster",'Average cluster score','Proportional score'])
    plt.show()
