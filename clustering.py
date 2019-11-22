# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:38:41 2018

@author: srujana
"""
import sklearn 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scipy
import os
import sys
import glob
import numpy as np
from sklearn.externals import joblib
from random import shuffle
from utils1 import GENRE_DIR,GENRE_LIST 

def read_ceps(genre_list, base_dir):
	X=[]
	y=[]
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
			ceps = np.load(fn)
			num_ceps = len(ceps)
			X.append(np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
			#X.append(ceps)
			y.append(label)
	
	print(np.array(X).shape)
	print(len(y))
	return np.array(X), np.array(y)

  #Kmeans clustering
def kmeans_clustering(X_train, y_train, X_test, y_test, genre_list):
    scalar = StandardScaler()
    scalar.fit(X_train,y_train)
    new_data = scalar.transform(X_train)
    kmeans = KMeans(init='k-means++',n_init=10,n_clusters=4, max_iter=300)
    rVal = kmeans.fit(X_train,y_train)
    kmeans_predictions = kmeans.predict(X_test)
    print("the randomized score is : ",metrics.adjusted_rand_score(y_test,kmeans_predictions))
    print("the normalized mutual info score is : ",metrics.normalized_mutual_info_score(y_test,kmeans_predictions))
    print("the mutual info score is : ",metrics.mutual_info_score(y_test,kmeans_predictions))
    print("the homogenity, completeness and v measure score is : ",metrics.homogeneity_completeness_v_measure(y_test,kmeans_predictions))
    print("the fowlkes mallows score is : ",metrics.fowlkes_mallows_score(y_test,kmeans_predictions))
    labels = kmeans.labels_
    print("the silhouette score is :",metrics.silhouette_score(X_test,kmeans_predictions,metric='euclidean'))
    print(kmeans_predictions)
    print(y_test)
    centers = rVal.cluster_centers_
    distances = pairwise_distances(new_data, centers, metric='euclidean')
    clusters = np.argmin(distances,axis=1)
    print(len(clusters))
    plotSamples = PCA(n_components=2).fit_transform(new_data)
    plotClusters(plotSamples,clusters,kmeans)
    joblib.dump(kmeans,'saved_models/model_kmeans.pkl')
    
def plotClusters(plotSamples,clusters,kmeans ):
    x = plotSamples[:,0]
    y = plotSamples[:,1]
    fig = plt.figure(1,figsize=(5,4))
    """
    LABEL_COLOR_MAP = {0 : 'r',
                       1 : 'k',
                       2 : 'g',
                       3 : 'b'
                       }
    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    """
    plt.scatter(x,y,c=kmeans.labels_.astype(float),alpha=0.5)
    plt.savefig("./plots/clusters")
    
def main():
	
	base_dir_fft  = GENRE_DIR
	base_dir_mfcc = GENRE_DIR
	#genre_list = [ "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
	
	genre_list = ["classical","jazz","metal","pop"]

	#use FFT
	"""X, y = read_fft(genre_list, base_dir_fft)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
	print('\n******USING FFT******')
	kmeans_clustering(X_train, y_train, X_test, y_test, genre_list)
	print('*********************\n')
"""
	#use MFCC
	X,y= read_ceps(genre_list,"C:/Users/srujana/Desktop/genres")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .10)
	print("new1",X_train.shape)
	print('******USING MFCC******')
	kmeans_clustering(X_train, y_train, X_test, y_test, genre_list)
	print('*********************')

if __name__ == "__main__":
	main()