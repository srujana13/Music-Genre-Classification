# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:07:05 2018

@author: srujana
"""
import sklearn 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from utils1 import GENRE_DIR, GENRE_LIST
from sklearn.externals import joblib
from random import shuffle

"""reads FFT-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required FFT-files
base_dir must contain genre_list of directories
"""
def read_fft(genre_list, base_dir):
	X = []
	y = []
	for label, genre in enumerate(genre_list):
		# create UNIX pathnames to id FFT-files.
		genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		# get path names that math genre-dir
		file_list = glob.glob(genre_dir)
		for file in file_list:
			fft_features = np.load(file)
			X.append(fft_features)
			y.append(label)
	
	return np.array(X), np.array(y)


"""reads MFCC-files and prepares X_train and y_train.
genre_list must consist of names of folders/genres consisting of the required MFCC-files
base_dir must contain genre_list of directories
"""
def read_ceps(genre_list, base_dir):
	X= []
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

def learn_and_classify(X_train, y_train, X_test, y_test, genre_list):

	
    print(len(X_train))
    print(len(X_train[0]))
    
    #Logistic Regression classifier

    logistic_classifier = linear_model.logistic.LogisticRegression()
    logistic_classifier.fit(X_train, y_train)
    
    #K-Nearest neighbour classifier

    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    
    #Multi Layer Perceptron classifier
    
    mlp = MLPClassifier(max_iter=300,learning_rate='adaptive')
    mlp.fit(X_train,y_train)

    #Poly kernel SVM
    Psvm = SVC(kernel='poly')
    Psvm.fit(X_train,y_train)
    
    #ensemble classifier
    
    eclf = VotingClassifier(estimators=[('mlp',mlp),('Psvm',Psvm),('knn',knn_classifier),('lr',logistic_classifier)],voting='hard',weights=[4,3,2,1])
    eclf.fit(X_train,y_train)
    eclf_predictions = eclf.predict(X_test)
    eclf_accuracy = accuracy_score(y_test,eclf_predictions)
    eclf_cm = confusion_matrix(y_test,eclf_predictions)
    print("Ensemble Classifier accuracy = "+str(eclf_accuracy))
    print("eclf_cm:")
    print(eclf_cm)              
    
    target_names = ['classical','jazz','metal','pop']
    print(classification_report(y_test,eclf_predictions,target_names=target_names))
    
    joblib.dump(eclf,'saved_models/model_mfcc_eclf_nn.pkl')
    
    #plot_confusion_matrix(mlp_cm, "Confusion matrix for multi-layered perceptron classification", genre_list)
    #plot_confusion_matrix(Psvm_cm,"Confusion matrix for Poly kernel SVM",genre_list)
    plot_confusion_matrix(eclf_cm, "Confusion matrix for ensemble classifier", genre_list)
    
def plot_confusion_matrix(cm, title, genre_list, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
"""
def ensemble_classifier(clf1,clf2,clf3,clf4):
    eclf = VotingClassifier(estimators=[('mlp',clf1),('Psvm',clf2),('knn',clf3),('some',clf4)],voting='hard',weights=[4,3,2,1])
    eclf.fit(X_train,y_train)
    eclf    
"""
def main():
    
    base_dir_mfcc = GENRE_DIR
    
    """list of genres (these must be folder names consisting .wav of respective genre in the base_dir)
    Change list if needed.
    """
    #genre_list = [ "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
    
    genre_list = ["classical","jazz","metal","pop"]
    
    #use MFCC
    X,y= read_ceps(genre_list, base_dir_mfcc)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20)
    print("new1",X_train.shape)
    print('******USING MFCC******')
    learn_and_classify(X_train, y_train, X_test, y_test, genre_list)
    print('*********************')

if __name__ == "__main__":
	main()