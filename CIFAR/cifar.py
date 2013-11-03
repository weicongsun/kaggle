from scipy.misc  import imread, imresize
import numpy as np 
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import os
from time import time
import logging
from sklearn import linear_model,metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import csv

def load_imgs(folder_path='./train/'):
	filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
	n_imgs = len(filepaths)
	data = np.zeros((n_imgs, 32*32))
	target_classes = []
	for i, f in enumerate(filepaths):
		print f
		img = np.asarray(imread(f), dtype=np.float32)
		img = img.mean(2)
		img /= 255.0
		index = int(f.split('/')[2].split('.')[0])
		data[index - 1, ...] = img.ravel()

	targetfile = csv.reader(open('trainLabels.csv', 'rb'))
	header = targetfile.next()
	for i, row in enumerate(targetfile):
		if i >= 5000 :
			break

		target_classes.append(row[1])

	target_names = np.unique(target_classes)
	target = np.searchsorted(target_names, target_classes)


	return data, target, target_names


X, y, target_names = load_imgs('./train_5k/')
print X.shape
print y.shape
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# Compute a PCA (eigenfaces) 
n_components = 150
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components,32,32))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print X_train_pca.shape

###############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))

