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
		# if i >= 5000 :
		# 	break

		target_classes.append(row[1])

	target_names = np.unique(target_classes)
	target = np.searchsorted(target_names, target_classes)

	return data, target, target_names


X, Y, target_names = load_imgs('./train_bk/')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)
# Models we will use
logistic = linear_model.LogisticRegression(C=1000.0)
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training
# Pipeline(logistic=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty=l2, random_state=None, tol=0.0001),
#      logistic__C=1.0, logistic__class_weight=None, logistic__dual=False,
#      logistic__fit_intercept=True, logistic__intercept_scaling=1,
#      logistic__penalty=l2, logistic__random_state=None,
#      logistic__tol=0.0001,
#      rbm=BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=256, n_iter=10,
#        random_state=0, verbose=True),
#      rbm__batch_size=10, rbm__learning_rate=0.1, rbm__n_components=256,
#      rbm__n_iter=10, rbm__random_state=0, rbm__verbose=True)
# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
# rbm.learning_rate = 0.1
# rbm.n_iter = 10
# # More components tend to give better prediction performance, but larger
# # fitting time
# rbm.n_components = 256
# logistic.C = 1.0

# clf = classifier

print("Fitting the classifier to the training set")
param_grid = {}
clf = GridSearchCV(classifier, param_grid)

clf.fit(X_train, Y_train)

print("Best estimator found by grid search:")
print(clf.best_estimator_)
###############################################################################
# Evaluation

print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        clf.predict(X_test))))
