from scipy.misc  import imread, imresize
import numpy as np 
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
from time import time
import logging
from sklearn import linear_model,metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# Display progress logs on stdout
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

def load_imgs(folder_path='./tmp/', slices=(slice(70, 195), slice(78, 172))):
	filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
	n_imgs = len(filepaths)
	data = np.zeros((n_imgs, 64*64))
	target = np.zeros(n_imgs)
	for i, f in enumerate(filepaths):
		print f
		img = imread(f)
		if img.shape[0] > 195 and img.shape[1] > 172 :
			img = img[slices]

		img = np.asarray(img, dtype=np.float32)
		img = imresize(img, (64,64))	
		img = img.mean(2)
		img /= 255.0
		data[i, ...] = img.ravel()

		if f.find('cat') >= 0 :
			target[i] = 0
		else:
			target[i] = 1



	indices = np.arange(n_imgs)	
	np.random.RandomState(42).shuffle(indices)
	data, target = data[indices], target[indices]
	return data, target

trainData, trainTarget = load_imgs('./train/')
testData, testTarget = load_imgs('./test/')

# trainData = (trainData - np.min(trainData, 0)) / (np.max(trainData, 0) + 0.0001)
# testData = (testData - np.min(testData, 0)) / (np.max(testData, 0) + 0.0001)
# 
n_components =  100
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(trainData)

eigenfaces = pca.components_.reshape((n_components, 64, 64))
train_pca = pca.transform(trainData)
test_pca = pca.transform(testData)

# train_pca = (train_pca - np.min(train_pca, 0)) / (np.max(train_pca, 0) + 0.0001)
# test_pca = (test_pca - np.min(test_pca, 0)) / (np.max(test_pca, 0) + 0.0001)

logistic = linear_model.LogisticRegression()
# rbm = BernoulliRBM(random_state=0, verbose=True)

# classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

# rbm.learning_rate = 0.06
# rbm.n_iter = 50
# rbm.n_components = 100
# logistic.C = 6000.0

print 'begin training'
# Training RBM-Logistic Pipeline
classifier = logistic
classifier.fit(train_pca, trainTarget)

print 'end'
print()
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(
        testTarget,
        classifier.predict(test_pca))))

	
# print("Fitting the classifier to the training set")
# t0 = time()
# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
# clf = clf.fit(train_pca, trainTarget)
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)

# print("Predicting n_imgs on the test set")
# t0 = time()
# y_pred = clf.predict(test_pca)
# print("done in %0.3fs" % (time() - t0))

# print(classification_report(testTarget, y_pred))