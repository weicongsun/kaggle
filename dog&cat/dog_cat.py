from scipy.misc  import imread, imresize
import numpy as np 
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def load_imgs(folder_path='./tmp/', slices):
	filepaths = [join(folder_path, f) for f in listdir(folder_path)]
	n_imgs = len(filepaths)
	data = np.zeros((n_imgs, 64*64)
	target = np.zeros(n_imgs)
	for i, f in enumerate(filepaths):
		print f
		img = np.asarray(imread(f), dtype=np.float32)
		img = imresize(img, (64,64))	
		img = img.mean(2)
		img /= 255.0
		data[i, ...] = img.ravel()

		if f.startswith('cat') :
			target[i] = 0
		else:
			target[i] = 1



	indices = np.arrange(n_imgs)	
	np.random.RandomState(42).shuffle(indices)
	data, target = data[indices], target[indices]
	return data, target

trainData, trainTarget = load_data.load('./train/')
testData, testTarget = load_data.load('./test/')

n_components =  100
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(trainData)

eigenfaces = pca.commonents_.reshape((n_components, 64, 64))
train_pca = pca.transform(trainData)
test_pca = pca.transform(testData)

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(train_pca, trainTarget)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))






