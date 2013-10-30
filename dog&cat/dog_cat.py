from scipy.misc  import imread, imresize
import numpy as np 

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
		data[i, ...] = img

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


