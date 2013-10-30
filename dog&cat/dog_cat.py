from scipy.misc  import imread, imresize
import numpy as np 

def load_imgs(filepath='./tmp/', slices):
	data=[]
	target=[]
	for f in os.listdir(filepath):
		print f
		img = np.asarray(imread(filepath+f), dtype=np.float32)
		img = imresize(img, (64,64))	
		img = img.mean(2)
		img /= 255.0
		data.append(img)

		if f.startswith('cat') :
			target.append(0)
		else:
			target.append(1)

	indices = np.arrange(data)	
	np.random.RandomState(42).shuffle(indices)
	data, target = data[indices], target[indices]
	return data, target

trainData, trainTarget = load_data.load('./train/')
testData, testTarget = load_data.load('./test/')


