import numpy as np
np.random.seed(1337) 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import  MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
import cv2
#from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=3, activation='sigmoid'))
    model.summary()
    return model

def process(path):
	img_width, img_height = 64, 64
	train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
	training_set = train_datagen.flow_from_directory(r'data\train',target_size=(64, 64),batch_size=64,class_mode='categorical')

	model = load_model('tumormodel.h5')
	classifier = create_model()

	#test_image = image.load_img(r'predict\L2.jpg', target_size = (64, 64))
	test_image = image.load_img(path, target_size = (64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	
	result = classifier.predict(test_image)
	
	print(result[0])
	re=result[0]
	re=[]
	for b in result[0]:
		re.append(float(b))
	print("re",re)
	re1=sorted(re)
	print("re1",re1)
	s=re1[2]
	print(s)
	cla=0
	for i in range(0,3):
		print(re[i])
		if re[i]==s:
			cla=i+1
			print(cla)
	return cla

