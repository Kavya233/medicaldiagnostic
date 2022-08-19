from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def process(train,test):
	classifier = Sequential()
	classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
	classifier.add(MaxPooling2D(pool_size=(2, 2)))
	classifier.add(Conv2D(32, (3, 3), activation='relu'))
	classifier.add(MaxPooling2D(pool_size=(2, 2)))
	classifier.add(Flatten())
	classifier.add(Dense(units=128, activation='relu'))
	classifier.add(Dense(units=3, activation='sigmoid'))
	classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1. / 255)
	training_set = train_datagen.flow_from_directory(train,target_size=(64, 64),batch_size=64,class_mode='categorical')
	test_set = test_datagen.flow_from_directory(test,target_size=(64, 64),batch_size=32,class_mode='categorical')
	history=classifier.fit_generator(training_set,steps_per_epoch=100,epochs=5,validation_data=test_set,validation_steps = 715 / 64 )
	classifier.save('tumormodel.h5');
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	#plt.title('model accuracy')
	plt.title('Training and validation accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('results/Training and validation accuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	#plt.title('model loss')
	plt.title('Training and validation Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('results/Training and validation Loss.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
