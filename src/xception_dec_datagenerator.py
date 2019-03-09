import numpy as np
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class XceptionDataGenerator(keras.utils.Sequence):
	def __init__(self, image_paths, labels, base_xception_model, base_xception_model_input_size, batch_size=32,
	             dim=(2048,), n_channels=1, n_classes=2, shuffle=True):
		# 'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.base_xception_model = base_xception_model
		self.base_xception_model_input_size = base_xception_model_input_size
		self.image_paths = image_paths
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		# 'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.image_paths))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, image_paths_temp):
		# 'Generates data containing batch_size samples'
		# X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty(self.batch_size, dtype=int)

		# Generate data
		for i, image_path in enumerate(image_paths_temp):
			# Store sample
			X[i,] = self.base_xception_model.predict(
				img_to_array(load_img(image_path, target_size=self.base_xception_model_input_size)))

			# Store class
			y[i] = self.labels[image_path]

		return X, y if self.n_classes == 2 else keras.utils.to_categorical(y, num_classes=self.n_classes)

	def __len__(self):
		# 'Denotes the number of batches per epoch'
		return int(np.floor(len(self.image_paths) / self.batch_size))

	def __getitem__(self, index):
		# 'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		# Find list of IDs
		image_paths_temp = [self.image_paths[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(image_paths_temp)

		return X, y


def main():
	print('Hello World')


if __name__ == '__main__':
	main()
