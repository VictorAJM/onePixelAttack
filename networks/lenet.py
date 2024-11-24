import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from networks.train_plot import PlotLearning

# Code taken from https://github.com/BIGBALLON/cifar-10-cnn
class LeNet:
  def __init__(self, epochs = 200, batch_size = 128, load_weights=True):
    self.name = 'lenet'
    self.model_filename = 'networks/models/lenet.keras'
    self.num_classes = 10
    self.input_shape = 32, 32, 3
    self.batch_size = batch_size
    self.epochs = epochs
    self.iterations = 391
    self.weight_decay = 0.0001
    self.log_filepath = r'networks/models/lenet/'
    
    if load_weights:
      try:
        self._model = load_model(self.model_filename)
        print('Successfully loaded', self.name)
      except (ImportError, ValueError, OSError):
        print('Failed to load', self.name)
    
  def count_params(self):
    return self._model.count_params()
  
  def color_preprocessing(self, x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
      x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) /std[i]
      x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) /std[i]
    return x_train, x_test
  
  def build_model(self):
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(32, 32, 3)))
    # 15 Max Pool Layer
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # 13 Conv Layer
    model.add(Conv2D(13, kernel_size=(3,3), padding='valid', activation='relu'))
    # 6 Max Pool Layer
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    # Flatten the Layer for transitioning to the Fully Connected Layers
    model.add(Flatten())
    # 120 Fully Connected Layer
    model.add(Dense(120, activation='relu'))
    # 84 Fully Connected Layer
    model.add(Dense(86, activation='relu'))
    # 10 Output
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(learning_rate=.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
  
  def scheduler(epoch):
    if epoch < 100:
      return 0.01
    if epoch < 150:
      return 0.005
    return 0.001
  
  def train(self):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, self.num_classes)
    y_test = keras.utils.to_categorical(y_test, self.num_classes)
    
    x_train, x_test = self.color_preprocessing(x_train, x_test)
    
    model = self.build_model()
    print(model.summary())
    
    checkpoint = ModelCheckpoint( filepath=self.model_filename, 
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='auto')
    plot_callback = PlotLearning()
    tb_cb = TensorBoard(log_dir=self.log_filepath, histogram_freq = 0)
    cbks = [checkpoint, plot_callback, tb_cb]
    
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant',cval=0.)
    datagen.fit(x_train)
    
    model.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                        steps_per_epoch = self.iterations,
                        epochs = self.epochs,
                        callbacks=[checkpoint, tb_cb],
                        validation_data=(x_test,y_test))
    
    model.save(self.model_filename)
    self._model = model
    
  def color_process(self, imgs):
    if imgs.ndim < 4:
      imgs = np.array([imgs])
    imgs = imgs.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for img in imgs:
      for i in range(3):
        img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
    return imgs
  
  def predict(self, img):
    processed = self.color_process(img)
    return self._model.predict(processed, batch_size = self.batch_size,verbose=0)
  
  def predict_one(self, img):
    return self.predict(img)[0]
  
  def accuracy(self):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, self.num_classes)
    y_test = keras.utils.to_categorical(y_test, self.num_classes)

    # color preprocessing
    x_train, x_test = self.color_preprocessing(x_train, x_test)

    return self._model.evaluate(x_test, y_test, verbose=0)[1]
