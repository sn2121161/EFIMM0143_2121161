# coding:utf-8


from keras.layers import Input, Convolution2D, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD , Adam
from keras.initializations import normal
from keras.utils.visualize_util import plot# apply a 3x3 convolution with 64 output filters on a 256x256 image:

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th',input_shape=(3, 256, 256)))# now
model.output_shape == (None, 64, 256, 256)# add a 3x3 convolution on top, with 32 output filters:
model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='th'))# now
adam = Adam(lr=1e-6)
model.compile(loss='mse',optimizer=adam)
print("We finish building the model")

plot(model, to_file='model1.png', show_shapes=True)
