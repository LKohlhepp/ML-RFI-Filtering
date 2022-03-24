import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import custom_layer
import train_gen
import compare
import pickle

def log_amps(data):
    #data[:, :, :, 0] = np.log10(data[:, :, :, 0])
    #data[:, :, :, 2] = np.log10(data[:, :, :, 2])
    data = np.log(data)
    return data


def discard_angle(data):
    data[:, :, :, 1] = 0
    data[:, :, :, 3] = 0
    return data

# define and compile models

n_times = 9
sigmoid = tf.nn.sigmoid
kernel1 = 5
kernel_snip1 = 2

kernel2 = 3
kernel_snip2 = 1

l2 = tf.keras.regularizers.l2(1E-14)
kernel1 = 9
kernel_snip1 = 4
kernel2 = 5
kernel_snip2 = 2

# list of diffent models

model1 = keras.Sequential([
    custom_layer.rip_and_norm((n_times, 435, 256, 1)),
    custom_layer.build_Conv_3D_padd_2D((9, 9, 9), filters=32, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
    custom_layer.build_Conv_3D_padd_2D((1, 5, 5), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
    keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_regularizer=l2, use_bias=True)
    ])

model2 = keras.Sequential([
    keras.layers.InputLayer(input_shape=(n_times, 435, 256, 1)),
    #keras.layers.LayerNormalization(input_shape=(n_times, 435, 256, 1), trainable=False),
    custom_layer.build_Conv_3D_padd_2D((9, 5, 5), filters=32, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
    custom_layer.build_Conv_3D_padd_2D((1, 3, 3), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
    keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_regularizer=l2, use_bias=True)
    ])

def m3():
    input_layer = keras.Input((n_times, 717, 256, 1))
    layer1 = custom_layer.build_Conv_3D_padd_2D((9, 5, 5), filters=32, activation=sigmoid, kernel_regularizer=l2, use_bias=True)(input_layer)
    branch1 = custom_layer.build_Conv_3D_padd_2D((1, 5, 5), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True)(layer1)
    branch2 = custom_layer.build_Conv_3D_padd_2D((1, 3, 3), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True)(layer1)
    rejoin = keras.layers.Concatenate(axis=-1, trainable=False)([branch1, branch2])
    last = keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_regularizer=l2, use_bias=True)(rejoin)
    return keras.Model(input_layer, last)

model3 = m3()

model_mono = keras.Sequential([
    keras.layers.InputLayer(input_shape=(n_times, 717, 256, 1)),
    #keras.layers.LayerNormalization(input_shape=(n_times, 435, 256, 1), trainable=False),
    keras.layers.Conv3D(kernel_size=(9, 1, 1), filters=1, padding='valid')
    ])

# global training parameters
my_model = model3
batch_size = 4
val_batches = 11
epoch_batches = 96
epochs = 300
split = 0.1
patience = 10
log_amplitudes = True
train_abs = True
baselines = True


# network compilation and summary
opt1 = keras.optimizers.Adam(1E-7)
my_model.compile(optimizer=opt1, loss='mean_squared_error')
my_model.summary()


# load data
data_in = np.load('raw_data/data_complex.npy')#[:50]
flags = np.load('raw_data/flag_tfcrop_r.npy')
#flags = np.load('raw_data/flag_ant89_r.npy')
if train_abs:
    data_in = np.abs((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
else:
     data_in = np.angle((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
#print(data_in)
flags = np.array(np.array(flags[:, :, :, 0:1]) + np.array(flags[:, :, :, 1:2]), dtype=np.bool_) * 10 
if baselines:
    data_in = np.transpose(data_in, axes=(1, 0, 2, 3))
    flags = np.transpose(flags, axes=(1, 0, 2, 3))
#data_in = discard_angle(data_in)
if train_abs and log_amplitudes:
    data_in = log_amps(data_in)
print(data_in.shape)
print(flags.shape)
print(data_in[n_times-1:].shape)
#data_in[n_times-1:].reshape(713, 1, 435, 256, 2)


# build the generators from data
gen, valid_gen, comp_gen, truth_indizes = train_gen.build_generators(data_in, flags, split, 665, batch_size, n_times)


# define the early stopping 
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
# train
hist = my_model.fit(gen(), validation_data=valid_gen(), validation_steps=val_batches, epochs=epochs, steps_per_epoch=epoch_batches, callbacks=[callback])


# make quality evaluation (true positve, true negative, false postive, false negative content) and save all relevant data
np.save('t_index', truth_indizes)
quality = compare.evaluate_model(my_model, comp_gen, val_batches, batch_size)
names = ['tp', 'tn', 'fp', 'fn']
[print(quality[i], names[i]) for i in range(4)]
np.save('quality.npy', quality)

my_model.save("model01.h5")
with open('hist.pkl', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)
