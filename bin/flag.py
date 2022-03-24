import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession 

import custom_layer
import pickle
import sys

from astropy.io import fits
import os

from tqdm import tqdm


def log_amps(data):
    data = np.log(data)
    return data


# define information for models that will be loaded in

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


n_times = 9


def load_data(path, log_amp=True, from_fits=False):
    """
    Load the data to predict flags for form a fits file or np.ndarray (that is saved on the disk)
    :param path: String = relative path, where the file is located
    :param log_amp: bool = if the logarithm needs to be applied to the amplitude (can have a signifcant impact on performace)
    :param from_fits: bool = if False data is expected to come form a saved np.ndarray if True from a .fits file (of GMRT telescope)
    :return: Tuple(np.ndarray, np.ndarray)
    """
    try:
        if from_fits:
            with fits.open(path) as file:
                data_in = file[0].data['DATA'].reshape(-1, 435, 256, 2, 3)
                data_in = data_in[:, :, :, :, 0] + 1j * data_in[:, :, :, :, 1]
        else:
            data_in = np.load(path)
    except FileNotFoundError:
        print(f'File "{path}" not found: Aborting!')
        exit()
    amps = np.abs((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
    phase = np.angle((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
    if log_amp:
        amps = log_amps(amps)
    return amps, phase


def load_amp():
    """
    Load the model that will flag RFI using the amplitudes
    :return: keras.Model
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_times, 435, 256, 1)),
        custom_layer.build_Conv_3D_padd_2D((9, 5, 5), filters=32, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
        custom_layer.build_Conv_3D_padd_2D((1, 3, 3), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
        keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_regularizer=l2, use_bias=True)
        ])

    model.load_weights('model/model51.h5')
    return model


def load_angle():
    """
    Load the model that will flag dead antennae using the phases
    :return: keras.Model
    """
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_times, 435, 256, 1)),
        custom_layer.build_Conv_3D_padd_2D((9, 5, 5), filters=32, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
        custom_layer.build_Conv_3D_padd_2D((1, 3, 3), filters=128, activation=sigmoid, kernel_regularizer=l2, use_bias=True),
        keras.layers.Conv3D(filters=1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', kernel_regularizer=l2, use_bias=True)
        ])

    model.load_weights('model/model53.h5')
    return model


def predict(model, data, cutoff, desc=None):
    """
    Predicts the flags for each point using the network and given cutoff value. Batchsize is always 1.
    :param model: keras.Model = the model to predict
    :param data: np.ndarray = the data to predict the flags for
    :param cutoff: float = if the value is exteeded by the networks prediction a point is flagged
    :param desc: string = displayed name in the progressbar for the process
    :return: np.ndarray
    """
    prediction = np.ones(shape=data.shape, dtype=np.bool_)
    for i in tqdm(range(len(data)-9), desc=desc):
        prediction[i+9] = (model.predict(data[i:i+9].reshape(1, 9, 435, 256, 1)) > cutoff).reshape(1, 435, 256, 1)
    return prediction


def save_in_fits(path, flag):
    """
    Save flags in the casa format into a already existing fits file.
    :param path: string = path of the existing file
    :param flag: np.ndarray = flag table to convert and save (0 is no-flag, 1 is flag)
    :return: None
    """
    with fits.open(path) as file:
        # shape without last dimension (would be 3 in that case)
        s = file[0].data['DATA'].shape[:-1]
        # replace True and False with -1 and 1 and write it into the flag place of the fits file
        file[0].data['DATA'][tuple([slice(0, e, 1) for e in s]) + (-1, )] = np.where(flag.reshape(s), -1.0, 1.0)

        # write to disk
        file.writeto(path[:-5] + 'flagged' + '.fits', overwrite=True)


def main(path):
    """
    Main routine of loading fits-file from path, estimating the flags with suitable cutoffs (same as in the thesis), 
    and saving the given flags in the given fits file (but renaming it to a flagged version)
    :param path: string = relative path of the fits-file to flag
    :return: None
    """
    print(f'loading file {path}')
    amps, phase = load_data(path, from_fits=True)
    print(f'file loaded sucessfully')
    amp_flag = predict(load_amp(), amps, 0.25, desc='flagging amplitudes')
    print(f'{np.sum(amp_flag)/np.prod(amp_flag.shape)*100}% of data points flagged by amplitudes')


    ph_flag = predict(load_angle(), phase, 2.5, desc='flagging phases')
    print(f'{np.sum(ph_flag)/np.prod(ph_flag.shape)*100}% of data points flagged by phases')


    flag = np.logical_or(ph_flag, amp_flag)

    print(f'{np.sum(flag)/np.prod(flag.shape)*100}% of data points flagged in total')
    print('writing to disk')
    flag = np.concatenate((flag, flag), axis=-1)
    save_in_fits(path, flag)
    print('Done')


if __name__ == '__main__':
    # if the program is started as main program, read the command line arguemnts
    # to get the name of the fits-file
    if len(sys.argv) > 1:
        # assume the fits-file is the data folder
        path = 'data/' + sys.argv[1]
    else:
        print('No path specified: Aborting!')
        exit()

    main(path)
