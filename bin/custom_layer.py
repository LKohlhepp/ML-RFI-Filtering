import numpy as np 
import tensorflow as tf 
from tensorflow import keras 


def build_Conv_3D_padd_2D(kernel, input_shape=None, filters=1, activation=None, kernel_regularizer=None, use_bias=False, name=None):
    """
    Creates and returns a 3D-convolution layer with periodic padding, that was created as a keras.Sequental object.
    :param kernel: Tupel[int, int, int] = kernel of the 3D-Conv layer, only odd kernel sizes in each direction are allowed
    :param input_shape: Tupel[int, int, int, int, Union(int, None)] = Input shape of the 3D conv layer (deprecated)
    :param filters: int = filters of 3D-conv layer
    :param activation: Union(func, String) = activation function of the 3D-conv layer, is passed straight through, see activation for tf-layers
    :param kernel_regularizer: tf.regularizer = regularizer of the 3D-conv layer, is passed through, see regularizers for tf-layers
    :param use_bias: bool = decides if a bias is used, is passed through, see bias for tf-layers
    :param name: String = name of the layer, is passed through; keep None for automatic naming
    :return: keras.Sequential
    """

    # calculates amout of paddding done infront and behind the main body
    kernel_snip = (np.array(kernel)-1) //2
    return keras.Sequential([
        # padds in baseline direction
        keras.layers.Lambda(lambda tensor: tf.concat([tensor[:, :, tensor.shape[-3]-kernel_snip[1]:], tensor, tensor[:, :, :kernel_snip[1]]], -3)),
        # padds in channel direction
        keras.layers.Lambda(lambda tensor: tf.concat([tensor[:, :, :, tensor.shape[-2]-kernel_snip[2]:], tensor, tensor[:, :, :, :kernel_snip[2]]], -2)),
        # 3D-conv layer
        keras.layers.Conv3D(filters=filters, kernel_size=kernel, strides=(1, 1, 1), padding='valid', activation=activation, kernel_regularizer=kernel_regularizer, use_bias=use_bias)
    ], name=name)


def build_LogLayer():
    """
    Deprecated: Logs the amplitude, and amplitude only if amps and phases are given at once. Not compatible with loss
    :return: keras.layers.Lambda
    """
    log10 = 1/np.log(10)
    return keras.layers.Lambda(lambda tensor: tf.concat([tf.math.log(tensor[:, :, :, :, 0:1])*log10, tensor[:, :, :, :, 1:2], tf.math.log(tensor[:, :, :, :, 2:3])*log10, tensor[:, :, :, :, 3:4]], 4))



def build_NormMean(sigma=1):
    """
    Norms each tensor per input (not batch), with the equation:
        x_i = (x_i - <x>)/(std(x) * sigma),
    where <x> is the expeted value over one input, std(x) is the standard deviation of one input,
    x_i is one data point of the input, and sigma is the parameter sigma
    :param sigma: float = factor for scaling the norming convention
    :return: keras.layers.Lambda
    """
    norm = lambda tensor: (tensor-tf.math.reduce_mean(tensor))/(sigma*tf.math.reduce_std(tensor))
    return keras.layers.Lambda(lambda tensor: tf.map_fn(norm, tensor))


class NormalisationLayer(keras.layers.Layer):
    """
    Deprecated / Not implemented properly. Does the same as build_NormMean, but instead of returning a
    lambda layer tf.map_fn is used instead.
    :pram sigma: float = factor for scaling the norming convention
    :return: NormalisationLayer
    """
    def __init__(self, sigma):
        super(NormalisationLayer, self).__init__()
        self.sigma = sigma

    def build(self, input_shape):
        #self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])
        pass

    def call(self, input):
        return tf.map_fn(lambda elem: (elem-tf.math.reduce_mean(elem))/(self.sigma*tf.math.reduce_std(elem)), input)
    

# norms the amplitude features, while ignoring the angle features
# !!! DOES NOT keep the features in the same order !!!
def rip_and_norm(input_shape, name=None, sigma=1):
    """
    Normalisation of the amplitude filters only. Does not keep the filters in the same order, but goes from amp, ph, amp, ph, 
    to amp, amp, ph, ph.
    Used for the "initial"-style of network with 4 input filters, thus it is deprecated for the newest networks
    :param input_shape: Tupel[int, int, int, int, None] = Shape of the input following tf standards
    :param name: String = Name of the Model and thus layer in the bigger network
    :param sigma:
    :return: keras.Model
    """

    # create input tensor
    input_layer = keras.Input(input_shape)
    # split input tensor into two, one containing only amps one only phases
    split_amps = keras.layers.Lambda(lambda tensor: tf.concat([tensor[:, :, :, :, 0:1], tensor[:, :, :, :, 2:3]], axis=-1), trainable=False)(input_layer)
    split_angles = keras.layers.Lambda(lambda tensor: tf.concat([tensor[:, :, :, :, 1:2], tensor[:, :, :, :, 3:4]], axis=-1), trainable=False)(input_layer)

    # norm the amplitudes:
    # these different lines all do (roughly) the same, only enable one

    #normed = keras.layers.LayerNormalization(axis=[1, 2, 3], trainable=False)(split_amps)
    #normed = NormalisationLayer(1)(split_amps)
    normed = build_NormMean(sigma=sigma)(split_amps)

    # reunite the now normed amps with the phases into one tensor
    rejoin = keras.layers.Concatenate(axis=-1, trainable=False)([normed, split_angles])

    # build a model, that can be used as a layer itself from the input and output tensor.
    model = keras.Model(input_layer, rejoin, name=name)
    return model



