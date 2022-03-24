import numpy as np 
import matplotlib.pyplot as plt 
import sys 
from astropy.io import fits
import os
import display_quality 

path = os.getcwd()

"""
This file, produces plots where the binary classifiers are shown in the
dependence of the cutoff alpha. Therefore the array which the prediction of
the validation data is saved in is used. 
"""

def load(name):
    """
    Load the predction and truth for the validation data, which are both saved after training
    in prediction.npy
    :param name: Union(int, string) = name or number of the network, with which all information saved
    during training is saved under
    :return: Tuple(np.ndarray, np.ndarray)
    """
    solutions = np.load(path + f'/trained/prediction{name}.npy')
    sol_T = solutions.transpose(0,1,4,2,3)
    return sol_T[:, 0], sol_T[:, 1]


def check(truth, guess, cutoff):
    """
    calcuates the four categries of TP, TN, FP, and FN for a given cutoff, normed
    in a way that the sum over all four is 1.
    :param truth: np.ndarray = ground truth for the validation data
    :param guess: np.ndarray = prediction of the network for the validation data
    :param cutoff: float = cutoff alpha at which a point of the prediction is flagged 
    :return: Tuple(float, float, float, float)
    """
    truth = np.array(truth, dtype=np.bool_)
    tp = np.sum(np.logical_and(truth, guess > cutoff))/np.prod(guess.shape)
    tn = np.sum(np.logical_and(~truth, ~(guess > cutoff)))/np.prod(guess.shape)
    fp = np.sum(np.logical_and(~truth, guess > cutoff))/np.prod(guess.shape)
    fn = np.sum(np.logical_and(truth, ~(guess > cutoff)))/np.prod(guess.shape)

    return tp, tn, fp, fn


if __name__ == '__main__':
    model_name = 53
    guess, sol = load(model_name)
    # produce list of alphas, to calculate the binaray classifiers for
    cut = np.arange(0, 7.5, 0.2)
    truth = np.zeros((len(cut), 4))
    # calculates the new binary classifiers for each alpha
    for i in range(len(cut)):
        truth[i] = check(sol, guess, cut[i])

    print(truth)
    # plot the version with a larger alpha range for the phase network
    display_quality.plot_all_in_one(truth.transpose(1, 0), 'Phase Network', cut)
    # plot the default version for model 51, as this is the Amplitude Network
    display_quality.plot_all_in_one(display_quality.get_array(51), 'Amplitude Network')
