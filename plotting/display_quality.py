import numpy as np 
import matplotlib.pyplot as plt 
import sys 


"""
This file can be used to create plots where the binary classifiers are shown in the
dependence of the cutoff alpha. 

For this already computed arrays with the four categories of TP, TN, FP, and FN
are required. These are generated at the end of training. Otherwise this file
is called by redo_quality.py which calculates these properies anew.
"""


# Definitions for the binary classifiers (usable with np.ndarrays)

def sensitivity(qual_array):
    """
    Calculates the sensitivity or true positive rate:
        sens = TP / (TP + FN)
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :return: np.ndarray(dtype=float, shape=(int, ))
    """
    return qual_array[0]/(qual_array[0]+qual_array[3])


def specifity(qual_array):
    """
    Calculates the specifity or true negative rate:
        spec = TN / (TN + FP)
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :return: np.ndarray(dtype=float, shape=(int, ))
    """
    return qual_array[1]/(qual_array[1]+qual_array[2])


def precision(qual_array):
    """
    Calculates the precision or positive prediction value:
        prec = TP / (TP + FP)
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :return: np.ndarray(dtype=float, shape=(int, ))
    """
    return qual_array[0]/(qual_array[0]+qual_array[2])


def neg_pred_value(qual_array):
    """
    Calculates the negative prediction value:
        npv = TN / (TN + FN)
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :return: np.ndarray(dtype=float, shape=(int, ))
    """
    return qual_array[1]/(qual_array[1]+qual_array[3])


def accuracy(qual_array):
    """
    Calculates the accuracy:
        acc = TP + TN
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :return: np.ndarray(dtype=float, shape=(int, ))
    """
    return qual_array[0]+qual_array[1]


def get_array(model_name):
    """
    Load array with classifiers from disc by model name
    :param model_name: string = name of the model (not path)
    :return: np.ndarray(dtype=float, shape=(4, int))
    """
    return np.load(f'trained/quality{model_name}.npy')


def plot_all_in_one(qual_array, model_name, cutoffs=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], linewidth=0.5):
    """
    Plot all binary classifiers into one graph and save the graph to disc.
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :param model_name: string = name of the model. Used for titeling the graph.
    :param cutoffs: List = same length as qual_array.shape[1], alphas correponding to the categories in qual_array
    :param linewidth: float = with of the connecting line in the plot.
    :return: None
    """ 

    # sensitvity
    plt.plot(cutoffs, sensitivity(qual_array), marker='+', linewidth=linewidth, label='sensitivity')
    # specifity
    plt.plot(cutoffs, specifity(qual_array), marker='x', linewidth=linewidth, label='specifity')
    # precision
    plt.plot(cutoffs, precision(qual_array), marker='.', linewidth=linewidth, label='precision')
    # negativ prediction
    plt.plot(cutoffs, neg_pred_value(qual_array), marker='^', linewidth=linewidth, label='negative prediction rate')
    # accuracy
    plt.plot(cutoffs, accuracy(qual_array), marker='v', linewidth=linewidth, label='accuracy')

    # labels
    plt.ylabel('value of quality measurement')
    plt.xlabel('cutoff value $\\alpha$')
    plt.title(f'{model_name}')
    plt.legend()
    plt.savefig(f'qual_plot{model_name}')
    plt.close()
    #plt.show()


def plot_all_in_one_invert(qual_array, model_name, cutoffs=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], linewidth=0.5): 
    """
    Plot all binary classifiers into one graph and save the graph to disc. Ins6tead of plotting normally 1 - the classifiers is plotted into a lin-log plot.
    This has to an effect, that classifiers close to 1 are better resolved.
    :param qual_array: np.ndarray(dtype=float, shape=(4, int)) = array with the four categories x n with TP, TN, FP, FN (in that order)
    :param model_name: string = name of the model. Used for titeling the graph.
    :param cutoffs: List = same length as qual_array.shape[1], alphas correponding to the categories in qual_array
    :param linewidth: float = with of the connecting line in the plot.
    :return: None
    """

    # sensitvity
    plt.plot(cutoffs, 1-sensitivity(qual_array), marker='+', linewidth=linewidth, label='sensitivity')
    # specifity
    plt.plot(cutoffs, 1-specifity(qual_array), marker='x', linewidth=linewidth, label='specifity')
    # precision
    plt.plot(cutoffs, 1-precision(qual_array), marker='.', linewidth=linewidth, label='precision')
    # negativ prediction
    plt.plot(cutoffs, 1-neg_pred_value(qual_array), marker='^', linewidth=linewidth, label='negative prediction rate')
    # accuracy
    plt.plot(cutoffs, 1-accuracy(qual_array), marker='v', linewidth=linewidth, label='accuracy')

    # labels
    plt.yscale('log')
    plt.ylabel('value of quality measurement')
    plt.xlabel('cutoff value')
    plt.title(f'inverted quality (1-x) of model{model_name}')
    plt.legend()
    plt.savefig(f'qual_plot_inv{model_name}')
    plt.close()
    #plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    print(f'displaying qualitys for model{model_name}')
    plot_all_in_one(get_array(model_name), model_name)
    plot_all_in_one_invert(get_array(model_name), model_name)
