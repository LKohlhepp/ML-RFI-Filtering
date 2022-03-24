import numpy as np 
from astropy.io import fits 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
This programm allows plotting of the network predicted flag table and the ground truth flag table,
alongside the amplitdue of the data.
"""


def load_data(path, log_amp=True, from_fits=False):
    """
    Loads the data, as well as the network flags form the fits-file generated from the mini-pipeline.
    :param path: string = location of the file
    :param log_amp: bool = if the logarithm of the amplitdue is used for displaying. 
    :param from_fits: bool = if the data is loaded from a fits-file or np.array
    :return: Tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    try:
        if from_fits:
            with fits.open(path) as file:
                data_in = file[0].data['DATA'].reshape(-1, 435, 256, 2, 3)
                flag = data_in[:, :, :, 0, 2]
                data_in = data_in[:, :, :, :, 0] + 1j * data_in[:, :, :, :, 1]
        else:
            data_in = np.load(path)
    except FileNotFoundError:
        print(f'File "{path}" not found: Aborting!')
        exit()
    amps = np.abs((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
    phase = np.angle((data_in[:, :, :, 0:1] + data_in[:,  :, :, 1:2])/2)
    if log_amp:
        amps = np.log(amps)
    return amps, phase, flag

def make_plot(amps, flag_net, flag_tf, name):
    """
    Wrapper function that creates the actual plot function. Here the amplitude, both flag tables, and
    the name of the plot are adjusted, so that a function can be defined which has only the time step as a
    parameter.
    :param amps: np.ndarray(dtype=float, shape=(int, int, int)) = logarithm of the amplitdue
    :param flag_net: np.ndarray(dtype=bool, shape=(int, int, int)) = flag table predicted by the network
    :param flag_tf: np.ndarray(dtype=bool, shape=(int, int, int)) = flag table used as ground truth
    :return: function(int)
    """
    def plot(time):
        """
        Plots and saves graph showing the logarithm of the amplitdue and both flag tables, alongside colorbars 
        for one time step of the given data.
        :param time: int = time step to plot
        :return: None
        """

        # define subplots, the first and last for the colorbars are smaller and 
        #the others are for the amps and flags
        fig, axs = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1, 10, 10, 10, 1]}, figsize=(16, 8))
        p = [None] *3
        # amps are plotted in another color than the flags
        p[0] = axs[1].imshow(amps[time], cmap='afmhot', aspect='auto')

        plt.colorbar(p[0], cax=axs[0], orientation='vertical', label='Logarithm of the Amplitude')
        
        p[1] = axs[2].imshow(flag_net[time], cmap='viridis', aspect='auto')
        p[2] = axs[3].imshow(flag_tf[time], cmap='viridis', aspect='auto')
        
        # only one colorbar for both flag tables as they use the same color coding
        plt.colorbar(p[2], cax=axs[4], orientation='vertical', label='flag: 1 = flag; 0 = no-flag')

        # label all 3 plots
        for i in range(3):
            axs[1+i].set_ylabel('Baselines')
            axs[1+i].set_xlabel('Channels')
        axs[1].set_title('Amplitude')
        axs[2].set_title('Network Flagging')
        axs[3].set_title('Tfcrop Flagging')

        fig.suptitle(f'Flagging of Time-step {time}', fontsize=20)

        # fix layout; important!
        fig.tight_layout()
        plt.savefig(f'flagtable{name}-{time}.png', dpi=300)
        plt.close()
    return plot


if __name__ == '__main__':
    amps, phase, flag_net = load_data('data/28_029_15MAY2015_8S.LTA_RRLL.RRLLFITSflagged.fits', from_fits=True)
    amps = amps.reshape(-1, 435, 256)

    flag_tf = np.load('data/flag_ant89_r.npy') + np.load('data/flag_tfcrop_r.npy')
    flag_tf = np.sum(flag_tf, axis=-1)

    print(f'Data loaded with {amps.shape[0]} time steps')


    plot15 = make_plot(amps, np.array(-flag_net+1, np.bool_), np.array(flag_tf, np.bool_), '15')
    plot15(240)
    plot15(506)


