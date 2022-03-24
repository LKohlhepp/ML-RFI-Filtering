# Plotting

This folder contains some of the scripts used to dislay the quality of the network.
To archive a clearer structure the plotting is seperated form the training code
in this realese. This was not the case in the original creation. The does not 
change executablity of the files as they do not reference any of the files used
for training. But this means that the required data for plotting needs to be 
copied into this folder, or the references need to be changed. As these plotting
functions are only required if the network is retrained, a person requiring them
will likely change the files displayed anyways. 

## view.py

Creates a plot containing the logarithm of the amplitudes of the correlation
function, the networks flag table and the ground truth flag table, for one time
step. 

## display_quality.py

Plot the binary classifiers. But requires TP, TN, FP, and FN to be already 
clalculated, in the way the training is outputting them in the end (qualtiy.npy). 

## redo_quality.py

Also plots the binary classifiers (utilizes display_quality.py), but calculates
the categories from an array containing prediciton of the network and the 
corresponding ground truth (prediction.npy). 
