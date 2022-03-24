# Pipeline

This folder contains the mini-pipline with which further obeservations can be
flagged using the network. The output is in a format that the radio astronomy
case can be understand. 

## Quick Guide 

Download this folder and copy a fits-file containing data from the GMRT telescope
in the data-folder. All paths given will read from this folder.

Now, flag.py can be executed from the commandline. As an additional argument the
fits-file, which is to be flagged, is requried.

Open this folder in the commandline and execute:
```
python flag.py observation.fits
```

where "observation.fits" is the name of your fits-file, located in the data-folder.
If you are only Linux, you might need to write `python3` instead, to use the 
correct interpreter. 

This is a mini-pipeline as such there are no protections against misspelling file
names. Furthermore, if the file name is missing the code will simply abort. 

After the code is run, which can take a moment as the pipeline is not optimized 
for speed, the flagged data is saved in the data folder, with "flagged" appended
to its name. In the example of "observation.fits", "observationflagged.fits" is 
saved in the data folder.

When the newly created fits-file is loaded into casa, the flags are already 
applied and no further "flagfile" needs to be specified when "importgmrt" is used
in casa. 