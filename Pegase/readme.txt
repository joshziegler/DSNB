The files in IMF and Scenarios provide the IMF input and example scenario files used with Pegase.3.0.1
to generate our results.
SSPs were generated using the default options, given one of the IMFs.

With a working build of Pegase.3.0.1, to generate the spectra, we ran the following commands from the
bin_dir directory:
./SSPs
    (follow prompts, identifying the appropriate IMF file, and then accepting the default option 
      until it runs.)
./spectra
./calibrate
    (only needs to be run once, and can be run at any point before running colors)
./colors
    (follow prompts, specifically identifying souce spectra based on target locations in 
      'scenarios.txt' files)

The provided scenario files can be used to generate spectra associated with the IMF in 'IMF_2.35.txt'.
In order to generate the spectra associated with a different IMF, replace the first line with the 
appropriate SSPs file (should match the IMF), and change the names of the output files.

We have also added the calibration file that we used in order to calculate the FIR luminosity.
To use, from a clean install of Pegase, replace the default file 'list_filters.txt' in the calib_dir 
directory with the one here, add the filter file 'FIR.txt' to the calib_dir directory, and run ./calib
from the bin_dir directory.
