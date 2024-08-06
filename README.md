# Overview
This repository contains the code modules developed for high-throughput (HTP) screening of biotic-abiotic materials (as defined in the DMREF grant) to test for specific performance targets. The modules delivers additional metrics to build out a material fingerprint for any materials tested through the code contained in this repository. After screening the files provided, the program will output a file titled summary.csv in the highest level directory you provide, or will output in the same directory as the specific file you examine. Additionally, graphs to explain the evaluation of the HTP modules are written into the same subdirectory as the original file examined.
This code runs in Python and requires Python 3 or higher.

# How to Run This Code
To run this code, copy this repository into the directory for which you wish to apply the HTP screening algorithms to. If you have Github enabled, this can be completed through the command ```git clone https://github.com/asriram31/htp-screening.git```. Edit the .yaml configuration file (as described below) to adjust which channels and modules you would like to run through the code.
Using the command line, type in ```python htp-screening/Scripts/main.py directory_name``` to run the code on the directory you wish to examine.

## Required Modules
This code was developed in Python 3.12 and requires Python 3.12 or higher to operate. Additionally, the following packages are required to run this code.
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Numpy](https://pypi.org/project/numpy/)
- [ImageIO](https://pypi.org/project/imageio/)
- [Scikit Image](https://pypi.org/project/scikit-image/)
- [ND2Reader](https://pypi.org/project/nd2reader/)
- [Pims](https://pypi.org/project/PIMS/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [Scipy](https://pypi.org/project/scipy/)

## Adjusting Parameters
By default, this program will run through all video in a directory, running through each channel for a video and running all three modules. Additionally, the program by default will accept dim images, and provide the output graphs listed above. To adjust the settings, you can adjust the config,yaml file provided in the Scripts directory or create your own. Following the format of the config.yaml file, the reader category determines settings for how the program screens the videos provided, while the other categories determine the individual settings for each of the modules. 
To set the program to only run through one channel, for the parameter channel, replace the -1 with the number of the channel you wish to examine. To reset the program back to running all channels, replace the channel number with -1 once again.
