ChipAnalyser - Image-Processing for Machining Chip Analysis
===========================================================

# Overview
ChipAnalyser is an image-processing program designed to analyze machining chips from a series of images or a video file. It measures the following characteristics of the chip in each image:
- Tool-chip contact length
- Average peak thickness
- Average valley thickness
Measurement results are saved in a CSV file. Additionally, the program can generate graphical renderings to illustrate the measurement process.

[![Watch the demo video](https://raw.githubusercontent.com/VictorLaugt/MachiningChipAnalyser/master/demo_video/thumbnail.jpg)](https://raw.githubusercontent.com/VictorLaugt/MachiningChipAnalyser/master/demo_video/demo.mp4)


# Usage
```shell
python3 ChipAnalyser [-S] -i INPUT_IMAGES -o OUTPUT_DIRECTORY [-s SCALE] [-b BATCH_SIZE] [-r]
```
`-S, --silent`: if this option is enabled, the program does not display progress bar

`-i INPUT_IMAGES`: path to the directory containing the images to be analyzed, or path to the video whose images are to be analyzed.

`-o OUTPUT_DIRECTORY`: path to the directory where output files will be written

`-s SCALE`: length of a pixel in µm (1 by default).

`-b BATCH_SIZE`: size of the input image batches (10 by default).

`-r`: if this option is enabled, the program produces graphical renderings of the feature extractions, else, no rendering is done and the program simply extracts the features from the inputs.


# Dependencies installation
## With Anaconda, in a virtual environment
```shell
conda create -n machining-chip-analysis
conda activate machining-chip-analysis
conda install numpy scipy scikit-image matplotlib opencv
```
## With pip
```shell
pip install numpy scipy scikit-image matplotlib opencv-python
```
