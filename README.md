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


# Dependencies
- NumPy: Perform vectorized operation on large arrays
- OpenCV: Implementation of famous image processing algorithms
- SciPy: Signal processing to extract peaks and valleys after chip thickness measurement
- Scikit-image: Line rasterization algorithm used for ray tracing to detect the inside contour of the chip
- Matplotlib: Generate output plots
- Imageio: Read/write video and image files
- PyAv: Python binding to the FFmpeg library

## Install dependencies in a virtual environment with conda
```shell
# create and activate the virtual environment
conda create -n machining-chip-analysis
conda activate machining-chip-analysis

# install dependencies inside the virtual environment
TODO: make a definitive list of dependencies

# read image files: imageio
# read video files: imageio
# write video files: scikit-video
conda install numpy opencv -c conda-forge
conda install matplotlib scikit-video imageio av -c conda-forge
conda install scipy scikit-image -c anaconda

# read image files: scikit-image
# read video files: scikit-video
# write video files: scikit-video
conda install "numpy<1.24" opencv -c conda-forge
conda install matplotlib scikit-video -c conda-forge
conda install scipy scikit-image -c anaconda
```

## Install dependencies with pip
```shell
pip install numpy scipy scikit-image matplotlib opencv-python
```


# tests de portabilité
LINUX:
- main version:
  - build: OK
  - no rendering:
  - image inputs: OK
  - video inputs: OK
- alternative version:
  - build: OK
  - no rendering: OK
  - image inputs: OK
  - video inputs: OK

WINDOWS:
- main version:
  - build:
  - image inputs:
  - video inputs:
- alternative version:
  - build:
  - image inputs:
  - video inputs:
