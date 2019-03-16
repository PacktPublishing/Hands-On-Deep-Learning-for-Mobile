# Chapter 1: Basics of Deep Learning

Welcome to the code repo for Chapter 1 of the book. The code requires Jupyter notebook installed, along with TensorFlow. Standard libraries like `numpy`, `scipy`, `matplotlib` etc are also required. If you have followed the instructions from the appendix, then you should be all set!

## Data set
For this chapter, we will use the [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) data set. This data set will also be used in Chapter 5 later. `emnist-analysis.ipynb` will help you download and set the data up.
Since the data set is large, a contained version is provided in the `./data` directory.
Please note that if you want to manage large files, you will need to install `git-lfs` package.

## Setting up Python Environment
[Anaconda](https://www.anaconda.com/distribution/#download-section) for Python 3 is used to manage the dependencies for Python packages. Using a virtual environment is *highly* recommended. To setup the
development environment, please execute

`conda env create -n <your_env_name> -f conda_env.yml `

This should setup a new virtual environment with the exact same libraries and versions used in writing this book.

## Main Files

* `emnist_analysis.ipynb`: Code for analysing the data set and it's properties.
* `convert_emnist_to_png.py`: A utility script for reading the data set and creating individual image files by labels on the hard disk.
* `emnist_fc_net.ipynb` :This is the main code file for this chapter with definitions of the first deep learning model. 
