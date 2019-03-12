import os
import struct
import sys

from array import array
from os import path

import png
import numpy as np

base_dir = "./data/"  # training and test images will be stored there
train_dir = "train"   # all training images will be inside this directotyr
test_dir  = "test"    # all test images will be inside this directory
data_dir = "./data/"   # files will be loaded from this directory

train_file = "emnist-bymerge-train-images-idx3-ubyte"
train_labels = "emnist-bymerge-train-labels-idx1-ubyte"

test_file = "emnist-bymerge-test-images-idx3-ubyte"
test_labels = "emnist-bymerge-test-labels-idx1-ubyte"

label_mappings = "emnist-bymerge-mapping.txt"

# Lets read the labels so that directories can be named appropriately
# Lets read the labels so that directories can be named appropriately
def map_labels(data_dir, label_mappings):
    labels_dict = {}
    with open(data_dir + label_mappings, 'rb') as f:
        # each row of the file has the label first and ascii code next
        for line in f:
            items = line.split()
            # note that data is in bytes, so need to convert
            labels_dict.update({int(items[0]): chr(int(items[1]))})
    return labels_dict


def read(dataset = "train", path = data_dir):
    if dataset == "train":
        fname_lbl = data_dir + train_labels
        fname_img = data_dir + train_file
    elif dataset == "test":
        fname_lbl = data_dir + test_labels
        fname_img = data_dir + test_file
    else:
        print("Not supported")
        sys.exit()

    with open(fname_lbl, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        lbl = array("b", flbl.read())

    with open(fname_img, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = array("B", fimg.read())

    return lbl, img, size, rows, cols


# This method reads a data set, create directories to store inidividual files
def write_dataset(labels_dict, labels, data, size, rows, cols, output_dir):
    # create output directories, named as 'label-character'
    output_dirs = [
        path.join(output_dir, str(key) + '-' + labels_dict[key])
        for key in list(labels_dict.keys())
    ]
    # ensure the directories exist
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    # write data
    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".png")
        print("Saving: " + output_filename)
        with open(output_filename, "wb") as h:
            w = png.Writer(cols, rows, greyscale=True)
            data_i = [
                data[ (i*rows*cols + j*cols) : (i*rows*cols + (j+1)*cols) ]
                for j in range(rows)
            ]
            img_data = np.asarray(data_i, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(rows, cols)
            flipped = np.fliplr(img_data)  # images are horiontally flipped and
            img_data = np.rot90(flipped)   # rotated 90 degrees
            w.write(h, img_data.tolist())


if __name__ == "__main__":
    # lets get the labels and their corresponding characters. Then,
    # iterate over the set of images and convert and store them
    labels_dict = map_labels(data_dir, label_mappings)

    for dataset in ["train", "test"]:
        labels, data, size, rows, cols = read(dataset)  # assumes it is run from Chapter1 directory
        outdir = base_dir + dataset  # Place to store outputs
        write_dataset(labels_dict, labels, data, size, rows, cols,
                      outdir)
