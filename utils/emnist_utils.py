"""
Utilities file that provides functions for:
    - loading EMNIST data and labels
    - confusion matrices
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras  # for normalization
import gzip as gz
import matplotlib.pyplot as plt


# convenience function to read EMNIST data into numpy array
def read_emnist(images_path, labels_path):
    with gz.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gz.open(images_path,'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                        .reshape(length, 784) \
                        .reshape(length, 28, 28)
        features = features.astype(float)
        flip = features[:,:, ::-1,...]       # note that images are flipped
        features = np.rot90(flip, 1, (1,2))  # and rotated 90deg

    return features, labels


# Lets read the labels so that directories can be named appropriately
def map_emnist_labels(label_mappings="../chap1/data/emnist-bymerge-mapping.txt"):
    labels_dict = {}
    with open(label_mappings, 'rb') as f:
        # each row of the file has the label first and ascii code next
        for line in f:
            items = line.split()
            # note that data is in bytes, so need to convert
            labels_dict.update({int(items[0]): chr(int(items[1]))})
    return labels_dict


# convenience function to display a grid of random images from emnist
def display_emnist_images(features, labels, mapping):
    # Now, lets try and generate some sample images to see what the data looks like
    fig=plt.figure(figsize=(9, 9))  # show 8in X 8in image
    columns = 4  # 4 images per row
    rows = 5  # lay out images on 5 rows
    num_images = features.shape[0]   # total # of images in supplied data set
    for i in range(1, columns * rows + 1):
        img_id = np.random.randint(0, num_images)
        fig.add_subplot(rows, columns, i)
        img_data = features[img_id].squeeze()
        plt.title('Label: %d Char: %s' % ( labels[img_id], mapping[labels[img_id]]))
        plt.imshow(img_data, cmap='gray')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Generates the confusion matrix for classification problems using
# sklearn.metrics package. First input is the set of predicted labels
# and the second input is the ground truth labels.
def generate_confusion_matrix(predictions, truth):
    return confusion_matrix(predictions, truth)


# Convenience function to plot a confusion matrix
# This confusion matrix is returned from a call to generate_confusion_matrix()
# above. Labels input is a dictionary of code and label. See map_emnist_labels()
# for an example of how it is generated.
def plot_confusion_matrix(confusion, labels):
    num_labels = len(labels)
    fig, ax = plt.subplots(figsize=(15, 15))
    # Note: Normalization is required to balance the class imbalance
    # i.e., all labels may not have the same number of samples
    im = ax.imshow(keras.utils.normalize(confusion), cmap="YlGn")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.set_ylabel("Percentage of Samples", rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(num_labels))
    ax.set_yticks(np.arange(num_labels))
    # ... and label them with the respective list entries
    ax.set_xticklabels(list(labels.values()))  # NOTE: The axis labels are hard coded here
    ax.set_yticklabels(list(labels.values()))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(num_labels + 1)- 0.5, minor=True)
    ax.set_yticks(np.arange(num_labels + 1)- 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)

    fig.tight_layout()

    plt.show()
