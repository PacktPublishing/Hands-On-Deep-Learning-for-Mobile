# Chapter 5: Convolutional Neural Networks

<<tbd add outline of chapter from Packt>>

ToC

## Computer Vision Applications
- get from medium article

## Convolutional Neural Networks for Classification
 -
overview of inspiration from visual cortex. Talk about translation invariance and scaling? (check in litreature of the two properties of cnn). 
 
 ### Filters and Convolutions
  - intuition behind filters like detecting lines or edges
  - show simple example of edge detection filter using a 3x3 applied to a sample EMNIST image
  - generalise to show we can learn different types of such filters
  - highlight that the key here is building better heirarchical features
  - explain convolution math to explain how to compute padding/no padding etc
  
  ### Pooling
  
  - why pooling? How does it help?
  - how to implement pooling in Keras etc
  
  ### EMNIST with CNNs
  
  - code up a simple extension to EMNIST code, introduce the utilities lib
  - show how trainable parameters have increased.
  - Ensure two convolutional layers atleast.
  - Plot confusion matrix and highlight any differences from chapt 1
 
 ### Visualizing Convolutions
 
  - Using TensorBoard
  - Migrating training real time plots to TensorBoard
  - visualizing output of each convolutional layer
  - show how weights are shifting in dense layers maybe?
  - Show case training and test errors
  
## Generalization through Regularization
The art of minimizing the difference between training and test error.  what is the difference between optimization and machine learning? Explain the purpose of regularization. Talk about constraining weights through L2 norms for dense networks. But say focus is on two specific methods used in CNN - Drop Out and BatchNorm.

### Drop Out
Intuitiuon behind drop out, forcing units to do more work on learning relationships, not depending on everything being present.
 Add to the code and see the difference. 
 Explain how to build code so that drop out is used only in training and not in inference.

### Batch Normalization
talk about that this is technically not a regularization method,but is often considered so. the intuition behind it, normalizing the weights in a time and space efficient manner. speeds up training and gets to better minima.
Add to code and see difference.
Talk about Deployment considerations.

## Mobile Optimization Part I
Show how to save and convert to mobile format. talk about building object detection to read lines of text from camera.

## Object Dection using CNN? (check what yolo uses)
see if we can isolate characters from a line of text.

### Detecting characters from a line of text from mobile Camera


### Smile Detection Selfie Taking App

  
  
  ## Questions
  1. You may have noticed that the number of trainable parameters have increased significantly when using a CNN. You may wonder if adding more dense layers which lead to same number of trainable parameters would lead to similar accuracy. This is a good exercise to try. Try two architectures:
       - Deeper: Add more layers rather than units in the layers
       - Wider: Add more units in the layers than adding more layers
       and see what you learn about how these networks perform. 