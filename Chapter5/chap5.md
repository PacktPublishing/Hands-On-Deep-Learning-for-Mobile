# Chapter 5: Convolutional Neural Networks

Computer vision is a very exciting and challenging field. While research on computer vision has been happening for decades, deep learning has driven the most recent advances providing order of magnitude improvements since 2012\. In fact, in 2015, deep learning based models surpassed human accuracy at classification tasks. Convolutional Neural Network or CNN is the main model architecture that has driven these improvements. It is widely used in the industry today. From tagging friends in photos to detecting fractures x-ray, there are lots of applications for computer vision. We will continue the EMNIST example from Chapter 1 and see how CNNs can increase accuracy. Specifically, this chapter will:

- Understand key computer vision application areas
- Explain the architecture of Convolutional Neural Networks and build examples
- Use regularization techniques to improve generalization of models
- Detect objects and landmarks in images
- // Include page numbers for ease of reference

# Technical Requirements

This chapter uses Python, TensorFlow 2.0, Jupyter Notebooks for building and training the models. Data files from training are reused from Chapter 1 Github location. All the code for this chapter is in

<github-repo>/Chapter5.</github-repo>

For the mobile application pieces, an iOS-based mobile app will be built. It will be developed using XCode running on MacOS 10.13.6 or above. The models developed will be converted for mobile use as demonstrated in previous chapters, using TensorFlow Lite. Further, MLKit, part of Firebase, will be used to put the trained model into the app. // list technologies and installations required here.

// Provide Github URL for the code in the chapter (setup instructions should be on the Github page). Create a Github folder named, "chX", where X is the chapter number. For example, ch1

# H1: Computer Vision Application Areas

Before building deep learning networks for computer vision tasks, it would useful to get an overview of key problems in this area. Main areas of interest in computer vision, amongst others, are:

- Image classification
- Object detection
- Landmark detection or keypoint detection
- Image labelling and captioning
- Super Resolution and Compression

## H2 Image Classification

Interest in application of deep learning in computer vision started with massive gains evidenced in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Figure 5-1 below shows the improvements with the arrival of deep learning in 2012 with the landmark AlexNet paper. ![Figure 5-1: Improvements in Top-5 Classification Rates](images/chap5-ilsvrc_error_rates.png "Figure 5-1: Improvements in ILSVRC Top-5 Classification Rates") Note the huge drop of about 10% between 2011 and 2012 results. This was the advent of deep learning techniques to image recognition. Prior to this, there were very small movements in accuracy. In 2015, CNNs surpassed human level accuracy of 5.1%.

> Info Box: Top-5 Classification Error rate: this measures whether the actual label for a given image was one of the top 5 labels predicted by the model.

In fact, this particular challenge has been retired since 2017 given that human level performance has been surpassed and replaced with object detection described below.

EMNIST is an image classification task. We will build a better image classification model using CNNs in this chapter.

## H2 Object Detection

This is a very important aspect of computer vision, being in focus due to interest and advances in self-driving or autonomous vehicles. Objective here is to identify key objects in an image. I t can be applied to videos as well, by parsing a frame at a time. An example is shown in Fig 5-2 below.

![Figure 5-2: Object Detection](images/chap5-1600px-Detected-with-YOLO--Schreibtisch-mit-Objekten.jpg "Figure 5-2: Object Detection")

ImageNet mentioned above has moved to this as the key benchmark for reporting progress now. This is a complex task as objects may be partially occluded and of different sizes and orientations. Further, for use in autonomous vehicles, the models used must be very fast. This would allow multiple frames of the video to be processed every second leading to higher reliability and safety of the vehicle.

We will build an object detection network later in this chapter to detect character bounding boxes in an image and hook it up to the EMNIST detection model to recognize them.

## H2: Landmark Detection or Keypoint Detection

A key point or a landmark marks an important part or feature of an image. If the image is of a face, then these can be different facial features like eyebrows, eyes, nose, lips etc. Fig 5-3 shows an example from Google's MLKit. Later in this chapter, we will try to build an app that takes selfies based on how wide a person is smiling. _TODO: See if it is feasible_

![Figure 5-3: Facial Landmark Detection](images/chap5-face_contours.svg "Figure 5-3: Facial Landmark Detection") (Image source: <https://firebase.google.com/docs/ml-kit/detect-faces>, CC 3.0 permissive license)

TODO: <https://webrtchacks.com/ml-kit-smile-detection/>

In other applications, human poses can also be marked. This is a key development used in motion capture techniques that are used in movies and animations. most notable movie that used this live motion capture technique is Avatar directed by James Cameron. Similar techniques were used by Microsoft Kinect to understand human poses to control video games.

![Figure 5-4: Author's image with pose](images/chap5-post-estimation.png "Figure 5-4: Pose estimation of Author cheering for you")

## H2: Image Captioning and Labeling

This is a very exciting development in the world of deep learning. It combines two areas, namely computer vision and natural language process (NLP). The task is to generate human readable labels given an image. It can be used to convert images into textual features and answer arbitrary questions about objects in the image. It can be used as an assistive technology for visually impaired persons. Applications are endless. This is a very complex field that is an active area of research. ![Figure 5-5: Image captioning from Google AI Blog](images/chap5-google-ai-blog-image-caption.png "Figure 5-5: Image captioning from Google AI Blog") Source: <https://ai.googleblog.com/2016/09/show-and-tell-image-captioning-open.html>

Chapter 7 of this book is devoted to this application area and will cover it in detail.

## H2: Super Resolution and Compression

Objective of these types of computer vision tasks are to either compress the image to a small size and then reconstruct the high resolution image on the different device, or to take an image and increase it's resolution and level of detail at the same time. There are additional use cases in converting an image with low detail, possibly shot during low light or night scene and add detail to it. In Chapter 8, we will unsupervised networks like Generative Adversarial Network (GAN) and Auto-encoders to perform this task. Image compression is especially useful in mobile settings as it can save bandwidth for communication while not comprising on quality.

Now that you have a good overview of the exciting application areas in computer vision, lets start working on the first of these - image classification. We will start with understanding Convolutional Neural Networks (CNN) architecture, and build a new classifier for EMNIST using this architecture.

## Convolutional Neural Networks for Classification

CNNs were inspired by the work of neurophysiologists David Hubel and Torsten Weisel, who eventually won a Nobel Prize. Their work put forth theories on how the _primary visual cortex_ functions in the mammalian brain. Signals that stimulate the retina result in simple pre-processing and then these signals are transferred to the primary visual cortex at the back of the head. These signals, as they move through layers of the brain and processed, follow the following structure:

- A sense of a 2-dimensional or spatial map is preserved about the image
- Simpler cells do simple detections like edges, curves, and colors. These cells work on small localized areas of the spatial map
- Complex cells aggregate inputs from the simpler cells to detect higher level concepts like faces. These cells have some resistance to the position of an object (like a face, or a car) in an image. As complex cells aggregate or _pool_ inputs, they can also become immune to changes in contrast or lighting.

CNNs emulate these key properties of the visual cortex in the following ways:

- Hierarchical representation: Recall from Chapter 1 that use of multiple layers in a deep learning network results in a hierarchical representation of features. This emulates the behaviour of layers of simple and complex cells. Complex cells, or units in the later layers of the network take inputs of neighbouring units for aggregation or _pooling_.
- Convolutions for locality sensitivity or sparse interactions: In a spatial map of the image, consider a random pixel. Chances are the pixels left, right, up, down and diagonally around that pixel are highly correlated to that pixel. This property of _locality sensitivity_ is very important. Conversely, it implies that connections of all units in a layer are not connected to every unit in the successive layer. This results in _sparse interactions_. Recall that in our first EMNIST model, all the pixels were fed in to the network with no notion of similarity between adjacent or close pixels. Structure of image data allows such locality sensitivity to be exploited through _convolutions_. Convolutions are described in more detail in the next section.
- Translation Invariance: This is a key property which allows the object to be detected to be located in different places in the image and yet be classified. Fig 5-6 shows an example of translation invariance to illustrate the concept. This property allows labels to be associated with entire images, instead of identifying the exact location of the object in the image. While this simplifies collection of data sets and training, it is crucial for the widespread success of CNNs. It enables the actual test images to differ from the training images in terms of the position of the object and still be able to detect it.

![Figure 5-6: Translation invariance](images/chap5-translation-invariance.png "Figure 5-6: Translation Invariance")

While we have been using 2 dimensional images as the main use case, these concepts generally apply to one dimensional data, such as an audio signal or a time series equally. It is important to see if a given problem has translation invariance and locality sensitivity properties. If it does, then CNNs would be a great fit for that problem.

Next two sections describe the concepts of convolutions and pooling in detail. These two are the core concepts of CNN architectures.

### Filters and Convolutions

In traditional computer vision prior to the advent of deep learning, filters were used to detect features like edges. These filters were hand crafted by scientists and engineers. Output of these features was used to feed into successive machine learning algorithms to aid in detection. Usually, these filters are 3x3 or 5x5 matrices that are _convolved_ with the image to produce a resultant image. The convolution example shall be illustrated with code. As an example, consider the Sobel Filter. This filter can be used to detect horizontal and vertical edges.

$$ G_{vertical} = \begin{bmatrix} -1 & 0 & +1 \ -2 & 0 & +2 \ -1 & 0 & +1 \end{bmatrix} * X $$

$$ G_{horizontal} = \begin{bmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ +1 & +2 & +1 \end{bmatrix} * X $$

- represents the convolution operation. $G_{vertical}$ represents a vertical edge detection filter while $G_{horizontal}$ can be used to detect horizontal edges. These filters are also called _kernels_, _masks_ or _convolution matrix_. _X_ contains the pixels of the input image represented as a 2D matrix. Fig 5-7 below shows the calculation on an example for one cell of the output matrix. The convolutional matrix or the kernel is moved over the input left to right and top to bottom as indicated by the arrows.

![Figure 5-7: Convolution calculation](images/chap5-convolution.png "Figure 5-7: Convolution example")

To see how this works, open up `convolution_arithmetic.ipynb` and run the following pieces of code. To see the impact of the Sobel filter defined above, we will use an image from the author's collection.

> Info box: This code example requires installation of Pillow image management library in Python. Installation in `conda` environment can be done through `$ conda install Pillow` . `scipy.signal` provides a `convolve2d` function which is used to implement this example. Example file used can be found in `githubrepo/Chapter5/images/chap5-tulip.jpg`.

```
# Load the image from the directory
tulip = Image.open("images/chap5-tulip.jpg")

#convert to gray scale image
tulip_grey = tulip.convert('L')
tulip_ar = np.array(tulip_grey)

# show the image
plt.imshow(tulip_grey)
```

These lines load the image, convert into grayscale and put it into a Numpy array. Next, the two filters shown above are defined.

```
kernel_1 = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])   # Vertical edge detection kernel / filter
kernel_2 = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])   # Horizontal edge detection kernel / filter
```

Now, to compute the result and visualize it, run the following piece of code.

```
from scipy.signal import convolve2d
out1 = convolve2d(tulip_ar, kernel_1)  # vertical filter output
out2 = convolve2d(tulip_ar, kernel_2)  # horizontal filter output
```

This will produce an output as shown in Fig 5-8.

![Figure 5-8: Result of Sobel filter for detecting edges using convolutions](images/chap5-tulip-edge-detection.png "Figure 5-8: Result of Sobel filter for detecting edges using convolutions")

Figure 5-8: Result of Sobel filter for detecting edges using convolutions

One of the challenges in computer vision earlier was hand-constructing these filters. In CNNs, these filter parameters are learnt automatically. Secondly, multiple filters are stacked on top of each other to create multiple outputs for the same input pixels. The idea here is to learn different types of representations in terms of simple and complex units through these stacked filters. Just like weights were learned in a fully connected network, the values for these filters are learned in CNN architectures.

There are some key hyper parameters used in convolutional layers that need to be chosen:

- _Filter depth_: This is the number of filters that are evaluated at a particular layer. For example, if the horizontal and vertical filters above are stacked together, then this would give a depth of 2.
- _Kernel size_: In the Sobel filter above, the kernel size was 3x3\. However, different kernel sizes like 5x5, 7x7 or 11x11 can also be used.
- _Stride size_: In the previous example, the filter was moved one pixel at time to the right and down. These values can be changed. Having larger stride size reduces computational and memory requirements and may help bring further apart features together in higher layers.
- _Padding_: It seems that kernel strides are limited by the bounds of the image. However, it is possible to pad the edges of the image to allow different stride sizes and resulting image sizes. There are two common settings for paddings:

  - Same: This padding ensures that the result has the same number of values as the input. This is shown in Fig 5-9.
  - Valid: This is same as saying no padding. In this case, the result of the computation has fewer values than input, as shown in Fig 5-10.

![Figure 5-9: Stride size 1 and same padding fo convolution](images/chap5-same-padding-unit-stride.png) Figure 5-9: Stride size 1 and same padding for convolution (Source: <https://github.com/vdumoulin/conv_arithmetic>) ![Figure 5-10: Stride size 1 and valid padding for convolution](images/chap5-unit-strides-valid-padding.png) Figure 5-10: Stride size 1 and valid padding for convolution (Source: <https://github.com/vdumoulin/conv_arithmetic>)

> Infobox: _A guide to convolutional arithmetic for deep learning_ paper provides in depth coverage of impact and meaning of these hyper parameters. It also provides formulae It can be found on <https://arxiv.org/pdf/1603.07285.pdf>

_Receptive field_ of a unit after convolution operation denotes the inputs that have influenced the value of that unit. By stacking multiple layers, each unit can have a wide effective receptive field, which allows CNNs to learn relationships. This idea is demonstrated in Fig 5-11\. One unit in the top most orange layer is composed of inputs from a very large area in the bottom most layer. This allows the composition of simple and complex cells as explained earlier.

![Figure 5-11: Receptive fields](images/chap5-receptive-field.png "Figure 5-11: Receptive fields") Figure 5-11: Receptive fields (NOTE: Please redraw this image)

After the outputs of the convolution operation are calculated, a nonlinear activation is applied. These activation functions, like ReLU, sigmoid, tanh etc, are same as the ones discussed in Chapter 1\. In the Sobel filter example above, no such function was used. It is common to have a third stage after a few convolutional layers called pooling. This is the focus of the next section.

### Pooling

Consider the top green outputs of the convolution step depicted in Fig 5-10\. A non-linear activation function will be applied to each of the four cells. A pooling layer would summarize neighboring outputs. It can be considered a way of down-sampling. Obvious benefits of this include reduction in the size of data flowing through the network leading to faster training and reduced memory usage, it further strengthens small translation invariance.

Pooling has a kernel size parameter. This parameter specifies a matrix size on which to apply the pooling operation. There are different options for implementing pooling. The most common option is called _max pooling_ and is shown in Fig 5-12\. Pooling kernel is of size (2,2) in this example. Max pooling operation picks the maximum value in the cells the kernel is operating on, and makes that the output value. another common pooling operation is _average pooling_, where the cells are averaged and this average becomes the output value. Max pooling is the most commonly used choice.

![Figure 5-12: Max pooling](images/chap5-Max_pooling.png "Figure 5-12: Max Pooling") Figure 5-12: Max Pooling (Image source: <https://commons.wikimedia.org/wiki/File:Max_pooling.png>)

A pooling kernel size of (2, 2) will halve the size of the input. Different kernel sizes can be chosen for the pooling operation. Similar to the convolution operation above, a stride size and padding option can be specified for the pooling operation. By default, the stride size is selected to be same as size of the kernel. This can be seen in Fig 5-12, where the pooling kernel is (2, 2) and is moving 2 steps right and down to produce the output. These default values are fairly common to start training the first CNN.

Now that we know the basics building blocks of CNNs, let's build an CNN to recognize EMNIST.

### EMNIST with CNNs

Code for this example can be found in `<TODO:git path>/Chapter5/mobile_cnn_model.ipynb`. A set of utilities to simplify loading and visualizing EMNIST data are available in `<TODO:git path>/utils/emnist_utils.py` file.

> Warning / Tip CNNs will cause a significant increase in number of trainable parameter and training time. To speed up training, consider using a GPU machine. You can build your own or use a cloud provided version.

After loading and normalizing the data, we will define the convolution and pooling layers. TensorFlow and Keras both have extensive support for CNNs.

```
# (1) First the input layer
inputs = keras.Input(shape=(28,28,), name='emnist_inp')
x = layers.Reshape((28, 28, 1))(inputs)  # since images are gray scale, they have only one channel

# (2) Learn 64 different filters, each 3x3 in size, with valid pooling, and (1,1) stride size
x = layers.Conv2D(64, (3, 3), activation='relu')(x)

# (3) Pooling layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# (4) Dimensions after pooling are 13x13x64\. The 28x28 image is now 13x13 with 64 filters
x = layers.Flatten()(x)

# (5) Traditional dense/FC layers to use these inputs for classification
# This part is similar to previous model
x = layers.Dense(256, activation='relu', name='dense_1')(x)
x = layers.Dense(128, activation='relu', name='dense_2')(x)
outputs = layers.Dense(47, activation='softmax', name='predictions')(x)

cnn = keras.Model(inputs=inputs, outputs=outputs, name='cnn_model_1')
cnn.summary()
```

Given the amount of complex theory that goes into CNNs, Keras and TensorFlow do a wonderful job of hiding this complexity and allowing ML engineers to focus on building the network. Lets analyze the code above. (1) sets up the input layer and shows that each image is 28x28\. Then, it is reshaped to 28x28x1 tensor. This is because it is a gray scale image. the added dimension just denotes the 'gray' channel. If this was a color image like RGB, then this could be reformatted into 28x28x3 to denote the three channels. Note that the number of channels is last in this data format. this is called `channels_last` or NHWC. Here, 'N' refers to number of data samples, 'H' refers to height of each image, 'W' is width of the image and 'C' refers to the channels. It is also possible to have data in NCHW format. `layers.reshape` method can be used to convert between these formats.

> Infobox: It is common mistake to not be careful with the data format sizes and get errors. Another tip: NCHW format is more efficient on NVIDIA GPUs while NHWC is more efficient for CPUs. More tips on performance can be found on <https://www.tensorflow.org/guide/performance/overview>

In (2) and (3), a convolutional layer is added with 64 filters followed by a max pooling layer are added. The kernel is of size 3x3\. By default, _valid_ padding is used if a padding is not supplied. Since _stride size_ is also not provided, it is assumed as (1,1), which means 1 pixel in the right and down directions. `tf.keras.layers.Conv2D` takes many other options and you are encouraged to look into the API and play with some of these.

T0 prepare these for input into the dense classification layers, (4) flattens this _volume_ of 13x13x64 into a single layer of 10,816 units. These dimensions can be verified by the summary shown below. (5) shows the two dense layers followed by an output layer. Note that this is identical to the fully connected network designed in Chapter 1\. One key point of difference is that that network took 28x28 or 784 values as inputs. Through stages of CNN, these have become over ten thousand inputs.

```
Model: "cnn_model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
emnist_inp (InputLayer)      [(None, 28, 28)]          0         
_________________________________________________________________
reshape (Reshape)            (None, 28, 28, 1)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 64)        640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 10816)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               2769152   
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
predictions (Dense)          (None, 47)                6063      
=================================================================
Total params: 2,808,751
Trainable params: 2,808,751
Non-trainable params: 0
_________________________________________________________________
```

Note that the summary above has a lot more parameters than the previous model. This is around 2.8 million trainable parameters compared to approximately 240 thousand parameters in the previous model. Lets compile the model and train it as show below.

```
# Lets compile the model and train it
cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = cnn.fit(norm_train_features, one_hot_train_labels, epochs=10, batch_size=128)

Epoch 1/10
697932/697932 [==============================] - 33s 48us/sample - loss: 0.4692 - accuracy: 0.8421
Epoch 2/10
697932/697932 [==============================] - 30s 43us/sample - loss: 0.3198 - accuracy: 0.8846
\.
\.
\.
Epoc 10/10
697932/697932 [==============================] - 30s 43us/sample - loss: 0.2048 - accuracy: 0.9177

# Evaluate the model on the test set
cnn.evaluate(norm_test_features, one_hot_test_labels, 47)

116323/116323 [==============================] - 5s 47us/sample - loss: 0.3395 - accuracy: 0.8888
```

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

Activation atlas: <https://distill.pub/2019/activation-atlas/>

## Generalization through Regularization

The art of minimizing the difference between training and test error. what is the difference between optimization and machine learning? Explain the purpose of regularization. Talk about constraining weights through L2 norms for dense networks. But say focus is on two specific methods used in CNN - Drop Out and BatchNorm.

### Drop Out

Intuition behind drop out, forcing units to do more work on learning relationships, not depending on everything being present. Add to the code and see the difference. Explain how to build code so that drop out is used only in training and not in inference.

### Batch Normalization

talk about that this is technically not a regularization method,but is often considered so. the intuition behind it, normalizing the weights in a time and space efficient manner. speeds up training and gets to better minima. Add to code and see difference. Talk about Deployment considerations.

## Mobile Optimization Part I

Show how to save and convert to mobile format. talk about building object detection to read lines of text from camera.

## Object Dection / Landmark using CNN? (check what yolo uses)

Check from this article: <https://blog.netcetera.com/face-recognition-using-one-shot-learning-a7cf2b91e96c> see if we can isolate characters from a line of text.

### Detecting characters from a line of text from mobile Camera

### Smile Detection Selfie Taking App

## Questions

1. What are the two key properties of a CNN that make them so effective at computer vision tasks?
2. You may have noticed that the number of trainable parameters have increased significantly when using a CNN. You may wonder if adding more dense layers which lead to same number of trainable parameters would lead to similar accuracy. This is a good exercise to try. Try two architectures:

  - Deeper: Add more layers rather than units in the layers
  - Wider: Add more units in the layers than adding more layers and see what you learn about how these networks perform.
