# Android application to host TFLite models and evaluate performance

Application based of sample Java Android demo application from Tensorflow GitHub:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java/demo

![TFLite Inference in Action on a Smartphone]("inference_on_smartphone_in_action.jpg")

A simple Android example that demonstrates hosting TFLite model in an android application.
Default use case is image classification using real time came images.

## Building in Android Studio with TensorFlow Lite AAR from JCenter.
The build.gradle is configured to use TensorFlow Lite's nightly build.

If you see a build error related to compatibility with Tensorflow Lite's Java API (example: method X is
undefined for type Interpreter), there has likely been a backwards compatible
change to the API. You will need to pull new app code that's compatible with the
nightly build and may need to first wait a few days for our external and internal
code to merge.


