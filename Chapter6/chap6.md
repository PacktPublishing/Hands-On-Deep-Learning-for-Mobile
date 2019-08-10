Speech recognition and language processing capabilities have become mainstream with the popularity of digital assistants like Siri and Alexa and various chatbots. These advances have been made possible due to the use of deep learning techniques to convert speech into text and applying natural language processing techniques to it.

A specialized network architecture, called _Recurrent Neural Networks_ or RNNs, is used extensively for these tasks. This chapter will focus on explaining how an RNN works and building RNN-based applications in text generation and speech recognition. We will build a joke generator, and a wake word detection algorithm, similar to 'Hey Siri' or 'Ok Google'.

The following topics will be covered in this chapter:

- Motivations for modeling sequences
- Different types of RNN architectures and cells like LSTM and GRU
- Overview of contemporary NLP techniques
- Building vector representation of words for NLP
- Generating text or completing sentences using RNN and word vectors
- Detecting words in speech/audio using RNNs
- Advanced Topics in sequence modelling

# Technical Requirements

This chapter uses Python, TensorFlow 2.0, Jupyter Notebooks for building and training the models. Data files from training are reused from Chapter 1 Github location. All the code for this chapter is in

<github-repo>/Chapter6/</github-repo>

For the mobile application pieces, an android-based mobile app will be built. It will be developed using Android Studio running on MacOS 10.13.6 or above. The models developed will be converted for mobile use as demonstrated in previous chapters, using TensorFlow Lite. Further, MLKit, part of Firebase, will be used to put the trained model into the app.

// list technologies and installations required here.

// Provide Github URL for the code in the chapter (setup instructions should be on the Github page). Create a Github folder named, "chX", where X is the chapter number. For example, ch1

# Motivations for Recurrent Neural Networks

Fully connected deep learning networks, as introduced in Chapter 1, are doing well in a wide variety of cases. However, there are specific architectures that work better for specific types of data. In Chapter 5, we saw an example of CNN architecture which has higher accuracy in computer vision domain. However, they have a limitation that they can process only one image at a time. If they are fed a sequence of images (like frames of a video), CNNs wont be able to process them. This domain of processing sequences is very important and has plenty of practical applications. The conventional way to think about sequences is time-series data. Voice or speech data fits very well with this view. The audio waveform changes over time and is usually chunked into 10 ms pieces for use in deep learning. Another way to think of an audio file is to model it as a sequence of 10 ms chunks.

A not so obvious example of sequences may be language. A sentence in a language is a set of words. However, these words have a certain organization - that is these words make a sentence only if they are in a certain order or a _sequence_. So, each sentence is a sequence of words. Notice the similarity in structure between an audio file represented as a sequence of 10 ms chunks and a sentence represented as a sequence of words. With appropriate featurization, these can be fed to an RNN.

The concept of modeling sequences is a very important and powerful one. It can be abstracted to suit a very large class of difficult tasks. Here are some additional examples of problems modeled as sequence learning:

- Robotic control: A robotic hand picking up an object needs to execute a set of steps, like rotation, extension, opening and closing the jaw. These sequences of steps can be modeled as a sequence. Using RNN architecture, these sequences can be learned for robotic control
- Speech Synthesis: We talked about speech recognition earlier. Speech synthesis is the exact opposite problem to this - given a set of text, generate the audio for it. Speech synthesis has gained prominence with the advent of chatbots and assistants like Siri and Alexa. The synthesized voices need to sound normal and not robotic.
- Handwriting recognition: The intuition behind this is the characters follow some distribution based on the language. Hence, the characters of a word and words in a sentence can be considered sequences. CNNs are used to extract features and then coupled with RNNs to detect the right set of words. We will see this particular technique in this chapter by combining our CNN model from Chapter 5 with an RNN model to read sentences.
- Composing music: Given the periodic structure of music, it is a great application of RNN. It is one that brings the arts and computers closer. For some inspiration in composing music and art, check out

  <magenta.tensorflow.org>.</magenta.tensorflow.org>

> > Info Box: A seminal paper titled "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever, Oriol Vinyals, and Quoc Le laid the groundwork for solving a large class of problems through the sequence to sequence approach, available at: <https://arxiv.org/abs/1409.3215>. The most famous of these is language translation, also referred to as _Neural Machine Translation_ or NMT. Approaches proposed in this paper enabled Google Translate to be written in 9 months and perform better than 8-9 years of successive improvements. To read this fascinating story, this New York Times article is highly recommended: <https://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html>

Let's see whats makes RNN special so that they can help solve such a large class of important problems.

# RNN Architecture

RNNs are built to handle sequences and learn the structure from them. RNN does that by using the output generated after processing the previous item in the sequence along with the current item to generate the next output.

Mathematically, this can be expressed like so:

$$ f_t(x_t) = f(f_{t-1}(x_{t-1}), x_t; \theta) $$

This equation says that to compute the output at time _t_, the output at _t-1_ is used as an input along with the input data $x_t$ at the same time step. Along with this, a set of parameters or learned weights, represented by $\theta$, are also used in computing the output. The objective of training an RNN is to learn these weights $\theta$. This particular formulation of RNN is unique. In previous examples, we have never used the output of a batch to determine the output of a future batch.

Referring back to Fig 1-4, defining deep learning, the middle part of building hierarchical feature representations is performed by RNNs. There is still a need to use a classification or mapping layer at the end. However, this takes a slightly different form given the output of previous layers feeds into successive outputs. This aspect will be detailed later on in this chapter.

![Figure 6-1: High-Level RNN Architecture](images/chap6-rnn-architecture.png)

Figure 6-1 shows how the overall architecture is adapted for CNN and RNN architectures. In Chapter 5, we saw several data pre-processing and augmentation techniques to make the images ready for CNN. There are specific techniques that need to be used for natural language processing (NLP) and speech domain. These will be covered in a later section in this chapter [TODO: Maybe add a reference to it].

The middle part is what makes RNN and CNN architectures unique. The next section is focussed on explaining these unique characteristics. The last part of the architecture is the mapping or classification layer. This is quite similar across network architectures and is usually a _softmax_ layer. We have seen this layer in several examples across different chapters. Please note that softmax is appropriate for classification. If you are trying to solve a different problem, a different output layer may be needed.

## RNN Building Blocks

The previous section outlined the basic mathematical intuition of a recursive function that is a simplification of the RNN building block. Figure 6-2 represents a few time steps and also adds details to show different weights used for computation for a basic RNN building block or cell.

![figure 6-2: Basic RNN cell](images/chap6-rnn-unravelled.png)

TThe basic cell is shown on the left. Input vector at a specific time or sequence step _t_ is multiplied by a weight vector, represented in the diagram as _U_, to generate an activation in the middle part. The key part of this architecture is the loop in this activation part. The output of a previous step is multiplied by a weight vector, denoted by _V_ in the figure, and added to the activation. This activation can be multiplied by another weight vector, represented by _W_, to produce the output of that step shown at the top. In terms of sequence or time steps, this network can be unrolled. This unrolling is virtual. However, it is represented on the right side of the figure. Mathematically, activation at time step _t_ can be represented by:

$$ a_t = Ux_t + V.a_{t-1} $$

output at the same step can be computed like so:

$$ o_t = W.a_t $$

> > Info: This is very simplified mathematics to only provide intuition about RNNs.

Structurally, the network is very simple as it is a single unit. To exploit and learn the structure of inputs passing through, weight vectors U, V, and W are shared across time steps. The network does not have layers as seen in fully connected or convolutional networks. However, as it is unrolled over time steps, it can be thought of as having as many layers as steps in the input sequences. there are additional criteria that would need to be satisfied to make a _Deep RNN_. More on that later in this section. These networks are trained using backpropagation and stochastic gradient descent techniques. The key thing to note here is that backpropagation is happening through the sequence or times steps instead of layers.

Having this structure enables processing sequences of arbitrary lengths. However, as the length of sequences increases, there are a couple of challenges that emerge:

- Vanishing and exploding gradients: As the length of these sequences increase, the gradients going back will become smaller and smaller. This will cause the network to train slowly or not learn at all. This effect will be more pronounced as sequence lengths increase. In the previous chapter, we built a network of a handful of layers. Here, a sentence of 10 words would equate a network of 10 layers. A 1 min audio sample of 10ms would generate 6,000 steps! Conversely, gradients can also explode if the output is increasing. The simplest way to manage vanishing gradients is through the use of ReLUs (described in Chapter 1). For managing exploding gradients, a technique gradient clipping is used. This technique artificially clips gradients if their magnitude exceeds a threshold. This prevents gradients from becoming too large.
- Inability to manage long term dependencies: Let's say that the third word in an eleven-word sentence is highly informative. Here is a toy example: "I think _soccer_ is the most popular game across the world." As the processing reaches the end of the sentence, the contribution of words prior earlier in the sequence will become smaller and smaller due to repeated multiplication with the vector _V_ as shown above.

Two specific RNN cell designs mitigate these problems: Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU). These are described in the section next. However, note that TensorFlow provides implementations of both types of cells out of the box. So, building RNNs with these cell types is almost trivial.

> TIP: Training RNNs is a very complicated process fraught with many frustrations. Modern tools such as TensorFlow do a great job of managing the complexity and reducing the pain to a great extent. However, training RNNs still is a challenging task. But the rewards of getting it right are well worth it.

### Long-Short Term Memory (LSTM) cells

### Gated Recurrent Units (GRUs)

// Include examples, code, illustrations: explain complex concepts in clear, simple language

// Address the readers pain points: Address common pain points and areas of confusion.

## H2 - Subtopic B // Add explanation/essential concepts

// Include examples, code, illustrations: explain complex concepts in clear, simple language

// Address the readers pain points: Address common pain points and areas of confusion.

## H2 - Add further subtopics as required

# H1 - Topic B // Add explanation and overview here

// H2 - Subtopic B // Add explanation/essential concepts

// Include examples, code, illustrations: explain complex concepts in clear, simple language

// Address the readers pain points: Address common pain points and areas of confusion.

// H2 - Add further subtopics as required

# Advanced Topics in Sequence Modeling

Attention is all you need BeRT and Transformer architectures

# Summary

// Capture the key points and reinforce the structure of the book. You need to remind the reader what they have just learnt. Aim for a maximum of 200–300 words.

# Questions

// Add 7-10 questions here

// Include the answers in the Back Matter under a section titled, "Assessments"

# Further Reading

// Add additional references to useful Packt resources, or other information that might help explain a particular concept in further detail.
