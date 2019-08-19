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

A new Python library for managing filesystem and paths called `pathlib` will be used in this chapter. This can be installed using `pip install pathlib` from the command line in your virtual environment. For the mobile application pieces, an android-based mobile app will be built. It will be developed using Android Studio running on MacOS 10.13.6 or above. The models developed will be converted for mobile use as demonstrated in previous chapters, using TensorFlow Lite. Further, MLKit, part of Firebase, will be used to put the trained model into the app.

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

- Autocomplete: This application is one most people are familiar with. Commonly, mobile phone apps give suggestions for completing the word being typed as a part of the sentence. ![Figure 6-1: Autocomplete example. Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.](images/chap6-rnn-auto-complete.png) Figure 6-1: Autocomplete example. (Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.)

> > Info Box: A seminal paper titled "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever, Oriol Vinyals, and Quoc Le laid the groundwork for solving a large class of problems through the sequence to sequence approach, available at: <https://arxiv.org/abs/1409.3215>. The most famous of these is language translation, also referred to as _Neural Machine Translation_ or NMT. Approaches proposed in this paper enabled Google Translate to be written in 9 months and perform better than 8-9 years of successive improvements. To read this fascinating story, this New York Times article is highly recommended: <https://www.nytimes.com/2016/12/14/magazine/the-great-ai-awakening.html>

Let's see whats makes RNN special so that they can help solve such a large class of important problems.

# RNN Architecture

RNNs are built to handle sequences and learn the structure from them. RNN does that by using the output generated after processing the previous item in the sequence along with the current item to generate the next output.

Mathematically, this can be expressed like so:

$$ f_t(x_t) = f(f_{t-1}(x_{t-1}), x_t; \theta) $$

This equation says that to compute the output at time _t_, the output at _t-1_ is used as an input along with the input data $x_t$ at the same time step. Along with this, a set of parameters or learned weights, represented by $\theta$, are also used in computing the output. The objective of training an RNN is to learn these weights $\theta$. This particular formulation of RNN is unique. In previous examples, we have never used the output of a batch to determine the output of a future batch.

Referring back to Fig 1-4, defining deep learning, the middle part of building hierarchical feature representations is performed by RNNs. There is still a need to use a classification or mapping layer at the end. However, this takes a slightly different form given the output of previous layers feeds into successive outputs. This aspect will be detailed later on in this chapter.

![Figure 6-2: High-Level RNN Architecture](images/chap6-rnn-architecture.png) Figure 6-2: High-Level RNN Architecture

Figure 6-2 shows how the overall architecture is adapted for CNN and RNN architectures. In Chapter 5, we saw several data pre-processing and augmentation techniques to make the images ready for CNN. There are specific techniques that need to be used for natural language processing (NLP) and speech domain. These will be covered in a later section in this chapter [TODO: Maybe add a reference to it].

The middle part is what makes RNN and CNN architectures unique. The next section is focussed on explaining these unique characteristics. The last part of the architecture is the mapping or classification layer. This is quite similar across network architectures and is usually a _softmax_ layer. We have seen this layer in several examples across different chapters. Please note that softmax is appropriate for classification. If you are trying to solve a different problem, a different output layer may be needed.

## RNN Building Blocks

The previous section outlined the basic mathematical intuition of a recursive function that is a simplification of the RNN building block. Figure 6-3 represents a few time steps and also adds details to show different weights used for computation for a basic RNN building block or cell.

![figure 6-3: Basic RNN cell](images/chap6-rnn-unravelled.png) Figure 6-3: Basic RNN cell

The basic cell is shown on the left. Input vector at a specific time or sequence step _t_ is multiplied by a weight vector, represented in the diagram as _U_, to generate an activation in the middle part. The key part of this architecture is the loop in this activation part. The output of a previous step is multiplied by a weight vector, denoted by _V_ in the figure, and added to the activation. This activation can be multiplied by another weight vector, represented by _W_, to produce the output of that step shown at the top. In terms of sequence or time steps, this network can be unrolled. This unrolling is virtual. However, it is represented on the right side of the figure. Mathematically, activation at time step _t_ can be represented by:

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

### Long Short Term Memory (LSTM) Networks

Long Short Term Memory Networks were proposed in 1997, and improved upon and popularized by many researchers. They are widely used today for a variety of tasks and produce amazing results.

LSTM has four main parts:

- Cell Value or memory of the network, also referred to as the cell, stores accumulated knowledge
- Input Gate that controls how much of the input is used in computing the new cell value
- Output gate determines how much of the cell value is used in the output
- Forget gate determines how much of the current cell value is used for updating the cell value

These are shown in the figure below. ![Figure 6-4: LSTM Architecture (Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.)](images/chap6-lstm-web.svg)Figure 6-4: LSTM Cell (Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.)

### Gated Recurrent Units (GRUs)

![Figure 6-5: GRU Architecture (Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.)](images/chap6-rnn-gru-web.svg)Figure 6-5: GRU Architecture (Source: Madsen, "Visualizing memorization in RNNs", Distill, 2019.)

## Deep RNN

what makes an RNN deep, having multiple layers

- maybe cover bi-lstm

## Data Preparation for sequences

# Natural Language Processing using RNNs

## Word Vectors

- Word 2 Vectors
- Transformer models

# Recognizing Sentences with RNNs

Now that we understand how RNNs work, let us continue working on the EMNIST example. In Chapter 5, a Convolutional Neural Network was developed that could recognize a character or number at a time. What if we wanted to read entire sentences at a time? An example of where this may be useful is a Google Translate type application that can read text from a mobile phone camera, and then try to translate it. Another example may be an assistive app for visually impaired people which can read out road signs aloud for them.

The high-level solution would process the input image and find places where there is text, read characters using CNNs, and convert them into words and sentences using RNNs. If the desired output is translation or speech, then another RNN can be used. for purposes of this example, we will assume that first step of the solution is done. Our example will focus on having handwritten images of sentences that need to be read and converted into text.

## Data Preparation

Finding an appropriate data set with images and matching text sentences can be challenging. To overcome this challenge, we will use a creative solution. This solution involves downloading a set of sentences in English, and then using a piece of code to pick out the characters from the EMNIST data set to create images of sentences synthetically.

The data set that will be used for this purpose is called the _WikiSplit Dataset_. It can be downloaded from <https://github.com/google-research-datasets/wiki-split>. A copy of the test and training files are made available in the `**TODO/Github**/Chapter6/websplit` folder. Note that the `test.tsv` file is approximately 362MB unzipped, and 92MB zipped. The zipped version is supplied for space efficiency. this can be unzipped and expanded with any program, or with

`$ unzip test.tsv.zip`

from the command line on a unix machine.

This data set has a primary sentence and multiple split up sentences which are edits of the original sentence. The data is in tab separated files, with two columns. The first column is the original sentence. Second column has the split up sentences, each separated by `<::::>`. There are 989,944 original sentences in the training set and 5,000 sentence in the test set. For purposes of this exercise, only the split up sentences will be used, as they are usually shorter in length and suit our purpose better.

Code for this example is in the `TODO/Github/Chapter6/reading-sentences.ipynb` notebook file. First steps is to load the test samples and use the split sentences. This is shown in the code below:

```python
raw_sentences = []  # empty list to store sentences
with open("websplit/test.tsv", "r") as f:
    reader = csv.reader(f, delimiter='\t')  # read a tsv file
    for row in reader:
        # print("Original Sentence:", row[0])      # Uncomment to view data
        # print("Split sentences", row[1].split("<::::>"))
        # print("\n")
        raw_sentences.extend(row[1].split("<::::>"))
print("Total Sentences: ", len(raw_sentences))
```

This should result in 1000 sentences being loaded. A few samples should be inspected to ensure everything loaded properly. It can be seen that the sentences have punctuation, but our EMNIST data set doesn't have an punctuations. So, these need to be cleaned out like so:

```python
# As we see there are lots of punctuations which we dont have in EMNIST, so we are going to remove them,
# and replace multiple spaces with one
import re

sentences = []
table = str.maketrans({key: None for key in string.punctuation})  # translation table

for sentence in raw_sentences:
    # remove punctuation and non-ascii characters
    clean_sentence = re.sub('  +', ' ', sentence.translate(table)).\
                        encode("ascii", 'ignore').decode()  
    sentences.append(clean_sentence.strip())  # add to clean sentences

print(sentences[99], '\n', raw_sentences[99])  # to verify
```

This piece of code also shows one cleansed sample and it's original. The output should look like so:

```
He was was arrested and booked on charges of first degree murder and first degree robbery
  He was was arrested and booked on charges of first - degree murder and first - degree robbery .
```

Next step is to load in the EMINST data. Helper functions developed in Chapter 1 and used in Chapter 5 are used here. Feel free to review these in the IPython notebook. When building images for sentences, we would like to use randomly selected images for the same character. This requires building an index of images and the characters. Code below shows a simple way to construct this:

```python
image_index = {}  # where key is the char and value is a list of IDs
for idx, code in enumerate(train['labels'].tolist()):
    char = mappings[code]
    if char in image_index:
        # this character already exists
        image_index[char].append(idx)  # append index
    else:
        image_index[char] = [idx]  # initiate list with 1 item
```

Now, `numpy.random.choice` can be used to select a sentence from the sentences loaded:

```python
def get_sample_sentences(num_sentence=10):
    # Get a defined number of sentences from the data
    return np.random.choice(sentences, num_sentence)
```

The main workhorse is sampling characters from the sentence, and the corresponding images from the EMNIST data set to construct a composite image. Let's take an example sentence 'The samples were 2 good'. Ideally, the code should generate slightly different looking sentences every time the same character is sampled. Some sample images generated are shown below.

![Figure 6-XX: Samples automatically generated from EMNIST dataset](images/chap6-sentence-image-sample.png) Figure 6-XX: Samples automatically generated from EMNIST dataset

Steps to produce an image given a sentence is shown below:

```python
def get_generate_image(words, chars=train['features'], index=image_index):
    # sentence is string of char/numbers that needs to be converted into an image
    # chars is a data set of images that need to be used to compose, usually pass in train['features'] in here
    # index maps a character to indexes in the images, available as dictionary
    height, width = train['features'][0].shape # height and width of each character
    length = len(words) # total number of characters in the image

    # create an empty array to store the data
    image = np.zeros((height, width * length), np.float64)
    pos = 0  # starting index of the character

    for char in words:
        if char is ' ':
            pos += width # if space, move over
        else:
            if char in image_index:
                # pick a random item from all images for that char
                idx = np.random.choice(image_index[char])  
            else:
                # for some characters, there is only upper case
                idx = np.random.choice(image_index[char.upper()])  
            image[:, pos:(pos+width)] += chars[idx]
            pos += width

    return image
```

Our data preparation is now complete. We can generate a lot of synthetic data for our training and testing purposes. After that, we need build a network combining CNN and RNN to recognize these sentences. Here is a simple code solution to generate multiple training and test images and labels.

```python
train_sentences = sentences[:9000]
test_sentences = sentences[9000:]

# Lets assume that for each training sample, 2 variants will be generated

def generate_sentences(texts, chars,
                           index, num_variants=2, max_length=32):
    # this method takes input text lines, character samples and labels
    # and generates images. It can generate multiple images per sentence
    # as controlled by num_variants parameter. max_length parameter
    # ensures that all sentences are the same length

    # total number of samples to generate
    num_samples = len(texts) * num_variants
    height, width = chars[0].shape  # shape of image

    # setup empty array of the images
    images = np.zeros((num_samples, height, width * max_length), np.float64)
    labels = []

    for i, item in enumerate(texts):
        padded_item = item[0:max_length] if (len(item) > max_length) else item.ljust(max_length, ' ')

        for v in range(num_variants):
            img = get_generated_image(padded_item, chars, index)
            images[i*num_variants+v, :, :] += img
            labels.append(padded_item)

    return images, labels
```

This can be called with different data from the sentence data base and EMNIST data to generate test and training data. You can see that the list of sentences was split at the top of the previous code listing. Now, to generate the test and training data, the code would look like:

```python
train_images, train_labels = generate_sentences(train_sentences, train['features'], image_index)
test_images, test_labels = generate_sentences(test_sentences, train['features'], image_index)
```

This is an expensive, time consuming operation. These files should be stored so that this step doesn't have to be repeated. This is shown in the code below.

```python
# Now to save these models for easy loading
pp = pathlib.Path('.') / 'sentences'
pp.mkdir(exist_ok=True)  # create the directory

np.save(pp / 'train_images', train_images)
np.save(pp / 'test_images', test_images)
np.save(pp / 'train_labels', train_labels)
np.save(pp / 'test_labels', test_labels)
```

We have provided these generated files as part of the GitHub repository. Please not that you will need to use `git lfs` to get them as they are large files. If you followed instructions from Chapter 1, this is should already be installed for you. These generated files are in the `<github>/Chapter6/sentences` folder.

> TIP/WARNING: This files are approximately 233 MB in size. Upon unzipping, they will take approximately 3.8GB on disk. Please be mindful of your network usage and disk space needs.

## Building and Training the Network

To solve this problem, we need a convolutional neural architecture to featurize the image and a RNN architecture to convert the characters read into sentences. For the CNN, we are going to use the classic LeNet5 architecture. This paper was published by Yann LeCun in 1998 and one of the best performing architectures for recognizing digits and characters for a long time (till advent of GPUs). This paper can be accessed from `http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf`. For the RNN network, an LSTM based architecture will be used.

### LeNet5 CNN

LeNet5 is a 7 layer architecture, as shown in the figure below.

[[insert diagram here ]]

### LSTM based RNN

## Testing performance

// Include examples, code, illustrations: explain complex concepts in clear, simple language

// Address the readers pain points: Address common pain points and areas of confusion.

## Optimizing for Mobile

TBD: Get Aditya or Vikram to help out

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
