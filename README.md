# MNIST_Neural_Net
This is an educational project intended to classify handwritten digits from the MNIST database using only native Python and the NumPy library. Pillow (PIL fork) was used to display the handwritten images for demonstration purposes.

The network is made up of three layers: the input layer, the hidden layer, and the output layer. The input layer consists of 784 neurons, each which represents the intensity of a pixel with an activation between 0 and 1. The hidden layer contains 30 neurons. The third layer contains 10 neurons, each of which should represent the network's calculation of the possibility that the input image is a 0, 1, 2, etc. The neuron with the highest activation in the output layer corresponds to which digit the network "thinks" the image is.

MNIST DATA STRUCTURE
The program uses serialized (.pickle) lists to represent the MNIST database. For example, the training database file contains two parallel lists. One list contains 2D lists that represent the pixel values of each image. The other list contains the integer that the image represents in handwriting. 

WEIGHT AND BIAS INITIALIZATION
The weights are initialized in a random distribution by the WeightsAndBiasesGenerator.py program. The range of the random distribution was determined by an equation found on the forum StackExchange (https://datascience.stackexchange.com/questions/22093/why-should-the-initialization-of-weights-and-bias-be-chosen-around-0?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa).
The biases are initialized to 0.

FUNCTIONALITY AND BACKPROPAGATION
The program only partially works as intended and is still a work in progress. The second layer of weights and biases (between the hidden and output layer) was trained and caused an improvement of accuracy to about 60% depending on the initial weights (61.3% is the highest so far). The problem preventing further accuracy is in the approach to or implementation of backpropagation. The non-functional version of a backpropagation function is included in the NeuralNet.py file but is never called in the program. When the backprop function is called, it makes changes to the weights and biases but no significant improvements in the program's accuracy. The program was run entirely on a 2012 Macbook Air with no special drivers to harness GPU power, and therefore training and testing took a significant amount of time. Therefore systematically optimizing hypervariables and initialization of weights and biases was out of the scope of this project. 

RESOURCES
Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/ (I made it a point to not consult the code of this project)
3Blue1Brown series on Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

MNIST database: http://yann.lecun.com/exdb/mnist/
Martin Thoma's project "Classify MNIST with PyBrain" which I used to parse the MNIST dataset and create a Python list from it, which I then serialized:
https://martin-thoma.com/classify-mnist-with-pybrain/

The serialized (.pickle) MNIST lists used in the programs are too large to upload to Github. They are uploaded in this Google Drive folder:
https://drive.google.com/drive/folders/1vAlvIMT4lpOiHP3lEoO7SguoVWfMGsbB?usp=sharing
