# MNIST_Neural_Net
This is an educational project intended to classify handwritten digits from the MNIST database using only native Python and the NumPy library. Pillow (PIL fork) was used to display the handwritten images for demonstration purposes.

The network is made up of three layers: the input layer, the hidden layer, and the output layer. The input layer consists of 784 neurons, each which represents the intensity of a pixel with an activation between 0 and 1. The hidden layer contains 30 neurons. The third layer contains 10 neurons, each of which should represent the network's calculation of the possibility that the input image is a 0, 1, 2, etc. The neuron with the highest activation in the output layer corresponds to which digit the network "thinks" the image is.

The program uses serialized (.pickle) lists to represent the MNIST database. For example, the training database file contains two parallel lists. One list contains 2D lists that represent the pixel values of each image. The other list contains the integer that the image represents in handwriting. 

The program only partially works as intended. 


Educational resources used: 
Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
3Blue1Brown series on Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
