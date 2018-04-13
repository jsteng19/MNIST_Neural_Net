import pickle, numpy
from PIL import Image

mnist = [pickle.load(open("/Users/jstenger/Desktop/MNIST/mnist_train.pickle", "rb")), pickle.load(open("/Users/jstenger/Desktop/MNIST/mnist_test.pickle", "rb"))]
weights = pickle.load(open("/Users/jstenger/Desktop/MNIST/untrainedWeights.pickle", "rb")) 
biases = pickle.load(open("/Users/jstenger/Desktop/MNIST/untrainedBiases.pickle", "rb"))
learning_constant = 1
backprop_constant = 1

layers = [None, numpy.zeros(30), numpy.zeros(10)]

def main():
    print("accuracy:" , testAccuracy(1000))
    batch_size = 150 #the number of images included in each "batch." An average gradient is calculated and applied to the weights and biases for each batch
    batches = 400
    print("Batch size: ", batch_size, " learning constant: ", learning_constant)
    for image in range(0, batch_size * batches, batch_size):
        applyStoich(image, image + batch_size)
    print("accuracy:" ,testAccuracy(1000))

    #save parameters to a new .pickle file
    biases_pickle = open("/Users/jstenger/Desktop/MNIST/trainedBiases2.pickle", "wb")
    pickle.dump(biases, biases_pickle)
    biases_pickle.close()
    weights_pickle = open("/Users/jstenger/Desktop/MNIST/trainedWeights2.pickle", "wb")
    pickle.dump(weights, weights_pickle)
    biases_pickle.close()

def showImage(n, database):
    flat_list = [item for sublist in mnist[database][0][n] for item in sublist]
    im = Image.new("L", (28, 28))
    im.putdata(flat_list)
    im.show()

#tests accuracy of the current weight and bias matrix, default database is the test database
def testAccuracy(trials, database = 1):
    correct = 0
    for image in range(trials):
         processImage(image, database)
         if layers[2].argmax() == mnist[database][1][image]:
             correct += 1
    return correct / trials

#sets the part of the global list representing the input layer to a train of 
# the pixel values of the specified image
def setInputLayer(image, database):
    layers[0] = mnist[database][0][image].flatten()

def processImage(image, database):
    setInputLayer(image, database)
    
    for i in range (30):
        layers[1][i] = sigmoid(numpy.dot(weights[0][i], layers[0]) - biases[0][i])
    
    for i in range (10):
        layers[2][i] = sigmoid(numpy.dot(weights[1][i], layers[1]) - biases[1][i]) 

#applies a Stoichiastic adjustment on the weights and biases
#based on the partial derivatives of the weights and biases and
#a learning constant using a subset of the training data
#specified by the arguments
def applyStoich(start, end, database = 0):
    gradientSum = 0
    for image in range (start, end):
        processImage(image, database)
        gradientSum = numpy.add(gradientSum, getGradient(image, database))
    averageGradient = numpy.divide(gradientSum, end - start)
    global weights
    global biases
    weights = numpy.add(weights, -learning_constant * averageGradient[0])
    biases = numpy.add(biases, -learning_constant * averageGradient[1])
    
#returns two arrays identical in structure to the array of weights and biases
#containing the partial derivatives of cost with respect to each weight or bias
def getGradient(image, database):
    
    weightGradient = numpy.zeros_like(weights)
    biasGradient = numpy.zeros_like(biases)
    desired = numpy.zeros(10)
    for layer in range (len(layers) - 1, 1, -1): # In a fully operational network, this loop would iterate down to 0, training all the layers.
                                                 # By setting the end of the range to 1, I limited the program to training the final layer of 
                                                 # weights and biases. This means the backProp function never gets called.
        if layer == len(layers) - 1:
            desired[mnist[database][1][image]] = 1
        else:
            desired = backProp(layer, desired)
        weightGradient[layer - 1], biasGradient[layer - 1] = getGradientBetweenLayers(layer, desired)

    return numpy.array([weightGradient, biasGradient])

#returns two lists of lists representing the gradient of the weights and biases between
#layer outLayerIndex and layer outLayerIndex - 1
def getGradientBetweenLayers(outLayerIndex, desired):
    weightGradient = numpy.zeros_like(weights[outLayerIndex - 1])
    biasGradient = numpy.zeros_like(biases[outLayerIndex - 1])
    for node in range(len(layers[outLayerIndex])):
        for edge in range(len(layers[outLayerIndex - 1])):                                         
            weightGradient[node][edge] = dCdW(layers[outLayerIndex - 1][edge], layers[outLayerIndex][node], desired[node])
    
        biasGradient[node] = dCdB(layers[outLayerIndex][node], desired[node])
        
    return [weightGradient, biasGradient]

#returns the "desired" output of the layer n-1 given the layer n and the weights and biases. (See README)
def backProp(layer, lastDesired, database = 0):
    desired = [0]*len(layers[layer])
    for node in range (len(layers[layer])):
        gradientSum = 0
        for edge in range (len(layers[layer + 1])):
            gradientSum += dCdA(weights[layer][edge][node], layers[layer + 1][edge], lastDesired[edge])
        desired[node] = layers[layer][node] - (gradientSum / len(layers[layer + 1])) * backprop_constant
    return desired

# returns the partial derivative of the cost function with respect to a weight
def dCdW(activation0, activation1, desired):
    return 2 * (activation1 - desired) * activation0 * activation1 * (1 - activation1)
    
# returns the partial derivative of the cost function with respect to a bias
def dCdB(activation1, desired):
    return 2 * (activation1 - desired) * activation1 * (1 - activation1)

# returns the partial derivative of the cost function with respect to an activation
def dCdA(weight, activation1, desired): 
    return 2 * (activation1 - desired) * activation1 * (1 - activation1) * weight

def sigmoid(x):
  return 1.0 / (1.0 + numpy.exp(-x))

main()
