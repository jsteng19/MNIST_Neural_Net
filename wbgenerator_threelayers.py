import pickle, numpy, random

biases = [numpy.zeros(30), numpy.zeros(10)]

weights = [numpy.empty((30, 784)), numpy.empty((10, 30))]
#initialize a 3D list of weights. Each innermost list is a list of weights going to single perceptron 
for x in range (30):
    for y in range (784):
        weights[0][x][y] = random.uniform(-0.036, 0.036)

for x in range (10):
    for y in range (30):
        weights[1][x][y] = random.uniform(-0.036, 0.036)
        



weights_pickle = open("/Users/jstenger/Desktop/MNIST/weights_3layers.pickle", "wb")
pickle.dump(weights, weights_pickle)
weights_pickle.close()

biases_pickle = open("/Users/jstenger/Desktop/MNIST/biases_3layers.pickle", "wb")
pickle.dump(biases, biases_pickle)
biases_pickle.close()
