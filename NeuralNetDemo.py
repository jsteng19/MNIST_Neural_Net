import pickle, numpy
from PIL import Image
layers = [None, numpy.zeros(30), numpy.zeros(10)]
mnist = [pickle.load(open("/Users/jstenger/Desktop/MNIST/mnist_train.pickle", "rb")), pickle.load(open("/Users/jstenger/Desktop/MNIST/mnist_test.pickle", "rb"))]
weights = [numpy.array(pickle.load(open("/Users/jstenger/Desktop/MNIST/weights_3layerstrained2.pickle", "rb"))[0]), numpy.array(pickle.load(open("/Users/jstenger/Desktop/MNIST/weights_3layerstrained2.pickle", "rb"))[1])]
biases = [numpy.array(pickle.load(open("/Users/jstenger/Desktop/MNIST/biases_3layerstrained2.pickle", "rb"))[0]), numpy.array(pickle.load(open("/Users/jstenger/Desktop/MNIST/biases_3layerstrained2.pickle", "rb"))[1])]


def main():

    print(testAccuracy(1000))
##    continues displaying images with their data each time the user
##    hits enter.
    n = 0
    while n < 10000:
        processImage(n, 1)
        print("Computer thought: ",layers[2].argmax())
        print("correct:" , mnist[1][1][n][0])
        print("type \"quit\" to quit.")
        image = showImage(n, 1)
        image.show()
        keyboard = " "
        while keyboard != "":
            keyboard = input("")
            if keyboard == "quit":
                n = 10001
        image.close()
        n = n + 1

    
def showImage(n, database):
    print(mnist[database][1][n][0])
    flat_list = [item for sublist in mnist[database][0][n] for item in sublist]
    im = Image.new("L", (28, 28))
    im.putdata(flat_list)
    return im

def testAccuracy(trials, database = 1):
    correct = 0
    for image in range(trials):
         processImage(image, database)
         if layers[2].argmax() == mnist[database][1][image]:
             correct += 1
    return correct / trials


#sets the part of the global list representing the input layer to a train of
# the specified image
def setInputLayer(image, database):
    layers[0] = mnist[database][0][image].flatten()

def processImage(image, database):
    setInputLayer(image, database)
    
    for i in range (30):
        layers[1][i] = sigmoid(numpy.dot(weights[0][i], layers[0]) - biases[0][i])
    
    for i in range (10):
        layers[2][i] = sigmoid(numpy.dot(weights[1][i], layers[1]) - biases[1][i])

def sigmoid(x):
  return 1 / (1 + numpy.exp(-x))

main()
