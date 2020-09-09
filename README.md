# Multi-Layer-Perceptron-for-Cloth-Classification-using-tensorflow-from-scratch
Developed a MLP for MNIST Cloth Classification using tensorflow from scratch

# Requirements
    tensorflow
    numpy
    scikit-learn
# MNIST fashion dataset
    here is a snapshot of MNIST fashion dataset
    download data from https://github.com/zalandoresearch/fashion-mnist and keep it in data folder
![](img/fashion-mnist-sprite.png)


# MLP configuration
    input layer  - 784 neurons
    hidden layer1 - 128 neurons
    hidden layer2 - 128 neurons
    hidden layer3 - 128 neurons
    output layer - 10 neurons
    
    learning rate = 0.001
    batch size = 64
    no of epochs = 50
    no of hidden layers = 3
    no of classes = 10
    
 Relu activation function is applied every layer except last(output) layer.
 Softmax is applied at output layer with 10 classes [0-9]
 Catagorical cross entropy loss fuction is used.

