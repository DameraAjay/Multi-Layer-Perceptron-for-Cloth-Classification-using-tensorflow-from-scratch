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

### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

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

# Training and Testing
    python main.py --train
    python test.py --test
# output
    ********************************************
            MLP Training
    ********************************************
    [+] Train data shape :  (54000, 784)
    [+] Train labels shape : (54000, 10)
    [+] Validation data shape :  (6000, 784)
    [+] Validation labels shape :  (6000, 10)
    Train_Loss: 0.453  | Train_Accuracy: 0.912  | Validation_Loss: 0.511  | Validation_Accuracy:0.857


    ****************************************
    [+] Test data shape   :  (10000, 784)
    [+] Test labels shape :  (10000, 10)
    [+] MLP Test Accuracy : 0.857
    ****************************************
# Trainable parameters
    w1 = 100352 [784 * 128] (input - hidden1)
    w2 = 16384  [128 * 128] (hidden1 – hidden2)
    w3 = 16384  [128 * 128] (hidden2 - hidden3)
    w4 = 1280   [128 * 10]  (hidden3 - output)
    b1 = 128    [128]       (hidden1)
    b2 = 128    [128]       (hidden2)
    b3 = 128    [128]       (hidden3)
    b4 = 10     [128]       (output)
    Total Trainable Parameters : 134874
