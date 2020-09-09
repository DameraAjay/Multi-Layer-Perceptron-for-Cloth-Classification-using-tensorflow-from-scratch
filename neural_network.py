'''
		Author :Damera Ajay
		Date   :07-04-2020

'''
#!/usr/bin/env python


import numpy as np
import os
import sys
import gzip 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.python.framework import ops

#display ing images

def show_images(x_train,height=0,width=0):
    k = width * height
    for i in range(width):
        for j in range(height):
            plt.subplot2grid((width,height),(i,j))
            plt.imshow(x_train[k].reshape(28,28),cmap='Greys')
            plt.axis('off')
            k = k + 1
    plt.show()

def labels():
    labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    return labels;

#Neural Network configurations(hyperparameters)

n_input = 784
n_hidden1 = 128
n_hidden2 = 128
n_hidden3 = 128

n_class = 10
n_epoch = 50
learning_rate = 0.001
batch_size = 512

#neural network model with 3 hidden layers of size 128 each

def model(batch_x):
	#weights of neural network
	weights = {
	    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1),name="w1"),
	    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2],stddev=0.1),name="w2"),
	    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3],stddev=0.1),name="w3"),
	    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_class],stddev=0.1),name="wout"),
	}
	#biases of neural network
	biases={
	    'b1': tf.Variable(tf.zeros([n_hidden1]),name="b1"),
	    'b2': tf.Variable(tf.zeros([n_hidden2]),name="b2"),
	    'b3': tf.Variable(tf.zeros([n_hidden3]),name="b3"),
	    'out': tf.Variable(tf.zeros([n_class]),name="bout")
	}
	#neural netork operations
	#layer1
	layer_1 = tf.add(tf.matmul(batch_x, weights['w1']), biases['b1'])
	layer_1 = tf.maximum(0.0,layer_1)
	#layer2
	layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
	layer_2 = tf.maximum(0.0,layer_2)
	#layer3
	layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
	layer_3 = tf.maximum(0.0,layer_3)
	#output
	output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
	return output_layer

# funciton for computing categorical cross entropy
def compute_cross_entropy_loss(y_hat, y):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_hat,labels = y)
    cost = tf.reduce_mean(loss)
    return cost

#creating optimizer
def create_optimizer():
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optimizer

#this fucntion for converting test data into one hot encoding
def one_hot_encode(n_class, Y):
    return np.eye(n_class)[Y]

#training.....
def train(x_train, x_val, y_train, y_val, verbose = False):
    X = tf.placeholder(tf.float32, [None, 784], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")
    
    logits = model(X)

    loss = compute_cross_entropy_loss(logits, Y)
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name("w1:0")
    b1 = graph.get_tensor_by_name("b1:0")
    w2 = graph.get_tensor_by_name("w2:0")
    b2 = graph.get_tensor_by_name("b2:0")
    w3 = graph.get_tensor_by_name("w3:0")
    b3 = graph.get_tensor_by_name("b3:0")
    w4 = graph.get_tensor_by_name("wout:0")
    b4 = graph.get_tensor_by_name("bout:0")

    #Adding L2 regularization
    # Loss = loss + 1/2(sum(w**2))
    r1 = tf.reduce_sum(tf.square(w1))/2.0
    r2 = tf.reduce_sum(tf.square(w2))/2.0
    r3 = tf.reduce_sum(tf.square(w3))/2.0
    r4 = tf.reduce_sum(tf.square(w4))/2.0

    r5 = tf.reduce_sum(tf.square(b1))/2.0
    r6 = tf.reduce_sum(tf.square(b2))/2.0
    r7 = tf.reduce_sum(tf.square(b3))/2.0
    r8 = tf.reduce_sum(tf.square(b4))/2.0

    reg = r1+r2+r3+r4+r5+r6+r7+r8
     
    #final loss
    cost = tf.reduce_mean(loss + learning_rate * reg)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    validation_loss = compute_cross_entropy_loss(logits, Y)

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        print("\n********************************************")
        print("\t\tMLP Training")
        print("********************************************")
        print("[+] Train data shape : ",x_train.shape)
        print("[+] Train labels shape :",y_train.shape)
        print("[+] Validation data shape : ",x_val.shape)
        print("[+] Validation labels shape : ",y_val.shape)
        print()


       	for epoch in range(n_epoch):

            if epoch % 10 == 0:
                save_path = saver.save(sess,"weights/",global_step=epoch)

            epoch_loss = 0.

            num_batches = np.round(x_train.shape[0]/batch_size).astype(int)

            for i in (range(num_batches)):
            	#mini-batch
                mini_batch_x = x_train[(i*batch_size):((i+1)*batch_size),:]
                mini_batch_y = y_train[(i*batch_size):((i+1)*batch_size),:]

                _, batch_loss = sess.run([optimizer, cost],feed_dict = {X: mini_batch_x, Y:mini_batch_y})

                epoch_loss += batch_loss

            epoch_loss = epoch_loss/num_batches

            train_accuracy = sess.run(accuracy, feed_dict = {X: x_train, Y: y_train})

            val_loss = sess.run(validation_loss, feed_dict = {X: x_val ,Y: y_val})
            val_accuracy = sess.run(accuracy, feed_dict = {X: x_val, Y: y_val})

            print("[+] Epoch %02d/%d"%(epoch+1,n_epoch),
            	" | Train_Loss: %.3f"%((epoch_loss)),
            " | Train_Accuracy: %.3f"%(((float(train_accuracy)))),
            " | Validation_Loss: %.3f"%(((float(val_loss)))),
            " | Validation_Accuracy:%.3f"%(((float(val_accuracy)))))

        save_path = saver.save(sess, "weights/final_weights.ckpt")

        #counting all parameters
        print("\n***********************************************")
        print("[+] Calculating Trainable Parameters:")
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print("   [+] %s =  %d"%(variable.name,variable_parameters))
            total_parameters += variable_parameters
        
        print("[+] Total Trainable Parameters : ",total_parameters)
        print("***********************************************\n")

        sess.close()

#Testing.....
def test(X_test, y_test):

    ops.reset_default_graph()
    saver = tf.train.import_meta_graph("weights/final_weights.ckpt.meta")

    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint("weights"))

        graph = tf.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        b1 = graph.get_tensor_by_name("b1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b2 = graph.get_tensor_by_name("b2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        b3 = graph.get_tensor_by_name("b3:0")
        w4 = graph.get_tensor_by_name("wout:0")
        b4 = graph.get_tensor_by_name("bout:0")


        X = tf.placeholder(tf.float32, [None, 784], name="X")
        Y = tf.placeholder(tf.float32, [None, 10], name="Y")

        l1 = tf.add(tf.matmul(X,w1),b1)
        l1 = tf.maximum(0.0,l1)

        l2 = tf.add(tf.matmul(l1,w2),b2)
        l2 = tf.maximum(0.0,l2)

        l3 = tf.add(tf.matmul(l2,w3),b3)
        l3 = tf.maximum(0.0,l3)

        logits = tf.add(tf.matmul(l3,w4),b4)

        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        test_accuracy = sess.run(accuracy, feed_dict = {X: X_test, Y: y_test})
        print("\n\n****************************************")
        print("[+] Test data shape   : ",X_test.shape)
        print("[+] Test labels shape : ",y_test.shape)
        print("[+] MLP Test Accuracy : %.3f"%(test_accuracy))
        print("****************************************\n")
        sess.close()







