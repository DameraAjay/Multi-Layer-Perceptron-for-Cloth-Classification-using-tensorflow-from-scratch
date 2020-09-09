'''
		Author : Damera Ajay
		Date   : 07-04-2020
 
'''


import os
from sklearn.model_selection import train_test_split

#importing python fuctions from other python files
from neural_network import *
from data_loader import *
from logistic_regression import *

if __name__ == '__main__':

	#dividing the data into train,test and validation
	x_train,y_train = load_data('train')
	x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.1,random_state=42)
	x_test,y_test = load_data('t10k')
	#checking for commandline argument
	if len(sys.argv) < 2:
		print("\n*************************************************")
		print("[-] Enter valid command line arguments")
		print("[-] For train \'--train\'")
		print("[-] For test \'--test\'")
		print("**************************************************\n")
	elif sys.argv[1] == '--train':
		#Neural Network traing
		y_train =  one_hot_encode(n_class, y_train)
		y_val = one_hot_encode(n_class, y_val)
		
		#show_images(x_train,height=10,width=10)
		
		train(x_train, x_val, y_train, y_val)

	elif sys.argv[1] == '--test':
		#tesing Neural Network
		y_test = one_hot_encode(n_class, y_test)
		test(x_test,y_test)
	else:
		#invalid command line arguments
		print("\n*************************************************")
		print("[-] Enter valid command line arguments")
		print("[-] For train \'--train\'")
		print("[-] For test \'--test\'")
		print("**************************************************\n")
