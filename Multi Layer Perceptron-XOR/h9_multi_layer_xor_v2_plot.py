# tis code is develop of v3 
import math
import matplotlib.pyplot as plt
import numpy as np

# activation function is sigmoid
def activation_func(x):
    return 1 / (1 + np.exp(-x))

# neural network class definition
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.error_list = np.array([])
        self.count = 0
        self.error_sum = 0
        minValue = -1

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to j in the next layer
        # w11 w21
        # w12 w22
        
        self.wih = np.random.uniform(size=self.inodes * self.hnodes, low=minValue, high=1).reshape(self.hnodes, self.inodes)
        self.who = np.random.uniform(size=self.hnodes * self.onodes, low=minValue, high=1).reshape(self.hnodes,self.onodes)        

        self.bih = np.random.uniform(size=self.hnodes, low=minValue, high=1).reshape(self.hnodes, 1)
        self.bho = np.random.uniform(size=self.onodes, low=minValue, high=1).reshape(self.onodes, 1)
        
        # learning rate 
        self.lr = learningRate
        pass

    # train the NN
    def train(self, input_list, target_list):
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2)

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih
        hidden_outputs = activation_func(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(np.transpose(hidden_outputs),self.who) + self.bho
        final_outputs = activation_func(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs

        self.error_sum += np.power(output_errors, 2)         

        # hidden layer error is the output_errors , split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who, np.transpose(output_errors))

        # update the weights for the links between the hidden and output layer
        self.who += self.lr * np.dot((hidden_outputs), (output_errors * final_outputs * (1.0 - final_outputs)))
        
        # sum_bho = self.lr * output_errors * final_outputs * (1.0 - final_outputs)
        # self.bho += np.sum(sum_bho) / 4

        self.bho += self.lr * output_errors * final_outputs * (1.0 - final_outputs)
               
        # update the weights for the links between the input and hidden layer
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        # sum_bih = self.lr * hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        # self.bih[0][0] += np.sum(sum_bih[0][:]) / 4
        # self.bih[1][0] += np.sum(sum_bih[1][:]) / 4
        
        self.bih += self.lr * hidden_errors * hidden_outputs * (1.0 - hidden_outputs)

        self.count += 1

        if (self.count > 3):
            self.error_sum = self.error_sum / 4
            self.error_list = np.append(self.error_list, self.error_sum)
            self.count = 0
            self.error_sum = 0     
        pass

     # quaery the NN
    def query(self, input_list):
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih
        hidden_outputs = activation_func(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(np.transpose(hidden_outputs),self.who) + self.bho
        final_outputs = activation_func(final_inputs)

        return final_outputs

    def print_weight(self):
        print("Wih=", self.wih)
        print("Who=", self.who)
        print("bih=", self.bih)
        print("bho=", self.bho)
    
    def plot_error(self):        
        plt.figure()
        plt.xlabel("İterasyon")
        plt.ylabel("Hatanın Değişimi")
        plt.grid(True)        
        plt.plot(self.error_list,label="Error", color='C0', lw=3)      
        plt.legend()
        plt.show()

# initialise NN parameters
input_nodes = 2
hidden_nodes = 2
output_nodes = 1

learning_rate = 0.1
inp_val = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
out_val = np.array([[0.0], [1.0], [1.0], [0.0]])
# create instance of NN
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#print(n.query([0.0, 0.0]))
print(n.query(inp_val))
n.print_weight()

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 5000
t = 0

for e in range(epochs):
    for x in inp_val:
        inputs = x
        target = out_val[t]
        n.train(inputs, target)
        
        t += 1
        if t > len(out_val)-1:
            t=0
   

#print(n.query([0.0, 0.0]))
print(n.query(inp_val))
n.print_weight()
n.plot_error()