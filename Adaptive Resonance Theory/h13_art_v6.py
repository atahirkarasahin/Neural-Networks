from cmath import sqrt
import math
import matplotlib.pyplot as plt
import numpy as np

c=0
temp = 0
input_data = 63

class adaptiveResonanceTheory:
    # initialise the ART
    def __init__(self, inputNodes, outputNodes, learningRate, vigilance_parameter):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputNodes
        self.onodes = outputNodes
        self.lr = learningRate
        self.p = vigilance_parameter
        self.yj = 1
        self.Oj = 1
        maxValue = self.lr / (self.lr - 1 + self.inodes)
        #maxValue = 1 / (1 + self.inodes)
        # bottom-up weight
        #self.bij = np.random.uniform(size=self.inodes * self.onodes, low=minValue, high=maxValue).reshape(self.inodes, self.onodes)
        self.bij = np.zeros(self.inodes*self.onodes).reshape(self.inodes, self.onodes)
        self.bij[:,:] = maxValue
        # top-down weight
        self.tji = np.ones(self.inodes*self.onodes).reshape(self.onodes,self.inodes)
    
    def querry(self, input_list):
        return np.argmax(np.dot(input_list,self.bij))
    
    def train(self, input_list):
        s_sum = np.sum(input_list)
        yj = 0
        if self.Oj != -1:
            yj = np.dot(input_list,self.bij)
            winner_weight = np.argmax(yj) 
            #xi = np.dot(input_list,np.transpose(self.tji[winner_weight,:])) # ?
            xi = input_list * np.transpose(self.tji[winner_weight,:])
            xi_sum = np.sum(xi)

            if s_sum == 0:
                s_sum = 0.000001

            if (xi_sum / s_sum) >= self.p:
                self.bij[:,winner_weight] = (self.lr * np.transpose(xi))/(self.lr - 1 + xi_sum)
                self.tji[winner_weight,:] = xi    
            else:
                self.Oj = -1  
                c = -2
                while(self.Oj == -1):
                    new_list = np.sort(yj)
                    new_winner = new_list[c]
                    
                    for y in range(len(yj)):
                        if yj[y] == new_winner:
                            new_winner = y
                            break
                                        
                    xi = input_list * np.transpose(self.tji[new_winner,:])
                    xi_sum = np.sum(xi)
                    if s_sum == 0:
                        s_sum = 0.000001
                    if (xi_sum / s_sum) >= self.p:
                        self.bij[:,new_winner] = (self.lr * np.transpose(xi))/(self.lr - 1 + xi_sum)
                        self.tji[new_winner,:] = xi
                        self.Oj = 1
                    c -= 1
                    if c < -3:
                        self.Oj = 1
        
      
# input_neuron number
n = 63
# cluster output neuron
m = 3
# vigilance parameter
p = 0.6
# learning rate
L = 2

nn = adaptiveResonanceTheory(n, m, L, p)

# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 100

# load the wall-robot training data CSV file into a list
training_data_file = open("data/grayscale_binary_3.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(epochs):
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[:-1]))        
        nn.train(inputs)

t = 0
count = 0
result_list = []

test_data_file = open("data/grayscale_binary_3_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[:-1]))  
    result = nn.querry(inputs)

    result_list.append(result)  

    target = all_values[-1]    

print(result_list)
#print(nn.bij)