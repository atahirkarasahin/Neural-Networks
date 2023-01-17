import numpy as np

class NeuralNetwork:
    def __init__(self,learningRate):
      

        self.wih = [[ 0.875707, -0.935321, 0.147852],[0.54123, 0.407679, -0.258963]]
        self.who = [[ -0.591154, 0.760786]]
        self.bih = [[0.528915], [0.415760]]
        self.bho = [[0.705415]]
      
        self.lr = learningRate

    # activation function is sigmoid
    def activation_func(self, x):
        return 1 / (1 + np.exp(-x))
    
    def print_weight(self):
        return (print("Wih:",self.wih, "Bh:",self.bih, "Who:", self.who,"Bo:",self.bho))

    # train the NN
    def train(self, input_list, target_list):
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2)

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, np.transpose(inputs)) + self.bih
        hidden_outputs = self.activation_func(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(np.transpose(hidden_outputs),np.transpose(self.who)) + self.bho
        final_outputs = self.activation_func(final_inputs)

        ### v2 multi layer network ###
        E_m = targets - final_outputs

        delta_m = final_outputs * (1 - final_outputs) * E_m
        hidden_error = np.dot(np.transpose(self.who), delta_m)

        delta_Ajm = np.dot(hidden_outputs, self.lr * delta_m)  
        self.who += np.transpose(delta_Ajm)

        delta_Bm = self.lr * delta_m
        self.bho += delta_Bm

        delta_j = hidden_outputs * (1 - hidden_outputs) * hidden_error 

        delta_Akj = np.dot(self.lr * delta_j, inputs)
        self.wih += delta_Akj

        delta_Bj = self.lr * delta_j
        self.bih += delta_Bj


    def query(self, input_list):
        # convert input list to 2d array
        inputs = np.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs) + self.bih
        hidden_outputs = self.activation_func(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(np.transpose(hidden_outputs),np.transpose(self.who)) + self.bho
        final_outputs = self.activation_func(final_inputs)

        return final_outputs


input = np.array([[1.0,0.0,1.0]]).T
target  = np.array([[1.0]])

n = NeuralNetwork(0.5)
n.train(input,target)
n.print_weight()
