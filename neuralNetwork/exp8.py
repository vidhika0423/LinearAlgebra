# softmax - exponential function 
import numpy as np 
import nnfs
nnfs.init()

layer_ouput = [[4.8, 1.21, 2.385],
               [8.9, -1.81, 0.2],
               [1.41, 1.051, 0.026]]


""" method 1"""
# exp_values = np.exp(layer_ouput)
# feature_sum = np.sum(exp_values, axis=1, keepdims=True)
# norm_values=np.zeros(np.shape(layer_ouput))

# for i in range(len(layer_ouput)):
#     for j in range(len(layer_ouput[i])):
#         norm_values[i][j] = exp_values[i][j] / feature_sum[i][0]


""" method 2"""
# exp_values = np.exp(layer_ouput)
# norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

""" softmax activation class"""
class activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = norm_values

"""
layer1 = dense_layer(2,5)
activation1 = activation_ReLU()

layer2 = dense_layer(5,5)
activation2 = activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)
"""