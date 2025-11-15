# batch , layers
#  batches runs code in parallel ; helps in generalization 

import numpy as np 
inputs = [[1.6,2,3.2,4],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]      #shape = (3,4)

weights = [[0.2, 0.8, -0.5, 0.1],
           [0.5, -0.91, -0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]      #shape = (3,4)

biases = [2, 3, 0.5]


output1 = np.dot(inputs, np.array(weights).T) + biases
print(output1)

weights2 = [[0.25, 0.85, -0.65],
           [0.5, -0.1, -0.6],
           [-0.26, -0.2, 0.37]]

biases2 = [-1, 3, 0.25]


layer1_outputs = np.dot(inputs, np.array(weights).T) + biases     
# op_layer1_shape = (3,3)

# shape of weights2 = (3,3)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)
