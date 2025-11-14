# code a neuron layer
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 0.1]
weights2 = [0.5, -0.91, -0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5


# for 1 neuron
#  op = sum(inputs . weights) + bias
# output = inputs[0]*weights[0] +  inputs[1]*weights[1] +  inputs[2]*weights[2]+  inputs[3]*weights[3] + bias
# print(output)

# 3 neurons with 4 inputs
# input set will remain same for all 3 neuron , i.e set of 4 inputs
# for each neuron there will b diff weights for each output
# each neuron will have 4 weights (one for each input)
# as we have 3 neuron , each have 4 weights set 
# shape of weights (3, 4)

output = [inputs[0]*weights1[0] +  inputs[1]*weights1[1] +  inputs[2]*weights1[2]+  inputs[3]*weights1[3] + bias1, 
          inputs[0]*weights2[0] +  inputs[1]*weights2[1] +  inputs[2]*weights2[2]+  inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] +  inputs[1]*weights3[1] +  inputs[2]*weights3[2]+  inputs[3]*weights3[3] + bias3]
print(output)

