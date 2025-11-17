# loss calculation with categorical cross entropy 

import math 
import numpy as np 

# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1,0,0]

# loss = -(math.log(softmax_output[0])*target_output[0] +
#          math.log(softmax_output[1])*target_output[1] +
#          math.log(softmax_output[2])*target_output[2])
# print(loss)

"""
classes = [0,1]
op= [0,1,1]
"""

softmax_output = np.array([[0.7,0.1,0.2],
                           [0.1,0.5,0.4],
                           [0.02,0.9,0.08]])
class_target = [0,1,1]

"""
print(softmax_output[[0,1,2], class_target]) 
# sotmax_output[[list no(0,1,2)], inexing into each list(0,1,1)]
# result = [(0,0) , (1,1) , (2,1)]
"""

# loss = -(np.log(softmax_output[range(len(softmax_output)), class_target]))
# print(loss)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1, keepdims=True)
        negative_log_likelihoods = -np.log(correct_confidences)

loss_function = Loss_CategoricalCrossEntropy()
# loss = loss_function.calculate(y' ,y)