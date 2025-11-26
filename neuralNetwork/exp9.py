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
print(softmax_output[[0,1,2], class_target]) zipping
# sotmax_output[[list of index], interested array] 
# result = [(0,0) , (1,1) , (2,1)] -> [0.7, 0.5, 0.9] -> confidence
"""

# loss = -(np.log(softmax_output[range(len(softmax_output)), class_target])) -> -log on each item in list
# print(loss) 

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)  #output-from model, y=target, forward=type of loss
        data_loss = np.mean(sample_losses)
        return data_loss
    
# loss function for regression losses


class Loss_MeanSquareError(Loss):
    def forward(self, y_pred, y_true):
        errors = (y_pred - y_true)**2
        # number of elements
        samples = errors.size
        loss = errors.sum() / samples
        return loss


class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        error = np.subtract(y_pred, y_true)
        error = np.abs(error)
        # number of elements
        samples = len(error.flatten())
        loss = np.sum(error) / samples
        return loss

# huber loss -> MSE when errors are small ->MAE when errors are large 
# If |e| ≤ δ → Small error
class Loss_huberLoss(Loss):
    def forward(self, y_pred, y_true):
        error = np.subtract(y_pred, y_true)
        abs_error= np.abs(error)
        delta = np.median(abs_error)
        # masks
        small_mask = abs_error <= delta
        large_mask = abs_error > delta
        # select only small_error elements
        if np.any(small_mask):
            y_pred_small = y_pred[small_mask]
            y_true_small = y_true[small_mask]
            loss_mse = Loss_MeanSquareError().forward(y_pred_small, y_true_small)
        else:
            loss_mse = 0.0

        # select only large_error elements
        if np.any(large_mask):
            y_pred_large = y_pred[large_mask]
            y_true_large = y_true[large_mask]
            loss_mae = Loss_MeanAbsoluteError().forward(y_pred_large, y_true_large)
        else:
            loss_mae = 0.0

        total_elements = len(y_pred.flatten())
        num_small = np.sum(small_mask)
        num_large = np.sum(large_mask)

        loss = ( loss_mse*num_small + loss_mae*num_large) / total_elements
        return loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:      #class's scalar value
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:        #one hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1, keepdims=True)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



loss_function = Loss_CategoricalCrossEntropy()
# loss = loss_function.calculate(y' ,y)