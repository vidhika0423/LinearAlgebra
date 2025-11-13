# basic neural network

import numpy as np

input_size = 3
hidden_size = 4
output_size = 1

# initialize weights and biases randomly
np.random.seed(42)      #for reproducibility
W1 =np.random.randn(hidden_size, input_size) * 0.01     #shape (4,3)
b1 = np.zeros((hidden_size,1))                          #shape (4,1)
W2 = np.random.randn(output_size, hidden_size)          #shape (1,4)
b2 = np.zeros((output_size,1))                          #shape (1,1)

# activation function 
def sigmoid(z):
    """sigmoid function for output layer"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU for hidden layer"""
    return np.maximum(0,z)

# forward propagation 
def forward_propagation(X):
    """
    perform forward pass through the network 
    X -> Input data,shape (input size, number of samples)
    """
    Z1 = np.dot(W1, X) + b1                 #linear step for layer 1
    A1 = relu(Z1)                           # activation (ReLu)
    Z2 = np.dot(W2, A1) + b2                # linear step for layer 2
    A2 = sigmoid(Z2)                        # activation (sigmoid)
    return Z1, A1, Z2, A2

# loss
def compute_loss(A2, Y):
    """
    Cross-entropy loss for binary classification
    Y -> true label, shape (1, number of samples)
    """
    m = Y.shape[1]                          # number of examples
    cost = -(1/m) * np.sum(Y*np.log(A2) + (1 - Y)*np.log(1 - A2))
    return np.squeeze(cost)                 #scalar cost

# backward propagation 
def backward_propagation(X, Y, Z1, A1, A2):
    """
    compute gradiants to update weights and biases
    """

    m= X.shape[1]
    dZ2 = A2 - Y                            #derivative of loss wrt Z2
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)                 # derivative wrt A1
    dZ1 = dA1 * (Z1>0)
    dW1= (1/m) * np.dot(dZ1, X.T)
    db1= (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2 

# update parameters 
def update_parameters(W1, b1, W2, b2, dW1, db1,dW2,db2, learning_rate=0.01):
    """
    gradiant descent step to update weights and biases
    """
    W1 -= learning_rate*dW1
    b1 -= learning_rate* db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2
    return W1,b1,W2,b2

# data simulation
# each column is training example
X = np.random.randn(3,5)
Y = np.array([[1,0,1,0,1]])

# train the network 
for i in range(1000):
    Z1,A1,Z2,A2 = forward_propagation(X)
    cost = compute_loss(A2,Y)
    dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, A2)
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2)

    if 1%100 == 0:
        print(f'Iteration {i} | cost: {cost:.4f}')
    
# make predictions
predictions = (A2>0.5).astype(int)
print('\nPredictions:', predictions)

